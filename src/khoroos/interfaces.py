"""Public extension contracts. Implement algorithms here without checkpoint assumptions.

Model instances are reused across runs. Trackers and video readers are created per run.
All boxes crossing these interfaces use pixel xyxy coordinates; annotation formats belong
at the I/O boundary. Tensor annotations are imported only for type checking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from khoroos.annotations import AnnotationCodec, Detections
from khoroos.pipeline.types import Track, VideoInfo

if TYPE_CHECKING:
    import torch
    from PIL import Image


class Detector(ABC):
    """Implement ``detect``; return one Detections object per input RGB frame.

    Boxes have shape (N, 4), scores (N,), including empty frames. Implementations own
    preprocessing, confidence filtering, NMS and padding. Legacy tuple outputs are
    normalized only at the pipeline and annotation boundaries.
    """

    @abstractmethod
    def detect(
        self,
        images: Sequence[Image.Image | np.ndarray | torch.Tensor] | torch.Tensor,
        confidence_threshold: float = 0.30,
        nms_threshold: float = 0.60,
        box_padding: float = 0.0,
    ) -> list[Detections]:
        """Return detections in input order, in pixel xyxy coordinates."""
        raise NotImplementedError

    def detect_annotations(
        self,
        images,
        codec: AnnotationCodec,
        *,
        image_sizes: Sequence[tuple[int, int]],
        **options,
    ) -> list[Any]:
        """Detect and encode each image using a codec and its explicit (width, height).

        Tuple outputs get category ID zero. Return Detections to preserve model category IDs.
        """
        results = self.detect(images, **options)
        return [
            codec.encode(
                Detections.coerce(result),
                width=width,
                height=height,
            )
            for result, (width, height) in zip(results, image_sizes, strict=True)
        ]


class VideoClassifier(ABC):
    """Implement ``classify`` and declare classes in probability-column order.

    Input clips are RGB uint8 (T, C, H, W) tensors. Set num_frames to request temporal
    thinning during extraction, or leave None to consume all frames. Preprocessing and
    any additional sampling belong to the implementation.
    """

    classes: Sequence[str]
    num_frames: int | None = None

    @abstractmethod
    def classify(self, clips: list[torch.Tensor]) -> np.ndarray:
        """Return finite probabilities with shape (len(clips), len(classes))."""
        raise NotImplementedError

    def classify_batched(self, clips: list[torch.Tensor], batch_size: int = 4) -> np.ndarray:
        """Convenience batching shared by classifier implementations."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not clips:
            return np.zeros((0, len(self.classes)), dtype=np.float32)
        return np.concatenate(
            [self.classify(clips[i : i + batch_size]) for i in range(0, len(clips), batch_size)]
        )


class Tracker(ABC):
    """Implement update and finalize. Each instance processes exactly one video.

    Updates arrive only on detector frames, with source frame indices and timestamps.
    Final tracks must have unique IDs and observations ordered by source frame index.
    """

    lost_track_count: int = 0

    @abstractmethod
    def update(
        self,
        frame_index: int,
        time_seconds: float,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> list[tuple[int, np.ndarray]]:
        """Consume one frame; return confirmed (track_id, xyxy box) states."""
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> list[Track]:
        """Return all confirmed tracks after the last frame."""
        raise NotImplementedError


class VideoReader(ABC):
    """Random-access RGB reader. Implement frame access, batching, and cleanup."""

    info: VideoInfo

    def time_of(self, frame_index: int) -> float:
        return frame_index / self.info.fps

    @abstractmethod
    def get_frames(self, indices: list[int] | np.ndarray) -> torch.Tensor:
        """Return RGB uint8 frames as (T, C, H, W)."""
        raise NotImplementedError

    @abstractmethod
    def iter_batches(
        self,
        batch_size: int = 16,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> Iterator[tuple[list[int], torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


def component_name(component: object) -> str:
    """Optional explicit name, legacy checkpoint name, or implementation class name."""
    name = getattr(component, "name", None)
    if name:
        return str(name)
    checkpoint = getattr(component, "checkpoint", None)
    return Path(checkpoint).name if checkpoint is not None else type(component).__name__


def model_card(component: object) -> dict[str, Any]:
    """Custom models may expose a card dict; checkpoint-backed models retain file cards."""
    card = getattr(component, "model_card", None)
    if card is not None:
        return dict(card)
    checkpoint = getattr(component, "checkpoint", None)
    if checkpoint is None:
        return {}
    from khoroos.models.card import load_model_card

    return load_model_card(checkpoint)
