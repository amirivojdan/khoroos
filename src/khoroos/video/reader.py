"""Video decoding built on torchcodec.

Chosen over OpenCV deliberately: no aarch64 wheel friction, it is already a dependency of
the training pipeline, and it gives cheap random access via ``get_frames_at`` — which the
tracklet stage needs to pull scattered frames without re-decoding the whole file.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from khoroos.pipeline.types import VideoInfo

logger = logging.getLogger(__name__)


class VideoReadError(RuntimeError):
    """Raised when a video cannot be opened or decoded."""


class VideoSource:
    """Random-access reader over a video file.

    Frames are returned as ``(C, H, W)`` uint8 tensors in RGB, matching what the
    training-time decoder produced.
    """

    def __init__(self, path: str | Path) -> None:
        try:
            from torchcodec.decoders import VideoDecoder
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise VideoReadError(
                "torchcodec is required to decode video. It needs FFmpeg shared libraries "
                "on the system; install FFmpeg and reinstall torchcodec."
            ) from exc

        self.path = Path(path)
        if not self.path.exists():
            raise VideoReadError(f"Video not found: {self.path}")

        try:
            self._decoder = VideoDecoder(str(self.path))
        except Exception as exc:
            raise VideoReadError(f"Could not open video {self.path}: {exc}") from exc

        meta = self._decoder.metadata
        fps = float(meta.average_fps or 0.0)
        if fps <= 0:
            raise VideoReadError(f"Video {self.path} reports an invalid frame rate.")

        num_frames = int(meta.num_frames or 0)
        if num_frames <= 0:
            raise VideoReadError(f"Video {self.path} reports no frames.")

        height = int(meta.height)
        width = int(meta.width)
        duration = float(meta.duration_seconds or (num_frames / fps))

        self.info = VideoInfo(
            filename=self.path.name,
            fps=fps,
            duration_seconds=duration,
            width=width,
            height=height,
            num_frames=num_frames,
        )

    # -- properties --------------------------------------------------------

    @property
    def fps(self) -> float:
        return self.info.fps

    @property
    def num_frames(self) -> int:
        return self.info.num_frames

    @property
    def width(self) -> int:
        return self.info.width

    @property
    def height(self) -> int:
        return self.info.height

    def time_of(self, frame_index: int) -> float:
        """Presentation time in seconds for a frame index."""
        return frame_index / self.info.fps

    # -- access ------------------------------------------------------------

    def get_frames(self, indices: list[int] | np.ndarray) -> torch.Tensor:
        """Fetch specific frames as a ``(T, C, H, W)`` uint8 tensor."""
        idx = [int(i) for i in indices]
        if not idx:
            return torch.zeros((0, 3, self.height, self.width), dtype=torch.uint8)
        clamped = [max(0, min(self.num_frames - 1, i)) for i in idx]
        return self._decoder.get_frames_at(indices=clamped).data

    def get_frame(self, index: int) -> torch.Tensor:
        """Fetch a single frame as a ``(C, H, W)`` uint8 tensor."""
        return self.get_frames([index])[0]

    def iter_batches(
        self,
        batch_size: int = 16,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> Iterator[tuple[list[int], torch.Tensor]]:
        """Iterate the video in batches of every ``stride``-th frame.

        Yields ``(frame_indices, frames)`` where ``frames`` is ``(B, C, H, W)`` uint8.
        """
        if stride < 1:
            raise ValueError("stride must be >= 1")
        limit = self.num_frames if max_frames is None else min(self.num_frames, max_frames)
        all_indices = list(range(0, limit, stride))

        for start in range(0, len(all_indices), batch_size):
            batch = all_indices[start : start + batch_size]
            if not batch:
                continue
            yield batch, self.get_frames(batch)

    def close(self) -> None:
        self._decoder = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup on failed pipelines
        self.close()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"VideoSource({self.path.name}, {self.width}x{self.height}, "
            f"{self.fps:.2f}fps, {self.num_frames} frames)"
        )


def frames_to_numpy(frames: torch.Tensor) -> np.ndarray:
    """Convert ``(T, C, H, W)`` uint8 tensor to ``(T, H, W, C)`` numpy for PIL/encoding."""
    return frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()
