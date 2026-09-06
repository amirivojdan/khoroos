"""Action recognition: V-JEPA2 ViT-L fine-tuned on the ChickenAct ethogram.

The model is a frozen-encoder linear probe over 64-frame, 256 px clips. Frame sampling here
must match training exactly (uniform indices over the clip, last-frame padding when short) —
any mismatch silently degrades accuracy rather than raising.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from khoroos.config import Settings, get_settings
from khoroos.interfaces import VideoClassifier
from khoroos.models.metadata import labels_from_mapping
from khoroos.models.registry import resolve_checkpoint
from khoroos.models.selection import ClassSelection

logger = logging.getLogger(__name__)

#: Fallback when the checkpoint config does not declare it.
DEFAULT_FRAMES_PER_CLIP = 64


def sample_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    """Uniformly sample ``num_frames`` indices from ``total_frames``.

    Identical to the training-time sampler: when the clip is shorter than the model's
    window, the final frame is repeated rather than looping.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if total_frames <= 0:
        raise ValueError("Clip has no frames.")
    if total_frames >= num_frames:
        return np.round(np.linspace(0, total_frames - 1, num_frames)).astype(int)
    indices = np.arange(total_frames)
    pad = np.full(num_frames - total_frames, total_frames - 1)
    return np.concatenate([indices, pad])


class ActionRecognizer(VideoClassifier):
    """Classifies clips using checkpoint labels, optionally selecting an ordered subset."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        settings: Settings | None = None,
        device: str | None = None,
        classes: list[str] | None = None,
    ) -> None:
        from transformers import AutoModelForVideoClassification, AutoVideoProcessor

        self.settings = (settings or get_settings()).with_overrides(device=device)
        self.checkpoint = resolve_checkpoint("action", checkpoint, self.settings)
        self.device = torch.device(self.settings.resolved_device())

        self.processor = AutoVideoProcessor.from_pretrained(
            str(self.checkpoint), local_files_only=True,
        )
        model = AutoModelForVideoClassification.from_pretrained(
            str(self.checkpoint), local_files_only=True,
        )
        self.model = model.to(self.device).eval()

        id2label = self.model.config.id2label
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.classes = labels_from_mapping(id2label)
        self.selection = ClassSelection(self.classes, classes)
        self.classes = self.selection.classes
        self.num_frames = int(
            getattr(self.model.config, "frames_per_clip", DEFAULT_FRAMES_PER_CLIP)
        )
        if self.num_frames < 1:
            raise ValueError("Checkpoint frames_per_clip must be positive")

    def _autocast(self):
        if self.settings.use_bf16 and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @torch.inference_mode()
    def classify(self, clips: list[torch.Tensor]) -> np.ndarray:
        """Classify a batch of clips.

        Each clip is a ``(T, C, H, W)`` uint8 tensor. Clips are resampled to the model's
        frame count before preprocessing.

        Returns an ``(N, num_classes)`` float32 array of softmax probabilities.
        """
        if not clips:
            return np.zeros((0, len(self.classes)), dtype=np.float32)

        resampled = []
        for clip in clips:
            if clip.ndim != 4:
                raise ValueError(f"Expected clip of shape (T, C, H, W), got {tuple(clip.shape)}")
            idx = sample_frame_indices(clip.shape[0], self.num_frames)
            resampled.append(clip[torch.as_tensor(idx, dtype=torch.long)])

        inputs = self.processor(resampled, return_tensors="pt").to(self.device)
        with self._autocast():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits.float(), dim=-1)
        return self.selection.apply(probs.cpu().numpy(), len(clips))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ActionRecognizer(checkpoint={self.checkpoint}, device={self.device}, "
            f"classes={len(self.classes)})"
        )
