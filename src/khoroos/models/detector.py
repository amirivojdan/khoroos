"""Chicken detector: RT-DETRv2 fine-tuned on ChickenDet.

Ported from the original ``RTR.py`` script with three changes:

* the box-padding clamp used swapped height/width axes (``target_sizes`` is ``[h, w]``),
  so padded boxes near frame edges were clamped against the wrong dimension;
* inference runs under ``torch.inference_mode`` with optional bfloat16 autocast;
* detections are returned as numpy arrays rather than tensors, since nothing downstream
  needs autograd and the tracker is numpy-based.

Frames arrive from :class:`~khoroos.video.reader.VideoSource` as ``(C, H, W)`` uint8
tensors, and modern image processors are torchvision-backed — they resize tensors
natively. Converting those tensors to PIL images first, only for the processor to convert
them back cost more than the resize itself, and the cost grew with the *source* resolution
even though the model always sees 640x480. :meth:`ChickenDetector.detect` now hands tensors
straight to the processor when it can take them, and keeps the PIL path for everything
else: measured 1.3x / 1.7x / 2.9x on the whole detect stage for 720p / 1080p / 4K input.

The two paths are bit-identical — verified equal ``pixel_values``, equal model logits, and
equal boxes and scores, not merely close ones. That equality is why the resize stays on the
host: letting the processor resize on an accelerator is several times faster again, but its
bilinear kernel differs from the CPU one by one uint8 step, which moves logits enough to
change how many birds are detected.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

from khoroos.annotations import Detections
from khoroos.config import Settings, get_settings
from khoroos.interfaces import Detector
from khoroos.models.registry import resolve_checkpoint

logger = logging.getLogger(__name__)


def to_pil(image: Image.Image | np.ndarray | torch.Tensor) -> Image.Image:
    """Coerce a frame to a PIL RGB image.

    Handles the three shapes that reach the detector: PIL images, ``(H, W, C)`` numpy
    arrays, and the ``(C, H, W)`` uint8 tensors the video reader yields.
    """
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu()
        if array.ndim == 3 and array.shape[0] in (1, 3):  # CHW -> HWC
            array = array.permute(1, 2, 0)
        array = array.numpy()
    else:
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[2] not in (1, 3):
            array = np.transpose(array, (1, 2, 0))

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    return Image.fromarray(array).convert("RGB")


def as_uint8_frames(images) -> torch.Tensor | list[torch.Tensor] | None:
    """Return ``images`` unchanged if a tensor-native processor can take them, else None.

    The conditions are exactly the cases where feeding the processor directly produces
    what :func:`to_pil` would have produced: ``(C, H, W)`` uint8 RGB tensors. Anything
    else — float tensors needing the clip-and-cast, single-channel frames needing the
    grayscale expansion, numpy arrays, PIL images — goes down the PIL path, because for
    those the conversion is doing real work rather than a round trip.

    A batched ``(B, C, H, W)`` tensor is returned as-is: the processor handles it, and it
    is marginally faster than the equivalent list because nothing has to be re-stacked.
    """
    if isinstance(images, torch.Tensor):
        if images.ndim != 4 or images.dtype != torch.uint8 or images.shape[1] != 3:
            return None
        return images if images.device.type == "cpu" else images.cpu()

    frames: list[torch.Tensor] = []
    for image in images:
        if (
            not isinstance(image, torch.Tensor)
            or image.ndim != 3
            or image.dtype != torch.uint8
            or image.shape[0] != 3
        ):
            return None
        # Matching to_pil, which moves to host before conversion. Resizing on an
        # accelerator is *not* equivalent: its bilinear rounding differs from the CPU
        # kernel by one uint8 step, which measurably moves detector logits.
        frames.append(image if image.device.type == "cpu" else image.cpu())
    return frames


class ChickenDetector(Detector):
    """Batched single-class chicken detector."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        settings: Settings | None = None,
        device: str | None = None,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.settings = (settings or get_settings()).with_overrides(device=device)
        self.checkpoint = resolve_checkpoint("detector", checkpoint, self.settings)
        self.device = torch.device(self.settings.resolved_device())

        self.processor = AutoImageProcessor.from_pretrained(
            str(self.checkpoint), local_files_only=True,
        )
        model = AutoModelForObjectDetection.from_pretrained(
            str(self.checkpoint), local_files_only=True,
        )
        self.model = model.to(self.device).eval()
        self.id2label = dict(self.model.config.id2label)

    # -- internals ---------------------------------------------------------

    @cached_property
    def processor_takes_tensors(self) -> bool:
        """Whether this processor resizes tensors itself, so PIL can be skipped.

        Torchvision-backed ("fast") processors do; the older PIL-backed ones do not and
        would convert back to PIL internally, gaining nothing. Resolved from the instance
        rather than in ``__init__`` so it also works for a detector assembled around an
        injected processor. ``backend`` is the current attribute; ``is_fast`` is the
        pre-5.x spelling and is read only as a fallback, since touching it on a modern
        processor emits a deprecation warning.
        """
        backend = getattr(self.processor, "backend", None)
        if backend is not None:
            return backend == "torchvision"
        return bool(getattr(self.processor, "is_fast", False))

    def _preprocess(self, images):
        """Return ``(model_inputs, [(height, width), ...])`` for a batch of frames."""
        frames = as_uint8_frames(images) if self.processor_takes_tensors else None
        if frames is None:
            pil_images = [to_pil(image) for image in images]
            sizes = [(image.height, image.width) for image in pil_images]
            return self.processor(images=pil_images, return_tensors="pt"), sizes
        sizes = [(int(frame.shape[-2]), int(frame.shape[-1])) for frame in frames]
        return self.processor(images=frames, return_tensors="pt"), sizes

    def _autocast(self):
        if self.settings.use_bf16 and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def _pad_boxes(boxes: torch.Tensor, padding: float, height: int, width: int) -> torch.Tensor:
        """Grow each box by ``padding`` of its own size, clamped to the frame.

        The original implementation clamped x against the image height and y against the
        width; this version keeps the axes straight.
        """
        if boxes.numel() == 0:
            return boxes
        padding = max(0.0, padding)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        padded = boxes.clone()
        padded[:, 0] = torch.clamp(boxes[:, 0] - widths * padding, min=0, max=float(width))
        padded[:, 1] = torch.clamp(boxes[:, 1] - heights * padding, min=0, max=float(height))
        padded[:, 2] = torch.clamp(boxes[:, 2] + widths * padding, min=0, max=float(width))
        padded[:, 3] = torch.clamp(boxes[:, 3] + heights * padding, min=0, max=float(height))
        return padded

    # -- public API --------------------------------------------------------

    @torch.inference_mode()
    def detect(
        self,
        images: list[Image.Image] | list[np.ndarray] | list[torch.Tensor] | torch.Tensor,
        confidence_threshold: float = 0.30,
        nms_threshold: float = 0.60,
        box_padding: float = 0.0,
    ) -> list[Detections]:
        """Detect chickens in a batch of frames.

        Accepts PIL images, ``(H, W, C)`` numpy arrays, ``(C, H, W)`` uint8 tensors as
        produced by :class:`~khoroos.video.reader.VideoSource`, or one batched
        ``(B, C, H, W)`` uint8 tensor. Tensor input skips the PIL round trip when the
        processor can resize tensors itself; see :func:`as_uint8_frames`.

        Returns one ``Detections`` per image, unpackable as ``(boxes, scores)``. Boxes are an
        ``(N, 4)`` float32 array in ``xyxy`` pixel coordinates.
        """
        # A batched tensor is kept batched: splitting it here would force the fast path
        # to re-stack it, which costs more than it saves.
        if len(images) == 0:
            return []

        inputs, sizes = self._preprocess(images)
        target_sizes = torch.tensor(sizes, device=self.device)

        inputs = inputs.to(self.device)
        with self._autocast():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, target_sizes=target_sizes
        )

        out: list[Detections] = []
        for index, result in enumerate(results):
            boxes = result["boxes"].float()
            scores = result["scores"].float()

            labels = result["labels"]
            # Invalid raw geometry must not enter NMS, even if clamping would hide it.
            valid = torch.isfinite(boxes).all(dim=1) & (boxes[:, 2:] > boxes[:, :2]).all(dim=1)
            dropped = int((~valid).sum())
            boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
            keep = nms(boxes, scores, iou_threshold=nms_threshold)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            height, width = sizes[index]
            boxes = self._pad_boxes(boxes, box_padding, height, width)
            inside = (boxes[:, 2:] > boxes[:, :2]).all(dim=1)
            dropped += int((~inside).sum())
            out.append(
                Detections(
                    boxes[inside].cpu().numpy(), scores[inside].cpu().numpy(),
                    labels[inside].cpu().numpy(), dropped_count=dropped,
                )
            )
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ChickenDetector(checkpoint={self.checkpoint}, device={self.device})"
