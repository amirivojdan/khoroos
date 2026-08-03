"""Chicken detector: RT-DETRv2 fine-tuned on ChickenDet.

Ported from the original ``RTR.py`` script with three changes:

* the box-padding clamp used swapped height/width axes (``target_sizes`` is ``[h, w]``),
  so padded boxes near frame edges were clamped against the wrong dimension;
* inference runs under ``torch.inference_mode`` with optional bfloat16 autocast;
* detections are returned as numpy arrays rather than tensors, since nothing downstream
  needs autograd and the tracker is numpy-based.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

from khoroos.config import Settings, get_settings
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


class ChickenDetector:
    """Batched single-class chicken detector."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        settings: Settings | None = None,
        device: str | None = None,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.settings = settings or get_settings()
        self.checkpoint = resolve_checkpoint("detector", checkpoint, self.settings)
        self.device = torch.device(device or self.settings.resolved_device())

        self.processor = AutoImageProcessor.from_pretrained(str(self.checkpoint))
        model = AutoModelForObjectDetection.from_pretrained(str(self.checkpoint))
        self.model = model.to(self.device).eval()
        self.id2label = dict(self.model.config.id2label)

    # -- internals ---------------------------------------------------------

    def _autocast(self):
        if self.settings.use_bf16 and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def _pad_boxes(
        boxes: torch.Tensor, padding: float, height: int, width: int
    ) -> torch.Tensor:
        """Grow each box by ``padding`` of its own size, clamped to the frame.

        The original implementation clamped x against the image height and y against the
        width; this version keeps the axes straight.
        """
        if padding <= 0.0 or boxes.numel() == 0:
            return boxes
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        padded = boxes.clone()
        padded[:, 0] = torch.clamp(boxes[:, 0] - widths * padding, min=0)
        padded[:, 1] = torch.clamp(boxes[:, 1] - heights * padding, min=0)
        padded[:, 2] = torch.clamp(boxes[:, 2] + widths * padding, max=float(width))
        padded[:, 3] = torch.clamp(boxes[:, 3] + heights * padding, max=float(height))
        return padded

    # -- public API --------------------------------------------------------

    @torch.inference_mode()
    def detect(
        self,
        images: list[Image.Image] | list[np.ndarray] | list[torch.Tensor] | torch.Tensor,
        confidence_threshold: float = 0.30,
        nms_threshold: float = 0.60,
        box_padding: float = 0.0,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Detect chickens in a batch of frames.

        Accepts PIL images, ``(H, W, C)`` numpy arrays, or ``(C, H, W)`` uint8 tensors as
        produced by :class:`~khoroos.video.reader.VideoSource`.

        Returns one ``(boxes, scores)`` pair per input image, where ``boxes`` is an
        ``(N, 4)`` float32 array in ``xyxy`` pixel coordinates.
        """
        if isinstance(images, torch.Tensor):
            images = list(images)
        if len(images) == 0:
            return []

        pil_images = [to_pil(img) for img in images]
        target_sizes = torch.tensor(
            [(img.height, img.width) for img in pil_images], device=self.device
        )

        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        with self._autocast():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, target_sizes=target_sizes
        )

        out: list[tuple[np.ndarray, np.ndarray]] = []
        for index, result in enumerate(results):
            boxes = result["boxes"].float()
            scores = result["scores"].float()

            if boxes.numel() == 0:
                out.append(
                    (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32))
                )
                continue

            keep = nms(boxes, scores, iou_threshold=nms_threshold)
            boxes, scores = boxes[keep], scores[keep]

            height, width = pil_images[index].height, pil_images[index].width
            boxes = self._pad_boxes(boxes, box_padding, height, width)

            out.append(
                (
                    boxes.cpu().numpy().astype(np.float32),
                    scores.cpu().numpy().astype(np.float32),
                )
            )
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ChickenDetector(checkpoint={self.checkpoint}, device={self.device})"
