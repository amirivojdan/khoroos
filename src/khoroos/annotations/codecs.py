"""Per-image YOLO text and COCO JSON bounding-box interchange.

These codecs handle detection annotations, not segmentation or keypoints. COCO category
IDs are preserved explicitly; YOLO IDs are supplied by a caller-defined ordered mapping.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from khoroos.annotations.detections import Detections


def _size(width: int, height: int):
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")


def _bounds(detections: Detections, width: int, height: int):
    _size(width, height)
    boxes = detections.boxes
    if (
        np.any(boxes < -1e-4)
        or np.any(boxes[:, [0, 2]] > width + 1e-4)
        or np.any(boxes[:, [1, 3]] > height + 1e-4)
    ):
        raise ValueError("Annotation boxes must lie within image bounds")


class AnnotationCodec(ABC):
    """Implement encode/decode to add another per-image annotation protocol."""

    @abstractmethod
    def encode(self, detections: Detections, *, width: int, height: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def decode(self, payload: Any, *, width: int, height: int) -> Detections:
        raise NotImplementedError


class YoloCodec(AnnotationCodec):
    """Standard five-column YOLO labels: class cx cy w h, normalized to image size.

    category_ids maps YOLO's contiguous zero-based IDs to canonical category IDs. Standard
    YOLO labels have no scores: encoding drops scores and decoding assigns 1.0.
    """

    def __init__(self, category_ids: Sequence[int] | None = None):
        self.category_ids = None if category_ids is None else list(category_ids)
        if self.category_ids is not None and (
            not self.category_ids
            or len(set(self.category_ids)) != len(self.category_ids)
            or any(not isinstance(i, int) or i < 0 for i in self.category_ids)
        ):
            raise ValueError("category_ids must be unique nonnegative integers")

    def encode(self, detections, *, width, height) -> str:
        _bounds(detections, width, height)
        lines = []
        for box, category in zip(detections.boxes, detections.class_ids, strict=True):
            category = int(category)
            if self.category_ids is not None:
                category = self.category_ids.index(category)
            x1, y1, x2, y2 = box
            values = [
                (x1 + x2) / (2 * width),
                (y1 + y2) / (2 * height),
                (x2 - x1) / width,
                (y2 - y1) / height,
            ]
            lines.append(f"{category} " + " ".join(f"{v:.9g}" for v in values))
        return "\n".join(lines) + ("\n" if lines else "")

    def decode(self, payload: str, *, width, height) -> Detections:
        _size(width, height)
        boxes, ids = [], []
        for line in payload.splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError("YOLO rows must contain class cx cy width height")
            category = int(fields[0])
            if category < 0:
                raise ValueError("YOLO class IDs must be nonnegative")
            if self.category_ids is not None:
                if category >= len(self.category_ids):
                    raise ValueError(f"Unknown YOLO class ID {category}")
                category = self.category_ids[category]
            cx, cy, w, h = map(float, fields[1:])
            if not all(np.isfinite(v) and 0 <= v <= 1 for v in (cx, cy, w, h)):
                raise ValueError("YOLO coordinates must be finite and normalized to [0, 1]")
            boxes.append(
                [
                    (cx - w / 2) * width,
                    (cy - h / 2) * height,
                    (cx + w / 2) * width,
                    (cy + h / 2) * height,
                ]
            )
            ids.append(category)
        result = Detections(np.asarray(boxes).reshape(-1, 4), np.ones(len(boxes)), np.array(ids))
        _bounds(result, width, height)
        return result


class CocoCodec(AnnotationCodec):
    """COCO per-image annotation records with pixel [x, y, width, height] boxes.

    Use image_id to select records from a dataset's annotations list. IDs generated during
    encoding start at annotation_id; callers assembling datasets must allocate unique IDs.
    Scores are preserved when present, defaulting to 1.0 for ground-truth records.
    """

    def __init__(self, image_id: int = 0, annotation_id: int = 1):
        self.image_id = image_id
        self.annotation_id = annotation_id

    def encode(self, detections, *, width, height) -> list[dict]:
        _bounds(detections, width, height)
        records = []
        for i, (box, score, category) in enumerate(
            zip(detections.boxes, detections.scores, detections.class_ids, strict=True)
        ):
            x1, y1, x2, y2 = map(float, box)
            records.append(
                {
                    "id": self.annotation_id + i,
                    "image_id": self.image_id,
                    "category_id": int(category),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                    "score": float(score),
                }
            )
        return records

    def decode(self, payload: list[dict], *, width, height) -> Detections:
        _size(width, height)
        boxes, scores, ids = [], [], []
        for row in payload:
            if row["image_id"] != self.image_id:
                continue
            x, y, w, h = row["bbox"]
            boxes.append([x, y, x + w, y + h])
            scores.append(row.get("score", 1.0))
            ids.append(row["category_id"])
        result = Detections(np.asarray(boxes).reshape(-1, 4), np.array(scores), np.array(ids))
        _bounds(result, width, height)
        return result


def annotation_codec(name: str, **options) -> AnnotationCodec:
    """Select a built-in codec by name; custom codecs can be passed directly."""
    codecs = {"yolo": YoloCodec, "coco": CocoCodec}
    try:
        codec = codecs[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown annotation protocol {name!r}; choose yolo or coco") from None
    return codec(**options)
