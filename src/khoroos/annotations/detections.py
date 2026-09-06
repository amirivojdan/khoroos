"""Canonical detections and explicit normalization of legacy detector outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

InvalidBoxes = Literal["error", "drop"]


def _arrays(boxes, scores, class_ids):
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),):
        raise ValueError("Expected boxes (N, 4) and scores (N,)")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite and in [0, 1]")
    ids = np.zeros(len(boxes), dtype=np.int64) if class_ids is None else np.asarray(class_ids)
    if (
        ids.shape != (len(boxes),)
        or not np.isfinite(ids).all()
        or np.any(ids < 0)
        or np.any(ids != np.floor(ids))
    ):
        raise ValueError("class_ids must be nonnegative integers with shape (N,)")
    return boxes, scores, ids.astype(np.int64)


def _valid_boxes(boxes):
    return np.isfinite(boxes).all(axis=1) & (boxes[:, 2:] > boxes[:, :2]).all(axis=1)


def _readonly(array):
    # A bytes-backed copy cannot be changed through input aliases or setflags(write=True).
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class Detections:
    """Immutable pixel xyxy boxes, scores, and nonnegative integer category IDs.

    Unpacking yields boxes and scores for the single-population tracking interface.
    dropped_count records boxes explicitly discarded during normalization; it is never
    inferred from confidence filtering or NMS. Array shapes, scores and category IDs are
    always strict. Geometry can be discarded only via from_raw(invalid_boxes="drop").
    """

    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray | None = None
    dropped_count: int = 0

    def __post_init__(self):
        boxes, scores, ids = _arrays(self.boxes, self.scores, self.class_ids)
        if not _valid_boxes(boxes).all():
            raise ValueError("Boxes must be finite with positive width and height")
        if not isinstance(self.dropped_count, int) or self.dropped_count < 0:
            raise ValueError("dropped_count must be a nonnegative integer")
        for name, array in (("boxes", boxes), ("scores", scores), ("class_ids", ids)):
            object.__setattr__(self, name, _readonly(array))

    @classmethod
    def from_raw(
        cls, boxes, scores, class_ids=None, *, invalid_boxes: InvalidBoxes = "error"
    ) -> Detections:
        """Normalize raw arrays, optionally dropping invalid geometry with a recorded count."""
        if invalid_boxes not in ("error", "drop"):
            raise ValueError("invalid_boxes must be 'error' or 'drop'")
        if invalid_boxes == "error":
            return cls(boxes, scores, class_ids)
        boxes, scores, ids = _arrays(boxes, scores, class_ids)
        valid = _valid_boxes(boxes)
        return cls(boxes[valid], scores[valid], ids[valid], dropped_count=int((~valid).sum()))

    @classmethod
    def coerce(cls, value, *, invalid_boxes: InvalidBoxes = "error") -> Detections:
        """Accept canonical detections or a legacy (boxes, scores) pair at the boundary.

        Canonical objects are returned unchanged, preserving category IDs and drop counts.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("Detector output must be Detections or a (boxes, scores) pair")
        return cls.from_raw(*value, invalid_boxes=invalid_boxes)

    def __iter__(self):
        yield self.boxes
        yield self.scores

    def __len__(self):
        return len(self.boxes)
