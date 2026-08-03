"""Multi-object tracker producing globally consistent bird IDs.

Unlike the dataset-generation script, which reset the tracker every 2-second window, this
runs a single tracker across the whole video so each bird keeps one ID throughout. That is
what makes per-bird timelines and per-bird behaviour budgets possible.

Association is IoU + Hungarian matching (SORT), with a constant-velocity Kalman filter
carrying tracks through frames where the detector was not run or missed a bird.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

from khoroos.pipeline.types import Track, TrackObservation
from khoroos.tracking.kalman import BoxKalmanFilter

logger = logging.getLogger(__name__)


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``xyxy`` boxes, shape ``(len(a), len(b))``."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a[:, None, :]  # (N, 1, 4)
    b = boxes_b[None, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)

    return (inter / (area_a + area_b - inter + 1e-9)).astype(np.float32)


class _ActiveTrack:
    """Internal bookkeeping for a track that is still being followed."""

    __slots__ = ("track_id", "kf", "observations", "hits", "age", "time_since_update")

    def __init__(self, track_id: int, box: np.ndarray) -> None:
        self.track_id = track_id
        self.kf = BoxKalmanFilter(box)
        self.observations: list[TrackObservation] = []
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def record(self, frame_index: int, time_seconds: float, box: np.ndarray, score: float) -> None:
        self.observations.append(
            TrackObservation(
                frame_index=frame_index,
                time_seconds=time_seconds,
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                score=float(score),
            )
        )


class BirdTracker:
    """SORT-style tracker over chicken detections."""

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_age: int = 15,
        min_hits: int = 3,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self._active: list[_ActiveTrack] = []
        self._finished: list[_ActiveTrack] = []
        self._next_id = 1
        #: Counts how often a confirmed track was lost, a proxy for ID-switch risk.
        self.lost_track_count = 0

    def update(
        self,
        frame_index: int,
        time_seconds: float,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> list[tuple[int, np.ndarray]]:
        """Advance the tracker by one detected frame.

        Returns ``(track_id, box)`` for every confirmed track present in this frame.
        """
        # 1. Predict where every active track should be now.
        predicted = np.array(
            [t.kf.predict() for t in self._active], dtype=np.float32
        ).reshape(-1, 4)
        for track in self._active:
            track.age += 1
            track.time_since_update += 1

        # 2. Associate detections to predictions by IoU.
        matches, unmatched_dets = self._associate(predicted, boxes)

        # 3. Update matched tracks.
        for track_idx, det_idx in matches:
            track = self._active[track_idx]
            track.kf.update(boxes[det_idx])
            track.hits += 1
            track.time_since_update = 0
            track.record(frame_index, time_seconds, track.kf.box, scores[det_idx])

        # 4. Spawn tracks for unmatched detections.
        for det_idx in unmatched_dets:
            track = _ActiveTrack(self._next_id, boxes[det_idx])
            self._next_id += 1
            track.record(frame_index, time_seconds, boxes[det_idx], scores[det_idx])
            self._active.append(track)

        # 5. Retire tracks that have gone unseen for too long.
        still_active = []
        for track in self._active:
            if track.time_since_update > self.max_age:
                if track.hits >= self.min_hits:
                    self._finished.append(track)
                    self.lost_track_count += 1
            else:
                still_active.append(track)
        self._active = still_active

        return [
            (t.track_id, t.kf.box)
            for t in self._active
            if t.hits >= self.min_hits and t.time_since_update == 0
        ]

    def _associate(
        self, predicted: np.ndarray, detections: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[int]]:
        """Hungarian matching on IoU, rejecting pairs below the IoU threshold."""
        if len(predicted) == 0 or len(detections) == 0:
            return [], list(range(len(detections)))

        ious = iou_matrix(predicted, detections)
        track_idx, det_idx = linear_sum_assignment(-ious)

        matches: list[tuple[int, int]] = []
        matched_dets: set[int] = set()
        for t, d in zip(track_idx, det_idx, strict=True):
            if ious[t, d] >= self.iou_threshold:
                matches.append((int(t), int(d)))
                matched_dets.add(int(d))

        unmatched = [i for i in range(len(detections)) if i not in matched_dets]
        return matches, unmatched

    def finalize(self) -> list[Track]:
        """Close the tracker and return every confirmed track, ordered by ID."""
        for track in self._active:
            if track.hits >= self.min_hits:
                self._finished.append(track)
        self._active = []

        tracks = [
            Track(track_id=t.track_id, observations=t.observations)
            for t in self._finished
            if len(t.observations) >= self.min_hits
        ]
        tracks.sort(key=lambda t: t.track_id)
        logger.info("Tracker produced %d confirmed tracks", len(tracks))
        return tracks
