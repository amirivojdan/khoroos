"""Tracker and tracklet tests.

These use synthetic motion rather than real footage so they stay fast and deterministic,
and so each failure mode (jitter, teleport, drift) can be exercised in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from khoroos.config import AnalysisParams
from khoroos.pipeline.types import Track, TrackObservation
from khoroos.tracking.kalman import BoxKalmanFilter, box_to_measurement, measurement_to_box
from khoroos.tracking.tracker import BirdTracker, iou_matrix
from khoroos.tracking.tracklets import (
    box_iou_xyxy,
    build_tracklets,
    interpolate_track,
    track_is_unstable,
)

FPS = 25.0
FRAME_W, FRAME_H = 1280, 720


# ---------------------------------------------------------------------------
# Kalman
# ---------------------------------------------------------------------------


def test_measurement_roundtrip():
    box = np.array([100.0, 200.0, 180.0, 260.0])
    assert np.allclose(measurement_to_box(box_to_measurement(box)), box)


def test_measurement_never_collapses():
    """A degenerate box must not produce a zero-area crop downstream."""
    box = measurement_to_box(np.array([50.0, 50.0, 0.0, 0.0]))
    assert box[2] > box[0] and box[3] > box[1]


def test_kalman_tracks_constant_velocity():
    kf = BoxKalmanFilter(np.array([0.0, 0.0, 40.0, 40.0]))
    for step in range(1, 25):
        kf.predict()
        kf.update(np.array([step * 5.0, 0.0, step * 5.0 + 40.0, 40.0]))

    predicted = kf.predict()
    # After learning a 5 px/frame drift, the next prediction should lead the last update.
    assert predicted[0] > 24 * 5.0
    assert abs((predicted[2] - predicted[0]) - 40.0) < 8.0


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


def test_iou_matrix_matches_scalar_helper():
    a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10]], dtype=np.float32)
    matrix = iou_matrix(a, b)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[1, 0] == pytest.approx(box_iou_xyxy(a[1], b[0]), abs=1e-5)


def test_iou_matrix_handles_empty():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


def _walk(tracker: BirdTracker, n_frames: int, boxes_fn):
    for frame in range(n_frames):
        boxes = np.asarray(boxes_fn(frame), dtype=np.float32).reshape(-1, 4)
        scores = np.full(len(boxes), 0.9, dtype=np.float32)
        tracker.update(frame, frame / FPS, boxes, scores)


def test_single_bird_keeps_one_id():
    tracker = BirdTracker(min_hits=3, max_age=10)
    _walk(tracker, 40, lambda f: [[100 + f * 2, 100, 160 + f * 2, 160]])
    tracks = tracker.finalize()
    assert len(tracks) == 1
    assert len(tracks[0]) >= 35


def test_two_separated_birds_get_distinct_ids():
    tracker = BirdTracker(min_hits=3, max_age=10)
    _walk(
        tracker,
        40,
        lambda f: [[100 + f, 100, 160 + f, 160], [600, 400 + f, 660, 460 + f]],
    )
    tracks = tracker.finalize()
    assert len(tracks) == 2
    assert {t.track_id for t in tracks} == {1, 2}


def test_short_flicker_is_not_confirmed():
    """A one-frame false positive must not become a bird."""
    tracker = BirdTracker(min_hits=3, max_age=5)
    _walk(tracker, 20, lambda f: [[10, 10, 40, 40]] if f == 5 else [])
    assert tracker.finalize() == []


def test_track_survives_a_brief_occlusion():
    def boxes(frame):
        if 10 <= frame < 14:  # bird hidden behind a feeder
            return []
        return [[100 + frame * 2, 100, 160 + frame * 2, 160]]

    tracker = BirdTracker(min_hits=3, max_age=10)
    _walk(tracker, 40, boxes)
    tracks = tracker.finalize()
    assert len(tracks) == 1, "occlusion shorter than max_age must not split the track"


def test_long_gap_splits_the_track():
    def boxes(frame):
        if 10 <= frame < 30:
            return []
        return [[100, 100, 160, 160]]

    tracker = BirdTracker(min_hits=3, max_age=5)
    _walk(tracker, 45, boxes)
    tracks = tracker.finalize()
    assert len(tracks) == 2
    assert tracker.lost_track_count >= 1


# ---------------------------------------------------------------------------
# Stability gates
# ---------------------------------------------------------------------------


def _smooth_track(n=40, size=120, step=2):
    boxes, centers = [], []
    for i in range(n):
        x = 100 + i * step
        boxes.append([x, 100, x + size, 100 + size])
        centers.append([x + size / 2, 100 + size / 2])
    return np.array(centers, dtype=np.float32), np.array(boxes, dtype=np.float32)


def test_smooth_track_is_stable():
    centers, boxes = _smooth_track()
    assert not track_is_unstable(centers, boxes, FRAME_W, FRAME_H)


def test_teleporting_track_is_unstable():
    centers, boxes = _smooth_track()
    centers[20] = [1200.0, 700.0]
    boxes[20] = [1140.0, 640.0, 1260.0, 760.0]
    assert track_is_unstable(centers, boxes, FRAME_W, FRAME_H)


def test_too_short_track_is_unstable():
    centers, boxes = _smooth_track(n=5)
    assert track_is_unstable(centers, boxes, FRAME_W, FRAME_H)


def test_jittering_track_is_unstable():
    centers, boxes = _smooth_track(n=40, size=60)
    rng = np.random.default_rng(0)
    jitter = rng.normal(0, 60, size=centers.shape).astype(np.float32)
    centers = centers + jitter
    boxes = np.stack(
        [centers[:, 0] - 30, centers[:, 1] - 30, centers[:, 0] + 30, centers[:, 1] + 30], axis=1
    )
    assert track_is_unstable(centers, boxes, FRAME_W, FRAME_H)


# ---------------------------------------------------------------------------
# Tracklet windows
# ---------------------------------------------------------------------------


def _track_from(boxes: np.ndarray, stride: int = 1) -> Track:
    observations = [
        TrackObservation(
            frame_index=i * stride,
            time_seconds=i * stride / FPS,
            box=tuple(float(v) for v in box),
            score=0.9,
        )
        for i, box in enumerate(boxes)
    ]
    return Track(track_id=1, observations=observations)


def test_windows_are_built_for_a_clean_track():
    _, boxes = _smooth_track(n=100, size=150)
    params = AnalysisParams(window_seconds=2.0, stride_seconds=1.0)
    tracklets, rejects = build_tracklets(_track_from(boxes), params, FPS, FRAME_W, FRAME_H)

    assert tracklets, f"expected windows, got rejects={rejects}"
    for tracklet in tracklets:
        assert tracklet.duration_seconds == pytest.approx(2.0, abs=0.15)
        assert tracklet.crop_size[0] >= params.min_crop_size


def test_tiny_birds_are_rejected_on_geometry():
    """Distant birds in a wide shot are too small for the action model to read."""
    _, boxes = _smooth_track(n=100, size=30)
    params = AnalysisParams()
    tracklets, rejects = build_tracklets(_track_from(boxes), params, FPS, FRAME_W, FRAME_H)
    assert not tracklets
    assert rejects["geometry"] > 0


def test_full_frame_box_is_rejected():
    """One spurious detection covering the frame must not become a 'bird' clip."""
    boxes = np.tile(np.array([0, 0, FRAME_W, FRAME_H], dtype=np.float32), (100, 1))
    params = AnalysisParams()
    tracklets, rejects = build_tracklets(_track_from(boxes), params, FPS, FRAME_W, FRAME_H)
    assert not tracklets
    assert rejects["geometry"] > 0


def test_crop_size_is_robust_to_one_giant_detection():
    """A single oversized box must not dictate the crop for the whole window."""
    _, boxes = _smooth_track(n=100, size=150)
    boxes[50] = [0, 0, FRAME_W, FRAME_H]
    params = AnalysisParams()
    tracklets, _ = build_tracklets(_track_from(boxes), params, FPS, FRAME_W, FRAME_H)
    assert tracklets
    for tracklet in tracklets:
        assert tracklet.crop_size[0] < FRAME_W * params.max_crop_frame_fraction


def test_strided_detections_are_interpolated():
    """With detection_stride > 1 the window must still be densely covered."""
    _, boxes = _smooth_track(n=50, size=150, step=4)
    params = AnalysisParams(window_seconds=2.0, stride_seconds=1.0)
    tracklets, rejects = build_tracklets(
        _track_from(boxes, stride=2), params, FPS, FRAME_W, FRAME_H
    )
    assert tracklets, f"interpolation should fill strided frames; rejects={rejects}"
    assert len(tracklets[0].frame_indices) >= int(2.0 * FPS * 0.9)


def test_interpolated_track_has_a_box_on_every_frame():
    _, boxes = _smooth_track(n=10, size=150, step=4)
    dense = interpolate_track(_track_from(boxes, stride=4))
    assert set(dense) == set(range(37))
