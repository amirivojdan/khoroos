"""Batch boundaries must preserve crop identity through skips, exports, and cancellation."""

from pathlib import Path

import numpy as np
import pytest
import torch

from khoroos.interfaces import VideoClassifier
from khoroos.models.selection import SelectedClasses
from khoroos.pipeline.analyze import CancelledError
from khoroos.pipeline.classification import classify_batch
from khoroos.pipeline.types import Tracklet
from khoroos.video.tracklet_export import TrackletExporter


class RecordingExporter(TrackletExporter):
    def __init__(self):
        self.saved = []

    def save(self, kind, index, tracklet, clip, prediction=None, raw_path=None):
        self.saved.append((kind, index, tracklet.track_id, prediction, raw_path))
        return Path(f"{kind}-{index}.mp4")


class ContentClassifier(VideoClassifier):
    classes = ["feeding", "resting"]
    num_frames = 2

    def __init__(self):
        self.calls = 0

    def classify(self, clips):
        self.calls += 1
        return np.array([[0.8, 0.2] if clip[0, 0, 0, 0] == 1 else [0.3, 0.7]
                         for clip in clips])


def window(track_id):
    return Tracklet(
        track_id, 0, 1, [0, 1], np.array([[0, 0, 8, 8]] * 2),
        (8, 8), np.array([[4, 4]] * 2),
    )


@pytest.mark.parametrize("skip", [None, torch.empty((0, 3, 8, 8), dtype=torch.uint8)])
def test_skipped_crop_preserves_prediction_and_export_identity(skip):
    classifier = ContentClassifier()
    exporter = RecordingExporter()
    source = object()

    def extract(reader, tracklet, target_frames):
        assert reader is source
        assert target_frames == 2
        if tracklet.track_id == 2:
            return skip
        return torch.full((2, 3, 8, 8), tracklet.track_id, dtype=torch.uint8)

    predictions = classify_batch(
        source, [window(1), window(2), window(3)], SelectedClasses(classifier),
        start_index=10, min_confidence=0.75, extract_clip=extract,
        exporter=exporter, check_cancel=lambda: None,
    )
    assert classifier.calls == 1
    assert [(p.track_id, p.label) for p in predictions] == [(1, "feeding"), (3, "uncertain")]
    assert predictions[1].probabilities["resting"] == pytest.approx(0.7)
    assert [(kind, index, track) for kind, index, track, _, _ in exporter.saved] == [
        ("raw", 10, 1), ("raw", 12, 3), ("classified", 10, 1), ("classified", 12, 3),
    ]
    assert [row[4] for row in exporter.saved[2:]] == [Path("raw-10.mp4"), Path("raw-12.mp4")]


def test_empty_prepared_batch_does_not_call_classifier_or_exporter():
    classifier = ContentClassifier()
    exporter = RecordingExporter()
    assert classify_batch(
        object(), [window(1)], SelectedClasses(classifier), start_index=0,
        min_confidence=0.5, extract_clip=lambda *args, **kwargs: None,
        exporter=exporter, check_cancel=lambda: None,
    ) == []
    assert classifier.calls == 0
    assert exporter.saved == []


def test_cancel_after_preparation_preserves_raw_export_and_skips_inference():
    classifier = ContentClassifier()
    exporter = RecordingExporter()

    def check_cancel():
        if exporter.saved:
            raise CancelledError("Cancelled after saving a crop")

    with pytest.raises(CancelledError):
        classify_batch(
            object(), [window(1)], SelectedClasses(classifier), start_index=0,
            min_confidence=0.5,
            extract_clip=lambda *args, **kwargs: torch.zeros((2, 3, 8, 8), dtype=torch.uint8),
            exporter=exporter, check_cancel=check_cancel,
        )
    assert classifier.calls == 0
    assert len(exporter.saved) == 1
    assert exporter.saved[0][0] == "raw"
