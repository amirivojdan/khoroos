"""Splitting a batch over several devices must change only the wall clock.

Nothing here needs a second GPU: a replica is any object with the model's method, so the
splitting, ordering, concurrency and failure behaviour are all testable on one machine.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

from khoroos.config import ACTION_CLASSES, Settings
from khoroos.interfaces import component_name
from khoroos.models.parallel import (
    ParallelClassifier,
    ParallelDetector,
    ReplicaPool,
    build_detector,
    build_recognizer,
    split_evenly,
)


class RecordingDetector:
    """Reports which replica saw which frames, and returns one detection per frame."""

    checkpoint = "stub-detector"

    def __init__(self, tag: str, device: str = "cpu") -> None:
        self.tag = tag
        self.device = device
        self.seen: list[list[int]] = []

    def detect(self, images, confidence_threshold=0.3, nms_threshold=0.6, box_padding=0.0):
        frames = [int(image) for image in images]
        self.seen.append(frames)
        return [
            (
                np.array([[frame, 0.0, frame + 1.0, 1.0]], dtype=np.float32),
                np.array([0.9], dtype=np.float32),
            )
            for frame in frames
        ]


class RecordingClassifier:
    """One probability row per clip, with the clip's own value in the first column."""

    checkpoint = "stub-action"
    num_frames = 16

    def __init__(self, tag: str, device: str = "cpu", classes=None) -> None:
        self.tag = tag
        self.device = device
        self.classes = list(classes or ACTION_CLASSES)
        self.seen: list[list[int]] = []

    def classify(self, clips):
        values = [int(clip.flatten()[0].item()) for clip in clips]
        self.seen.append(values)
        probs = np.zeros((len(clips), len(self.classes)), dtype=np.float32)
        for row, value in enumerate(values):
            probs[row, 0] = value / 1000.0
        return probs


def clip_of(value: int) -> torch.Tensor:
    return torch.full((4, 3, 8, 8), value, dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "parts", "expected"),
    [
        (8, 2, [(0, 4), (4, 8)]),
        (7, 2, [(0, 4), (4, 7)]),  # the remainder goes to the earliest span
        (5, 3, [(0, 2), (2, 4), (4, 5)]),
        (1, 4, [(0, 1)]),  # fewer items than devices uses fewer devices
        (0, 2, []),
        (4, 1, [(0, 4)]),
    ],
)
def test_shares_are_contiguous_and_cover_the_batch(count, parts, expected):
    spans = split_evenly(count, parts)
    assert spans == expected
    assert sum(stop - start for start, stop in spans) == count
    assert all(start < stop for start, stop in spans)


# ---------------------------------------------------------------------------
# Order and coverage
# ---------------------------------------------------------------------------


def test_detections_come_back_in_input_order_across_devices():
    replicas = [RecordingDetector("a"), RecordingDetector("b"), RecordingDetector("c")]
    detector = ParallelDetector(replicas)
    try:
        results = detector.detect(list(range(10)))
    finally:
        detector.close()

    # The caller sees one detection per frame, in the order it submitted them.
    assert [int(boxes[0][0]) for boxes, _ in results] == list(range(10))
    # Each replica handled a distinct contiguous share, and together they covered the batch.
    assert [replica.seen for replica in replicas] == [[[0, 1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]]


def test_probabilities_are_concatenated_in_input_order():
    replicas = [RecordingClassifier("a"), RecordingClassifier("b")]
    classifier = ParallelClassifier(replicas)
    try:
        probs = classifier.classify([clip_of(value) for value in (10, 20, 30, 40, 50)])
    finally:
        classifier.close()

    assert probs.shape == (5, len(ACTION_CLASSES))
    assert [round(float(value) * 1000) for value in probs[:, 0]] == [10, 20, 30, 40, 50]
    assert [replica.seen for replica in replicas] == [[[10, 20, 30]], [[40, 50]]]


def test_an_empty_batch_asks_no_device_for_anything():
    replicas = [RecordingDetector("a"), RecordingDetector("b")]
    detector = ParallelDetector(replicas)
    classifier = ParallelClassifier([RecordingClassifier("a"), RecordingClassifier("b")])
    try:
        assert detector.detect([]) == []
        empty = classifier.classify([])
        assert empty.shape == (0, len(ACTION_CLASSES))
    finally:
        detector.close()
        classifier.close()
    assert [replica.seen for replica in replicas] == [[], []]


def test_a_single_device_batch_is_split_over_fewer_devices():
    """The last batch of most runs is smaller than the device count."""
    replicas = [RecordingDetector("a"), RecordingDetector("b"), RecordingDetector("c")]
    detector = ParallelDetector(replicas)
    try:
        assert len(detector.detect([7, 8])) == 2
    finally:
        detector.close()
    assert [replica.seen for replica in replicas] == [[[7]], [[8]], []]


# ---------------------------------------------------------------------------
# Concurrency and failure
# ---------------------------------------------------------------------------


def test_devices_run_at_the_same_time():
    """Each replica waits for the others, so serialized execution would time out."""
    barrier = threading.Barrier(3, timeout=10)

    class Synchronized(RecordingDetector):
        def detect(self, images, **options):
            barrier.wait()
            return super().detect(images, **options)

    detector = ParallelDetector([Synchronized(tag) for tag in "abc"])
    try:
        results = detector.detect(list(range(6)))
    finally:
        detector.close()
    assert [int(boxes[0][0]) for boxes, _ in results] == list(range(6))


def test_one_device_failing_fails_the_batch_after_every_share_finishes():
    finished: list[str] = []

    class Failing(RecordingDetector):
        def detect(self, images, **options):
            raise RuntimeError("out of memory on this device")

    class Slow(RecordingDetector):
        def detect(self, images, **options):
            result = super().detect(images, **options)
            finished.append(self.tag)
            return result

    detector = ParallelDetector([Failing("a"), Slow("b")])
    try:
        with pytest.raises(RuntimeError, match="out of memory"):
            detector.detect([1, 2, 3, 4])
    finally:
        detector.close()
    # The healthy device is not abandoned mid-forward-pass; its share completes first.
    assert finished == ["b"]


def test_replicas_must_agree_on_the_class_vocabulary():
    with pytest.raises(ValueError, match="identical classes"):
        ParallelClassifier(
            [RecordingClassifier("a"), RecordingClassifier("b", classes=["feeding", "resting"])]
        )


def test_a_pool_needs_at_least_one_replica():
    with pytest.raises(ValueError, match="at least one replica"):
        ReplicaPool([])


def test_closing_releases_the_worker_threads():
    before = threading.active_count()
    detector = ParallelDetector([RecordingDetector("a"), RecordingDetector("b")])
    detector.detect([1, 2])
    detector.close()
    assert threading.active_count() <= before
    # Still usable afterwards, just without the extra threads.
    assert len(detector.detect([1, 2])) == 2


# ---------------------------------------------------------------------------
# Identity and construction
# ---------------------------------------------------------------------------


def test_a_parallel_model_is_named_for_its_checkpoint_not_the_wrapper():
    """Which devices ran a model is a deployment detail, not a different model."""
    detector = ParallelDetector([RecordingDetector("a"), RecordingDetector("b")])
    classifier = ParallelClassifier([RecordingClassifier("a"), RecordingClassifier("b")])
    try:
        assert component_name(detector) == "stub-detector"
        assert component_name(classifier) == "stub-action"
        assert classifier.classes == list(ACTION_CLASSES)
        assert classifier.num_frames == 16
        assert detector.devices == ["cpu", "cpu"]
    finally:
        detector.close()
        classifier.close()


def test_one_device_builds_a_plain_model_and_several_build_replicas(monkeypatch):
    built: list[str] = []

    class FakeModel:
        checkpoint = "fake"
        classes = list(ACTION_CLASSES)
        num_frames = 16

        def __init__(self, settings=None):
            self.device = settings.device
            built.append(settings.device)

    monkeypatch.setattr("khoroos.models.detector.ChickenDetector", FakeModel, raising=False)
    monkeypatch.setattr("khoroos.models.action.ActionRecognizer", FakeModel, raising=False)

    single = build_detector(Settings(device="cpu"), devices=["cpu"])
    assert isinstance(single, FakeModel)
    assert built == ["cpu"]

    built.clear()
    both = build_recognizer(Settings(device="cpu"), devices=["cuda:0", "cuda:1"])
    try:
        assert isinstance(both, ParallelClassifier)
        # One model per device, each pinned to its own — not to the configured spec.
        assert built == ["cuda:0", "cuda:1"]
        assert both.devices == ["cuda:0", "cuda:1"]
    finally:
        both.close()


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


class SteadyDetector:
    """Two fixed boxes per frame — stable enough to build tracks and clips from."""

    checkpoint = "stub-detector"
    device = "cpu"

    def detect(self, images, confidence_threshold=0.3, nms_threshold=0.6, box_padding=0.0):
        if isinstance(images, torch.Tensor):
            images = list(images)
        boxes = np.array([[60, 150, 210, 300], [360, 150, 510, 300]], dtype=np.float32)
        scores = np.array([0.9, 0.85], dtype=np.float32)
        return [(boxes.copy(), scores.copy()) for _ in images]


class ContentClassifier:
    """A label derived from the clip's own pixels, so it cannot depend on call order."""

    checkpoint = "stub-action"
    num_frames = 16
    device = "cpu"

    def __init__(self) -> None:
        self.classes = list(ACTION_CLASSES)

    def classify(self, clips):
        probs = np.zeros((len(clips), len(self.classes)), dtype=np.float32)
        for row, clip in enumerate(clips):
            index = int(clip.float().mean().item() * 100) % len(self.classes)
            probs[row, index] = 0.92
        return probs


def analyse_with(detector, recognizer, video):
    from khoroos.config import params_for_preset
    from khoroos.pipeline.analyze import VideoAnalyzer

    analyzer = VideoAnalyzer(
        settings=Settings(device="cpu"), detector=detector, recognizer=recognizer
    )
    return analyzer.analyze(video, params=params_for_preset("balanced"))


def test_splitting_a_video_across_devices_preserves_the_analysis(synthetic_video):
    """Two devices are a throughput change, not an analysis change.

    The stubs here are batch-shape independent, so this pins down what the wiring owes:
    every frame detected, every clip classified, and both reassembled in order. Real models
    additionally shift in their last decimal places when the batch shape changes — which
    is also true of changing the batch size on one device — so this is an equivalence of
    the pipeline, not a promise of bit-identical scores.
    """
    single = analyse_with(SteadyDetector(), ContentClassifier(), synthetic_video)

    parallel_detector = ParallelDetector([SteadyDetector(), SteadyDetector()])
    parallel_classifier = ParallelClassifier([ContentClassifier(), ContentClassifier()])
    try:
        split = analyse_with(parallel_detector, parallel_classifier, synthetic_video)
    finally:
        parallel_detector.close()
        parallel_classifier.close()

    assert [track.track_id for track in split.tracks] == [t.track_id for t in single.tracks]
    assert len(split.predictions) == len(single.predictions) > 2
    # Without several distinct labels, a reassembly that shuffled the shares would still
    # match and the comparison below would prove nothing.
    assert len({prediction.label for prediction in single.predictions}) > 1
    for got, expected in zip(split.predictions, single.predictions, strict=True):
        assert (got.track_id, got.start_seconds, got.label) == (
            expected.track_id,
            expected.start_seconds,
            expected.label,
        )
    assert split.metrics["time_budget"] == single.metrics["time_budget"]
    # The model is reported by its checkpoint either way, so results stay comparable.
    assert split.model.detector == single.model.detector
