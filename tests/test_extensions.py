"""Extension contracts exercised without weights or a video decoder."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from khoroos import (
    AnalysisParams,
    Detector,
    PipelineComponents,
    SelectedClasses,
    Tracker,
    VideoAnalyzer,
    VideoClassifier,
    VideoReader,
)
from khoroos.annotations import Detections
from khoroos.pipeline.analyze import CancelledError
from khoroos.pipeline.types import Track, Tracklet, TrackObservation, VideoInfo


class MemoryReader(VideoReader):
    info = VideoInfo("memory", 10, 1, 100, 60, 10)

    def __init__(self, path):
        self.closed = False

    def get_frames(self, indices):
        return torch.zeros((len(indices), 3, 60, 100), dtype=torch.uint8)

    def iter_batches(self, batch_size=16, stride=1, max_frames=None):
        indices = list(range(0, min(max_frames or 10, 10), stride))
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            yield batch, self.get_frames(batch)

    def close(self):
        self.closed = True


class OwnDetector(Detector):
    def detect(self, images, **kwargs):
        return [Detections([[10, 10, 30, 30]], [0.9], [7]) for _ in images]


class OwnClassifier(VideoClassifier):
    classes = ["eating", "sleeping", "moving"]

    def classify(self, clips):
        return np.tile([0.2, 0.1, 0.7], (len(clips), 1))


class OwnTracker(Tracker):
    def __init__(self):
        self.observations = []

    def update(self, frame_index, time_seconds, boxes, scores):
        self.observations.append(
            TrackObservation(frame_index, time_seconds, tuple(boxes[0]), scores[0])
        )
        return [(42, boxes[0])]

    def finalize(self):
        return [Track(42, self.observations)]


def own_windows(track, params, fps, width, height):
    return [
        Tracklet(
            42, 0, 1, [0, 1], np.array([[10, 10, 30, 30]] * 2), (20, 20), np.array([[20, 20]] * 2)
        )
    ], {}


@pytest.fixture
def custom_pipeline():
    readers, trackers = [], []

    def source_factory(path):
        readers.append(MemoryReader(path))
        return readers[-1]

    def tracker_factory(params):
        trackers.append(OwnTracker())
        return trackers[-1]

    analyzer = VideoAnalyzer(
        detector=OwnDetector(),
        recognizer=OwnClassifier(),
        components=PipelineComponents(
            source_factory=source_factory,
            tracker_factory=tracker_factory,
            tracklet_builder=own_windows,
        ),
    )
    return analyzer, readers, trackers


def test_custom_components_need_no_checkpoint_and_tracker_is_fresh(custom_pipeline):
    analyzer, readers, trackers = custom_pipeline
    first = analyzer.analyze("memory")
    second = analyzer.analyze("memory")
    assert first.predictions[0].label == "moving"
    assert first.model.action == "OwnClassifier"
    assert first.model.detector == "OwnDetector"
    assert first.model.classes == OwnClassifier.classes
    assert "indicators" not in first.metrics
    assert "alerts" not in first.metrics
    assert len(first.tracks[0].observations) == len(second.tracks[0].observations)
    assert trackers[0] is not trackers[1]
    assert all(reader.closed for reader in readers)


def test_selection_preserves_order_and_original_confidence(custom_pipeline):
    analyzer, _, _ = custom_pipeline
    params = AnalysisParams(action_classes=["sleeping", "eating"])
    result = analyzer.analyze("memory", params=params)
    assert result.model.classes == ["sleeping", "eating"]
    prediction = result.predictions[0]
    assert prediction.is_uncertain
    assert prediction.confidence == pytest.approx(0.2)
    assert list(prediction.probabilities) == ["sleeping", "eating"]
    assert sum(prediction.probabilities.values()) == pytest.approx(0.3)
    # Selection is per-run and must not mutate the cached classifier.
    assert analyzer.analyze("memory").predictions[0].label == "moving"


def test_classifier_wrapper_and_batching():
    selected = SelectedClasses(OwnClassifier(), ["moving", "eating"])
    values = selected.classify_batched([None] * 5, batch_size=2)
    np.testing.assert_allclose(values, [[0.7, 0.2]] * 5)
    assert selected.classify_batched([]).shape == (0, 2)
    with pytest.raises(ValueError, match="batch_size"):
        selected.classify_batched([], batch_size=0)


@pytest.mark.parametrize("labels", [[], ["eating", "eating"], ["unknown"], ["uncertain"]])
def test_invalid_class_selection_fails(labels):
    with pytest.raises(ValueError):
        SelectedClasses(OwnClassifier(), labels)


@pytest.mark.parametrize(
    "values",
    [
        np.array([[np.nan, 0, 0]]),
        np.array([[0.1, 0.2]]),
        np.array([[2, 0, 0]]),
        np.array([[0.8, 0.8, 0.8]]),
    ],
)
def test_invalid_classifier_outputs_fail_and_close_reader(custom_pipeline, values):
    analyzer, readers, _ = custom_pipeline
    analyzer.recognizer.classify = lambda clips: values
    with pytest.raises(ValueError, match="Classifier"):
        analyzer.analyze("memory")
    assert readers[0].closed


def test_cancellation_closes_reader(custom_pipeline):
    analyzer, readers, _ = custom_pipeline
    with pytest.raises(CancelledError):
        analyzer.analyze("memory", should_cancel=lambda: True)
    assert readers[0].closed


def test_generator_close_closes_reader(custom_pipeline):
    analyzer, readers, _ = custom_pipeline
    events = analyzer.iter_analyze("memory")
    next(events)
    next(events)
    events.close()
    assert readers[0].closed


def test_metrics_and_clip_extraction_are_replaceable(custom_pipeline):
    analyzer, _, _ = custom_pipeline
    calls = []

    def metrics(predictions, video, counts, **options):
        calls.append(options["classes"])
        return {"custom": len(predictions)}

    analyzer.components = replace(
        analyzer.components,
        metrics=metrics,
        clip_extractor=lambda source, tracklet, target_frames: source.get_frames([0]),
    )
    assert analyzer.analyze("memory").metrics == {"custom": 1}
    assert calls == [OwnClassifier.classes]


def test_class_selection_parses_cli_text():
    assert AnalysisParams(action_classes="eating, sleeping").action_classes == [
        "eating",
        "sleeping",
    ]


@pytest.mark.parametrize("parent", [Detector, VideoClassifier, Tracker, VideoReader])
def test_parent_requires_implementation(parent):
    with pytest.raises(TypeError):
        parent()


def test_runner_uses_custom_export_and_overlay(custom_pipeline, tmp_path):
    from khoroos import AnalysisRunner

    analyzer, _, _ = custom_pipeline
    calls = []

    def export(result, directory):
        calls.append("export")
        return {"custom": directory / "custom.json"}

    def overlay(video, result, path, *, max_seconds):
        calls.append(("overlay", max_seconds))
        return path

    artifacts = AnalysisRunner(analyzer, exporter=export, overlay_renderer=overlay).run(
        "memory",
        tmp_path,
        render_overlay=True,
    )
    assert calls == ["export", ("overlay", 1)]
    assert artifacts.paths == {
        "custom": tmp_path / "custom.json",
        "overlay": tmp_path / "annotated.mp4",
    }


def test_unknown_class_fails_before_detection_and_closes_source(custom_pipeline):
    analyzer, readers, trackers = custom_pipeline
    with pytest.raises(ValueError, match="Unknown action classes"):
        analyzer.analyze("memory", params=AnalysisParams(action_classes=["typo"]))
    assert not trackers
    assert readers[0].closed


def test_default_overlay_uses_injected_reader(custom_pipeline, tmp_path):
    from khoroos import AnalysisRunner
    from khoroos.video.reader import VideoSource

    analyzer, readers, _ = custom_pipeline
    artifacts = AnalysisRunner(analyzer).run("memory", tmp_path, render_overlay=True)
    assert len(readers) == 2
    assert all(reader.closed for reader in readers)
    with VideoSource(artifacts.overlay) as source:
        assert source.width == 100
        assert source.height == 60
        assert source.num_frames == 10


def test_overlay_closes_injected_reader_when_encoder_setup_fails(
    custom_pipeline, tmp_path, monkeypatch
):
    import av

    from khoroos import AnalysisRunner

    analyzer, readers, _ = custom_pipeline

    def fail(*args, **kwargs):
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(av, "open", fail)
    with pytest.raises(RuntimeError, match="encoder failed"):
        AnalysisRunner(analyzer).run("memory", tmp_path, render_overlay=True)
    assert len(readers) == 2
    assert all(reader.closed for reader in readers)


def test_pipeline_explicitly_drops_bad_legacy_boxes(custom_pipeline):
    analyzer, _, _ = custom_pipeline
    analyzer.detector.detect = lambda images, **kwargs: [
        (np.array([[10, 10, 30, 30], [1, 1, 0, 0]]), np.array([0.9, 0.8])) for _ in images
    ]
    with pytest.raises(ValueError, match="Boxes"):
        analyzer.analyze("memory")
    result = analyzer.analyze("memory", params=AnalysisParams(invalid_boxes="drop"))
    assert result.warnings == ["Dropped 5 invalid detection boxes."]
    assert all(count == 1 for _, count in result.frame_counts)


def test_custom_groups_are_recorded_and_used(custom_pipeline):
    analyzer, _, _ = custom_pipeline
    groups = {"activity": ["moving", "eating"]}
    result = analyzer.analyze("memory", params=AnalysisParams(behaviour_groups=groups))
    assert result.metrics["behaviour_groups"]["activity"]["seconds"] == 1
    assert result.params["behaviour_groups"] == groups
    result = analyzer.analyze("memory", params=AnalysisParams(behaviour_groups={}))
    assert result.metrics["behaviour_groups"] == {}


def test_pipeline_without_result_raises_explicit_error(custom_pipeline):
    analyzer, _, _ = custom_pipeline
    analyzer.iter_analyze = lambda *args, **kwargs: iter(())
    with pytest.raises(RuntimeError, match="without producing a result"):
        analyzer.analyze("memory")
