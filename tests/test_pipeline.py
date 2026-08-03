"""End-to-end pipeline tests using stub models over a generated video."""

from __future__ import annotations

import csv
import json

import pytest

from khoroos.config import params_for_preset
from khoroos.pipeline.types import AnalysisResult, ProgressEvent
from khoroos.video.reader import VideoReadError, VideoSource

# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def test_reader_reports_metadata(synthetic_video):
    with VideoSource(synthetic_video) as source:
        assert source.width == 640
        assert source.height == 480
        assert source.fps == pytest.approx(25.0, abs=0.5)
        assert source.num_frames > 90


def test_reader_returns_chw_uint8(synthetic_video):
    with VideoSource(synthetic_video) as source:
        frames = source.get_frames([0, 5, 10])
        assert frames.shape == (3, 3, 480, 640)
        assert frames.dtype.is_floating_point is False


def test_reader_clamps_out_of_range_indices(synthetic_video):
    with VideoSource(synthetic_video) as source:
        frames = source.get_frames([-5, 10**6])
        assert frames.shape[0] == 2


def test_reader_rejects_a_missing_file(tmp_path):
    with pytest.raises(VideoReadError):
        VideoSource(tmp_path / "nope.mp4")


def test_iter_batches_respects_stride(synthetic_video):
    with VideoSource(synthetic_video) as source:
        indices = [i for batch, _ in source.iter_batches(batch_size=8, stride=4) for i in batch]
        assert indices == list(range(0, source.num_frames, 4))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_pipeline_produces_a_result(stub_analyzer, synthetic_video):
    params = params_for_preset("balanced", detection_stride=1)
    result = stub_analyzer.analyze(synthetic_video, params=params)

    assert isinstance(result, AnalysisResult)
    assert len(result.tracks) == 2, "the two synthetic birds should be tracked separately"
    assert result.predictions, "clips should have been classified"
    assert result.metrics["time_budget"]["total_bird_seconds"] > 0
    assert result.runtime_seconds > 0


def test_progress_events_are_monotonic_and_complete(stub_analyzer, synthetic_video):
    events: list[ProgressEvent] = []
    result = None
    for item in stub_analyzer.iter_analyze(
        synthetic_video, params=params_for_preset("balanced", detection_stride=1)
    ):
        if isinstance(item, ProgressEvent):
            events.append(item)
        else:
            result = item

    assert result is not None
    assert events
    progresses = [e.progress for e in events]
    assert progresses == sorted(progresses), "progress must never go backwards"
    assert progresses[-1] == pytest.approx(1.0, abs=0.01)
    assert {"probe", "detect", "track", "clips", "classify", "metrics"} <= {e.stage for e in events}


def test_cancellation_stops_the_run(stub_analyzer, synthetic_video):
    from khoroos.pipeline.analyze import CancelledError

    with pytest.raises(CancelledError):
        stub_analyzer.analyze(synthetic_video, should_cancel=lambda: True)


def test_max_duration_limits_work(stub_analyzer, synthetic_video):
    params = params_for_preset("balanced", detection_stride=1, max_duration_seconds=1.0)
    result = stub_analyzer.analyze(synthetic_video, params=params)
    assert max(t for t, _ in result.frame_counts) <= 1.05
    assert result.video.duration_seconds == pytest.approx(1.0)
    assert result.video.num_frames == 25


@pytest.mark.parametrize(("detection_batch", "action_batch"), [(7, 2), (3, 1)])
def test_batch_sizes_govern_the_forward_passes(
    stub_analyzer, synthetic_video, detection_batch, action_batch
):
    """The configured batch sizes must drive the actual calls, not merely be recorded.

    Both models are wrapped so every call's batch length is captured. A run's final batch
    is short whenever the work does not divide evenly, so the check is that no batch
    exceeds the setting and that at least one full batch was formed.
    """
    detector_batches: list[int] = []
    action_batches: list[int] = []

    real_detect = stub_analyzer.detector.detect
    real_classify = stub_analyzer.recognizer.classify

    def detect(images, **kwargs):
        detector_batches.append(len(images))
        return real_detect(images, **kwargs)

    def classify(clips):
        action_batches.append(len(clips))
        return real_classify(clips)

    stub_analyzer.detector.detect = detect
    stub_analyzer.recognizer.classify = classify

    params = params_for_preset(
        "balanced",
        detection_stride=1,
        detection_batch_size=detection_batch,
        action_batch_size=action_batch,
    )
    stub_analyzer.analyze(synthetic_video, params=params)

    assert detector_batches, "the detector was never called"
    assert action_batches, "the action model was never called"
    assert max(detector_batches) == detection_batch
    assert all(n <= detection_batch for n in detector_batches)
    assert max(action_batches) == action_batch
    assert all(n <= action_batch for n in action_batches)


def test_low_confidence_becomes_uncertain(stub_analyzer, synthetic_video):
    params = params_for_preset("balanced", detection_stride=1, min_confidence=0.99)
    result = stub_analyzer.analyze(synthetic_video, params=params)

    assert result.predictions
    assert all(p.is_uncertain for p in result.predictions)
    assert all(p.label == "uncertain" for p in result.predictions)
    assert result.metrics["time_budget"]["uncertain_share"] == pytest.approx(1.0)


def test_result_is_json_serialisable(stub_analyzer, synthetic_video):
    result = stub_analyzer.analyze(
        synthetic_video, params=params_for_preset("balanced", detection_stride=1)
    )
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["schema_version"] == "1.0"
    assert {"video", "params", "model", "predictions", "metrics", "tracks"} <= set(payload)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_bundle_is_written(stub_analyzer, synthetic_video, tmp_path):
    from khoroos.welfare.export import export_all

    result = stub_analyzer.analyze(
        synthetic_video, params=params_for_preset("balanced", detection_stride=1)
    )
    paths = export_all(result, tmp_path / "out")

    for name in ("result", "predictions", "time_budget", "per_bird", "metrics"):
        assert paths[name].exists(), f"{name} was not written"

    with paths["predictions"].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.predictions)
    assert {"track_id", "start_s", "label", "confidence"} <= set(rows[0])


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


#: The overlay's fixed header bar, drawn once per frame regardless of any detection.
HEADER_RECT = (0, 0, 330, 26)


def _render_recording_rectangles(monkeypatch, stub_analyzer, video, output, min_confidence):
    """Render an overlay, returning every rectangle drawn that is not the header bar."""
    from PIL import ImageDraw

    from khoroos.video import writer as writer_module

    real_draw = ImageDraw.Draw
    drawn: list[tuple] = []

    class RecordingDraw:
        def __init__(self, *args, **kwargs):
            self._draw = real_draw(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._draw, name)

        def rectangle(self, xy, *args, **kwargs):
            drawn.append(tuple(xy))
            return self._draw.rectangle(xy, *args, **kwargs)

    monkeypatch.setattr(writer_module.ImageDraw, "Draw", RecordingDraw)

    result = stub_analyzer.analyze(
        video, params=params_for_preset("balanced", detection_stride=1, min_confidence=min_confidence)
    )
    assert result.predictions, "the fixture produced no predictions to draw"
    # Long enough to cover the first classified window, which starts at t=1s once the
    # tracks are confirmed — a shorter render would draw nothing either way.
    limit = min(p.end_seconds for p in result.predictions) + 0.5
    writer_module.render_overlay(video, result, output, max_seconds=limit)
    return [rect for rect in drawn if rect != HEADER_RECT]


def test_overlay_draws_confident_actions(stub_analyzer, synthetic_video, tmp_path, monkeypatch):
    marks = _render_recording_rectangles(
        monkeypatch, stub_analyzer, synthetic_video, tmp_path / "confident.mp4", 0.5
    )
    assert marks, "confident predictions should be marked on the video"


def test_overlay_omits_uncertain_actions(stub_analyzer, synthetic_video, tmp_path, monkeypatch):
    """An uncertain window is an absence of a finding, so nothing is drawn for it."""
    marks = _render_recording_rectangles(
        monkeypatch, stub_analyzer, synthetic_video, tmp_path / "uncertain.mp4", 0.99
    )
    assert marks == [], f"uncertain windows were marked on the video: {marks[:3]}"


def test_overlay_video_is_playable(stub_analyzer, synthetic_video, tmp_path):
    from khoroos.video.writer import render_overlay

    result = stub_analyzer.analyze(
        synthetic_video, params=params_for_preset("balanced", detection_stride=1)
    )
    output = render_overlay(synthetic_video, result, tmp_path / "annotated.mp4", max_seconds=1.0)

    assert output.exists() and output.stat().st_size > 0
    with VideoSource(output) as source:
        assert source.num_frames > 0
        assert (source.width, source.height) == (640, 480)
