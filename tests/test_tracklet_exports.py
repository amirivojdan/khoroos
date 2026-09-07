"""Tracklet exports must preserve clips and predictions without overwriting earlier runs."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from khoroos.config import AnalysisParams, params_for_preset
from khoroos.pipeline.types import ActionPrediction, Tracklet
from khoroos.video.tracklet_export import TrackletExporter, write_clip


def test_clip_encoding_preserves_duration_and_pads_odd_dimensions(tmp_path):
    import av

    path = tmp_path / "clip.mp4"
    clip = torch.full((16, 3, 15, 17), 128, dtype=torch.uint8)
    write_clip(clip, 3.0, path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(video=0))
        assert len(frames) == 16
        assert (frames[0].width, frames[0].height) == (18, 16)
        assert float(stream.duration * stream.time_base) == pytest.approx(3.0, abs=0.02)
        assert np.mean(frames[0].to_ndarray(format="rgb24")) == pytest.approx(128, abs=3)
    assert not list(tmp_path.glob("*.partial.mp4"))


@pytest.mark.parametrize("raw,classified", [(False, False), (True, False), (False, True), (True, True)])
def test_pipeline_exports_only_requested_tracklets(
    stub_runner, synthetic_video, tmp_path, raw, classified
):
    raw_root, classified_root = tmp_path / "raw", tmp_path / "classified"
    params = params_for_preset(
        "fast", max_duration_seconds=3,
        raw_tracklets_dir=str(raw_root) if raw else None,
        classified_tracklets_dir=str(classified_root) if classified else None,
        min_confidence=0.99,  # all stub predictions should be filed as uncertain
    )
    result = stub_runner.run(synthetic_video, tmp_path / "results", params=params).result
    assert result.predictions
    assert raw_root.exists() == raw
    assert classified_root.exists() == classified
    paths = result.params["tracklet_exports"]
    assert set(paths) == ({"raw"} if raw else set()) | ({"classified"} if classified else set())
    for kind, directory in paths.items():
        files = sorted(Path(directory).rglob("*.mp4"))
        assert len(files) == len(result.predictions)
        for file, prediction in zip(files, result.predictions, strict=True):
            metadata = json.loads(file.with_suffix(".json").read_text())
            assert metadata["track_id"] == prediction.track_id
            assert metadata["encoded_frames"] == 16
            assert metadata["start_s"] == prediction.start_seconds
            assert metadata["end_s"] == prediction.end_seconds
            if kind == "classified":
                assert file.parent.name == "uncertain"
                assert metadata["prediction"]["uncertain"] is True
                assert metadata["probabilities"] == prediction.probabilities
                if raw:
                    assert file.read_bytes() == (Path(paths["raw"]) / file.name).read_bytes()
            else:
                assert "prediction" not in metadata


def test_raw_clips_survive_a_classifier_failure(stub_analyzer, synthetic_video, tmp_path):
    def fail(clips):
        raise RuntimeError("Classification failed")

    stub_analyzer.recognizer.classify = fail
    params = params_for_preset("fast", raw_tracklets_dir=str(tmp_path / "raw"))
    with pytest.raises(RuntimeError, match="Classification failed"):
        stub_analyzer.analyze(synthetic_video, params=params)
    assert list((tmp_path / "raw").rglob("*.mp4"))


def test_run_isolation_and_custom_labels_cannot_escape_export_directory(tmp_path):
    params = AnalysisParams(classified_tracklets_dir=str(tmp_path))
    first = TrackletExporter(params, "farm.mp4")
    second = TrackletExporter(params, "farm.mp4")
    assert first.directories != second.directories
    tracklet = Tracklet(1, 0.0, 1.0, [0], np.array([[0, 0, 16, 16]]), (16, 16), np.array([[8, 8]]))
    label = "../../outside"
    prediction = ActionPrediction(1, 0.0, 1.0, label, 1.0, {label: 1.0}, (0, 0, 16, 16))
    clip = torch.zeros((2, 3, 16, 16), dtype=torch.uint8)
    path = first.save("classified", 0, tracklet, clip, prediction)
    assert path.is_relative_to(Path(first.directories["classified"]))
    assert path.parent.parent == Path(first.directories["classified"])
    assert json.loads(path.with_suffix(".json").read_text())["prediction"]["label"] == label
