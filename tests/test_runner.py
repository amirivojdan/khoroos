"""The shared runner — the sequence both the CLI and the web worker drive."""

from __future__ import annotations

import pytest

from khoroos.config import params_for_preset


@pytest.fixture
def quick_params():
    """Short window over the 4-second synthetic clip, so runs stay fast."""
    return params_for_preset("fast", max_duration_seconds=3.0)


def test_run_produces_result_and_exports(stub_runner, synthetic_video, tmp_path, quick_params):
    artifacts = stub_runner.run(synthetic_video, output_dir=tmp_path, params=quick_params)

    assert artifacts.result.predictions
    assert set(artifacts.exports) == {
        "result",
        "predictions",
        "time_budget",
        "per_bird",
        "metrics",
    }
    for path in artifacts.exports.values():
        assert path.exists() and path.stat().st_size > 0
    assert artifacts.overlay is None
    assert artifacts.export_error is None


def test_progress_reaches_the_export_stage(stub_runner, synthetic_video, tmp_path, quick_params):
    """The runner's own stages must reach callers, or the UI stalls at 100% mid-write."""
    stages = []
    stub_runner.run(
        synthetic_video,
        output_dir=tmp_path,
        params=quick_params,
        on_progress=lambda event: stages.append(event.stage),
    )

    assert "detect" in stages
    assert stages[-1] == "export"


def test_progress_never_moves_backwards(stub_runner, synthetic_video, tmp_path, quick_params):
    values = []
    stub_runner.run(
        synthetic_video,
        output_dir=tmp_path,
        params=quick_params,
        on_progress=lambda event: values.append(event.progress),
    )

    assert values == sorted(values)
    assert values[-1] == pytest.approx(1.0)


def test_unwritable_output_fails_when_exports_are_required(
    stub_runner, synthetic_video, tmp_path, quick_params
):
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the output directory should be")

    with pytest.raises(OSError):
        stub_runner.run(synthetic_video, output_dir=blocked, params=quick_params)


def test_export_failure_is_survivable_for_the_web_worker(
    stub_runner, synthetic_video, tmp_path, quick_params
):
    """A job that cannot write downloads still returns its analysis."""
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the output directory should be")

    artifacts = stub_runner.run(
        synthetic_video, output_dir=blocked, params=quick_params, require_exports=False
    )

    assert artifacts.result.predictions
    assert artifacts.exports == {}
    assert artifacts.export_error


def test_cancellation_stops_the_run(stub_runner, synthetic_video, tmp_path, quick_params):
    from khoroos.pipeline.analyze import CancelledError

    with pytest.raises(CancelledError):
        stub_runner.run(
            synthetic_video,
            output_dir=tmp_path,
            params=quick_params,
            should_cancel=lambda: True,
        )
