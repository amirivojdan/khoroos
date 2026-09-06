"""CLI tests.

The CLI is a thin surface over the shared library, so these check the wiring — argument
parsing, error reporting, exit codes — rather than re-testing the pipeline.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from khoroos.cli import app

runner = CliRunner()


@pytest.fixture
def cli(monkeypatch, stub_analyzer):
    """Make ``khoroos analyze`` build its runner around stub models."""
    import khoroos.pipeline.runner as runner_module

    real = runner_module.AnalysisRunner
    monkeypatch.setattr(runner_module, "AnalysisRunner", lambda **_: real(analyzer=stub_analyzer))
    return runner


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def test_analyze_writes_the_export_bundle(cli, synthetic_video, tmp_path):
    output = tmp_path / "out"
    result = cli.invoke(
        app,
        [
            "analyze",
            str(synthetic_video),
            "-o",
            str(output),
            "--preset",
            "fast",
            "-s",
            "max_duration_seconds=3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "predictions.csv").exists()
    assert (output / "metrics.json").exists()
    assert "Summary" in result.output


def test_analyze_accepts_parameter_overrides(cli, synthetic_video, tmp_path):
    result = cli.invoke(
        app,
        [
            "analyze",
            str(synthetic_video),
            "-o",
            str(tmp_path / "out"),
            "-s",
            "window_seconds=1.5",
            "-s",
            "max_duration_seconds=3",
        ],
    )

    assert result.exit_code == 0, result.output
    params = json.loads((tmp_path / "out" / "result.json").read_text())["params"]
    assert params["window_seconds"] == 1.5


def test_analyze_accepts_batch_size_flags(cli, synthetic_video, tmp_path):
    result = cli.invoke(
        app,
        [
            "analyze",
            str(synthetic_video),
            "-o",
            str(tmp_path / "out"),
            "--detection-batch-size",
            "7",
            "--action-batch-size",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    params = json.loads((tmp_path / "out" / "result.json").read_text())["params"]
    assert params["detection_batch_size"] == 7
    assert params["action_batch_size"] == 2


def test_set_wins_over_its_named_shorthand(cli, synthetic_video, tmp_path):
    """`--set` is applied after the named flags, which is what makes it a general escape hatch."""
    result = cli.invoke(
        app,
        [
            "analyze",
            str(synthetic_video),
            "-o",
            str(tmp_path / "out"),
            "--detection-batch-size",
            "7",
            "-s",
            "detection_batch_size=3",
        ],
    )

    assert result.exit_code == 0, result.output
    params = json.loads((tmp_path / "out" / "result.json").read_text())["params"]
    assert params["detection_batch_size"] == 3


def test_analyze_rejects_a_zero_batch_size(cli, synthetic_video, tmp_path):
    result = cli.invoke(
        app, ["analyze", str(synthetic_video), "-o", str(tmp_path), "--action-batch-size", "0"]
    )

    assert result.exit_code == 2
    assert "action_batch_size" in result.output


def test_analyze_rejects_an_unknown_parameter(cli, synthetic_video, tmp_path):
    result = cli.invoke(
        app, ["analyze", str(synthetic_video), "-o", str(tmp_path), "-s", "windo_seconds=2"]
    )

    assert result.exit_code == 2
    assert "unknown analysis parameter" in result.output
    assert "window_seconds" in result.output  # the message lists the valid names


def test_analyze_rejects_malformed_overrides(cli, synthetic_video, tmp_path):
    result = cli.invoke(
        app, ["analyze", str(synthetic_video), "-o", str(tmp_path), "-s", "window_seconds"]
    )

    assert result.exit_code == 2
    assert "name=value" in result.output


def test_analyze_reports_a_missing_video_without_loading_models(tmp_path):
    """No stub fixture here: the check must fire before anything expensive happens."""
    result = runner.invoke(app, ["analyze", str(tmp_path / "nope.mp4"), "-o", str(tmp_path)])

    assert result.exit_code == 2
    assert "No such video" in result.output


def test_analyze_rejects_an_unknown_preset(cli, synthetic_video, tmp_path):
    result = cli.invoke(
        app, ["analyze", str(synthetic_video), "-o", str(tmp_path), "--preset", "nonsense"]
    )

    assert result.exit_code == 2
    assert "nonsense" in result.output


# ---------------------------------------------------------------------------
# info / version
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_metadata", [False, True])
def test_info_json_is_the_description_the_web_also_serves(monkeypatch, tmp_path, with_metadata):
    """`khoroos info --json` and GET /api/config both render describe_environment()."""
    import khoroos.config as config
    from khoroos.environment import describe_environment

    settings = config.Settings(device="cpu", cache_dir=tmp_path, action_path=tmp_path)
    monkeypatch.setattr(config, "_settings", settings)
    if with_metadata:
        (tmp_path / "config.json").write_text(json.dumps({
            "id2label": dict(enumerate(config.ACTION_CLASSES)),
        }))

    result = runner.invoke(app, ["info", "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert payload == describe_environment(settings)
    assert payload["classes"] == (list(config.ACTION_CLASSES) if with_metadata else [])
    assert set(payload["presets"]) == {"fast", "balanced", "thorough"}


def test_info_is_human_readable():
    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0, result.output
    assert "Presets" in result.output
    assert "balanced" in result.output
    assert "Alert thresholds" not in result.output


def test_version_reports_the_package_version():
    from khoroos import __version__

    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_class_selection_reaches_pipeline(cli, synthetic_video, tmp_path):
    response = cli.invoke(
        app,
        [
            "analyze",
            str(synthetic_video),
            "-o",
            str(tmp_path),
            "-s",
            "action_classes=feeding,drinking",
        ],
    )
    assert response.exit_code == 0, response.output
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["model"]["classes"] == ["feeding", "drinking"]
    assert "indicators" not in result["metrics"]


def test_welfare_threshold_option_is_removed():
    response = runner.invoke(app, ["analyze", "--help"])
    assert response.exit_code == 0
    assert "--threshold" not in response.output


def test_device_typo_fails_before_runner_construction(synthetic_video, monkeypatch):
    from khoroos.pipeline import runner as runner_module

    def fail(**kwargs):
        pytest.fail("Runner must not be constructed for an invalid device")

    monkeypatch.setattr(runner_module, "AnalysisRunner", fail)
    result = runner.invoke(app, ["analyze", str(synthetic_video), "--device", "banana"])
    assert result.exit_code == 2
    assert "device" in result.output


def test_cli_device_override_does_not_mutate_global(cli, synthetic_video, tmp_path):
    from khoroos.config import get_settings

    original = get_settings().device
    result = cli.invoke(app, ["analyze", str(synthetic_video), "-o", str(tmp_path), "--device", "cpu"])
    assert result.exit_code == 0, result.output
    assert get_settings().device == original
