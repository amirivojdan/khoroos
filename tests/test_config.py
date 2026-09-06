"""The shared configuration layer both surfaces build their parameters from."""

from __future__ import annotations

import pytest

from khoroos.config import (
    PRESETS,
    AnalysisParams,
    params_for_preset,
    parse_overrides,
)


def test_every_preset_builds_valid_params():
    for name in PRESETS:
        assert isinstance(params_for_preset(name), AnalysisParams)


def test_none_overrides_fall_back_to_the_preset():
    """Unset CLI flags and absent form fields arrive as None and must not clobber."""
    params = params_for_preset("fast", min_confidence=None, detection_stride=None)
    assert params.detection_stride == PRESETS["fast"]["detection_stride"]


def test_overrides_beat_the_preset():
    assert params_for_preset("fast", detection_stride=1).detection_stride == 1


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="Unknown preset"):
        params_for_preset("nonsense")


def test_unknown_parameter_is_rejected_not_ignored():
    """A typo has to fail loudly: silently ignoring it returns default-run results."""
    with pytest.raises(ValueError) as exc:
        params_for_preset("fast", windo_seconds=2.0)

    assert "unknown analysis parameter" in str(exc.value)
    assert "window_seconds" in str(exc.value)  # the valid names are listed


def test_out_of_range_parameter_is_rejected():
    with pytest.raises(ValueError, match="min_confidence"):
        params_for_preset("fast", min_confidence=1.5)


def test_non_positive_duration_limit_is_rejected():
    with pytest.raises(ValueError, match="max_duration_seconds"):
        params_for_preset("fast", max_duration_seconds=0)


def test_string_values_are_coerced():
    """`--set window_seconds=3` arrives as text and must land as a float."""
    assert params_for_preset("fast", window_seconds="3").window_seconds == 3.0


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# name=value parsing
# ---------------------------------------------------------------------------


def test_parse_overrides_reads_pairs():
    assert parse_overrides(["a=1", " b = two "]) == {"a": "1", "b": "two"}


def test_parse_overrides_keeps_values_containing_equals():
    assert parse_overrides(["note=a=b"]) == {"note": "a=b"}


@pytest.mark.parametrize("bad", ["window_seconds", "", "=3"])
def test_parse_overrides_rejects_malformed_input(bad):
    with pytest.raises(ValueError, match="name=value"):
        parse_overrides([bad])


@pytest.mark.parametrize("value", [3, {}, [""], ["feeding", "feeding"], ["uncertain"]])
def test_invalid_action_class_lists_are_validation_errors(value):
    with pytest.raises(ValueError, match="action_classes"):
        AnalysisParams(action_classes=value)


@pytest.mark.parametrize("device", ["banana", "cuda:-1", "cuda:x", "cpu:2", ""])
def test_invalid_devices_rejected_at_creation_and_assignment(device):
    from khoroos.config import Settings

    with pytest.raises(ValueError, match="device"):
        Settings(device=device)
    settings = Settings(device="cpu")
    with pytest.raises(ValueError, match="device"):
        settings.device = device
    assert settings.device == "cpu"


@pytest.mark.parametrize("device", ["cpu", "auto", "mps", "cuda", "cuda:1"])
def test_valid_device_syntax_without_loading_models(device):
    from khoroos.config import Settings

    assert Settings(device=device).device == device


def test_settings_overrides_are_validated_copies():
    from khoroos.config import Settings

    original = Settings(device="cpu")
    changed = original.with_overrides(device="cuda:1", behaviour_groups={"custom": ["a"]})
    assert original.device == "cpu"
    assert changed.device == "cuda:1"
    assert "custom" not in original.behaviour_groups
    with pytest.raises(ValueError):
        original.with_overrides(device="banana")


@pytest.mark.parametrize("groups", [{"a": ["x"], "b": ["x"]}, {"a": []}, {"uncertain": ["x"]}])
def test_invalid_behavior_groups_rejected(groups):
    with pytest.raises(ValueError):
        AnalysisParams(behaviour_groups=groups)


def test_group_overrides_parse_cli_json():
    assert AnalysisParams(behaviour_groups='{"activity":["moving"]}').behaviour_groups == {"activity": ["moving"]}
