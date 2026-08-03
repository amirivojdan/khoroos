"""The shared configuration layer both surfaces build their parameters from."""

from __future__ import annotations

import pytest

from khoroos.config import (
    PRESETS,
    AnalysisParams,
    WelfareThresholds,
    params_for_preset,
    parse_overrides,
    thresholds_from_overrides,
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


def test_thresholds_default_when_nothing_is_overridden():
    assert thresholds_from_overrides() == WelfareThresholds()
    assert thresholds_from_overrides({}) == WelfareThresholds()


def test_thresholds_apply_only_the_named_override():
    thresholds = thresholds_from_overrides({"min_comfort_share": 0.2})
    assert thresholds.min_comfort_share == 0.2
    assert thresholds.max_inactive_share == WelfareThresholds().max_inactive_share


def test_unknown_threshold_is_rejected():
    with pytest.raises(ValueError, match="unknown welfare threshold"):
        thresholds_from_overrides({"min_comfort": 0.2})


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
