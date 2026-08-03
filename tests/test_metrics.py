"""Welfare metric tests.

The emphasis is on the honesty guarantees: uncertain time is never folded into a class,
thin samples never raise alerts, and unreliable classes never raise alerts.
"""

from __future__ import annotations

import pytest

from khoroos.config import UNCERTAIN_LABEL, WelfareThresholds
from khoroos.pipeline.types import ActionPrediction, VideoInfo
from khoroos.welfare.metrics import (
    activity_timeline,
    bout_stats,
    compute_metrics,
    evaluate_alerts,
    group_shares,
    per_bird_budgets,
    time_budget,
)

VIDEO = VideoInfo(
    filename="test.mp4", fps=25.0, duration_seconds=60.0, width=1280, height=720, num_frames=1500
)


def make_prediction(
    track_id=1, start=0.0, end=2.0, label="resting", confidence=0.9, uncertain=False
) -> ActionPrediction:
    return ActionPrediction(
        track_id=track_id,
        start_seconds=start,
        end_seconds=end,
        label=UNCERTAIN_LABEL if uncertain else label,
        confidence=confidence,
        probabilities={label: confidence},
        box=(10.0, 10.0, 110.0, 110.0),
        is_uncertain=uncertain,
    )


# ---------------------------------------------------------------------------
# Time budget
# ---------------------------------------------------------------------------


def test_time_budget_shares_sum_to_one():
    predictions = [
        make_prediction(label="resting", start=0, end=2),
        make_prediction(label="feeding", start=2, end=4),
        make_prediction(label="preening", start=4, end=6),
    ]
    budget = time_budget(predictions)
    assert budget["total_bird_seconds"] == pytest.approx(6.0)
    # Shares are rounded to 4 dp for the JSON payload, so they sum to 1 only approximately.
    assert sum(v["share"] for v in budget["by_class"].values()) == pytest.approx(1.0, abs=1e-3)


def test_uncertain_time_is_reported_not_absorbed():
    predictions = [
        make_prediction(label="resting", start=0, end=2),
        make_prediction(start=2, end=4, uncertain=True),
    ]
    budget = time_budget(predictions)

    assert budget["uncertain_share"] == pytest.approx(0.5)
    assert budget["confident_bird_seconds"] == pytest.approx(2.0)
    # 'resting' is all of the *confident* time but only half the total.
    assert budget["by_class"]["resting"]["share"] == pytest.approx(0.5)
    assert budget["by_class"]["resting"]["share_of_confident"] == pytest.approx(1.0)


def test_empty_budget_does_not_divide_by_zero():
    budget = time_budget([])
    assert budget["total_bird_seconds"] == 0
    assert budget["uncertain_share"] == 0.0


def test_group_shares_use_confident_time_only():
    predictions = [
        make_prediction(label="preening", start=0, end=2),   # comfort
        make_prediction(label="walking", start=2, end=4),    # locomotion
        make_prediction(start=4, end=8, uncertain=True),
    ]
    groups = group_shares(time_budget(predictions))
    assert groups["comfort"]["share"] == pytest.approx(0.5)
    assert groups["locomotion"]["share"] == pytest.approx(0.5)


def test_overlapping_windows_do_not_double_count_bird_time():
    predictions = [
        make_prediction(label="resting", start=0, end=2),
        make_prediction(label="feeding", start=1, end=3),
    ]
    budget = time_budget(predictions)

    assert budget["total_bird_seconds"] == pytest.approx(3.0)
    assert budget["by_class"]["resting"]["seconds"] == pytest.approx(1.5)
    assert budget["by_class"]["feeding"]["seconds"] == pytest.approx(1.5)


def test_overlapping_windows_of_one_class_cover_the_union_once():
    predictions = [
        make_prediction(label="resting", start=0, end=2),
        make_prediction(label="resting", start=1, end=3),
        make_prediction(label="resting", start=2, end=4),
    ]
    budget = time_budget(predictions)
    assert budget["total_bird_seconds"] == pytest.approx(4.0)
    assert budget["by_class"]["resting"]["seconds"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Per-bird and bouts
# ---------------------------------------------------------------------------


def test_per_bird_budgets_split_by_track():
    predictions = [
        make_prediction(track_id=1, label="feeding", start=0, end=2),
        make_prediction(track_id=2, label="resting", start=0, end=2),
        make_prediction(track_id=2, label="resting", start=2, end=4),
    ]
    per_bird = per_bird_budgets(predictions)
    assert set(per_bird) == {1, 2}
    assert per_bird[2]["observed_seconds"] == pytest.approx(4.0)
    assert per_bird[2]["dominant"] == "resting"


def test_consecutive_windows_merge_into_one_bout():
    predictions = [
        make_prediction(label="feeding", start=0, end=2),
        make_prediction(label="feeding", start=2, end=4),
        make_prediction(label="feeding", start=4, end=6),
    ]
    bouts = bout_stats(predictions)
    assert bouts["feeding"]["n_bouts"] == 1
    assert bouts["feeding"]["mean_duration_s"] == pytest.approx(6.0)


def test_separated_windows_are_separate_bouts():
    predictions = [
        make_prediction(label="feeding", start=0, end=2),
        make_prediction(label="resting", start=2, end=4),
        make_prediction(label="feeding", start=4, end=6),
    ]
    bouts = bout_stats(predictions)
    assert bouts["feeding"]["n_bouts"] == 2


def test_bouts_do_not_merge_across_birds():
    predictions = [
        make_prediction(track_id=1, label="feeding", start=0, end=2),
        make_prediction(track_id=2, label="feeding", start=2, end=4),
    ]
    assert bout_stats(predictions)["feeding"]["n_bouts"] == 2


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def test_timeline_bins_cover_the_video():
    predictions = [make_prediction(label="resting", start=0, end=10)]
    timeline = activity_timeline(predictions, duration_seconds=60.0, bin_seconds=5.0)
    assert len(timeline["bins"]) == 12
    assert sum(sum(b["seconds"]) for b in timeline["bins"]) == pytest.approx(10.0, abs=0.01)


def test_timeline_splits_a_window_across_bins():
    predictions = [make_prediction(label="resting", start=3, end=7)]
    timeline = activity_timeline(predictions, duration_seconds=20.0, bin_seconds=5.0)
    assert timeline["bins"][0]["seconds"][0] == pytest.approx(2.0)
    assert timeline["bins"][1]["seconds"][0] == pytest.approx(2.0)


def test_timeline_does_not_double_count_overlapping_windows():
    predictions = [
        make_prediction(label="resting", start=0, end=2),
        make_prediction(label="feeding", start=1, end=3),
    ]
    timeline = activity_timeline(predictions, duration_seconds=3.0, bin_seconds=1.0)
    assert sum(sum(values for values in b["seconds"]) for b in timeline["bins"]) == pytest.approx(
        3.0
    )


# ---------------------------------------------------------------------------
# Alerts — the honesty guarantees
# ---------------------------------------------------------------------------


def test_thin_sample_never_raises_a_threshold_alert():
    predictions = [make_prediction(label="resting", start=0, end=4)]
    budget = time_budget(predictions)
    alerts = evaluate_alerts(budget=budget, groups=group_shares(budget), thresholds=WelfareThresholds())

    assert [a["code"] for a in alerts] == ["insufficient_data"]
    assert all(a["level"] != "warning" for a in alerts)


def test_low_comfort_raises_a_warning_when_well_sampled():
    predictions = [make_prediction(label="resting", start=i * 2, end=i * 2 + 2) for i in range(60)]
    budget = time_budget(predictions)
    alerts = evaluate_alerts(budget=budget, groups=group_shares(budget), thresholds=WelfareThresholds())

    codes = {a["code"] for a in alerts if a["level"] == "warning"}
    assert "comfort_behaviour_low" in codes
    assert "inactivity_high" in codes


def test_alert_is_suppressed_for_an_unreliable_class():
    predictions = [make_prediction(label="resting", start=i * 2, end=i * 2 + 2) for i in range(60)]
    budget = time_budget(predictions)
    weak = dict.fromkeys(
        ["preening", "dust_bathing", "wing_flapping", "stretching", "body_shaking"], 0.1
    )
    alerts = evaluate_alerts(
        budget=budget, groups=group_shares(budget), thresholds=WelfareThresholds(), per_class_f1=weak
    )

    codes = {a["code"] for a in alerts}
    assert "comfort_behaviour_low_suppressed" in codes
    assert "comfort_behaviour_low" not in codes


def test_high_uncertainty_is_flagged():
    # Clearly past the >50% trigger, not sitting on the boundary.
    predictions = [make_prediction(start=i * 2, end=i * 2 + 2, uncertain=True) for i in range(60)]
    predictions += [make_prediction(label="resting", start=200 + i * 2, end=202 + i * 2) for i in range(40)]
    budget = time_budget(predictions)
    alerts = evaluate_alerts(budget=budget, groups=group_shares(budget), thresholds=WelfareThresholds())
    assert any(a["code"] == "high_uncertainty" for a in alerts)


# ---------------------------------------------------------------------------
# Full metric set
# ---------------------------------------------------------------------------


def test_compute_metrics_is_json_serialisable():
    import json

    predictions = [
        make_prediction(track_id=i % 3, label="feeding", start=i * 2, end=i * 2 + 2)
        for i in range(20)
    ]
    metrics = compute_metrics(predictions, VIDEO, [(t / 25.0, 5) for t in range(0, 100, 5)])
    # Keys are used verbatim by the frontend, so a rename must break a test.
    assert {"time_budget", "indicators", "timeline", "per_bird", "spatial", "alerts"} <= set(metrics)
    json.dumps(metrics)


def test_compute_metrics_handles_no_predictions():
    metrics = compute_metrics([], VIDEO, [])
    assert metrics["time_budget"]["total_bird_seconds"] == 0
    assert metrics["indicators"]["comfort_index"] == 0.0
