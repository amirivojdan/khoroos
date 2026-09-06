"""Descriptive statistics derived from classified tracklets.

Durations and proportions describe model-assigned observations. No domain conclusions,
alert rules, or normative reference values are applied.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from khoroos.config import BEHAVIOUR_GROUPS, UNCERTAIN_LABEL
from khoroos.pipeline.types import ActionPrediction, VideoInfo

logger = logging.getLogger(__name__)


def _effective_intervals(
    predictions: list[ActionPrediction],
) -> list[tuple[ActionPrediction, float, float]]:
    """Assign overlapping windows non-overlapping temporal support.

    Action windows deliberately overlap so short behaviours are less likely to fall on a
    boundary. Summing every window's full duration would therefore double-count bird-time
    (or quadruple it in the thorough preset). For each track, adjacent windows split their
    overlap halfway between their centres. Gaps remain gaps and are never invented as
    observation time.
    """
    by_track: dict[int, list[ActionPrediction]] = defaultdict(list)
    for prediction in predictions:
        if prediction.end_seconds > prediction.start_seconds:
            by_track[prediction.track_id].append(prediction)

    effective: list[tuple[ActionPrediction, float, float]] = []
    for track_predictions in by_track.values():
        ordered = sorted(
            track_predictions,
            key=lambda p: (0.5 * (p.start_seconds + p.end_seconds), p.start_seconds),
        )
        centres = [0.5 * (p.start_seconds + p.end_seconds) for p in ordered]
        for index, prediction in enumerate(ordered):
            start = prediction.start_seconds
            end = prediction.end_seconds
            if index > 0:
                start = max(start, 0.5 * (centres[index - 1] + centres[index]))
            if index + 1 < len(ordered):
                end = min(end, 0.5 * (centres[index] + centres[index + 1]))
            if end > start:
                effective.append((prediction, start, end))
    return effective


def time_budget(predictions: list[ActionPrediction]) -> dict[str, Any]:
    """Share of observed bird-time spent in each behaviour.

    Uncertain predictions are counted separately rather than being folded into a class,
    so the budget always states how much of the footage the model declined to label.
    """
    seconds: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    total = 0.0

    for p, start, end in _effective_intervals(predictions):
        duration = end - start
        label = UNCERTAIN_LABEL if p.is_uncertain else p.label
        seconds[label] += duration
        counts[label] += 1
        total += duration

    confident_total = total - seconds.get(UNCERTAIN_LABEL, 0.0)

    return {
        "total_bird_seconds": round(total, 2),
        "confident_bird_seconds": round(confident_total, 2),
        "uncertain_share": round(seconds.get(UNCERTAIN_LABEL, 0.0) / total, 4) if total else 0.0,
        "by_class": {
            label: {
                "seconds": round(secs, 2),
                "share": round(secs / total, 4) if total else 0.0,
                # Share among confidently labelled time only — the number to quote when
                # comparing behaviour composition across videos.
                "share_of_confident": (
                    round(secs / confident_total, 4)
                    if confident_total and label != UNCERTAIN_LABEL
                    else 0.0
                ),
                "n_windows": counts[label],
            }
            for label, secs in sorted(seconds.items(), key=lambda kv: -kv[1])
        },
    }


def group_shares(
    budget: dict[str, Any],
    classes: list[str] | None = None,
    behaviour_groups: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate the class budget into ethological behaviour groups."""
    by_class = budget["by_class"]
    confident = budget["confident_bird_seconds"]

    out: dict[str, dict[str, Any]] = {}
    groups = BEHAVIOUR_GROUPS if behaviour_groups is None else behaviour_groups
    for group, members in groups.items():
        if classes is not None:
            members = tuple(m for m in members if m in classes)
            if not members:
                continue
        secs = sum(by_class.get(m, {}).get("seconds", 0.0) for m in members)
        out[group] = {
            "seconds": round(secs, 2),
            "share": round(secs / confident, 4) if confident else 0.0,
            "members": list(members),
        }
    return out


def per_bird_budgets(predictions: list[ActionPrediction]) -> dict[int, dict[str, Any]]:
    """Behaviour budget for each tracked bird.

    Approximate by construction: identity switches in a dense flock mean a "bird" here is
    a track, not necessarily one animal for the whole video.
    """
    by_track: dict[int, list[ActionPrediction]] = defaultdict(list)
    for p in predictions:
        by_track[p.track_id].append(p)

    out: dict[int, dict[str, Any]] = {}
    for track_id, preds in by_track.items():
        preds.sort(key=lambda p: p.start_seconds)
        budget = time_budget(preds)
        out[track_id] = {
            "track_id": track_id,
            "start_s": round(preds[0].start_seconds, 2),
            "end_s": round(preds[-1].end_seconds, 2),
            "observed_seconds": budget["total_bird_seconds"],
            "by_class": {k: v["share"] for k, v in budget["by_class"].items()},
            "dominant": next(iter(budget["by_class"]), None),
            "n_windows": len(preds),
        }
    return out


def activity_timeline(
    predictions: list[ActionPrediction],
    duration_seconds: float,
    bin_seconds: float = 5.0,
) -> dict[str, Any]:
    """Behaviour composition over video time, binned for plotting."""
    if duration_seconds <= 0:
        return {"bin_seconds": bin_seconds, "bins": [], "classes": []}

    n_bins = max(1, int(np.ceil(duration_seconds / bin_seconds)))
    labels = sorted({UNCERTAIN_LABEL if p.is_uncertain else p.label for p in predictions})
    matrix = np.zeros((n_bins, len(labels)), dtype=np.float32)
    label_index = {name: i for i, name in enumerate(labels)}

    for p, effective_start, effective_end in _effective_intervals(predictions):
        label = UNCERTAIN_LABEL if p.is_uncertain else p.label
        # Spread the window's duration across every bin it overlaps.
        start_bin = int(effective_start // bin_seconds)
        end_bin = int(max(effective_start, effective_end - 1e-6) // bin_seconds)
        for b in range(start_bin, min(end_bin, n_bins - 1) + 1):
            bin_start = b * bin_seconds
            bin_end = bin_start + bin_seconds
            overlap = min(effective_end, bin_end) - max(effective_start, bin_start)
            if overlap > 0:
                matrix[b, label_index[label]] += overlap

    return {
        "bin_seconds": bin_seconds,
        "classes": labels,
        "bins": [
            {
                "t": round(i * bin_seconds, 2),
                "seconds": [round(float(v), 3) for v in matrix[i]],
            }
            for i in range(n_bins)
        ],
    }


def spatial_activity(
    predictions: list[ActionPrediction],
    video: VideoInfo,
    grid: int = 12,
) -> dict[str, Any]:
    """Occupancy and dominant behaviour per spatial cell.

    Counts classified windows by the center of their representative box.
    """
    cell_w = video.width / grid
    cell_h = video.height / grid
    counts = np.zeros((grid, grid), dtype=np.int32)
    per_cell_labels: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    for p in predictions:
        x1, y1, x2, y2 = p.box
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        col = min(grid - 1, max(0, int(cx // cell_w)))
        row = min(grid - 1, max(0, int(cy // cell_h)))
        counts[row, col] += 1
        if not p.is_uncertain:
            per_cell_labels[(row, col)][p.label] += p.end_seconds - p.start_seconds

    dominant: dict[str, str] = {}
    for (row, col), labels in per_cell_labels.items():
        dominant[f"{row},{col}"] = max(labels.items(), key=lambda kv: kv[1])[0]

    return {
        "grid": grid,
        "cell_width": round(cell_w, 1),
        "cell_height": round(cell_h, 1),
        "counts": counts.tolist(),
        "dominant_behaviour": dominant,
    }


def population_stats(frame_counts: list[tuple[float, int]]) -> dict[str, Any]:
    """Bird-count statistics from raw per-frame detections."""
    if not frame_counts:
        return {"mean": 0.0, "min": 0, "max": 0, "series": []}
    values = np.array([c for _, c in frame_counts], dtype=np.float32)
    return {
        "mean": round(float(values.mean()), 2),
        "median": round(float(np.median(values)), 2),
        "min": int(values.min()),
        "max": int(values.max()),
        "series": [[round(t, 2), int(c)] for t, c in frame_counts],
    }


def bout_stats(predictions: list[ActionPrediction]) -> dict[str, Any]:
    """Bout counts and durations per behaviour.

    A bout is a run of consecutive windows of the same behaviour within one track, so
    "10 short feeding visits" reads differently from "one long feeding period".
    """
    intervals_by_track: dict[int, list[tuple[ActionPrediction, float, float]]] = defaultdict(list)
    for p, start, end in _effective_intervals(predictions):
        if not p.is_uncertain:
            intervals_by_track[p.track_id].append((p, start, end))

    bouts: dict[str, list[float]] = defaultdict(list)
    for intervals in intervals_by_track.values():
        intervals.sort(key=lambda item: item[1])
        current_label: str | None = None
        bout_start = 0.0
        bout_end = 0.0
        for p, start, end in intervals:
            contiguous = current_label == p.label and start <= bout_end + 1e-6
            if contiguous:
                bout_end = max(bout_end, end)
                continue
            if current_label is not None:
                bouts[current_label].append(bout_end - bout_start)
            current_label = p.label
            bout_start, bout_end = start, end
        if current_label is not None:
            bouts[current_label].append(bout_end - bout_start)

    return {
        label: {
            "n_bouts": len(durations),
            "mean_duration_s": round(float(np.mean(durations)), 2),
            "total_s": round(float(np.sum(durations)), 2),
        }
        for label, durations in sorted(bouts.items())
    }


def compute_metrics(
    predictions: list[ActionPrediction],
    video: VideoInfo,
    frame_counts: list[tuple[float, int]],
    bin_seconds: float = 5.0,
    classes: list[str] | None = None,
    behaviour_groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Return descriptive summaries for any classifier vocabulary.

    classes records the selected vocabulary; observed labels drive the statistics.
    Behavior groups are configurable, descriptive aggregations; an empty mapping disables them.
    """
    budget = time_budget(predictions)
    return {
        "time_budget": budget,
        "behaviour_groups": group_shares(budget, classes, behaviour_groups),
        "population": population_stats(frame_counts),
        "timeline": activity_timeline(predictions, video.duration_seconds, bin_seconds),
        "per_bird": per_bird_budgets(predictions),
        "bouts": bout_stats(predictions),
        "spatial": spatial_activity(predictions, video),
    }
