"""Exporting results to formats animal scientists actually work in."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from khoroos.pipeline.types import AnalysisResult


def predictions_to_csv(result: AnalysisResult, path: str | Path) -> Path:
    """One row per classified window — the tidy format for stats software."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "video",
                "track_id",
                "start_s",
                "end_s",
                "duration_s",
                "label",
                "confidence",
                "uncertain",
                "box_x1",
                "box_y1",
                "box_x2",
                "box_y2",
            ]
        )
        for p in result.predictions:
            writer.writerow(
                [
                    result.video.filename,
                    p.track_id,
                    f"{p.start_seconds:.3f}",
                    f"{p.end_seconds:.3f}",
                    f"{p.end_seconds - p.start_seconds:.3f}",
                    p.label,
                    f"{p.confidence:.4f}",
                    int(p.is_uncertain),
                    *(f"{v:.1f}" for v in p.box),
                ]
            )
    return path


def time_budget_to_csv(result: AnalysisResult, path: str | Path) -> Path:
    """Flock-level behaviour budget, one row per behaviour."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_class = result.metrics.get("time_budget", {}).get("by_class", {})

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["video", "behaviour", "bird_seconds", "share", "share_of_confident", "n_windows"]
        )
        for label, values in by_class.items():
            writer.writerow(
                [
                    result.video.filename,
                    label,
                    values["seconds"],
                    values["share"],
                    values["share_of_confident"],
                    values["n_windows"],
                ]
            )
    return path


def per_bird_to_csv(result: AnalysisResult, path: str | Path) -> Path:
    """Per-track behaviour budget. Approximate — identity switches are possible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    per_bird = result.metrics.get("per_bird", {})
    classes = result.model.classes or []

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["video", "track_id", "start_s", "end_s", "observed_s", "dominant", "n_windows"]
            + [f"share_{c}" for c in classes]
        )
        for values in per_bird.values():
            shares = values.get("by_class", {})
            writer.writerow(
                [
                    result.video.filename,
                    values["track_id"],
                    values["start_s"],
                    values["end_s"],
                    values["observed_seconds"],
                    values["dominant"],
                    values["n_windows"],
                ]
                + [shares.get(c, 0.0) for c in classes]
            )
    return path


def export_all(result: AnalysisResult, output_dir: str | Path) -> dict[str, Path]:
    """Write the full export bundle into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "result": result.save_json(output_dir / "result.json"),
        "predictions": predictions_to_csv(result, output_dir / "predictions.csv"),
        "time_budget": time_budget_to_csv(result, output_dir / "time_budget.csv"),
        "per_bird": per_bird_to_csv(result, output_dir / "per_bird.csv"),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    paths["metrics"] = metrics_path
    return paths
