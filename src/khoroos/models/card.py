"""Model card: held-out reliability figures shown next to every prediction.

The action model is a frozen-encoder linear probe trained on an imbalanced dataset, so a
raw class label alone over-states certainty. When a ``model_card.json`` sits beside the
checkpoint, its per-class F1 and support are surfaced in the UI as descriptive evaluation metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CARD_FILENAME = "model_card.json"


def load_model_card(checkpoint: str | Path) -> dict[str, Any]:
    """Load the model card next to a checkpoint, or an empty card when absent."""
    path = Path(checkpoint) / CARD_FILENAME
    if not path.exists():
        logger.debug("No model card at %s; reliability figures unavailable.", path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read model card %s: %s", path, exc)
        return {}


def per_class_f1(card: dict[str, Any]) -> dict[str, float]:
    """Extract ``{class_name: f1}`` from a card, tolerating a missing or partial card."""
    report = card.get("classification_report") or {}
    out: dict[str, float] = {}
    for name, values in report.items():
        if isinstance(values, dict) and "f1-score" in values:
            out[name] = float(values["f1-score"])
    return out


def per_class_support(card: dict[str, Any]) -> dict[str, int]:
    """Extract ``{class_name: n_test_clips}`` from a card."""
    report = card.get("classification_report") or {}
    out: dict[str, int] = {}
    for name, values in report.items():
        if isinstance(values, dict) and "support" in values:
            out[name] = int(values["support"])
    return out


def write_model_card(
    checkpoint: str | Path,
    classification_report: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    notes: str = "",
) -> Path:
    """Write a model card beside a checkpoint.

    ``classification_report`` is the dict form of sklearn's
    ``classification_report(..., output_dict=True)``.
    """
    path = Path(checkpoint) / CARD_FILENAME
    payload = {
        "classification_report": classification_report,
        "metrics": metrics or {},
        "notes": notes,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
