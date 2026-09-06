"""Read classifier vocabulary from local JSON metadata without loading model weights."""

import json
from pathlib import Path

from khoroos.config import Settings
from khoroos.models.registry import local_config_path
from khoroos.models.selection import validate_classes


def labels_from_mapping(id2label: dict) -> list[str]:
    """Check output-column IDs and return labels in their declared model order."""
    if not isinstance(id2label, dict):
        raise ValueError("Checkpoint id2label must be a mapping")
    try:
        mapping = {int(k): v for k, v in id2label.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint label IDs must be integers") from exc
    if len(mapping) != len(id2label) or sorted(mapping) != list(range(len(mapping))):
        raise ValueError("Checkpoint label IDs must be contiguous starting at zero")
    return validate_classes([mapping[i] for i in range(len(mapping))])


def checkpoint_classes(settings: Settings) -> tuple[list[str], Path | None]:
    """Return local checkpoint labels, or an empty vocabulary when metadata is absent."""
    path = local_config_path("action", settings)
    if path is None:
        return [], None
    config = json.loads(path.read_text(encoding="utf-8"))
    return labels_from_mapping(config.get("id2label")), path
