"""What this installation can do: version, device, presets, ethogram, checkpoints.

The web UI renders its controls from this, and ``khoroos info`` prints it. Keeping one
description means the presets offered in the browser are by construction the presets the
CLI runs, and a missing checkpoint is reported the same way on both.
"""

from __future__ import annotations

from typing import Any

from khoroos.config import (
    ACTION_CLASSES,
    BEHAVIOUR_GROUPS,
    PRESETS,
    Settings,
    WelfareThresholds,
    get_settings,
)

DEFAULT_PRESET = "balanced"


def describe_environment(settings: Settings | None = None) -> dict[str, Any]:
    """Describe the installation as plain JSON-serialisable data."""
    settings = settings or get_settings()

    # Imported here so `khoroos.environment` stays importable from `khoroos/__init__`
    # without a cycle, and so importing this module does not drag in the model registry.
    from khoroos import __version__
    from khoroos.models.registry import checkpoint_status

    return {
        "version": __version__,
        "device": settings.resolved_device(),
        "presets": {name: dict(values) for name, values in PRESETS.items()},
        "default_preset": DEFAULT_PRESET,
        "classes": list(ACTION_CLASSES),
        "behaviour_groups": {name: list(members) for name, members in BEHAVIOUR_GROUPS.items()},
        "thresholds": WelfareThresholds().model_dump(),
        "models": checkpoint_status(settings),
        "max_upload_mb": settings.max_upload_mb,
        "cache_dir": str(settings.cache_dir),
    }
