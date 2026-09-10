"""What this installation can do: version, device, presets, ethogram, checkpoints.

The web UI renders its controls from this, and ``khoroos info`` prints it. Keeping one
description means the presets offered in the browser are by construction the presets the
CLI runs, and a missing checkpoint is reported the same way on both.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from khoroos.config import (
    PRESETS,
    Settings,
    get_settings,
)

DEFAULT_PRESET = "balanced"


def describe_environment(
    settings: Settings | None = None,
    *,
    classifier: object | None = None,
    detector: object | None = None,
) -> dict[str, Any]:
    """Describe the installation as plain JSON-serialisable data."""
    settings = settings or get_settings()

    # Imported here so `khoroos.environment` stays importable from `khoroos/__init__`
    # without a cycle, and so importing this module does not drag in the model registry.
    from khoroos import __version__
    from khoroos.devices import available_devices, resolve_device
    from khoroos.interfaces import component_name
    from khoroos.models.metadata import checkpoint_classes
    from khoroos.models.registry import checkpoint_status
    from khoroos.models.selection import validate_classes

    metadata_error = None
    if classifier is not None:
        classes = validate_classes(classifier.classes)
        classes_source = "classifier"
    else:
        try:
            classes, path = checkpoint_classes(settings)
            classes_source = str(path) if path else "unavailable"
        except (OSError, ValueError, AttributeError) as exc:
            classes, classes_source, metadata_error = [], "unavailable", str(exc)
    groups = {
        name: [label for label in members if label in classes]
        for name, members in settings.behaviour_groups.items()
        if any(label in classes for label in members)
    }

    models = checkpoint_status(settings)
    for kind, component in (("action", classifier), ("detector", detector)):
        if component is not None:
            models[kind] = {
                "available": True,
                "path": str(getattr(component, "checkpoint", None) or component_name(component)),
                "source": "injected",
            }

    default_device = settings.device
    if default_device == "cuda":
        # Keep an unavailable configured default visible rather than replacing it.
        with suppress(ValueError):
            default_device = resolve_device(default_device)

    return {
        "version": __version__,
        "device": settings.resolved_device(),
        "default_device": default_device,
        "devices": available_devices(),
        "presets": {name: dict(values) for name, values in PRESETS.items()},
        "default_preset": DEFAULT_PRESET,
        "classes": classes,
        "classes_source": classes_source,
        "metadata_error": metadata_error,
        "behaviour_groups": groups,
        "models": models,
        "max_upload_mb": settings.max_upload_mb,
        "cache_dir": str(settings.cache_dir),
        "tracklet_directories": {
            "raw": str((settings.cache_dir / "tracklets" / "raw").expanduser().resolve()),
            "classified": str(
                (settings.cache_dir / "tracklets" / "classified").expanduser().resolve()
            ),
        },
    }
