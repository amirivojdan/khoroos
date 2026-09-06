"""Khoroos — a toolkit for poultry behavior analysis.

Typical use::

    from khoroos import analyze_video

    result = analyze_video("farm.mp4", preset="balanced")
    print(result.metrics["time_budget"])

Or from the command line::

    khoroos ui                      # web interface
    khoroos analyze farm.mp4 -o out # batch analysis
"""

from khoroos.config import (
    ACTION_CLASSES,
    BEHAVIOUR_GROUPS,
    PRESETS,
    UNCERTAIN_LABEL,
    AnalysisParams,
    Settings,
    get_settings,
    params_for_preset,
)
from khoroos.pipeline.types import (
    ActionPrediction,
    AnalysisResult,
    ProgressEvent,
    Track,
    Tracklet,
    VideoInfo,
)


def _installed_version() -> str:
    """Read the version from package metadata, so pyproject.toml stays its one source."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("khoroos")
    except PackageNotFoundError:  # running from a source tree that was never installed
        return "0.0.0+unknown"


__version__ = _installed_version()

__all__ = [
    "ACTION_CLASSES",
    "BEHAVIOUR_GROUPS",
    "PRESETS",
    "UNCERTAIN_LABEL",
    "ActionPrediction",
    "Detector",
    "VideoClassifier",
    "Tracker",
    "VideoReader",
    "PipelineComponents",
    "SelectedClasses",
    "AnalysisParams",
    "AnalysisResult",
    "AnalysisRunner",
    "ProgressEvent",
    "RunArtifacts",
    "Settings",
    "Track",
    "Tracklet",
    "VideoAnalyzer",
    "VideoInfo",
    "__version__",
    "analyze_video",
    "describe_environment",
    "get_settings",
    "params_for_preset",
]

#: Names served lazily, and the module each comes from. Everything here transitively
#: imports torch.
_LAZY = {
    "Detector": "khoroos.interfaces",
    "VideoClassifier": "khoroos.interfaces",
    "Tracker": "khoroos.interfaces",
    "VideoReader": "khoroos.interfaces",
    "PipelineComponents": "khoroos.pipeline.components",
    "SelectedClasses": "khoroos.models.selection",
    "AnalysisRunner": "khoroos.pipeline.runner",
    "RunArtifacts": "khoroos.pipeline.runner",
    "VideoAnalyzer": "khoroos.pipeline.analyze",
    "analyze_video": "khoroos.pipeline.analyze",
    "describe_environment": "khoroos.environment",
}


def __getattr__(name: str):
    """Defer the heavy pipeline import until something actually needs it.

    Importing :mod:`khoroos` must stay cheap — the CLI touches it just to print a
    version, and pulling in torch and transformers for that costs seconds.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
