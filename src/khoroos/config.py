"""Runtime configuration for Khoroos.

Every tunable lives here so the CLI, the web UI and library callers all agree on defaults.
Settings can be overridden by environment variables prefixed with ``KHOROOS_``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Ethogram
# ---------------------------------------------------------------------------

#: The 15 behaviours the action model was trained on (ChickenAct ethogram).
ACTION_CLASSES: tuple[str, ...] = (
    "body_shaking",
    "drinking",
    "dust_bathing",
    "feeding",
    "head_scratching",
    "litter_pecking",
    "litter_scratching",
    "pooping",
    "preening",
    "resting",
    "running",
    "standing",
    "stretching",
    "walking",
    "wing_flapping",
)

#: Label used when the classifier is not confident enough to commit to a class.
UNCERTAIN_LABEL = "uncertain"

#: Descriptive groupings of related behavior labels.
BEHAVIOUR_GROUPS: dict[str, tuple[str, ...]] = {
    "maintenance": ("preening", "dust_bathing", "wing_flapping", "stretching", "body_shaking"),
    "locomotion": ("walking", "running"),
    "inactive": ("resting", "standing"),
    "ingestive": ("feeding", "drinking"),
    "foraging": ("litter_pecking", "litter_scratching"),
    "other": ("head_scratching", "pooping"),
}


def validate_groups(value):
    """Validate descriptive group membership; no reserved labels or duplicate memberships."""
    if not isinstance(value, dict):
        raise ValueError("behaviour_groups must be a mapping of names to label lists")
    seen = set()
    for name, members in value.items():
        if not isinstance(name, str) or not name.strip() or name in (UNCERTAIN_LABEL, "ungrouped"):
            raise ValueError("Group names must be nonempty and not reserved")
        if not isinstance(members, (list, tuple)) or not members:
            raise ValueError("Each behavior group must have a nonempty label list")
        for member in members:
            if not isinstance(member, str) or not member.strip() or member == UNCERTAIN_LABEL:
                raise ValueError("Group labels must be nonempty and exclude 'uncertain'")
            if member in seen:
                raise ValueError(f"Label {member!r} belongs to more than one group or is repeated")
            seen.add(member)
    return {name: list(members) for name, members in value.items()}


# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------


class AnalysisParams(BaseModel):
    """Parameters controlling a single video analysis run."""

    # Both surfaces let users name a parameter (``--set`` on the CLI, an overrides map in
    # the web API), so an unknown name has to be an error. Silently ignoring a typo would
    # hand back results that quietly used the defaults.
    model_config = ConfigDict(extra="forbid")

    # -- detection ---------------------------------------------------------
    detection_confidence: float = Field(0.30, ge=0.0, le=1.0)
    nms_threshold: float = Field(0.60, ge=0.0, le=1.0)
    box_padding: float = Field(0.10, ge=0.0, le=1.0)
    detection_batch_size: int = Field(16, ge=1)
    #: Run the detector on every Nth frame. Intermediate frames are carried by the
    #: Kalman filter, so this trades tracking precision for throughput.
    detection_stride: int = Field(2, ge=1)
    invalid_boxes: Literal["error", "drop"] = "error"
    #: None uses Settings.behaviour_groups; an empty mapping disables grouping.
    behaviour_groups: dict[str, list[str]] | None = None

    @field_validator("behaviour_groups", mode="before")
    @classmethod
    def parse_groups(cls, value):
        if isinstance(value, str):
            value = json.loads(value)
        return validate_groups(value) if value is not None else None

    # -- tracking ----------------------------------------------------------
    track_iou_threshold: float = Field(0.30, ge=0.0, le=1.0)
    track_max_age: int = Field(15, ge=1, description="Frames a track survives without detections.")
    track_min_hits: int = Field(3, ge=1, description="Detections before a track is confirmed.")

    # -- tracklet windows --------------------------------------------------
    window_seconds: float = Field(2.0, gt=0.0)
    stride_seconds: float = Field(1.0, gt=0.0)
    min_window_coverage: float = Field(
        0.6, ge=0.0, le=1.0, description="Fraction of window frames a track must be present for."
    )
    #: Stability gates ported from the dataset-generation cropper.
    stability_min_iou: float = Field(0.50, ge=0.0, le=1.0)
    stability_max_bad_frac: float = Field(0.10, ge=0.0, le=1.0)
    stability_jump_factor: float = Field(0.70, gt=0.0)
    min_crop_size: int = Field(100, ge=1)
    min_crop_aspect_ratio: float = Field(0.50, gt=0.0, le=1.0)
    #: Percentile of per-frame box size used for the fixed crop. Robust to the occasional
    #: oversized detection, which a plain max would let dictate the whole window.
    crop_size_percentile: float = Field(90.0, gt=0.0, le=100.0)
    #: A crop larger than this fraction of the frame is not a single bird.
    max_crop_frame_fraction: float = Field(0.50, gt=0.0, le=1.0)

    # -- classification ----------------------------------------------------
    action_batch_size: int = Field(4, ge=1)
    action_classes: list[str] | None = Field(
        None, description="Ordered classes of interest; None uses all checkpoint labels."
    )

    min_confidence: float = Field(
        0.50, ge=0.0, le=1.0, description="Below this the prediction is reported as 'uncertain'."
    )

    #: Optional export roots. Each analysis creates a unique subdirectory in each root.
    raw_tracklets_dir: str | None = None
    classified_tracklets_dir: str | None = None

    @field_validator("raw_tracklets_dir", "classified_tracklets_dir")
    @classmethod
    def check_tracklet_directory(cls, value):
        if value is not None and (not value.strip() or "\x00" in value):
            raise ValueError("Tracklet directory must be a nonempty path")
        return value

    # -- limits ------------------------------------------------------------
    max_duration_seconds: float | None = Field(
        None, gt=0.0, description="Analyse only the first N seconds. None = whole video."
    )

    @field_validator("action_classes", mode="before")
    @classmethod
    def parse_action_classes(cls, value):
        # CLI --set values arrive as strings; Python and JSON callers use lists.
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",")]
        if value is not None:
            if not isinstance(value, (list, tuple)):
                raise ValueError("action_classes must be a list or comma-separated string")
            if not value or any(not isinstance(c, str) or not c.strip() for c in value):
                raise ValueError("action_classes must contain nonempty labels")
            if len(set(value)) != len(value) or UNCERTAIN_LABEL in value:
                raise ValueError("action_classes must be unique and exclude 'uncertain'")
        return value

    def describe(self) -> str:
        return (
            f"conf={self.detection_confidence} stride={self.detection_stride} "
            f"window={self.window_seconds}s/{self.stride_seconds}s "
            f"min_conf={self.min_confidence}"
        )


#: Named presets trading throughput against temporal resolution.
PRESETS: dict[str, dict] = {
    "fast": {
        "detection_stride": 4,
        "window_seconds": 2.0,
        "stride_seconds": 2.0,
        "detection_batch_size": 24,
        "action_batch_size": 6,
    },
    "balanced": {
        "detection_stride": 2,
        "window_seconds": 2.0,
        "stride_seconds": 1.0,
        "detection_batch_size": 16,
        "action_batch_size": 4,
    },
    "thorough": {
        "detection_stride": 1,
        "window_seconds": 2.0,
        "stride_seconds": 0.5,
        "detection_batch_size": 12,
        "action_batch_size": 4,
    },
}

PresetName = Literal["fast", "balanced", "thorough"]


def params_for_preset(preset: str = "balanced", **overrides) -> AnalysisParams:
    """Build :class:`AnalysisParams` from a named preset plus explicit overrides.

    ``None`` overrides are dropped, so a caller can pass unset CLI flags or absent form
    fields straight through without filtering them first.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from {sorted(PRESETS)}.")
    values = dict(PRESETS[preset])
    values.update({k: v for k, v in overrides.items() if v is not None})
    return _build(AnalysisParams, values, "analysis parameter")


def parse_overrides(items: Iterable[str]) -> dict[str, str]:
    """Parse ``name=value`` strings into a mapping.

    Lets the CLI accept any parameter the library defines without growing a flag per
    field. Values stay strings; pydantic coerces them when the model is built.
    """
    parsed: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name.strip():
            raise ValueError(f"Expected name=value, got {item!r}.")
        parsed[name.strip()] = value.strip()
    return parsed


def _build[ModelT: BaseModel](model: type[ModelT], values: Mapping[str, Any], what: str) -> ModelT:
    """Instantiate ``model``, turning pydantic's report into a one-line message.

    The same wrong name can arrive from a CLI flag or an HTTP form, so the explanation —
    including the list of valid names — belongs here rather than in either surface.
    """
    try:
        return model(**values)
    except ValidationError as exc:
        known = ", ".join(sorted(model.model_fields))
        problems = []
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"]) or "?"
            if error["type"] == "extra_forbidden":
                problems.append(f"unknown {what} {field!r}")
            else:
                problems.append(f"{field}: {error['msg']}")
        raise ValueError(f"{'; '.join(problems)}. Valid names: {known}.") from exc


# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------


def _default_cache_dir() -> Path:
    if xdg := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg) / "khoroos"
    return Path.home() / ".cache" / "khoroos"


class Settings(BaseSettings):
    """Process-wide settings, overridable via ``KHOROOS_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="KHOROOS_", extra="ignore", validate_assignment=True
    )

    #: ``cuda`` / ``cpu`` / ``mps``; ``auto`` picks the best available.
    device: str = "auto"
    behaviour_groups: dict[str, list[str]] = Field(
        default_factory=lambda: {name: list(members) for name, members in BEHAVIOUR_GROUPS.items()}
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, value):
        if value not in ("auto", "cpu", "mps") and not re.fullmatch(r"cuda(?::[0-9]+)?", value):
            raise ValueError("device must be auto, cpu, mps, cuda, or cuda:<index>")
        return value

    @field_validator("behaviour_groups", mode="before")
    @classmethod
    def check_groups(cls, value):
        return validate_groups(value)

    def with_overrides(self, **overrides) -> Settings:
        """Return a validated copy; callers never need to mutate global settings."""
        values = self.model_dump()
        values.update({k: v for k, v in overrides.items() if v is not None})
        return type(self).model_validate(values)

    #: Use bfloat16 autocast on CUDA. Falls back to fp32 elsewhere.
    use_bf16: bool = True

    cache_dir: Path = Field(default_factory=_default_cache_dir)

    #: Local checkpoint directories. When unset, weights are fetched from the Hub.
    detector_path: Path | None = None
    action_path: Path | None = None

    detector_repo: str = "amirivojdan/chicken_rtdetrv2"
    action_repo: str = "amirivojdan/chicken_vjepa2_action"

    #: Web UI.
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    max_upload_mb: int = 4096
    job_ttl_hours: float = 24.0

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def jobs_dir(self) -> Path:
        return self.cache_dir / "jobs"

    @property
    def weights_dir(self) -> Path:
        return self.cache_dir / "weights"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    global _settings
    _settings = settings
