"""Core data types and the versioned result schema.

``AnalysisResult`` is the contract between the analysis pipeline, the CLI exporters and the
web UI. Bump ``SCHEMA_VERSION`` on any breaking change to its serialised form.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "2.0"


@dataclass(slots=True)
class VideoInfo:
    """Static properties of the analysed video."""

    filename: str
    fps: float
    duration_seconds: float
    width: int
    height: int
    num_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "fps": round(self.fps, 4),
            "duration_s": round(self.duration_seconds, 3),
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
        }


@dataclass(slots=True)
class FrameDetections:
    """Detector output for one frame.

    ``boxes`` is an ``(N, 4)`` float array in ``xyxy`` pixel coordinates,
    ``scores`` an ``(N,)`` float array.
    """

    frame_index: int
    time_seconds: float
    boxes: np.ndarray
    scores: np.ndarray

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


@dataclass(slots=True)
class TrackObservation:
    """A single track's state at one frame."""

    frame_index: int
    time_seconds: float
    box: tuple[float, float, float, float]
    score: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


@dataclass(slots=True)
class Track:
    """One bird followed across the video, identified by a global ``track_id``."""

    track_id: int
    observations: list[TrackObservation] = field(default_factory=list)

    @property
    def start_seconds(self) -> float:
        return self.observations[0].time_seconds

    @property
    def end_seconds(self) -> float:
        return self.observations[-1].time_seconds

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def __len__(self) -> int:
        return len(self.observations)

    def boxes_array(self) -> np.ndarray:
        return np.asarray([o.box for o in self.observations], dtype=np.float32)

    def centers_array(self) -> np.ndarray:
        return np.asarray([o.center for o in self.observations], dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.track_id,
            "start_s": round(self.start_seconds, 3),
            "end_s": round(self.end_seconds, 3),
            "n_frames": len(self.observations),
            "boxes": [
                [round(o.time_seconds, 3), *(round(float(v), 1) for v in o.box)]
                for o in self.observations
            ],
        }


@dataclass(slots=True)
class Tracklet:
    """A fixed-duration window of one track, the unit the action model consumes."""

    track_id: int
    start_seconds: float
    end_seconds: float
    #: Frame indices in the source video contributing to this tracklet.
    frame_indices: list[int]
    #: Per-frame boxes aligned with ``frame_indices``.
    boxes: np.ndarray
    #: Fixed crop box (x1, y1, x2, y2) applied across the window.
    crop_size: tuple[int, int]
    centers: np.ndarray

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(slots=True)
class ActionPrediction:
    """Classifier output for one tracklet."""

    track_id: int
    start_seconds: float
    end_seconds: float
    label: str
    confidence: float
    probabilities: dict[str, float]
    box: tuple[float, float, float, float]
    #: True when confidence fell below the threshold and ``label`` was set to "uncertain".
    is_uncertain: bool = False

    def to_dict(self, top_k: int = 3) -> dict[str, Any]:
        top = sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return {
            "track_id": self.track_id,
            "start_s": round(self.start_seconds, 3),
            "end_s": round(self.end_seconds, 3),
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "uncertain": self.is_uncertain,
            "top_k": [{"label": k, "p": round(v, 4)} for k, v in top],
            "box": [round(float(v), 1) for v in self.box],
        }


@dataclass(slots=True)
class ModelInfo:
    """Provenance and reliability of the models used for a run."""

    detector: str
    action: str
    classes: list[str]
    per_class_f1: dict[str, float] = field(default_factory=dict)
    per_class_support: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "action": self.action,
            "classes": self.classes,
            "per_class_f1": self.per_class_f1,
            "per_class_support": self.per_class_support,
        }


@dataclass(slots=True)
class AnalysisResult:
    """Everything a completed analysis produces."""

    video: VideoInfo
    params: dict[str, Any]
    model: ModelInfo
    tracks: list[Track]
    predictions: list[ActionPrediction]
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: Per-frame bird counts as ``(time_seconds, count)`` pairs.
    frame_counts: list[tuple[float, int]] = field(default_factory=list)
    runtime_seconds: float = 0.0
    schema_version: str = SCHEMA_VERSION

    def to_dict(self, include_tracks: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "video": self.video.to_dict(),
            "params": self.params,
            "model": self.model.to_dict(),
            "predictions": [p.to_dict() for p in self.predictions],
            "metrics": self.metrics,
            "warnings": self.warnings,
            "frame_counts": [[round(t, 3), c] for t, c in self.frame_counts],
            "runtime_s": round(self.runtime_seconds, 2),
        }
        if include_tracks:
            payload["tracks"] = [t.to_dict() for t in self.tracks]
        return payload

    def save_json(self, path: str | Path, include_tracks: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(include_tracks), indent=2), encoding="utf-8")
        return path


@dataclass(slots=True)
class ProgressEvent:
    """Streamed by the pipeline so callers can render progress."""

    stage: str
    progress: float  # 0..1 within the whole run
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "message": self.message,
            "detail": self.detail,
        }
