"""Composition of pipeline stages, with the existing algorithms as defaults.

Replace any callable independently. Factories receive the current run's parameters and
must create fresh state; model instances are injected separately into VideoAnalyzer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from khoroos.config import AnalysisParams
from khoroos.interfaces import Tracker, VideoReader
from khoroos.pipeline.types import ActionPrediction, Track, Tracklet, VideoInfo
from khoroos.statistics.metrics import compute_metrics
from khoroos.tracking.tracklets import build_tracklets, extract_clip
from khoroos.video.reader import VideoSource

if TYPE_CHECKING:
    import torch


class TrackletBuilder(Protocol):
    """Callable that selects windows from one track."""

    def __call__(
        self,
        track: Track,
        params: AnalysisParams,
        fps: float,
        frame_w: int,
        frame_h: int,
    ) -> tuple[list[Tracklet], dict[str, int]]: ...


class ClipExtractor(Protocol):
    """Callable that crops one tracklet into an RGB tensor."""

    def __call__(
        self,
        source: VideoReader,
        tracklet: Tracklet,
        target_frames: int | None = None,
    ) -> torch.Tensor | None: ...


class MetricsCalculator(Protocol):
    """Callable that aggregates predictions into a JSON-compatible metrics dictionary."""

    def __call__(
        self,
        predictions: list[ActionPrediction],
        video: VideoInfo,
        frame_counts: list[tuple[float, int]],
        *,
        classes: list[str],
        behaviour_groups: dict[str, list[str]],
        bin_seconds: float,
    ) -> dict[str, Any]: ...


def default_tracker(params: AnalysisParams) -> Tracker:
    from khoroos.tracking.tracker import BirdTracker

    return BirdTracker(
        iou_threshold=params.track_iou_threshold,
        # max_age counts updates; configuration measures source frames.
        max_age=max(1, int(np.ceil(params.track_max_age / params.detection_stride))),
        min_hits=params.track_min_hits,
    )


@dataclass(frozen=True)
class PipelineComponents:
    """Replaceable stages. See each default function for its input/output contract.

    tracker_factory(params) -> fresh Tracker
    source_factory(path) -> fresh VideoReader
    tracklet_builder(track, params, fps, width, height) -> (tracklets, rejection_counts)
    clip_extractor(source, tracklet, target_frames=...) -> RGB clip or None
    metrics(predictions, video, frame_counts, **options) -> JSON-compatible metric dict
    """

    tracker_factory: Callable[[AnalysisParams], Tracker] = default_tracker
    source_factory: Callable[[str | Path], VideoReader] = VideoSource
    tracklet_builder: TrackletBuilder = build_tracklets
    clip_extractor: ClipExtractor = extract_clip
    metrics: MetricsCalculator = compute_metrics
