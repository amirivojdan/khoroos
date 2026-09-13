"""The analysis orchestrator: video file in, :class:`AnalysisResult` out.

Runs as a generator of :class:`ProgressEvent` so the CLI and the web UI can both render
progress from the same source, then yields the finished result last.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from khoroos.annotations import Detections
from khoroos.config import (
    AnalysisParams,
    Settings,
    get_settings,
    params_for_preset,
)
from khoroos.interfaces import Detector, VideoClassifier, VideoReader, component_name, model_card
from khoroos.models.card import per_class_f1, per_class_support
from khoroos.models.selection import SelectedClasses
from khoroos.pipeline.classification import classify_batch
from khoroos.pipeline.components import PipelineComponents
from khoroos.pipeline.types import (
    ActionPrediction,
    AnalysisResult,
    ModelInfo,
    ProgressEvent,
    Track,
    Tracklet,
    VideoInfo,
)
from khoroos.video.tracklet_export import TrackletExporter

logger = logging.getLogger(__name__)

# Relative cost of each stage, used to turn per-stage progress into one overall bar.
_STAGE_WEIGHTS = {
    "probe": 0.01,
    "detect": 0.45,
    "track": 0.02,
    "clips": 0.12,
    "classify": 0.35,
    "metrics": 0.05,
}


def _cumulative_offsets() -> dict[str, float]:
    """Starting offset of each stage on the overall bar.

    Computed once, so a stage's end and the next stage's start are the *same* float.
    Re-summing the weights per event instead lets floating-point ordering make the bar
    tick backwards between stages.
    """
    offsets: dict[str, float] = {}
    running = 0.0
    for stage, weight in _STAGE_WEIGHTS.items():
        offsets[stage] = running
        running += weight
    return offsets


_STAGE_OFFSETS = _cumulative_offsets()


class CancelledError(RuntimeError):
    """Raised when a caller-supplied cancel check fires."""


class VideoAnalyzer:
    """Holds the loaded models so repeated analyses do not reload 1.7 GB of weights."""

    def __init__(
        self,
        settings: Settings | None = None,
        detector: Detector | None = None,
        recognizer: VideoClassifier | None = None,
        components: PipelineComponents | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._detector = detector
        self._recognizer = recognizer
        self.components = components or PipelineComponents()

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            from khoroos.models.parallel import build_detector

            self._detector = build_detector(self.settings)
        return self._detector

    @property
    def recognizer(self) -> VideoClassifier:
        if self._recognizer is None:
            from khoroos.models.parallel import build_recognizer

            self._recognizer = build_recognizer(self.settings)
        return self._recognizer

    def close(self) -> None:
        """Drop loaded models and the device worker threads a multi-device model holds.

        The analyzer reloads on next use, so this is about releasing hardware promptly —
        the web worker calls it before loading the same models onto a different device.
        """
        models = (self._detector, self._recognizer)
        self._detector = self._recognizer = None
        # Attempt both cleanups even if a custom component raises. Detaching first also
        # makes a repeated close harmless after a partially failed cleanup.
        with ExitStack() as cleanup:
            for model in reversed(models):
                close = getattr(model, "close", None)
                if close is not None:
                    cleanup.callback(close)

    def __enter__(self) -> VideoAnalyzer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def describe_environment(self, *, settings: Settings | None = None) -> dict[str, Any]:
        """Describe models without loading them, optionally using presentation settings.

        A web application can supply its configured device default while retaining metadata
        from this analyzer's injected components. Neither settings object is mutated.
        """
        from khoroos.environment import describe_environment

        return describe_environment(
            settings if settings is not None else self.settings,
            classifier=self._recognizer,
            detector=self._detector,
        )

    # -- main entry points -------------------------------------------------

    def analyze(
        self,
        video_path: str | Path,
        params: AnalysisParams | None = None,
        preset: str = "balanced",
        on_progress: Callable[[ProgressEvent], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AnalysisResult:
        """Analyse a video, returning the finished result."""
        result: AnalysisResult | None = None
        for item in self.iter_analyze(video_path, params, preset, should_cancel=should_cancel):
            if isinstance(item, ProgressEvent):
                if on_progress is not None:
                    on_progress(item)
            else:
                result = item
        if result is None:
            raise RuntimeError("Pipeline finished without producing a result")
        return result

    def iter_analyze(
        self,
        video_path: str | Path,
        params: AnalysisParams | None = None,
        preset: str = "balanced",
        should_cancel: Callable[[], bool] | None = None,
    ) -> Iterator[ProgressEvent | AnalysisResult]:
        """Run the pipeline, yielding progress events then the final result."""
        params = params or params_for_preset(preset)
        yield ProgressEvent("probe", 0.0, "Opening video")
        source = self.components.source_factory(video_path)
        try:
            yield from self._run(source, params, should_cancel)
        finally:
            source.close()

    def _run(
        self,
        source: VideoReader,
        params: AnalysisParams,
        should_cancel: Callable[[], bool] | None,
    ) -> Iterator[ProgressEvent | AnalysisResult]:
        """Coordinate stages for an open source; iter_analyze owns its lifetime."""
        started = time.perf_counter()
        warnings: list[str] = []

        def check_cancel() -> None:
            if should_cancel is not None and should_cancel():
                raise CancelledError("Analysis cancelled.")

        def progress(stage: str, fraction: float, message: str = "", **detail):
            overall = _STAGE_OFFSETS[stage] + _STAGE_WEIGHTS[stage] * min(max(fraction, 0.0), 1.0)
            return ProgressEvent(
                stage=stage, progress=min(overall, 1.0), message=message, detail=detail
            )

        selected = (
            SelectedClasses(self.recognizer, params.action_classes)
            if params.action_classes is not None
            else None
        )

        info = source.info
        tracklet_exporter = TrackletExporter(params, info.filename)
        yield progress(
            "probe",
            1.0,
            f"{info.width}x{info.height}, {info.fps:.1f} fps, {info.duration_seconds:.1f}s",
            video=info.to_dict(),
        )

        max_frames = None
        if params.max_duration_seconds is not None:
            max_frames = max(1, int(np.ceil(params.max_duration_seconds * info.fps)))

        limit = info.num_frames if max_frames is None else min(info.num_frames, max_frames)
        analyzed_info = VideoInfo(
            filename=info.filename,
            fps=info.fps,
            duration_seconds=min(info.duration_seconds, limit / info.fps),
            width=info.width,
            height=info.height,
            num_frames=limit,
        )

        # -- 2. detect + 3. track -------------------------------------------
        tracker = self.components.tracker_factory(params)
        frame_counts: list[tuple[float, int]] = []
        dropped_boxes = 0
        # The detector still sees the whole frame at the scale it was trained on; the region
        # decides which of its findings are ours.
        region = None if params.roi is None else roi_pixels(params.roi, info.width, info.height)
        outside_region = 0

        total_detect_frames = max(1, len(range(0, limit, params.detection_stride)))
        done = 0
        detect_started = time.perf_counter()

        yield progress("detect", 0.0, "Detecting birds")
        for indices, frames in source.iter_batches(
            batch_size=params.detection_batch_size,
            stride=params.detection_stride,
            max_frames=max_frames,
        ):
            check_cancel()
            detections = self.detector.detect(
                frames,
                confidence_threshold=params.detection_confidence,
                nms_threshold=params.nms_threshold,
                box_padding=params.box_padding,
            )
            for frame_index, raw in zip(indices, detections, strict=True):
                detected = Detections.coerce(raw, invalid_boxes=params.invalid_boxes)
                dropped_boxes += detected.dropped_count
                boxes, scores = detected
                if region is not None:
                    keep = inside_roi(boxes, region)
                    outside_region += int((~keep).sum())
                    boxes, scores = boxes[keep], scores[keep]
                t = source.time_of(frame_index)
                frame_counts.append((t, len(boxes)))
                tracker.update(frame_index, t, boxes, scores)

            done += len(indices)
            elapsed = time.perf_counter() - detect_started
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total_detect_frames - done) / rate if rate > 0 else None
            yield progress(
                "detect",
                done / total_detect_frames,
                f"Detecting birds — frame {done}/{total_detect_frames}",
                frames_done=done,
                frames_total=total_detect_frames,
                fps=round(rate, 2),
                eta_s=round(eta, 1) if eta else None,
            )

        if dropped_boxes:
            warnings.append(f"Dropped {dropped_boxes} invalid detection boxes.")
        if region is not None:
            kept = sum(count for _, count in frame_counts)
            if not kept:
                warnings.append(
                    "No detections fell inside the region of interest; check that it covers "
                    "the part of the frame the birds are in."
                )
            else:
                warnings.append(
                    f"Region of interest kept {kept} detections and ignored {outside_region} "
                    "outside it."
                )

        yield progress("track", 0.5, "Linking detections into tracks")
        tracks: list[Track] = tracker.finalize()
        if not tracks:
            warnings.append("No confirmed tracks were produced.")
        yield progress("track", 1.0, f"{len(tracks)} birds tracked", n_tracks=len(tracks))

        # -- 4. tracklet windows -------------------------------------------
        yield progress("clips", 0.0, "Selecting stable clips")
        tracklets: list[Tracklet] = []
        reject_totals: dict[str, int] = {}
        for i, track in enumerate(tracks):
            check_cancel()
            built, rejects = self.components.tracklet_builder(
                track, params, analyzed_info.fps, analyzed_info.width, analyzed_info.height
            )
            tracklets.extend(built)
            for key, value in rejects.items():
                reject_totals[key] = reject_totals.get(key, 0) + value
            if i % 10 == 0:
                yield progress("clips", (i + 1) / max(1, len(tracks)), "Selecting stable clips")

        total_windows = len(tracklets) + sum(reject_totals.values())
        if total_windows:
            reject_share = sum(reject_totals.values()) / total_windows
            if reject_share > 0.5:
                warnings.append(
                    f"{reject_share:.0%} of candidate clips were rejected as unstable or "
                    f"outside the crop geometry limits ({reject_totals})."
                )
        if not tracklets:
            warnings.append(
                "No clips passed the stability and size gates; no actions were classified."
            )
        yield progress(
            "clips",
            1.0,
            f"{len(tracklets)} clips selected",
            n_tracklets=len(tracklets),
            rejects=reject_totals,
        )

        # -- 5. classify ----------------------------------------------------
        predictions: list[ActionPrediction] = []
        classes: list[str] = []
        if tracklets:
            recognizer = selected or SelectedClasses(self.recognizer)
            classes = list(recognizer.classes)
            batch_size = params.action_batch_size
            classify_started = time.perf_counter()

            yield progress("classify", 0.0, "Recognising actions")
            for start in range(0, len(tracklets), batch_size):
                check_cancel()
                predictions.extend(
                    classify_batch(
                        source,
                        tracklets[start : start + batch_size],
                        recognizer,
                        start_index=start,
                        min_confidence=params.min_confidence,
                        extract_clip=self.components.clip_extractor,
                        exporter=tracklet_exporter,
                        check_cancel=check_cancel,
                    )
                )

                done_clips = min(start + batch_size, len(tracklets))
                elapsed = time.perf_counter() - classify_started
                rate = done_clips / elapsed if elapsed > 0 else 0.0
                eta = (len(tracklets) - done_clips) / rate if rate > 0 else None
                yield progress(
                    "classify",
                    done_clips / len(tracklets),
                    f"Recognising actions — clip {done_clips}/{len(tracklets)}",
                    clips_done=done_clips,
                    clips_total=len(tracklets),
                    clips_per_s=round(rate, 2),
                    eta_s=round(eta, 1) if eta else None,
                )

        # -- 6. metrics -----------------------------------------------------
        yield progress("metrics", 0.2, "Computing descriptive statistics")

        card = model_card(self.recognizer) if tracklets else {}
        f1 = {k: v for k, v in per_class_f1(card).items() if k in classes}
        model_info = ModelInfo(
            detector=component_name(self.detector),
            action=component_name(self.recognizer) if tracklets else "not-run",
            classes=classes,
            per_class_f1=f1,
            per_class_support={k: v for k, v in per_class_support(card).items() if k in classes},
        )

        configured_groups = (
            self.settings.behaviour_groups
            if params.behaviour_groups is None
            else params.behaviour_groups
        )
        groups = {name: list(members) for name, members in configured_groups.items()}
        metrics = self.components.metrics(
            predictions,
            analyzed_info,
            frame_counts,
            classes=classes,
            behaviour_groups=groups,
            bin_seconds=_pick_bin_seconds(analyzed_info.duration_seconds),
        )
        if tracker.lost_track_count:
            warnings.append(
                f"{tracker.lost_track_count} confirmed tracks were lost; "
                "later detections may receive new IDs. "
                "Track IDs do not necessarily correspond to unique animals."
            )

        result = AnalysisResult(
            video=analyzed_info,
            params=params.model_dump() | {
                "behaviour_groups": groups,
                "tracklet_exports": tracklet_exporter.directories,
            },
            model=model_info,
            tracks=tracks,
            predictions=predictions,
            metrics=metrics,
            warnings=warnings,
            frame_counts=frame_counts,
            runtime_seconds=time.perf_counter() - started,
        )

        yield progress("metrics", 1.0, "Done")
        yield result


def roi_pixels(roi: tuple[float, float, float, float], width: int, height: int):
    """Turn fractional ``(x1, y1, x2, y2)`` into pixel bounds for a frame of this size."""
    x1, y1, x2, y2 = roi
    return (x1 * width, y1 * height, x2 * width, y2 * height)


def inside_roi(boxes: np.ndarray, bounds) -> np.ndarray:
    """Mask of boxes whose centre lies within ``bounds``.

    Centre-inside rather than any-overlap: a bird on the boundary belongs to whichever side
    it is mostly on, and its membership does not flicker as the box grows and shrinks
    between frames — which it would if a single overlapping pixel were enough.
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    x1, y1, x2, y2 = bounds
    centres_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centres_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    return (centres_x >= x1) & (centres_x <= x2) & (centres_y >= y1) & (centres_y <= y2)


def _pick_bin_seconds(duration: float) -> float:
    """Keep the timeline around 100-200 bins regardless of video length."""
    if duration <= 60:
        return 1.0
    if duration <= 600:
        return 5.0
    if duration <= 3600:
        return 30.0
    return 60.0


def analyze_video(
    video_path: str | Path,
    preset: str = "balanced",
    params: AnalysisParams | None = None,
    settings: Settings | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    *,
    detector: Detector | None = None,
    recognizer: VideoClassifier | None = None,
    components: PipelineComponents | None = None,
) -> AnalysisResult:
    """Convenience wrapper: analyse one video with freshly loaded models."""
    analyzer = VideoAnalyzer(
        settings=settings, detector=detector, recognizer=recognizer, components=components
    )
    return analyzer.analyze(
        video_path,
        params=params,
        preset=preset,
        on_progress=on_progress,
    )
