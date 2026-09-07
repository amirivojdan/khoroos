"""The analysis orchestrator: video file in, :class:`AnalysisResult` out.

Runs as a generator of :class:`ProgressEvent` so the CLI and the web UI can both render
progress from the same source, then yields the finished result last.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from khoroos.annotations import Detections
from khoroos.config import (
    UNCERTAIN_LABEL,
    AnalysisParams,
    Settings,
    get_settings,
    params_for_preset,
)
from khoroos.interfaces import Detector, VideoClassifier, VideoReader, component_name, model_card
from khoroos.models.card import per_class_f1, per_class_support
from khoroos.models.selection import SelectedClasses
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
            from khoroos.models.detector import ChickenDetector

            self._detector = ChickenDetector(settings=self.settings)
        return self._detector

    @property
    def recognizer(self) -> VideoClassifier:
        if self._recognizer is None:
            from khoroos.models.action import ActionRecognizer

            self._recognizer = ActionRecognizer(settings=self.settings)
        return self._recognizer

    def describe_environment(self) -> dict:
        """Describe this composition using injected models or local checkpoint metadata."""
        from khoroos.environment import describe_environment

        return describe_environment(
            self.settings, classifier=self._recognizer, detector=self._detector
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
        started = time.time()
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

        total_detect_frames = max(1, len(range(0, limit, params.detection_stride)))
        done = 0
        detect_started = time.time()

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
                t = source.time_of(frame_index)
                frame_counts.append((t, len(boxes)))
                tracker.update(frame_index, t, boxes, scores)

            done += len(indices)
            elapsed = time.time() - detect_started
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
            classify_started = time.time()

            yield progress("classify", 0.0, "Recognising actions")
            for start in range(0, len(tracklets), batch_size):
                check_cancel()
                batch = tracklets[start : start + batch_size]
                clips = []
                kept: list[Tracklet] = []
                saved_raw = []
                clip_indices = []
                for offset, tracklet in enumerate(batch):
                    check_cancel()
                    clip = self.components.clip_extractor(
                        source, tracklet, target_frames=recognizer.num_frames
                    )
                    if clip is not None and clip.shape[0] > 0:
                        clips.append(clip)
                        kept.append(tracklet)
                        index = start + offset
                        clip_indices.append(index)
                        saved_raw.append(tracklet_exporter.save("raw", index, tracklet, clip))

                if clips:
                    probs = recognizer.classify(clips)
                    for index, tracklet, clip, raw_path, row in zip(
                        clip_indices, kept, clips, saved_raw, probs, strict=True
                    ):
                        check_cancel()
                        prediction = _to_prediction(tracklet, row, classes, params.min_confidence)
                        predictions.append(prediction)
                        tracklet_exporter.save(
                            "classified", index, tracklet, clip, prediction, raw_path
                        )

                done_clips = min(start + batch_size, len(tracklets))
                elapsed = time.time() - classify_started
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

        groups = (
            self.settings.behaviour_groups
            if params.behaviour_groups is None
            else params.behaviour_groups
        )
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
            runtime_seconds=time.time() - started,
        )

        yield progress("metrics", 1.0, "Done")
        yield result


def _pick_bin_seconds(duration: float) -> float:
    """Keep the timeline around 100-200 bins regardless of video length."""
    if duration <= 60:
        return 1.0
    if duration <= 600:
        return 5.0
    if duration <= 3600:
        return 30.0
    return 60.0


def _to_prediction(
    tracklet: Tracklet,
    probabilities: np.ndarray,
    classes: list[str],
    min_confidence: float,
) -> ActionPrediction:
    best = int(np.argmax(probabilities))
    confidence = float(probabilities[best])
    label = classes[best]
    uncertain = confidence < min_confidence

    # Representative box for the window: the middle frame, where the bird is most likely
    # to be centred in its own crop.
    mid = len(tracklet.boxes) // 2
    box = tuple(float(v) for v in tracklet.boxes[mid])

    return ActionPrediction(
        track_id=tracklet.track_id,
        start_seconds=tracklet.start_seconds,
        end_seconds=tracklet.end_seconds,
        label=UNCERTAIN_LABEL if uncertain else label,
        confidence=confidence,
        probabilities={c: float(p) for c, p in zip(classes, probabilities, strict=True)},
        box=box,  # type: ignore[arg-type]
        is_uncertain=uncertain,
    )


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
