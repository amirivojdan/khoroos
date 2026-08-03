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

from khoroos.config import (
    AnalysisParams,
    Settings,
    WelfareThresholds,
    get_settings,
    params_for_preset,
)
from khoroos.models.action import ActionRecognizer
from khoroos.models.card import load_model_card, per_class_f1, per_class_support
from khoroos.models.detector import ChickenDetector
from khoroos.pipeline.types import (
    ActionPrediction,
    AnalysisResult,
    ModelInfo,
    ProgressEvent,
    Track,
    Tracklet,
    VideoInfo,
)
from khoroos.tracking.tracker import BirdTracker
from khoroos.tracking.tracklets import build_tracklets, extract_clip
from khoroos.video.reader import VideoSource
from khoroos.welfare.metrics import compute_metrics

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
        detector: ChickenDetector | None = None,
        recognizer: ActionRecognizer | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._detector = detector
        self._recognizer = recognizer

    @property
    def detector(self) -> ChickenDetector:
        if self._detector is None:
            self._detector = ChickenDetector(settings=self.settings)
        return self._detector

    @property
    def recognizer(self) -> ActionRecognizer:
        if self._recognizer is None:
            self._recognizer = ActionRecognizer(settings=self.settings)
        return self._recognizer

    # -- main entry points -------------------------------------------------

    def analyze(
        self,
        video_path: str | Path,
        params: AnalysisParams | None = None,
        thresholds: WelfareThresholds | None = None,
        preset: str = "balanced",
        on_progress: Callable[[ProgressEvent], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AnalysisResult:
        """Analyse a video, returning the finished result."""
        result: AnalysisResult | None = None
        for item in self.iter_analyze(
            video_path, params, thresholds, preset, should_cancel=should_cancel
        ):
            if isinstance(item, ProgressEvent):
                if on_progress is not None:
                    on_progress(item)
            else:
                result = item
        assert result is not None, "pipeline finished without producing a result"
        return result

    def iter_analyze(
        self,
        video_path: str | Path,
        params: AnalysisParams | None = None,
        thresholds: WelfareThresholds | None = None,
        preset: str = "balanced",
        should_cancel: Callable[[], bool] | None = None,
    ) -> Iterator[ProgressEvent | AnalysisResult]:
        """Run the pipeline, yielding progress events then the final result."""
        params = params or params_for_preset(preset)
        thresholds = thresholds or WelfareThresholds()
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

        # -- 1. probe ------------------------------------------------------
        yield progress("probe", 0.0, "Opening video")
        source = VideoSource(video_path)
        info = source.info
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
        tracker = BirdTracker(
            iou_threshold=params.track_iou_threshold,
            # ``track_max_age`` is expressed in source frames; the tracker advances only
            # on detector frames, so preserve the meaning across presets.
            max_age=max(1, int(np.ceil(params.track_max_age / params.detection_stride))),
            min_hits=params.track_min_hits,
        )
        frame_counts: list[tuple[float, int]] = []

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
            for frame_index, (boxes, scores) in zip(indices, detections, strict=True):
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

        yield progress("track", 0.5, "Linking detections into tracks")
        tracks: list[Track] = tracker.finalize()
        if not tracks:
            warnings.append(
                "No birds were tracked. Check that the footage shows chickens and that the "
                "detection confidence is not set too high."
            )
        yield progress("track", 1.0, f"{len(tracks)} birds tracked", n_tracks=len(tracks))

        # -- 4. tracklet windows -------------------------------------------
        yield progress("clips", 0.0, "Selecting stable clips")
        tracklets: list[Tracklet] = []
        reject_totals: dict[str, int] = {}
        for i, track in enumerate(tracks):
            check_cancel()
            built, rejects = build_tracklets(
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
                    f"wrongly sized ({reject_totals}). Birds may be too small or too "
                    f"occluded in this footage for reliable action recognition."
                )
        if not tracklets:
            warnings.append(
                "No clips passed the stability and size gates, so no actions could be "
                "classified. Birds are likely too small in frame — a closer camera view or "
                "a higher-resolution recording is needed."
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
        if tracklets:
            recognizer = self.recognizer
            classes = recognizer.classes
            batch_size = params.action_batch_size
            classify_started = time.time()

            yield progress("classify", 0.0, "Recognising actions")
            for start in range(0, len(tracklets), batch_size):
                check_cancel()
                batch = tracklets[start : start + batch_size]
                clips = []
                kept: list[Tracklet] = []
                for tracklet in batch:
                    clip = extract_clip(source, tracklet, target_frames=recognizer.num_frames)
                    if clip is not None and clip.shape[0] > 0:
                        clips.append(clip)
                        kept.append(tracklet)

                if clips:
                    probs = recognizer.classify(clips)
                    for tracklet, row in zip(kept, probs, strict=True):
                        predictions.append(
                            _to_prediction(tracklet, row, classes, params.min_confidence)
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
        yield progress("metrics", 0.2, "Computing welfare indicators")

        card = load_model_card(self.recognizer.checkpoint) if tracklets else {}
        f1 = per_class_f1(card)
        model_info = ModelInfo(
            detector=str(self.detector.checkpoint.name),
            action=str(self.recognizer.checkpoint.name) if tracklets else "not-run",
            classes=list(self.recognizer.classes) if tracklets else [],
            per_class_f1=f1,
            per_class_support=per_class_support(card),
        )

        metrics = compute_metrics(
            predictions,
            analyzed_info,
            frame_counts,
            thresholds=thresholds,
            per_class_f1=f1,
            bin_seconds=_pick_bin_seconds(analyzed_info.duration_seconds),
        )
        if tracker.lost_track_count:
            warnings.append(
                f"{tracker.lost_track_count} tracks were lost and re-identified as new birds. "
                f"Per-bird figures are approximate; flock-level figures are unaffected."
            )

        result = AnalysisResult(
            video=analyzed_info,
            params=params.model_dump(),
            model=model_info,
            tracks=tracks,
            predictions=predictions,
            metrics=metrics,
            warnings=warnings,
            frame_counts=frame_counts,
            runtime_seconds=time.time() - started,
        )
        source.close()

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

    from khoroos.config import UNCERTAIN_LABEL

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
    thresholds: WelfareThresholds | None = None,
    settings: Settings | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
) -> AnalysisResult:
    """Convenience wrapper: analyse one video with freshly loaded models."""
    analyzer = VideoAnalyzer(settings=settings)
    return analyzer.analyze(
        video_path,
        params=params,
        thresholds=thresholds,
        preset=preset,
        on_progress=on_progress,
    )
