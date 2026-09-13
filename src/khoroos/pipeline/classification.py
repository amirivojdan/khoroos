"""Prepare, classify, and export one batch of tracklet crops.

The analyzer owns batch scheduling and progress. This module keeps each crop attached to
its source window and export identity, including when extraction skips a window. Tensors
live only for the duration of one call; returned predictions contain no image data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from khoroos.config import UNCERTAIN_LABEL
from khoroos.interfaces import VideoClassifier, VideoReader
from khoroos.pipeline.components import ClipExtractor
from khoroos.pipeline.types import ActionPrediction, Tracklet
from khoroos.video.tracklet_export import TrackletExporter

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class _PreparedClip:
    index: int
    tracklet: Tracklet
    tensor: torch.Tensor
    raw_path: Path | None


def classify_batch(
    source: VideoReader,
    tracklets: Sequence[Tracklet],
    recognizer: VideoClassifier,
    *,
    start_index: int,
    min_confidence: float,
    extract_clip: ClipExtractor,
    exporter: TrackletExporter,
    check_cancel: Callable[[], None],
) -> list[ActionPrediction]:
    """Classify accepted crops in input order, preserving their original export indices.

    Raw clips are saved before inference so they survive a classifier failure. Empty or
    missing crops are skipped without shifting later filenames or probability rows.
    The recognizer must validate its probabilities (the pipeline uses SelectedClasses).
    """
    prepared: list[_PreparedClip] = []
    for index, tracklet in enumerate(tracklets, start=start_index):
        check_cancel()
        clip = extract_clip(source, tracklet, target_frames=recognizer.num_frames)
        if clip is None or clip.shape[0] == 0:
            continue
        raw_path = exporter.save("raw", index, tracklet, clip)
        prepared.append(_PreparedClip(index, tracklet, clip, raw_path))

    if not prepared:
        return []

    check_cancel()
    probabilities = recognizer.classify([item.tensor for item in prepared])
    classes = list(recognizer.classes)
    predictions: list[ActionPrediction] = []
    for item, row in zip(prepared, probabilities, strict=True):
        check_cancel()
        prediction = _to_prediction(item.tracklet, row, classes, min_confidence)
        exporter.save(
            "classified", item.index, item.tracklet, item.tensor, prediction, item.raw_path
        )
        predictions.append(prediction)
    return predictions


def _to_prediction(
    tracklet: Tracklet,
    probabilities: np.ndarray,
    classes: list[str],
    min_confidence: float,
) -> ActionPrediction:
    best = int(np.argmax(probabilities))
    confidence = float(probabilities[best])
    uncertain = confidence < min_confidence

    # The middle frame represents the window in spatial summaries and exports.
    x1, y1, x2, y2 = (float(v) for v in tracklet.boxes[len(tracklet.boxes) // 2])
    return ActionPrediction(
        track_id=tracklet.track_id,
        start_seconds=tracklet.start_seconds,
        end_seconds=tracklet.end_seconds,
        label=UNCERTAIN_LABEL if uncertain else classes[best],
        confidence=confidence,
        probabilities={c: float(p) for c, p in zip(classes, probabilities, strict=True)},
        box=(x1, y1, x2, y2),
        is_uncertain=uncertain,
    )
