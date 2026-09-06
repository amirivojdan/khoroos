"""Turning tracks into classifiable clips.

Each track is sliced into overlapping fixed-duration windows; every window that survives
the stability and geometry gates becomes a :class:`~khoroos.pipeline.types.Tracklet`, which
is then cropped straight into a tensor for the action model.

The stability heuristics (``box_iou_xyxy``, ``track_is_unstable``) are ported unchanged
from the dataset-generation script — they encode tuning validated against real footage, and
the same gates that selected training clips should select inference clips.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from khoroos.config import AnalysisParams
from khoroos.interfaces import VideoReader
from khoroos.pipeline.types import Track, Tracklet

logger = logging.getLogger(__name__)


def box_iou_xyxy(a, b) -> float:
    """IoU of two ``xyxy`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-9
    return inter / union


def track_is_unstable(
    centers,
    boxes,
    frame_w: int,
    frame_h: int,
    min_iou: float = 0.5,
    max_bad_frac: float = 0.10,
    jump_factor: float = 0.7,
    min_length: int = 10,
) -> bool:
    """Detect tracks that jitter, teleport, or lose their target.

    ``centers`` is ``[T, 2]``, ``boxes`` is ``[T, 4]``. ``jump_factor`` scales the jump
    threshold relative to the median box diagonal.
    """
    centers = np.asarray([[float(c[0]), float(c[1])] for c in centers], dtype=np.float32)
    boxes = np.asarray([[float(x) for x in b] for b in boxes], dtype=np.float32)

    if len(centers) < min_length:
        return True  # too short to trust

    # Diagonal of each box, then a robust size anchor.
    diag = np.sqrt(
        np.maximum(0.0, (boxes[:, 2] - boxes[:, 0])) ** 2
        + np.maximum(0.0, (boxes[:, 3] - boxes[:, 1])) ** 2
    )
    diag_median = np.median(diag)
    frame_diag = np.hypot(frame_w, frame_h)
    anchor = diag_median if diag_median > 1.0 else 0.05 * frame_diag

    # Per-step centre jumps.
    step = np.linalg.norm(centers[1:] - centers[:-1], axis=1)

    # Robust baseline speed.
    med = np.median(step) + 1e-6
    mad = np.median(np.abs(step - med)) + 1e-6

    big_jump_abs = step > jump_factor * anchor
    big_jump_rob = (step - med) > 6.0 * mad

    ious = np.array(
        [box_iou_xyxy(a, b) for a, b in zip(boxes[:-1], boxes[1:], strict=True)],
        dtype=np.float32,
    )
    tiny_iou = ious < float(min_iou)

    bad_frac = (big_jump_abs | big_jump_rob | tiny_iou).mean()
    extreme_teleport = step.max() > 0.25 * frame_diag

    return bool(bad_frac > max_bad_frac or extreme_teleport)


def interpolate_track(track: Track) -> dict[int, np.ndarray]:
    """Densify a track to one box per frame.

    The detector may run on a stride, so tracks carry observations only on detected
    frames. Linear interpolation between them gives the per-frame boxes the crop stage
    needs, without inventing motion beyond the observed span.
    """
    obs = track.observations
    if not obs:
        return {}

    frames = np.array([o.frame_index for o in obs], dtype=np.int64)
    boxes = track.boxes_array()

    dense: dict[int, np.ndarray] = {}
    for i in range(len(frames) - 1):
        f0, f1 = int(frames[i]), int(frames[i + 1])
        b0, b1 = boxes[i], boxes[i + 1]
        span = f1 - f0
        dense[f0] = b0
        if span > 1:
            for step in range(1, span):
                alpha = step / span
                dense[f0 + step] = (1 - alpha) * b0 + alpha * b1
    dense[int(frames[-1])] = boxes[-1]
    return dense


def build_tracklets(
    track: Track,
    params: AnalysisParams,
    fps: float,
    frame_w: int,
    frame_h: int,
) -> tuple[list[Tracklet], dict[str, int]]:
    """Slice one track into windows and keep those passing the quality gates.

    Returns the accepted tracklets and a count of why windows were rejected.
    """
    rejects = {"short": 0, "unstable": 0, "geometry": 0, "coverage": 0}
    dense = interpolate_track(track)
    if not dense:
        return [], rejects

    window_frames = max(1, int(round(params.window_seconds * fps)))
    stride_frames = max(1, int(round(params.stride_seconds * fps)))

    first_frame = min(dense)
    last_frame = max(dense)

    tracklets: list[Tracklet] = []
    start = first_frame
    while start + window_frames - 1 <= last_frame:
        end = start + window_frames
        indices = [f for f in range(start, end) if f in dense]

        coverage = len(indices) / window_frames
        if coverage < params.min_window_coverage:
            rejects["coverage"] += 1
            start += stride_frames
            continue

        boxes = np.asarray([dense[f] for f in indices], dtype=np.float32)
        centers = np.stack(
            [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1
        )

        if len(indices) < 10:
            rejects["short"] += 1
            start += stride_frames
            continue

        if track_is_unstable(
            centers,
            boxes,
            frame_w,
            frame_h,
            min_iou=params.stability_min_iou,
            max_bad_frac=params.stability_max_bad_frac,
            jump_factor=params.stability_jump_factor,
        ):
            rejects["unstable"] += 1
            start += stride_frames
            continue

        # Fixed crop size across the window so the subject never changes scale mid-clip
        # (the action model is sensitive to that). A high percentile rather than the max:
        # one spurious oversized detection would otherwise blow the crop up to the whole
        # frame and drown the bird in background.
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        crop_w = int(np.percentile(widths, params.crop_size_percentile))
        crop_h = int(np.percentile(heights, params.crop_size_percentile))

        if crop_w < params.min_crop_size or crop_h < params.min_crop_size:
            rejects["geometry"] += 1
            start += stride_frames
            continue
        # A crop covering a large share of the frame is not one bird.
        if (
            crop_w > params.max_crop_frame_fraction * frame_w
            or crop_h > params.max_crop_frame_fraction * frame_h
        ):
            rejects["geometry"] += 1
            start += stride_frames
            continue
        ratio = min(crop_w, crop_h) / max(crop_w, crop_h)
        if ratio < params.min_crop_aspect_ratio:
            rejects["geometry"] += 1
            start += stride_frames
            continue

        tracklets.append(
            Tracklet(
                track_id=track.track_id,
                start_seconds=indices[0] / fps,
                end_seconds=(indices[-1] + 1) / fps,
                frame_indices=indices,
                boxes=boxes,
                crop_size=(crop_w, crop_h),
                centers=centers,
            )
        )
        start += stride_frames

    return tracklets, rejects


def extract_clip(
    source: VideoReader,
    tracklet: Tracklet,
    target_frames: int | None = None,
) -> torch.Tensor | None:
    """Crop a tracklet out of the video into a ``(T, C, H, W)`` uint8 tensor.

    Unlike the dataset-generation script, nothing is written to disk. Frames whose crop
    window would fall outside the image are clamped back inside rather than dropped, so a
    bird near a wall still yields a full-length clip.
    """
    crop_w, crop_h = tracklet.crop_size
    if crop_w <= 0 or crop_h <= 0:
        return None

    indices = tracklet.frame_indices
    centers = tracklet.centers

    # Optionally thin the window before decoding: the model only sees `target_frames`
    # anyway, so decoding more is wasted work on long windows.
    if target_frames is not None and len(indices) > target_frames:
        pick = np.round(np.linspace(0, len(indices) - 1, target_frames)).astype(int)
        indices = [indices[i] for i in pick]
        centers = centers[pick]

    frames = source.get_frames(indices)  # (T, C, H, W)
    if frames.shape[0] == 0:
        return None

    _, _, height, width = frames.shape
    crop_w = min(crop_w, width)
    crop_h = min(crop_h, height)

    crops = []
    for i in range(frames.shape[0]):
        cx, cy = float(centers[i][0]), float(centers[i][1])
        x1 = int(round(cx - crop_w / 2))
        y1 = int(round(cy - crop_h / 2))
        # Clamp the window inside the frame, preserving its size.
        x1 = max(0, min(x1, width - crop_w))
        y1 = max(0, min(y1, height - crop_h))
        crops.append(frames[i, :, y1 : y1 + crop_h, x1 : x1 + crop_w])

    return torch.stack(crops)
