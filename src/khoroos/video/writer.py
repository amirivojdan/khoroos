"""Annotated overlay rendering.

Encodes with PyAV rather than OpenCV: it produces H.264 in a faststart MP4 that plays
directly in the browser, which the UI needs for the results player.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from khoroos.config import UNCERTAIN_LABEL
from khoroos.interfaces import VideoReader
from khoroos.pipeline.types import AnalysisResult
from khoroos.tracking.tracklets import interpolate_track
from khoroos.video.reader import VideoSource

logger = logging.getLogger(__name__)

#: Distinct, colour-blind-considerate palette; index by class order for stability.
PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
    "#00a0b0",
    "#9b59b6",
    "#7f8c8d",
    "#c0392b",
    "#16a085",
    "#d35400",
    "#2c3e50",
]
UNCERTAIN_COLOUR = "#9aa3ad"
TRACK_PALETTE = [
    "#00c2e8",
    "#ffd166",
    "#ef476f",
    "#06d6a0",
    "#a78bfa",
    "#f97316",
    "#84cc16",
    "#f472b6",
    "#38bdf8",
    "#facc15",
]
TRAIL_SECONDS = 2.0


def colour_for(label: str, classes: list[str]) -> str:
    if label == UNCERTAIN_LABEL or label not in classes:
        return UNCERTAIN_COLOUR
    return PALETTE[classes.index(label) % len(PALETTE)]


def track_colour(track_id: int) -> str:
    return TRACK_PALETTE[abs(track_id) % len(TRACK_PALETTE)]


def _contrast_text(colour: str) -> str:
    red, green, blue = ImageColor.getrgb(colour)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#101010" if luminance > 150 else "#ffffff"


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 9.2 has no size argument
        return ImageFont.load_default()


def render_overlay(
    video_path: str | Path,
    result: AnalysisResult,
    output_path: str | Path,
    max_seconds: float | None = None,
    *,
    source_factory: Callable[[str | Path], VideoReader] | None = None,
) -> Path:
    """Draw tracked boxes and predicted actions onto the video.

    Each bird's box is coloured by the action predicted for the window it falls in, with
    its track ID and confidence. Uncertain windows are left unmarked. The reader factory
    is shared with analysis when called through AnalysisRunner.
    """
    try:
        import av  # noqa: F401 — check availability before opening the reader
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("PyAV is required to render annotated video.") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = (source_factory or VideoSource)(video_path)
    try:
        return _render_source(source, result, output_path, max_seconds)
    finally:
        source.close()


def _render_source(
    source: VideoReader,
    result: AnalysisResult,
    output_path: Path,
    max_seconds: float | None,
) -> Path:
    import av

    classes = list(result.model.classes)

    # Index predictions by track so the per-frame lookup is a bisect, not a scan.
    by_track: dict[int, list] = defaultdict(list)
    for p in result.predictions:
        by_track[p.track_id].append(p)
    for preds in by_track.values():
        preds.sort(key=lambda p: p.start_seconds)
    starts = {tid: [p.start_seconds for p in preds] for tid, preds in by_track.items()}

    def prediction_at(track_id: int, t: float):
        preds = by_track.get(track_id)
        if not preds:
            return None
        i = bisect_right(starts[track_id], t) - 1
        if i < 0:
            return None
        candidate = preds[i]
        return candidate if candidate.start_seconds <= t < candidate.end_seconds else None

    # Boxes per frame, from the tracks.
    boxes_by_frame: dict[int, list[tuple[int, tuple[float, ...]]]] = defaultdict(list)
    dense_tracks: dict[int, dict[int, np.ndarray]] = {}
    for track in result.tracks:
        dense = interpolate_track(track)
        dense_tracks[track.track_id] = dense
        for frame_index, box in dense.items():
            boxes_by_frame[frame_index].append(
                (track.track_id, tuple(float(value) for value in box))
            )

    fps = source.info.fps
    limit = source.info.num_frames
    if max_seconds is not None:
        limit = min(limit, int(max_seconds * fps))

    font = _load_font(max(14, source.info.width // 90))
    small_font = _load_font(max(12, source.info.width // 110))
    trail_frames = max(1, int(round(TRAIL_SECONDS * fps)))
    trail_step = max(1, int(round(fps / 8)))

    container = av.open(str(output_path), mode="w", options={"movflags": "+faststart"})
    try:
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
        stream.width = source.info.width
        stream.height = source.info.height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}

        for start in range(0, limit, 32):
            indices = list(range(start, min(start + 32, limit)))
            frames = source.get_frames(indices)
            arrays = frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()

            for offset, frame_index in enumerate(indices):
                image = Image.fromarray(arrays[offset])
                # RGBA drawing blends fading trail dots directly into the RGB frame.
                draw = ImageDraw.Draw(image, "RGBA")
                t = frame_index / fps

                present = boxes_by_frame.get(frame_index, [])
                for track_id, box in present:
                    trail_rgb = ImageColor.getrgb(track_colour(track_id))
                    dense = dense_tracks[track_id]
                    first_trail_frame = max(min(dense), frame_index - trail_frames)
                    trail_indices = list(range(first_trail_frame, frame_index + 1, trail_step))
                    if not trail_indices or trail_indices[-1] != frame_index:
                        trail_indices.append(frame_index)
                    for trail_index in trail_indices:
                        trail_box = dense.get(trail_index)
                        if trail_box is None:
                            continue
                        freshness = 1.0 - (frame_index - trail_index) / trail_frames
                        alpha = int(35 + 220 * max(0.0, freshness))
                        radius = 2 + int(4 * max(0.0, freshness))
                        tx = 0.5 * (float(trail_box[0]) + float(trail_box[2]))
                        ty = 0.5 * (float(trail_box[1]) + float(trail_box[3]))
                        draw.ellipse(
                            (tx - radius, ty - radius, tx + radius, ty + radius),
                            fill=(*trail_rgb, alpha),
                        )

                    prediction = prediction_at(track_id, t)
                    # Uncertain windows are left unmarked. An uncertain result is the
                    # absence of a finding, and drawing one on the footage invites it to be
                    # read as an observed behaviour. The trail still shows the bird, and the
                    # time is still reported in the budget, where it is labelled uncertain.
                    if prediction is None or prediction.is_uncertain:
                        continue
                    colour = colour_for(prediction.label, classes)
                    label = (
                        f"#{track_id} {prediction.label.replace('_', ' ')} "
                        f"{prediction.confidence:.0%}"
                    )

                    x1, y1, x2, y2 = (float(v) for v in box)
                    draw.rectangle((x1, y1, x2, y2), outline=(0, 0, 0, 180), width=5)
                    draw.rectangle((x1, y1, x2, y2), outline=colour, width=2)
                    confidence_x = x1 + max(0.0, min(1.0, prediction.confidence)) * (x2 - x1)
                    draw.rectangle((x1, y2 - 4, confidence_x, y2), fill=colour)

                    text_box = draw.textbbox((0, 0), label, font=small_font)
                    text_width = text_box[2] - text_box[0]
                    label_x = max(0.0, min(x1, source.info.width - text_width - 10))
                    label_y = max(0.0, y1 - 18)
                    draw.rectangle(
                        (label_x, label_y, label_x + text_width + 10, label_y + 17),
                        fill=colour,
                    )
                    draw.text(
                        (label_x + 5, label_y + 2),
                        label,
                        fill=_contrast_text(colour),
                        font=small_font,
                    )

                header = f"t={t:6.2f}s   birds in frame: {len(present)}"
                draw.rectangle((0, 0, 330, 26), fill="#000000")
                draw.text((6, 4), header, fill="#ffffff", font=font)

                av_frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    logger.info("Wrote annotated video to %s", output_path)
    return output_path
