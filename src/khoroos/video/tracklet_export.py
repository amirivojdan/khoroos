"""Optional MP4 exports of the cropped clips supplied to the action classifier."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from khoroos.config import AnalysisParams
from khoroos.pipeline.types import ActionPrediction, Tracklet

if TYPE_CHECKING:
    import torch


def _label_folder(label: str) -> str:
    """Keep ordinary labels readable; disambiguate custom labels with unsafe characters."""
    if re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", label):
        return label
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", label)[:60] or "behavior"
    digest = hashlib.sha256(label.encode()).hexdigest()[:12]
    return f"{slug}-{digest}"


def write_clip(clip: torch.Tensor, duration_seconds: float, path: Path) -> None:
    """Encode RGB uint8 crops at a rate preserving the source window's duration.

    Clips may have been temporally sampled for inference. Pad odd dimensions by one
    edge pixel for H.264 rather than resizing the subject. Publish only finished MP4s.
    """
    import av

    temporary = path.with_suffix(".partial.mp4")
    try:
        with av.open(str(temporary), "w", options={"movflags": "+faststart"}) as container:
            rate = Fraction(clip.shape[0] / duration_seconds).limit_denominator(100000)
            stream = container.add_stream("libx264", rate=rate)
            height, width = clip.shape[-2:]
            stream.width, stream.height = width + width % 2, height + height % 2
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "veryfast"}
            for tensor in clip:
                array = tensor.permute(1, 2, 0).contiguous().cpu().numpy()
                if height % 2 or width % 2:
                    array = np.pad(array, ((0, height % 2), (0, width % 2), (0, 0)), mode="edge")
                for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class TrackletExporter:
    """Write each run into isolated folders; keep clips bounded to the inference batch."""

    def __init__(self, params: AnalysisParams, source_filename: str) -> None:
        self.source_filename = source_filename
        self.directories: dict[str, str] = {}
        stem = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(source_filename).stem)[:60] or "video"
        run_name = f"{stem}-{uuid.uuid4().hex}"
        for kind in ("raw", "classified"):
            root = getattr(params, f"{kind}_tracklets_dir")
            if root is not None:
                directory = Path(root).expanduser().resolve() / run_name
                directory.mkdir(parents=True, exist_ok=True)
                self.directories[kind] = str(directory)

    def save(
        self,
        kind: str,
        index: int,
        tracklet: Tracklet,
        clip: torch.Tensor,
        prediction: ActionPrediction | None = None,
        raw_path: Path | None = None,
    ) -> Path | None:
        if kind not in self.directories:
            return None
        directory = Path(self.directories[kind])
        if prediction is not None:
            directory /= _label_folder(prediction.label)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"clip_{index:06d}_track_{tracklet.track_id}.mp4"
        if raw_path is None:
            write_clip(clip, tracklet.duration_seconds, path)
        else:
            temporary = path.with_suffix(".partial.mp4")
            try:
                shutil.copyfile(raw_path, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        metadata = {
            "source_filename": self.source_filename,
            "track_id": tracklet.track_id,
            "start_s": tracklet.start_seconds,
            "end_s": tracklet.end_seconds,
            "source_window_frame_indices": tracklet.frame_indices,
            "encoded_frames": int(clip.shape[0]),
            "crop_width": int(clip.shape[-1]),
            "crop_height": int(clip.shape[-2]),
            "playback_fps": clip.shape[0] / tracklet.duration_seconds,
        }
        if prediction is not None:
            metadata["prediction"] = prediction.to_dict()
            metadata["probabilities"] = prediction.probabilities
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return path
