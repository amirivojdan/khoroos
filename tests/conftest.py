"""Shared fixtures.

Real weights are 1.7 GB and real inference is GPU-bound, so the integration tests drive
the pipeline with stub models over a tiny generated video. That keeps the wiring — reader,
tracker, tracklets, metrics, export — under test on any machine, including CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from khoroos.config import ACTION_CLASSES


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    """A short video containing two clearly separated moving rectangles ("birds")."""
    av = pytest.importorskip("av")

    path = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    width, height, fps, n_frames = 640, 480, 25, 100

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "28", "preset": "ultrafast"}

    for frame_index in range(n_frames):
        canvas = np.full((height, width, 3), 40, dtype=np.uint8)
        for offset, speed in ((60, 2), (360, 1)):
            x = offset + frame_index * speed
            y = 150 + int(20 * np.sin(frame_index / 8))
            canvas[y : y + 150, x : x + 150] = (220, 200, 170)
        container.mux(stream.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")))

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


class StubDetector:
    """Returns two boxes tracking the synthetic video's rectangles."""

    checkpoint = Path("stub-detector")

    def __init__(self, *_, **__):
        self.calls = 0

    def detect(self, images, confidence_threshold=0.3, nms_threshold=0.6, box_padding=0.0):
        if isinstance(images, torch.Tensor):
            images = list(images)
        results = []
        for _ in images:
            frame = self.calls
            self.calls += 1
            boxes = np.array(
                [
                    [60 + frame * 2, 150, 210 + frame * 2, 300],
                    [360 + frame, 150, 510 + frame, 300],
                ],
                dtype=np.float32,
            )
            results.append((boxes, np.array([0.9, 0.85], dtype=np.float32)))
        return results


class StubRecognizer:
    """Deterministically alternates between two behaviours."""

    checkpoint = Path("stub-action")
    num_frames = 16

    def __init__(self, *_, **__):
        self.classes = list(ACTION_CLASSES)
        self._n = 0

    def classify(self, clips):
        probs = np.zeros((len(clips), len(self.classes)), dtype=np.float32)
        for row in range(len(clips)):
            label = "feeding" if self._n % 2 == 0 else "preening"
            probs[row, self.classes.index(label)] = 0.92
            probs[row, self.classes.index("standing")] = 0.08
            self._n += 1
        return probs

    def classify_batched(self, clips, batch_size=4):
        return self.classify(clips)


@pytest.fixture
def make_stub_analyzer():
    """Build independent :class:`VideoAnalyzer` instances wired to stub models.

    A test comparing two runs needs one analyzer each: the stubs count the calls they have
    seen and move their boxes accordingly, so a second run through the same instance would
    start where the first left off rather than reproducing it.

    Pinned to the CPU so the job manager sees the same device the tests ask jobs to run
    on. Left at the default ``auto``, an injected runner on a machine with a GPU would
    look like a runner that has to be moved before a ``cpu`` job could use it.
    """
    from khoroos.config import get_settings
    from khoroos.pipeline.analyze import VideoAnalyzer

    def build():
        return VideoAnalyzer(
            settings=get_settings().with_overrides(device="cpu"),
            detector=StubDetector(),
            recognizer=StubRecognizer(),
        )

    return build


@pytest.fixture
def stub_analyzer(make_stub_analyzer):
    return make_stub_analyzer()


@pytest.fixture
def stub_runner(stub_analyzer):
    """An :class:`AnalysisRunner` wired to stub models."""
    from khoroos.pipeline.runner import AnalysisRunner

    return AnalysisRunner(analyzer=stub_analyzer)
