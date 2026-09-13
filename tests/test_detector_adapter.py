"""The shipped detector's full output path, with fake inference and real NMS/formatting."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from khoroos.annotations import CocoCodec, YoloCodec
from khoroos.config import Settings
from khoroos.models.detector import ChickenDetector


@pytest.fixture
def detector():
    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        def __call__(self, images, return_tensors):
            return Inputs()

        def post_process_object_detection(self, *args, **kwargs):
            return [
                {
                    "boxes": torch.tensor(
                        [
                            [-5.0, 5.0, 20.0, 30.0],
                            [210.0, 5.0, 230.0, 30.0],
                            [2.0, 2.0, 1.0, 3.0],
                            [0.0, 0.0, float("inf"), 20.0],
                        ]
                    ),
                    "scores": torch.tensor([0.9, 0.8, 0.7, 0.6]),
                    "labels": torch.tensor([42, 7, 8, 9]),
                }
            ]

    detector = ChickenDetector.__new__(ChickenDetector)
    detector.settings = Settings(device="cpu")
    detector.device = torch.device("cpu")
    detector.processor = Processor()
    detector.model = lambda **kwargs: SimpleNamespace()
    return detector


@pytest.mark.parametrize("padding", [0, 0.1])
def test_detector_clamps_and_records_invalid_boxes(detector, padding):
    result = detector.detect([np.zeros((100, 200, 3), np.uint8)], box_padding=padding)[0]
    assert len(result) == 1
    assert result.boxes[0, 0] == 0
    assert result.class_ids.tolist() == [42]
    assert result.dropped_count == 3


@pytest.mark.parametrize("codec", [YoloCodec([42]), CocoCodec(image_id=1)])
def test_default_detect_annotations_handles_edge_boxes(detector, codec):
    output = detector.detect_annotations(
        [np.zeros((100, 200, 3), np.uint8)],
        codec,
        image_sizes=[(200, 100)],
    )[0]
    decoded = codec.decode(output, width=200, height=100)
    np.testing.assert_allclose(decoded.boxes, [[0, 5, 20, 30]], atol=1e-5)
    assert decoded.class_ids.tolist() == [42]


# ---------------------------------------------------------------------------
# Preprocessing path selection
#
# Frames reach the detector as uint8 CHW tensors and modern image processors resize
# tensors natively, so the PIL round trip is skipped when the processor can take them.
# The two paths must stay interchangeable: these tests pin which path is chosen and what
# the processor actually receives, since picking wrong is silent — it costs throughput,
# or (for a PIL-only processor handed tensors) changes what gets resized.
# ---------------------------------------------------------------------------


class RecordingProcessor:
    """Captures what ``detect`` handed it, and reports a configurable backend."""

    def __init__(self, backend=None, is_fast=None):
        if backend is not None:
            self.backend = backend
        if is_fast is not None:
            self.is_fast = is_fast
        self.received = None

    def __call__(self, images, return_tensors):
        self.received = images

        class Inputs(dict):
            def to(self, device):
                return self

        return Inputs()

    def post_process_object_detection(self, *args, **kwargs):
        return [
            {
                "boxes": torch.tensor([[1.0, 2.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]


def make_detector(processor):
    detector = ChickenDetector.__new__(ChickenDetector)
    detector.settings = Settings(device="cpu")
    detector.device = torch.device("cpu")
    detector.processor = processor
    detector.model = lambda **kwargs: SimpleNamespace()
    return detector


@pytest.mark.parametrize(
    ("processor", "expected"),
    [
        (RecordingProcessor(backend="torchvision"), True),
        (RecordingProcessor(backend="pil"), False),
        (RecordingProcessor(is_fast=True), True),  # pre-5.x spelling
        (RecordingProcessor(is_fast=False), False),
        (RecordingProcessor(), False),  # a processor declaring neither stays on PIL
    ],
)
def test_tensor_capability_is_read_from_the_processor(processor, expected):
    assert make_detector(processor).processor_takes_tensors is expected


def test_tensor_native_processor_receives_tensors_not_pil_images():
    processor = RecordingProcessor(backend="torchvision")
    detector = make_detector(processor)
    frames = torch.zeros((2, 3, 48, 64), dtype=torch.uint8)

    detector.detect(frames)

    # The batch is forwarded as-is: re-stacking a list costs more than it saves.
    assert processor.received is frames


def test_pil_backed_processor_still_receives_pil_images():
    processor = RecordingProcessor(backend="pil")
    detector = make_detector(processor)

    detector.detect(torch.zeros((2, 3, 48, 64), dtype=torch.uint8))

    assert all(isinstance(image, Image.Image) for image in processor.received)


@pytest.mark.parametrize(
    "images",
    [
        pytest.param([np.zeros((48, 64, 3), np.uint8)], id="numpy-hwc"),
        pytest.param([torch.zeros((3, 48, 64), dtype=torch.float32)], id="float-tensor"),
        pytest.param([torch.zeros((1, 48, 64), dtype=torch.uint8)], id="single-channel"),
        pytest.param([Image.new("RGB", (64, 48))], id="pil-image"),
    ],
)
def test_inputs_needing_real_conversion_take_the_pil_path(images):
    """Only a plain uint8 RGB tensor is a pure round trip; everything else is work."""
    processor = RecordingProcessor(backend="torchvision")
    detector = make_detector(processor)

    detector.detect(images)

    assert all(isinstance(image, Image.Image) for image in processor.received)


def test_target_sizes_come_from_the_tensor_on_the_fast_path():
    """Box padding clamps against the source frame, so the sizes must survive the switch."""
    processor = RecordingProcessor(backend="torchvision")
    detector = make_detector(processor)

    # A box wider than the frame is clamped to it; 64x48 here, not the other way round.
    result = detector.detect(torch.zeros((1, 3, 48, 64), dtype=torch.uint8), box_padding=5.0)[0]

    assert result.boxes[0].tolist() == [0.0, 0.0, 64.0, 48.0]


def test_as_uint8_frames_moves_accelerator_tensors_to_host():
    """Resizing on an accelerator is not bit-equivalent to the CPU kernel."""
    from khoroos.models.detector import as_uint8_frames

    frames = [torch.zeros((3, 8, 8), dtype=torch.uint8)]
    assert all(frame.device.type == "cpu" for frame in as_uint8_frames(frames))
    assert as_uint8_frames(torch.zeros((2, 3, 8, 8), dtype=torch.uint8)).device.type == "cpu"
