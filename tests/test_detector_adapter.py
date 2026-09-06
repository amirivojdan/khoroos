"""The shipped detector's full output path, with fake inference and real NMS/formatting."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

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
