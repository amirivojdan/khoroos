"""Annotation interchange tests use non-square frames and sparse category IDs."""

import numpy as np
import pytest

from khoroos.annotations import CocoCodec, Detections, YoloCodec, annotation_codec
from khoroos.interfaces import Detector


@pytest.fixture
def detections():
    return Detections([[20, 10, 100, 50], [0, 0, 200, 100]], [0.8, 0.9], [7, 42])


def test_yolo_known_coordinates_and_sparse_category_mapping(detections):
    codec = YoloCodec(category_ids=[42, 7])
    text = codec.encode(detections, width=200, height=100)
    np.testing.assert_allclose(
        [float(v) for v in text.splitlines()[0].split()], [1, 0.3, 0.3, 0.4, 0.4]
    )
    decoded = codec.decode(text, width=200, height=100)
    np.testing.assert_allclose(decoded.boxes, detections.boxes)
    np.testing.assert_array_equal(decoded.class_ids, [7, 42])
    np.testing.assert_array_equal(decoded.scores, [1, 1])


def test_coco_known_coordinates_and_image_filtering(detections):
    codec = CocoCodec(image_id=9, annotation_id=12)
    records = codec.encode(detections, width=200, height=100)
    assert records[0]["bbox"] == [20, 10, 80, 40]
    assert records[0]["id"] == 12
    assert records[0]["area"] == 3200
    records.append({"image_id": 99})
    decoded = codec.decode(records, width=200, height=100)
    np.testing.assert_allclose(decoded.boxes, detections.boxes)
    np.testing.assert_allclose(decoded.scores, detections.scores)
    np.testing.assert_array_equal(decoded.class_ids, detections.class_ids)


@pytest.mark.parametrize("name", ["yolo", "coco"])
def test_empty_annotations(name):
    codec = annotation_codec(name)
    empty = Detections(np.zeros((0, 4)), np.zeros(0))
    payload = codec.encode(empty, width=200, height=100)
    assert codec.decode(payload, width=200, height=100).boxes.shape == (0, 4)


@pytest.mark.parametrize(
    "row",
    [
        "0 .5 .5 0 .2",
        "0 nan .2 .2 .2",
        "0 .9 .9 .9 .9",
        "-1 .5 .5 .2 .2",
        "2 .5 .5 .2 .2",
        "0 .5 .5 .2 .2 .9",
    ],
)
def test_invalid_yolo_rows(row):
    with pytest.raises(ValueError):
        YoloCodec([7]).decode(row, width=200, height=100)


def test_protocol_switching_on_custom_detector(detections):
    class CustomDetector(Detector):
        def detect(self, images, **options):
            return [detections for _ in images]

    model = CustomDetector()
    yolo = model.detect_annotations([None], YoloCodec([7, 42]), image_sizes=[(200, 100)])[0]
    coco = model.detect_annotations([None], CocoCodec(), image_sizes=[(200, 100)])[0]
    np.testing.assert_allclose(
        YoloCodec([7, 42]).decode(yolo, width=200, height=100).boxes,
        CocoCodec().decode(coco, width=200, height=100).boxes,
    )


@pytest.mark.parametrize("width,height", [(0, 100), (100, -1)])
def test_invalid_image_size(width, height, detections):
    with pytest.raises(ValueError):
        YoloCodec().encode(detections, width=width, height=height)


def test_unknown_protocol_is_clear():
    with pytest.raises(ValueError, match="Unknown annotation protocol"):
        annotation_codec("pascal")


def test_canonical_coercion_preserves_ids_and_owns_immutable_arrays():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    result = Detections(boxes, [0.9], [42])
    boxes[:] = 0
    assert result.boxes[0, 2] == 10
    assert Detections.coerce(result) is result
    assert result.class_ids.tolist() == [42]
    for array in (result.boxes, result.scores, result.class_ids):
        with pytest.raises(ValueError):
            array.flat[0] = 0
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_geometry_drop_policy_preserves_alignment_and_counts():
    boxes = [[0, 0, 10, 10], [2, 2, 1, 3], [0, np.nan, 10, 10]]
    result = Detections.from_raw(boxes, [0.8, 0.7, 0.6], [42, 7, 9], invalid_boxes="drop")
    assert result.dropped_count == 2
    assert result.class_ids.tolist() == [42]
    with pytest.raises(ValueError, match="Boxes"):
        Detections.from_raw(boxes, [0.8, 0.7, 0.6])
    with pytest.raises(ValueError, match="scores"):
        Detections.from_raw(boxes, [0.8], invalid_boxes="drop")
    with pytest.raises(ValueError, match="Scores"):
        Detections.from_raw([[0, 0, 10, 10]], [np.nan], invalid_boxes="drop")
