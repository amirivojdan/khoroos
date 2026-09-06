"""Interchangeable per-image bounding-box annotation codecs."""

from khoroos.annotations.codecs import AnnotationCodec, CocoCodec, YoloCodec, annotation_codec
from khoroos.annotations.detections import Detections

__all__ = ["AnnotationCodec", "CocoCodec", "Detections", "YoloCodec", "annotation_codec"]
