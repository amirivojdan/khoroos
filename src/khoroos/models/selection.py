"""Class selection shared by model wrappers and the pipeline."""

from collections.abc import Sequence

import numpy as np

from khoroos.config import UNCERTAIN_LABEL
from khoroos.interfaces import VideoClassifier


def validate_classes(classes: Sequence[str]) -> list[str]:
    values = list(classes)
    if (
        not values
        or any(not isinstance(c, str) or not c.strip() for c in values)
        or len(set(values)) != len(values)
        or UNCERTAIN_LABEL in values
    ):
        raise ValueError("classes must be nonempty, unique labels excluding 'uncertain'")
    return values


def class_indices(classes: Sequence[str], selected: Sequence[str] | None) -> list[int]:
    available = validate_classes(classes)
    selected = available if selected is None else validate_classes(selected)
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown action classes: {sorted(unknown)}. Available: {available}")
    return [available.index(c) for c in selected]


def validate_probabilities(values: np.ndarray, batch: int, classes: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (batch, classes):
        raise ValueError(f"Classifier returned {values.shape}; expected {(batch, classes)}")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Classifier must return finite probabilities in [0, 1]")
    # Selected-class probabilities may sum to less than one, but never more than one.
    if np.any(values.sum(axis=1) > 1.0001):
        raise ValueError("Classifier probabilities must sum to at most one per clip")
    return values


class ClassSelection:
    """One implementation of ordered vocabulary selection and probability validation."""

    def __init__(self, available: Sequence[str], selected: Sequence[str] | None = None):
        self.indices = class_indices(available, selected)
        self.classes = [available[i] for i in self.indices]
        self.input_count = len(available)

    def apply(self, values: np.ndarray, batch_size: int) -> np.ndarray:
        return validate_probabilities(values, batch_size, self.input_count)[:, self.indices]


class SelectedClasses(VideoClassifier):
    """Expose an ordered subset of any classifier without renormalizing confidence.

    This selects the highest-scoring requested label. Unselected probability mass remains
    unassigned, so a weak requested class cannot become certain simply by filtering.
    """

    def __init__(self, classifier: VideoClassifier, classes: Sequence[str] | None = None) -> None:
        self.classifier = classifier
        self.selection = ClassSelection(classifier.classes, classes)
        self.classes = self.selection.classes
        self.num_frames = classifier.num_frames

    @property
    def name(self) -> str:
        from khoroos.interfaces import component_name

        return component_name(self.classifier)

    @property
    def model_card(self) -> dict:
        from khoroos.interfaces import model_card

        return model_card(self.classifier)

    def classify(self, clips):
        return self.selection.apply(self.classifier.classify(clips), len(clips))
