"""Exercise checkpoint label mapping and selection without downloading weights."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from khoroos.config import Settings
from khoroos.models.action import ActionRecognizer


@pytest.fixture
def checkpoint_loader(monkeypatch, tmp_path):
    import khoroos.models.action as module

    def load(labels, classes=None, frames=4):
        class Model:
            config = SimpleNamespace(id2label=labels, frames_per_clip=frames)

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, **inputs):
                return SimpleNamespace(
                    logits=torch.tensor([[1.0, 2.0, 3.0]]).repeat(inputs["n"], 1)
                )

        class Inputs(dict):
            def to(self, device):
                return self

        class Processor:
            def __call__(self, clips, return_tensors):
                assert all(clip.shape[0] == frames for clip in clips)
                return Inputs(n=len(clips))

        monkeypatch.setattr(module, "resolve_checkpoint", lambda *args: tmp_path)
        monkeypatch.setitem(
            sys.modules,
            "transformers",
            SimpleNamespace(
                AutoModelForVideoClassification=SimpleNamespace(
                    from_pretrained=lambda path, *, local_files_only: Model()
                ),
                AutoVideoProcessor=SimpleNamespace(
                    from_pretrained=lambda path, *, local_files_only: Processor()
                ),
            ),
        )
        return ActionRecognizer(settings=Settings(device="cpu"), classes=classes)

    return load


def test_custom_checkpoint_vocabulary_and_selection(checkpoint_loader):
    model = checkpoint_loader({"2": "c", "0": "a", "1": "b"}, classes=["c", "a"])
    clip = torch.zeros((2, 3, 8, 8), dtype=torch.uint8)
    values = model.classify([clip])
    expected = torch.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=0).numpy()[[2, 0]]
    np.testing.assert_allclose(values[0], expected)
    assert model.classes == ["c", "a"]
    assert model.classify([]).shape == (0, 2)
    assert model.classify_batched([clip, clip], 1).shape == (2, 2)


@pytest.mark.parametrize("labels", [{1: "a", 2: "b"}, {0: "a", 1: "a"}, {0: "uncertain"}])
def test_invalid_checkpoint_labels_fail(checkpoint_loader, labels):
    with pytest.raises(ValueError):
        checkpoint_loader(labels)


def test_unknown_selected_checkpoint_class_fails(checkpoint_loader):
    with pytest.raises(ValueError, match="Unknown action classes"):
        checkpoint_loader({0: "a", 1: "b", 2: "c"}, classes=["missing"])


def test_invalid_checkpoint_frame_count_fails(checkpoint_loader):
    with pytest.raises(ValueError, match="frames_per_clip"):
        checkpoint_loader({0: "a", 1: "b", 2: "c"}, frames=0)


def test_checkpoint_output_width_must_match_labels(checkpoint_loader):
    model = checkpoint_loader({0: "a", 1: "b"})
    with pytest.raises(ValueError, match="Classifier returned"):
        model.classify([torch.zeros((2, 3, 8, 8), dtype=torch.uint8)])
