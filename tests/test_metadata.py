"""Vocabulary discovery must not require inference or pretend custom checkpoints use defaults."""

import json
from types import SimpleNamespace

from khoroos.config import Settings
from khoroos.environment import describe_environment


def test_metadata_only_checkpoint_advertises_its_own_labels(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"id2label": {"1": "asleep", "0": "awake"}}))
    settings = Settings(device="cpu", action_path=tmp_path, behaviour_groups={"sleep": ["asleep"]})
    description = describe_environment(settings)
    assert description["classes"] == ["awake", "asleep"]
    assert description["behaviour_groups"] == {"sleep": ["asleep"]}
    assert description["classes_source"] == str(tmp_path / "config.json")
    assert not description["models"]["action"]["available"]  # no weights exist


def test_missing_custom_metadata_does_not_advertise_chickenact(tmp_path):
    description = describe_environment(Settings(device="cpu", action_path=tmp_path))
    assert description["classes"] == []
    assert description["classes_source"] == "unavailable"
    assert description["behaviour_groups"] == {}


def test_invalid_metadata_is_reported_without_fallback(tmp_path):
    (tmp_path / "config.json").write_text('{"id2label":{"1":"wrong"}}')
    description = describe_environment(Settings(device="cpu", action_path=tmp_path))
    assert description["classes"] == []
    assert "contiguous" in description["metadata_error"]


def test_injected_classifier_overrides_checkpoint_metadata(tmp_path):
    settings = Settings(
        device="cpu", action_path=tmp_path, behaviour_groups={"activity": ["moving"]}
    )
    description = describe_environment(
        settings, classifier=SimpleNamespace(classes=["moving", "still"])
    )
    assert description["classes"] == ["moving", "still"]
    assert description["classes_source"] == "classifier"
    assert description["models"]["action"]["source"] == "injected"
    assert description["models"]["action"]["available"]
    assert description["behaviour_groups"] == {"activity": ["moving"]}
