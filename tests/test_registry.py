"""Model selection follows the configured Hub repository, never old training folders."""

import json

import pytest

from khoroos.config import Settings
from khoroos.models import registry
from khoroos.models.metadata import checkpoint_classes

REPOS = {
    "detector": "amirivojdan/chicken_rtdetrv2",
    "action": "amirivojdan/chicken_vjepa2_action",
}


def checkpoint(path, label="chicken"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"id2label": {"0": label}}))
    (path / "model.safetensors").touch()
    return path


def snapshot(settings, repo_id, label="chicken"):
    """Create the actual Hub cache layout so offline lookups exercise the SDK."""
    repo_cache = settings.weights_dir / ("models--" + repo_id.replace("/", "--"))
    (repo_cache / "refs").mkdir(parents=True, exist_ok=True)
    revision = "a" * 40
    (repo_cache / "refs" / "main").write_text(revision)
    return checkpoint(repo_cache / "snapshots" / revision, label)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for name in ("DETECTOR_PATH", "ACTION_PATH", "DETECTOR_REPO", "ACTION_REPO"):
        monkeypatch.delenv(f"KHOROOS_{name}", raising=False)
    return Settings(device="cpu", cache_dir=tmp_path / "cache")


@pytest.mark.parametrize("kind", REPOS)
def test_default_download_uses_private_repo_and_hub_cache(kind, settings, monkeypatch):
    calls = []
    real_snapshot_download = registry.snapshot_download

    def download(repo_id, *, cache_dir, local_files_only=False):
        calls.append((repo_id, cache_dir, local_files_only))
        if local_files_only:
            return real_snapshot_download(
                repo_id, cache_dir=cache_dir, local_files_only=True,
            )
        return str(snapshot(settings, repo_id))

    monkeypatch.setattr(registry, "snapshot_download", download)
    path = registry.resolve_checkpoint(kind, settings=settings)
    assert path.is_dir()
    assert calls == [
        (REPOS[kind], settings.weights_dir, True),
        (REPOS[kind], settings.weights_dir, False),
    ]
    calls.clear()
    assert registry.resolve_checkpoint(kind, settings=settings, allow_download=False) == path
    assert calls == [(REPOS[kind], settings.weights_dir, True)]


@pytest.mark.parametrize("kind", REPOS)
def test_old_notebook_and_unscoped_cache_are_ignored(kind, settings, tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    module = root / "src/khoroos/models/registry.py"
    module.parent.mkdir(parents=True)
    module.touch()
    (root / "pyproject.toml").touch()
    monkeypatch.setattr(registry, "__file__", str(module))
    checkpoint(root / "notebooks/checkpoints/chicken-rtdetr-final", "old")
    checkpoint(root / "notebooks/checkpoints/vjepa2-chicken-action/final", "old")
    checkpoint(settings.weights_dir / kind, "old")

    with pytest.raises(registry.CheckpointNotFoundError, match=REPOS[kind]):
        registry.resolve_checkpoint(kind, settings=settings, allow_download=False)
    assert registry.local_config_path(kind, settings) is None
    status = registry.checkpoint_status(settings)[kind]
    assert not status["available"]
    assert status["hub_repo"] == REPOS[kind]


@pytest.mark.parametrize("kind", REPOS)
def test_switching_repo_cannot_reuse_previous_weights_or_metadata(kind, settings):
    old_path = snapshot(settings, REPOS[kind], "old")
    assert registry.resolve_checkpoint(kind, settings=settings, allow_download=False) == old_path
    settings = settings.with_overrides(**{f"{kind}_repo": "someone/replacement"})
    with pytest.raises(registry.CheckpointNotFoundError, match="someone/replacement"):
        registry.resolve_checkpoint(kind, settings=settings, allow_download=False)
    assert registry.local_config_path(kind, settings) is None

    new_path = snapshot(settings, "someone/replacement", "replacement")
    assert registry.resolve_checkpoint(kind, settings=settings, allow_download=False) == new_path
    assert registry.local_config_path(kind, settings) == new_path / "config.json"
    status = registry.checkpoint_status(settings)[kind]
    assert status["source"] == "hub"
    assert status["hub_repo"] == "someone/replacement"
    if kind == "action":
        assert checkpoint_classes(settings)[0] == ["replacement"]


@pytest.mark.parametrize("kind", REPOS)
def test_explicit_custom_paths_still_override_the_hub(kind, settings, tmp_path):
    snapshot(settings, REPOS[kind])
    configured = checkpoint(tmp_path / "configured", "configured")
    explicit = checkpoint(tmp_path / "explicit", "explicit")
    settings = settings.with_overrides(**{f"{kind}_path": configured})
    assert registry.resolve_checkpoint(kind, settings=settings) == configured
    assert registry.resolve_checkpoint(kind, explicit, settings) == explicit
    assert registry.local_config_path(kind, settings) == configured / "config.json"
    assert registry.checkpoint_status(settings)[kind]["source"] == "local"

    settings = settings.with_overrides(**{f"{kind}_path": tmp_path / "missing"})
    with pytest.raises(registry.CheckpointNotFoundError, match="Configured"):
        registry.resolve_checkpoint(kind, settings=settings)
    assert registry.local_config_path(kind, settings) is None


def test_incomplete_snapshot_requires_download(settings, monkeypatch):
    path = snapshot(settings, REPOS["action"])
    (path / "model.safetensors").unlink()
    with pytest.raises(registry.CheckpointNotFoundError):
        registry.resolve_checkpoint("action", settings=settings, allow_download=False)

    def download(repo_id, active_settings):
        assert repo_id == REPOS["action"]
        assert active_settings == settings
        return checkpoint(path)

    monkeypatch.setattr(registry, "_download_from_hub", download)
    assert registry.resolve_checkpoint("action", settings=settings) == path


def test_download_failure_explains_private_repo_authentication(settings, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("access denied")

    monkeypatch.setattr(registry, "snapshot_download", fail)
    with pytest.raises(registry.CheckpointNotFoundError, match="hf auth login.*HF_TOKEN"):
        registry._download_from_hub(REPOS["action"], settings)


@pytest.mark.parametrize("lookup", [registry.resolve_checkpoint, registry.local_config_path])
def test_invalid_kind_is_rejected(lookup, settings):
    with pytest.raises(ValueError, match="Unknown checkpoint kind"):
        lookup("unknown", settings=settings)
