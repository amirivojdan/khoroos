"""Checkpoint resolution.

Default models come from the configured Hugging Face repositories. Resolution order:

1. An explicit path passed by the caller.
2. ``KHOROOS_DETECTOR_PATH`` / ``KHOROOS_ACTION_PATH`` settings.
3. A cached Hub snapshot belonging to the configured repository.
4. The Hugging Face Hub (downloaded once, then cached).

Notebook checkpoints and legacy, unscoped weight directories are never searched.
"""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from khoroos.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Files that must be present for a directory to count as a usable checkpoint.
_REQUIRED = ("config.json",)


class CheckpointNotFoundError(RuntimeError):
    """Raised when a checkpoint cannot be located locally or fetched from the Hub."""


def _is_checkpoint(path: Path) -> bool:
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    return path.is_dir() and all((path / f).exists() for f in _REQUIRED) and has_weights


def _repo_id(kind: str, settings: Settings) -> str:
    if kind not in ("detector", "action"):
        raise ValueError(f"Unknown checkpoint kind {kind!r}")
    return getattr(settings, f"{kind}_repo")


def _cached_snapshot(repo_id: str, settings: Settings) -> Path | None:
    try:
        return Path(
            snapshot_download(
                repo_id=repo_id, cache_dir=settings.weights_dir, local_files_only=True,
            )
        )
    except LocalEntryNotFoundError:
        return None


def local_config_path(kind: str, settings: Settings) -> Path | None:
    """Resolve local metadata with the same precedence as inference, without weights.

    An explicitly configured directory is authoritative even when its metadata is missing.
    No downloads or model imports are performed.
    """
    repo_id = _repo_id(kind, settings)
    configured = getattr(settings, f"{kind}_path")
    if configured is not None:
        path = Path(configured).expanduser() / "config.json"
        return path if path.is_file() else None
    snapshot = _cached_snapshot(repo_id, settings)
    path = snapshot / "config.json" if snapshot else None
    return path if path is not None and path.is_file() else None


def _download_from_hub(repo_id: str, settings: Settings) -> Path:
    logger.info("Downloading %s from the Hugging Face Hub (this happens once)...", repo_id)
    try:
        path = snapshot_download(repo_id=repo_id, cache_dir=settings.weights_dir)
    except Exception as exc:
        raise CheckpointNotFoundError(
            f"Could not download {repo_id!r} from the Hugging Face Hub: {exc}. "
            "For private repositories, run `hf auth login` or set HF_TOKEN to a token "
            "with read access to this repository."
        ) from exc
    return Path(path)


def resolve_checkpoint(
    kind: str,
    explicit: str | Path | None = None,
    settings: Settings | None = None,
    allow_download: bool = True,
) -> Path:
    """Resolve a checkpoint directory for ``kind`` (``"detector"`` or ``"action"``)."""
    settings = settings or get_settings()
    repo_id = _repo_id(kind, settings)

    if explicit is not None:
        path = Path(explicit).expanduser()
        if not _is_checkpoint(path):
            raise CheckpointNotFoundError(f"{path} is not a valid {kind} checkpoint directory.")
        return path

    configured = settings.detector_path if kind == "detector" else settings.action_path
    if configured is not None:
        path = Path(configured).expanduser()
        if not _is_checkpoint(path):
            raise CheckpointNotFoundError(
                f"Configured {kind} checkpoint {path} is missing or incomplete."
            )
        return path

    cached = _cached_snapshot(repo_id, settings)
    if cached is not None and _is_checkpoint(cached):
        return cached

    if allow_download:
        path = _download_from_hub(repo_id, settings)
        if _is_checkpoint(path):
            return path

    raise CheckpointNotFoundError(
        f"No complete {kind} checkpoint for {repo_id!r} is cached. "
        "Run `khoroos models download` while authenticated with Hugging Face."
    )


def checkpoint_status(settings: Settings | None = None) -> dict[str, dict[str, object]]:
    """Report where each checkpoint would come from, without downloading anything."""
    settings = settings or get_settings()
    status: dict[str, dict[str, object]] = {}
    for kind in ("detector", "action"):
        try:
            path = resolve_checkpoint(kind, settings=settings, allow_download=False)
            source = "local" if getattr(settings, f"{kind}_path") is not None else "hub"
            status[kind] = {"available": True, "path": str(path), "source": source}
            if source == "hub":
                status[kind]["hub_repo"] = _repo_id(kind, settings)
        except CheckpointNotFoundError:
            repo = settings.detector_repo if kind == "detector" else settings.action_repo
            status[kind] = {"available": False, "path": None, "hub_repo": repo}
    return status
