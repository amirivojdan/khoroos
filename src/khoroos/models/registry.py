"""Checkpoint resolution.

Weights total ~1.7 GB and cannot ship inside the wheel, so they are resolved in order:

1. An explicit path passed by the caller.
2. ``KHOROOS_DETECTOR_PATH`` / ``KHOROOS_ACTION_PATH`` settings.
3. A checkpoint previously downloaded into the local cache.
4. Well-known local development locations inside the repo.
5. The Hugging Face Hub (downloaded once, then cached).
"""

from __future__ import annotations

import logging
from pathlib import Path

from khoroos.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Files that must be present for a directory to count as a usable checkpoint.
_REQUIRED = ("config.json",)

#: Locations checked during development, relative to the repository root.
_DEV_LOCATIONS = {
    "detector": ("notebooks/checkpoints/chicken-rtdetr-final",),
    "action": (
        "notebooks/checkpoints/vjepa2-chicken-action/final",
        "notebooks/checkpoints/vjepa2-chicken-action",
    ),
}


class CheckpointNotFoundError(RuntimeError):
    """Raised when a checkpoint cannot be located locally or fetched from the Hub."""


def _is_checkpoint(path: Path) -> bool:
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    return path.is_dir() and all((path / f).exists() for f in _REQUIRED) and has_weights


def _repo_root() -> Path:
    # src/khoroos/models/registry.py -> repo root is four levels up.
    return Path(__file__).resolve().parents[3]


def _search_dev_locations(kind: str) -> Path | None:
    root = _repo_root()
    for rel in _DEV_LOCATIONS.get(kind, ()):
        candidate = root / rel
        if _is_checkpoint(candidate):
            return candidate
    return None


def _download_from_hub(repo_id: str, target: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise CheckpointNotFoundError(
            "huggingface-hub is required to download model weights."
        ) from exc

    logger.info("Downloading %s from the Hugging Face Hub (this happens once)...", repo_id)
    target.mkdir(parents=True, exist_ok=True)
    try:
        path = snapshot_download(repo_id=repo_id, local_dir=str(target))
    except Exception as exc:
        raise CheckpointNotFoundError(
            f"Could not download {repo_id!r} from the Hugging Face Hub: {exc}. "
            "If the repository is private or not published, provide a local checkpoint "
            "with KHOROOS_DETECTOR_PATH or KHOROOS_ACTION_PATH."
        ) from exc
    return Path(path)


def resolve_checkpoint(
    kind: str,
    explicit: str | Path | None = None,
    settings: Settings | None = None,
    allow_download: bool = True,
) -> Path:
    """Resolve a checkpoint directory for ``kind`` (``"detector"`` or ``"action"``)."""
    if kind not in ("detector", "action"):
        raise ValueError(f"Unknown checkpoint kind {kind!r}")
    settings = settings or get_settings()

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

    cached = settings.weights_dir / kind
    if _is_checkpoint(cached):
        return cached

    if (dev := _search_dev_locations(kind)) is not None:
        logger.debug("Using development %s checkpoint at %s", kind, dev)
        return dev

    if allow_download:
        repo_id = settings.detector_repo if kind == "detector" else settings.action_repo
        path = _download_from_hub(repo_id, cached)
        if _is_checkpoint(path):
            return path

    raise CheckpointNotFoundError(
        f"Could not locate the {kind} checkpoint. Set KHOROOS_{kind.upper()}_PATH to a local "
        f"checkpoint directory, or run `khoroos models download`."
    )


def checkpoint_status(settings: Settings | None = None) -> dict[str, dict[str, object]]:
    """Report where each checkpoint would come from, without downloading anything."""
    settings = settings or get_settings()
    status: dict[str, dict[str, object]] = {}
    for kind in ("detector", "action"):
        try:
            path = resolve_checkpoint(kind, settings=settings, allow_download=False)
            status[kind] = {"available": True, "path": str(path)}
        except CheckpointNotFoundError:
            repo = settings.detector_repo if kind == "detector" else settings.action_repo
            status[kind] = {"available": False, "path": None, "hub_repo": repo}
    return status
