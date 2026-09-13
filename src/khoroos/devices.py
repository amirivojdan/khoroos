"""Discover inference devices visible to the server process.

A device *spec* is what a user writes: one device, ``auto``, ``all``, or several devices
as a comma-separated list. :func:`resolve_devices` turns a spec into the ordered list of
devices a run will actually use. More than one device means the run splits each batch
across them — see :mod:`khoroos.models.parallel`.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Spec meaning "every accelerator of the best kind this machine has".
ALL = "all"


def available_devices() -> list[dict[str, str]]:
    """Return CPU and available accelerators, using process-local GPU indices."""
    import torch

    devices = [{"id": "cpu", "label": "CPU"}]
    if torch.cuda.is_available():
        devices.extend(
            {"id": f"cuda:{index}", "label": f"GPU {index} — {torch.cuda.get_device_name(index)}"}
            for index in range(torch.cuda.device_count())
        )
    if torch.backends.mps.is_available():
        devices.append({"id": "mps", "label": "Apple GPU"})
    return devices


def device_choices() -> list[dict[str, str]]:
    """Hardware, plus the multi-GPU choice when there is more than one GPU.

    Separate from :func:`available_devices` because ``all`` is a spec rather than a piece
    of hardware: it is a valid thing to *ask for*, but never a device to load a model onto.
    Listed last so a picker shows real hardware first.
    """
    devices = available_devices()
    gpus = [device for device in devices if device["id"].startswith("cuda:")]
    if len(gpus) > 1:
        devices.append({"id": ALL, "label": f"All {len(gpus)} GPUs — split each video"})
    return devices


def _requested(device: str | Sequence[str]) -> list[str]:
    """Split a spec into its parts, accepting a string, a list, or a list of strings."""
    parts = [device] if isinstance(device, str) else [str(item) for item in device]
    return [part.strip() for item in parts for part in item.split(",") if part.strip()]


def resolve_devices(device: str | Sequence[str]) -> list[str]:
    """Resolve a spec to the ordered devices to run on, rejecting unavailable hardware.

    ``auto`` stays what it has always been — the single best device — so an existing
    configuration keeps its behaviour on a machine that has since grown a second GPU.
    Using every GPU is opt-in, via ``all`` or an explicit list like ``cuda:0,cuda:1``.
    """
    import torch

    ids = [item["id"] for item in available_devices()]
    gpus = [item for item in ids if item.startswith("cuda:")]

    def current_gpu() -> str:
        return f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cuda"

    def expand(item: str) -> list[str]:
        if item == "auto":
            if gpus:
                return [current_gpu()]
            return ["mps"] if "mps" in ids else ["cpu"]
        if item in (ALL, "cuda:all"):
            if gpus:
                return list(gpus)
            # "cuda:all" asked for GPUs specifically; report that none exist rather than
            # quietly running on the CPU. Plain "all" means whatever this machine has.
            if item == "cuda:all":
                return ["cuda"]
            return ["mps"] if "mps" in ids else ["cpu"]
        if item == "cuda":
            return [current_gpu()]
        return [item]

    requested = _requested(device)
    if not requested:
        raise ValueError("No device requested. Choose an available device.")

    # Deduplicated in the order asked for: a device named twice would otherwise load the
    # models twice and receive two shares of every batch.
    resolved = list(dict.fromkeys(item for part in requested for item in expand(part)))
    for item in resolved:
        if item not in ids:
            raise ValueError(
                f"Device {item!r} is unavailable on this server. Choose an available device."
            )
    return resolved


def resolve_device(device: str | Sequence[str]) -> str:
    """Resolve a spec to a single device — the first, when the spec names several."""
    return resolve_devices(device)[0]
