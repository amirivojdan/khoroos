"""Discover inference devices visible to the server process."""

from __future__ import annotations


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


def resolve_device(device: str) -> str:
    """Resolve aliases and reject hardware unavailable to this process."""
    import torch

    devices = {item["id"] for item in available_devices()}
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if "mps" in devices else "cpu"
    if device == "cuda":
        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cuda"
    if device not in devices:
        raise ValueError(
            f"Device {device!r} is unavailable on this server. Choose an available device."
        )
    return device
