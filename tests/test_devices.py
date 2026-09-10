"""Device discovery does not require accelerator hardware in CI."""

import pytest
import torch

from khoroos.devices import available_devices, resolve_device


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def test_cpu_only_discovery_and_automatic_selection(cpu_only):
    assert available_devices() == [{"id": "cpu", "label": "CPU"}]
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    for device in ("cuda", "cuda:0", "mps", "banana", "cuda:-1"):
        with pytest.raises(ValueError, match="unavailable"):
            resolve_device(device)


def test_multiple_gpus_and_current_device_alias(cpu_only, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"Test GPU {index}")
    assert available_devices()[2] == {"id": "cuda:1", "label": "GPU 1 — Test GPU 1"}
    assert resolve_device("auto") == "cuda:1"
    assert resolve_device("cuda") == "cuda:1"
    assert resolve_device("cuda:0") == "cuda:0"
    with pytest.raises(ValueError, match="unavailable"):
        resolve_device("cuda:2")


def test_apple_gpu_is_available_and_selected_without_cuda(cpu_only, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert available_devices()[-1] == {"id": "mps", "label": "Apple GPU"}
    assert resolve_device("auto") == "mps"
