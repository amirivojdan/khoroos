"""Device discovery does not require accelerator hardware in CI."""

import pytest
import torch

from khoroos.devices import available_devices, device_choices, resolve_device, resolve_devices


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


@pytest.fixture
def two_gpus(cpu_only, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"Test GPU {index}")


def test_every_gpu_is_used_only_when_asked_for(two_gpus):
    """A machine that grows a second GPU must not silently change what "auto" means."""
    assert resolve_devices("auto") == ["cuda:0"]
    assert resolve_devices("all") == ["cuda:0", "cuda:1"]
    assert resolve_devices("cuda:all") == ["cuda:0", "cuda:1"]
    assert resolve_devices("cuda:1,cuda:0") == ["cuda:1", "cuda:0"]
    assert resolve_devices(["cuda:0", "cuda:1"]) == ["cuda:0", "cuda:1"]


def test_a_device_named_twice_is_used_once(two_gpus):
    """Two shares of every batch on one GPU would load the models twice for no gain."""
    assert resolve_devices("cuda:1,cuda:0,cuda:1") == ["cuda:1", "cuda:0"]
    assert resolve_devices("all,cuda:0") == ["cuda:0", "cuda:1"]
    assert resolve_devices(" cuda:0 , , cuda:1 ") == ["cuda:0", "cuda:1"]


def test_one_unavailable_device_rejects_the_whole_list(two_gpus):
    for spec in ("cuda:0,cuda:2", "cpu,banana", "cuda:9"):
        with pytest.raises(ValueError, match="unavailable"):
            resolve_devices(spec)
    with pytest.raises(ValueError, match="No device requested"):
        resolve_devices(" , ")


def test_all_falls_back_to_whatever_the_machine_has(cpu_only):
    assert resolve_devices("all") == ["cpu"]
    # "cuda:all" asked for GPUs by name, so silently using the CPU would be wrong.
    with pytest.raises(ValueError, match="unavailable"):
        resolve_devices("cuda:all")


def test_the_multi_gpu_choice_is_offered_only_when_there_are_several(two_gpus, monkeypatch):
    assert device_choices()[-1]["id"] == "all"
    assert "2 GPUs" in device_choices()[-1]["label"]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert [choice["id"] for choice in device_choices()] == ["cpu", "cuda:0"]


def test_resolving_to_one_device_takes_the_first(two_gpus):
    assert resolve_device("cuda:1,cuda:0") == "cuda:1"
    assert resolve_device("all") == "cuda:0"
