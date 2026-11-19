"""Unit tests for hardware discovery (no GPU and no OpenVINO needed: both are simulated)."""

import pytest

pytest.importorskip("fastapi", reason="needs the 'studio' extra")

import torch  # noqa: E402

from nitid.studio import devices  # noqa: E402


def _patch_openvino(monkeypatch, result=None, error=None):
    import dfine.nn.openvino_runtime as runtime

    def fake():
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(runtime, "list_openvino_devices", fake)


def test_openvino_missing_is_not_an_error(monkeypatch):
    _patch_openvino(monkeypatch, error=ImportError("OpenVINO is not installed."))
    assert devices.openvino_devices() == {"available": False, "devices": []}


def test_openvino_devices_listed(monkeypatch):
    _patch_openvino(monkeypatch, result=["CPU", "GPU", "NPU"])
    assert devices.openvino_devices() == {"available": True, "devices": ["CPU", "GPU", "NPU"]}


def test_openvino_installed_without_devices_is_unavailable(monkeypatch):
    _patch_openvino(monkeypatch, result=[])
    assert devices.openvino_devices() == {"available": False, "devices": []}


def test_openvino_runtime_failure_is_swallowed(monkeypatch):
    _patch_openvino(monkeypatch, error=RuntimeError("driver exploded"))
    assert devices.openvino_devices() == {"available": False, "devices": []}


def test_openvino_device_names_are_strings(monkeypatch):
    _patch_openvino(monkeypatch, result=("CPU",))
    assert devices.openvino_devices()["devices"] == ["CPU"]


def test_torch_cpu_always_present_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert devices.torch_devices() == {"cpu": True, "cuda": []}


def test_cuda_devices_simulated(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: f"Fake GPU {i}")
    assert devices.torch_devices() == {
        "cpu": True,
        "cuda": [{"index": 0, "name": "Fake GPU 0"}, {"index": 1, "name": "Fake GPU 1"}],
    }


def test_cuda_name_failure_keeps_the_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def get_name(index):
        if index == 0:
            raise RuntimeError("no name")
        return "Good GPU"

    monkeypatch.setattr(torch.cuda, "get_device_name", get_name)
    cuda = devices.torch_devices()["cuda"]
    assert [d["index"] for d in cuda] == [0, 1]
    assert cuda[0]["name"] == "CUDA device 0"
    assert cuda[1]["name"] == "Good GPU"


def test_cuda_probe_failure_means_no_cuda(monkeypatch):
    def boom():
        raise RuntimeError("driver mismatch")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    assert devices.torch_devices() == {"cpu": True, "cuda": []}


def test_collect_devices_combines_both(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _patch_openvino(monkeypatch, result=["CPU"])
    assert devices.collect_devices() == {
        "torch": {"cpu": True, "cuda": []},
        "openvino": {"available": True, "devices": ["CPU"]},
    }
