"""Unit tests for dfine.nn.openvino_runtime device resolution."""

import pytest


def test_resolve_openvino_device_rejects_unknown_device(monkeypatch):
    from dfine.nn import openvino_runtime

    monkeypatch.setattr(openvino_runtime, "list_openvino_devices", lambda: ["CPU"])

    with pytest.raises(ValueError, match="not available on this machine"):
        openvino_runtime.resolve_openvino_device("NPU")


def test_resolve_openvino_device_auto_prefers_npu_then_gpu_then_cpu(monkeypatch):
    from dfine.nn import openvino_runtime

    monkeypatch.setattr(openvino_runtime, "list_openvino_devices", lambda: ["CPU", "GPU", "NPU"])
    assert openvino_runtime.resolve_openvino_device(None) == "NPU"
    assert openvino_runtime.resolve_openvino_device("auto") == "NPU"

    monkeypatch.setattr(openvino_runtime, "list_openvino_devices", lambda: ["CPU", "GPU"])
    assert openvino_runtime.resolve_openvino_device("auto") == "GPU"

    monkeypatch.setattr(openvino_runtime, "list_openvino_devices", lambda: ["CPU"])
    assert openvino_runtime.resolve_openvino_device("auto") == "CPU"


def test_resolve_openvino_device_auto_raises_when_nothing_available(monkeypatch):
    from dfine.nn import openvino_runtime

    monkeypatch.setattr(openvino_runtime, "list_openvino_devices", lambda: [])

    with pytest.raises(ValueError, match="No OpenVINO devices are available"):
        openvino_runtime.resolve_openvino_device("auto")


def test_resolve_openvino_device_is_case_insensitive(monkeypatch):
    from dfine.nn import openvino_runtime

    monkeypatch.setattr(openvino_runtime, "list_openvino_devices", lambda: ["CPU", "GPU"])
    assert openvino_runtime.resolve_openvino_device("gpu") == "GPU"


def test_missing_openvino_raises_clear_import_error(monkeypatch):
    import builtins

    from dfine.nn import openvino_runtime

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openvino":
            raise ImportError("no module named openvino")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="uv sync --extra openvino"):
        openvino_runtime.list_openvino_devices()
