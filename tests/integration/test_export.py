"""Integration tests for model export (ONNX, TorchScript, TensorRT)."""

import pytest


def test_export_onnx(tiny_checkpoint, tmp_path):
    import onnx

    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    # imgsz must match the model's eval_spatial_size (640) — anchors are pre-computed
    out = model.export(format="onnx", imgsz=640, simplify=False, verbose=False)
    assert out.exists()
    assert out.suffix == ".onnx"

    # Validate ONNX graph inputs and outputs
    onnx_model = onnx.load(str(out))

    # Assert inputs
    inputs = [inp.name for inp in onnx_model.graph.input]
    assert len(inputs) == 1
    assert inputs[0] == "images"

    # Assert outputs
    outputs = [ot.name for ot in onnx_model.graph.output]
    assert len(outputs) == 3
    assert "labels" in outputs
    assert "boxes" in outputs
    assert "scores" in outputs

    out.unlink()


def test_export_torchscript(tiny_checkpoint, tmp_path):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    out = model.export(format="torchscript", imgsz=640, verbose=False)
    assert out.exists()
    assert out.suffix == ".torchscript"
    out.unlink()


def test_export_invalid_format(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    with pytest.raises(ValueError, match="Unsupported export format"):
        model.export(format="banana")


# ── TensorRT ─────────────────────────────────────────────────────────────────
# TRT tests are skipped automatically when tensorrt is not installed.


def test_export_tensorrt(tiny_checkpoint, tmp_path):
    pytest.importorskip("tensorrt", reason="tensorrt not installed")
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    out = model.export(format="tensorrt", imgsz=640, verbose=False)
    assert out.exists()
    assert out.suffix == ".engine"
    out.unlink()


def test_export_tensorrt_half(tiny_checkpoint, tmp_path):
    pytest.importorskip("tensorrt", reason="tensorrt not installed")
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    out = model.export(format="tensorrt", imgsz=640, half=True, verbose=False)
    assert out.exists()
    assert out.suffix == ".engine"
    out.unlink()


def test_export_tensorrt_dynamic(tiny_checkpoint, tmp_path):
    pytest.importorskip("tensorrt", reason="tensorrt not installed")
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    out = model.export(format="tensorrt", imgsz=640, dynamic=True, verbose=False)
    assert out.exists()
    assert out.suffix == ".engine"
    out.unlink()


def test_export_tensorrt_missing_package(tiny_checkpoint, monkeypatch):
    """ImportError with install hint is raised when tensorrt is absent."""
    import sys

    monkeypatch.setitem(sys.modules, "tensorrt", None)
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    with pytest.raises(ImportError, match="uv sync --extra tensorrt"):
        model.export(format="tensorrt", imgsz=640, verbose=False)
