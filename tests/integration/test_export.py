"""Integration tests for model export (ONNX, TorchScript)."""
import pytest


def test_export_onnx(tiny_checkpoint, tmp_path):
    from dfine import DFINE
    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    # imgsz must match the model's eval_spatial_size (640) — anchors are pre-computed
    out = model.export(format="onnx", imgsz=640, simplify=False, verbose=False)
    assert out.exists()
    assert out.suffix == ".onnx"
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
