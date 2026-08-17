"""Integration tests for model export (ONNX, OpenVINO, TorchScript, TensorRT)."""

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


def test_export_segment_onnx_includes_masks(tiny_segment_checkpoint, tmp_path):
    import onnx

    from dfine import DFINE

    model = DFINE(tiny_segment_checkpoint, task="segment", device="cpu", verbose=False)
    output = tmp_path / "segment.onnx"
    out = model.export(
        format="onnx",
        imgsz=640,
        simplify=False,
        output=output,
        verbose=False,
    )

    graph = onnx.load(str(out)).graph
    assert [value.name for value in graph.output] == ["labels", "boxes", "scores", "masks"]


def test_export_semantic_onnx_returns_dense_logits(tiny_semantic_checkpoint, tmp_path):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    from dfine import DFINE

    model = DFINE(tiny_semantic_checkpoint, task="semantic", device="cpu", verbose=False)
    output = tmp_path / "semantic.onnx"
    inputs = torch.zeros(1, 3, 64, 64)
    out = model.export(
        format="onnx",
        imgsz=64,
        simplify=False,
        output=output,
        verbose=False,
    )

    graph = onnx.load(str(out)).graph
    assert [value.name for value in graph.output] == ["semantic_logits"]
    with torch.inference_mode():
        expected = model._model(inputs)["sem_seg_logits"].cpu().numpy()
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"images": inputs.numpy()})[0]
    assert actual.shape == (1, 3, 64, 64)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


def test_semantic_export_rejects_unvalidated_formats(tiny_semantic_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_semantic_checkpoint, task="semantic", device="cpu", verbose=False)
    with pytest.raises(ValueError, match="format='onnx' only"):
        model.export(format="torchscript", imgsz=64, verbose=False)


def test_export_openvino(tiny_checkpoint, tmp_path):
    ov = pytest.importorskip("openvino", reason="openvino not installed")
    import numpy as np

    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    out = model.export(
        format="openvino",
        imgsz=640,
        simplify=False,
        project=str(tmp_path),
        verbose=False,
    )

    assert out.suffix == ".xml"
    assert out.is_file()
    assert out.with_suffix(".bin").is_file()

    compiled = ov.Core().compile_model(out, "CPU")
    results = compiled([np.zeros((1, 3, 640, 640), dtype=np.float32)])
    assert len(results) == 3


def test_export_segment_openvino_includes_masks(tiny_segment_checkpoint, tmp_path):
    ov = pytest.importorskip("openvino", reason="openvino not installed")
    import numpy as np

    from dfine import DFINE

    model = DFINE(tiny_segment_checkpoint, task="segment", device="cpu", verbose=False)
    out = model.export(
        format="openvino",
        imgsz=640,
        simplify=False,
        project=str(tmp_path),
        verbose=False,
    )

    compiled = ov.Core().compile_model(out, "CPU")
    results = compiled([np.zeros((1, 3, 640, 640), dtype=np.float32)])
    assert len(results) == 4


def test_export_openvino_missing_package(tiny_checkpoint, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "openvino", None)
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    with pytest.raises(ImportError, match="uv sync --extra openvino"):
        model.export(format="openvino", imgsz=640, simplify=False, verbose=False)


def test_repeated_export_calls_increment_run_directory(tiny_checkpoint, tmp_path):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    first = model.export(
        format="onnx", imgsz=640, simplify=False, project=str(tmp_path), verbose=False
    )
    second = model.export(
        format="onnx", imgsz=640, simplify=False, project=str(tmp_path), verbose=False
    )

    assert first.parent == tmp_path / "exp"
    assert second.parent == tmp_path / "exp2"
    assert (first.parent / "args.yaml").exists()
    assert (second.parent / "environment.yaml").exists()


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
