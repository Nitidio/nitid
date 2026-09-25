"""Integration tests for model export (ONNX, OpenVINO, TorchScript, TensorRT)."""

import pytest


def _onnxruntime_outputs(path, images):
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {"images": images.numpy()})


def _torch_deploy_outputs(model, images):
    import torch

    from dfine.exporter import DeployModel
    from dfine.nn.build import build_postprocessor

    postprocessor = build_postprocessor(model._cfg)
    postprocessor.deploy()
    deployed_model = model._get_deployed_model()
    wrapped = DeployModel(
        deployed_model,
        postprocessor,
        semantic=str(model._cfg.get("task", "detect")).lower() == "semantic",
    ).eval()
    with torch.inference_mode():
        outputs = wrapped(images)
    return outputs if isinstance(outputs, tuple) else (outputs,)


def _sort_instances(*arrays):
    """
    Put instances in a canonical order so two runtimes can be compared row by row.

    Top-k returns the same instances in whatever order each runtime produced
    them, so the rows have to be realigned before they mean anything. Ordering
    on the box alone is not enough: the rows that share a query — same box,
    different class — stay in emission order, which leaves the label, score and
    mask rows compared against each other misaligned. Sorting on the label
    first makes the key a total order, and the box coordinates are rounded so
    that the ~1e-5 differences between runtimes cannot reorder the rows they
    are meant to be matching.
    """
    import numpy as np

    labels, boxes = arrays[0], arrays[1]
    order = np.lexsort(
        (
            boxes[0, :, 3].round(3),
            boxes[0, :, 2].round(3),
            boxes[0, :, 1].round(3),
            boxes[0, :, 0].round(3),
            labels[0],
        )
    )
    return tuple(array[:, order] for array in arrays)


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


def test_export_segment_onnxruntime_matches_torch_deploy(tiny_segment_checkpoint, tmp_path):
    import numpy as np
    import torch

    from dfine import DFINE

    model = DFINE(tiny_segment_checkpoint, task="segment", device="cpu", verbose=False)
    output = tmp_path / "segment_parity.onnx"
    inputs = torch.rand(1, 3, 640, 640, generator=torch.Generator().manual_seed(7))
    out = model.export(
        format="onnx",
        imgsz=640,
        simplify=False,
        output=output,
        verbose=False,
    )

    expected = _torch_deploy_outputs(model, inputs)
    actual = _onnxruntime_outputs(out, inputs)

    assert len(actual) == 4
    actual = _sort_instances(*actual)
    expected_np = tuple(value.detach().cpu().numpy() for value in expected)
    expected_np = _sort_instances(*expected_np)

    # Guard the precondition that makes this comparison meaningful. If the
    # fixture's random weights ever collapse the decoder's queries onto one
    # another, every top-k score ties, the two runtimes are free to keep
    # different queries, and this test either fails at random or passes while
    # comparing 300 copies of a single box. tests/conftest.py seeds the draw to
    # keep that from happening; fail here, clearly, if that ever stops holding.
    distinct_boxes = len(np.unique(expected_np[1][0].round(3), axis=0))
    assert distinct_boxes > 1, (
        f"the segmentation fixture produced {distinct_boxes} distinct box(es) across "
        "300 instances: its decoder queries have collapsed, so top-k selection is "
        "tied and export parity cannot be compared. Check _FIXTURE_SEED in "
        "tests/conftest.py."
    )
    assert sorted(actual[0].reshape(-1).tolist()) == sorted(expected_np[0].reshape(-1).tolist())
    for actual_value, expected_value in zip(actual[1:], expected_np[1:]):
        np.testing.assert_allclose(
            actual_value,
            expected_value,
            rtol=1e-3,
            atol=1e-4,
        )


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
    with pytest.raises(ValueError, match="format='onnx' or 'openvino' only"):
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


def test_export_semantic_openvino_returns_dense_logits(tiny_semantic_checkpoint, tmp_path):
    ov = pytest.importorskip("openvino", reason="openvino not installed")
    import numpy as np

    from dfine import DFINE

    model = DFINE(tiny_semantic_checkpoint, task="semantic", device="cpu", verbose=False)
    out = model.export(
        format="openvino",
        imgsz=64,
        simplify=False,
        project=str(tmp_path),
        verbose=False,
    )

    compiled = ov.Core().compile_model(out, "CPU")
    results = compiled([np.zeros((1, 3, 64, 64), dtype=np.float32)])
    values = list(results.values())
    assert len(values) == 1
    assert values[0].shape == (1, 3, 64, 64)
    assert np.isfinite(values[0]).all()


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
