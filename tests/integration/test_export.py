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
    import numpy as np

    boxes = arrays[1]
    order = np.lexsort((boxes[0, :, 3], boxes[0, :, 2], boxes[0, :, 1], boxes[0, :, 0]))
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


def test_export_pose_onnx_returns_keypoints(tiny_pose_checkpoint, tmp_path):
    import onnx

    from dfine import DFINE

    model = DFINE(tiny_pose_checkpoint, task="pose", device="cpu", verbose=False)
    output = tmp_path / "pose.onnx"
    out = model.export(
        format="onnx",
        imgsz=640,
        simplify=False,
        output=output,
        verbose=False,
    )

    graph = onnx.load(str(out)).graph
    assert [value.name for value in graph.output] == ["labels", "boxes", "scores", "keypoints"]


def test_export_pose_onnxruntime_outputs_keypoint_contract(tiny_pose_checkpoint, tmp_path):
    import torch

    from dfine import DFINE

    model = DFINE(tiny_pose_checkpoint, task="pose", device="cpu", verbose=False)
    output = tmp_path / "pose_parity.onnx"
    inputs = torch.rand(1, 3, 640, 640, generator=torch.Generator().manual_seed(11))
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
    assert actual[0].shape == (1, 10)
    assert actual[1].shape == (1, 10, 4)
    assert actual[2].shape == (1, 10)
    assert actual[3].shape == (1, 10, 17, 2)
    assert set(actual[0].reshape(-1).tolist()) <= {0}
    assert all(torch.isfinite(torch.from_numpy(value)).all() for value in actual)
    assert actual[0].tolist() == expected[0].detach().cpu().numpy().tolist()

    import numpy as np

    np.testing.assert_allclose(
        actual[2],
        expected[2].detach().cpu().numpy(),
        rtol=1e-2,
        atol=1e-2,
    )


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


def test_export_pose_openvino_returns_keypoints(tiny_pose_checkpoint, tmp_path):
    ov = pytest.importorskip("openvino", reason="openvino not installed")
    import numpy as np

    from dfine import DFINE

    model = DFINE(tiny_pose_checkpoint, task="pose", device="cpu", verbose=False)
    out = model.export(
        format="openvino",
        imgsz=640,
        simplify=False,
        project=str(tmp_path),
        verbose=False,
    )

    compiled = ov.Core().compile_model(out, "CPU")
    results = compiled([np.zeros((1, 3, 640, 640), dtype=np.float32)])
    values = list(results.values())
    assert len(values) == 4
    assert values[0].shape == (1, 10)
    assert values[1].shape == (1, 10, 4)
    assert values[2].shape == (1, 10)
    assert values[3].shape == (1, 10, 17, 2)
    assert set(values[0].reshape(-1).tolist()) <= {0}
    assert all(np.isfinite(value).all() for value in values)


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
