"""Integration tests: ONNX exports embed a nitid_meta that matches the real graph."""

from __future__ import annotations

import numpy as np
import pytest

from dfine import DFINE, __version__
from dfine.export_meta import META_KEY, META_SCHEMA_VERSION, read_onnx_meta

_DTYPES = {
    "tensor(float)": "float32",
    "tensor(int64)": "int64",
}


def _check_minimal_schema(meta: dict) -> None:
    """Minimal local schema (the studio JSON Schema lives on another branch)."""
    required = {
        "schema_version",
        "task",
        "imgsz",
        "input",
        "outputs",
        "names",
        "topk",
        "default_iou",
        "nitid_version",
    }
    assert required <= set(meta) <= required | {"model"}
    assert meta["schema_version"] == META_SCHEMA_VERSION
    assert meta["task"] in {"detect", "segment"}
    assert set(meta["input"]) == {"name", "shape", "dtype", "range", "color", "resize"}
    assert all(k.isdigit() and isinstance(v, str) and v for k, v in meta["names"].items())
    for output in meta["outputs"]:
        assert {"name", "dtype"} <= set(output)


def _session(path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _assert_meta_matches_runtime(path, meta, imgsz, *, batch=1):
    session = _session(path)
    (inp,) = session.get_inputs()
    assert inp.name == meta["input"]["name"]
    assert list(inp.shape) == meta["input"]["shape"] == [batch, 3, imgsz, imgsz]
    assert _DTYPES[inp.type] == meta["input"]["dtype"]
    outs = session.get_outputs()
    assert [o.name for o in outs] == [o["name"] for o in meta["outputs"]]
    assert [_DTYPES[o.type] for o in outs] == [o["dtype"] for o in meta["outputs"]]
    # run once and check real shapes against the meta
    image = np.random.default_rng(0).random((batch, 3, imgsz, imgsz), dtype=np.float32)
    values = session.run(None, {inp.name: image})
    for spec, value in zip(meta["outputs"], values):
        assert str(value.dtype) == spec["dtype"]
        if "stride" in spec:
            assert value.shape[-2:] == (imgsz // spec["stride"], imgsz // spec["stride"])
    return dict(zip([o.name for o in outs], values))


@pytest.mark.parametrize("simplify", [False, True])
def test_detect_meta_matches_graph(tiny_checkpoint, tmp_path, simplify):
    pytest.importorskip("onnxruntime")
    if simplify:
        pytest.importorskip("onnxsim")
    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    out = model.export(
        format="onnx", imgsz=640, simplify=simplify, output=tmp_path / "d.onnx", verbose=False
    )
    meta = read_onnx_meta(out)
    assert meta is not None
    _check_minimal_schema(meta)
    assert meta["task"] == "detect" and meta["imgsz"] == 640
    assert meta["model"] == "nitid1s"
    assert meta["nitid_version"] == __version__
    assert meta["default_iou"] == 0.85
    assert meta["names"] == {str(i): f"class_{i}" for i in range(80)}
    values = _assert_meta_matches_runtime(out, meta, 640)
    assert values["labels"].shape == (1, meta["topk"])
    assert values["boxes"].shape == (1, meta["topk"], 4)
    # boxes are xyxy in input pixels: bounded by the input size (+ tiny slack)
    assert values["boxes"][..., 2].max() <= 640 * 1.5


def test_segment_meta_matches_graph(tiny_segment_checkpoint, tmp_path):
    pytest.importorskip("onnxruntime")
    model = DFINE(tiny_segment_checkpoint, task="segment", device="cpu", verbose=False)
    out = model.export(
        format="onnx", imgsz=640, simplify=False, output=tmp_path / "s.onnx", verbose=False
    )
    meta = read_onnx_meta(out)
    assert meta is not None
    _check_minimal_schema(meta)
    assert meta["task"] == "segment"
    values = _assert_meta_matches_runtime(out, meta, 640)
    assert [o["name"] for o in meta["outputs"]] == ["labels", "boxes", "scores", "masks"]
    assert values["masks"].shape == (1, meta["topk"], 160, 160)


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_postprocess_false_meta_matches_raw_graph(
    tiny_checkpoint, tiny_segment_checkpoint, tmp_path, task
):
    pytest.importorskip("onnxruntime")
    checkpoint = tiny_checkpoint if task == "detect" else tiny_segment_checkpoint
    model = DFINE(checkpoint, task=task, device="cpu", verbose=False)
    out = model.export(
        format="onnx",
        imgsz=640,
        simplify=False,
        postprocess=False,
        output=tmp_path / "raw.onnx",
        verbose=False,
    )
    meta = read_onnx_meta(out)
    assert meta is not None
    _check_minimal_schema(meta)
    expected = ["pred_logits", "pred_boxes"] + (["pred_masks"] if task == "segment" else [])
    assert [o["name"] for o in meta["outputs"]] == expected
    _assert_meta_matches_runtime(out, meta, 640)


def test_two_identical_exports_have_identical_meta(tiny_checkpoint, tmp_path):
    import onnx

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    paths = [
        model.export(
            format="onnx", imgsz=640, simplify=False, output=tmp_path / f"{i}.onnx", verbose=False
        )
        for i in range(2)
    ]
    raw = [
        next(p.value for p in onnx.load(str(path)).metadata_props if p.key == META_KEY)
        for path in paths
    ]
    assert raw[0] == raw[1]
    assert read_onnx_meta(paths[0]) == read_onnx_meta(paths[1])


def test_meta_does_not_change_graph_outputs(tiny_checkpoint, tmp_path):
    """Embedding is additive: stripping metadata_props leaves a loadable, equal graph."""
    pytest.importorskip("onnxruntime")
    import onnx

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    out = model.export(
        format="onnx", imgsz=640, simplify=False, output=tmp_path / "m.onnx", verbose=False
    )
    proto = onnx.load(str(out))
    assert [p.key for p in proto.metadata_props] == [META_KEY]
    onnx.checker.check_model(proto)
    del proto.metadata_props[:]
    stripped = tmp_path / "stripped.onnx"
    onnx.save(proto, str(stripped))
    assert read_onnx_meta(stripped) is None
    image = np.random.default_rng(1).random((1, 3, 640, 640), dtype=np.float32)
    a = _session(out).run(None, {"images": image})
    b = _session(stripped).run(None, {"images": image})
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)


def test_semantic_export_has_no_meta(tiny_semantic_checkpoint, tmp_path):
    model = DFINE(tiny_semantic_checkpoint, task="semantic", device="cpu", verbose=False)
    out = model.export(
        format="onnx", imgsz=64, simplify=False, output=tmp_path / "sem.onnx", verbose=False
    )
    assert read_onnx_meta(out) is None


def test_topk_is_the_effective_row_count_for_small_models(tmp_path):
    """10 queries x 2 classes < 300: the graph returns 20 rows and the meta says so."""
    pytest.importorskip("onnxruntime")
    import torch

    from dfine.nn.build import build_model
    from dfine.nn.configs import make_model_config

    config = make_model_config("dfine_s", task="detect", num_classes=2, image_size=(64, 64))
    config["DFINETransformer"]["num_layers"] = 1
    config["DFINETransformer"]["num_queries"] = 10
    config["DFINETransformer"]["num_denoising"] = 0
    config["HybridEncoder"]["depth_mult"] = 0.1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        net = build_model(config)
    checkpoint = tmp_path / "small.pth"
    torch.save(
        {
            "model": net.state_dict(),
            "config": config,
            "names": {0: "class_0", 1: "class_1"},
            "epoch": -1,
            "metrics": {},
        },
        checkpoint,
    )
    model = DFINE(checkpoint, task="detect", device="cpu", verbose=False)
    out = model.export(
        format="onnx", imgsz=64, simplify=False, output=tmp_path / "small.onnx", verbose=False
    )
    meta = read_onnx_meta(out)
    assert meta is not None
    assert meta["topk"] == 20 and meta["names"] == {"0": "class_0", "1": "class_1"}
    values = _assert_meta_matches_runtime(out, meta, 64)
    assert values["labels"].shape == (1, 20)


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_simplified_export_has_no_stale_value_info(
    tiny_checkpoint, tiny_segment_checkpoint, tmp_path, task
):
    """onnxsim leaves a rank-0 ``value_info`` for a [4, 256] tensor in the segment graph.

    onnxruntime-web 1.31.1 refuses such a file, so the exporter drops ``value_info`` and
    the full ONNX checker must pass. Outputs must match an export that kept it.
    """
    pytest.importorskip("onnxsim")
    ort = pytest.importorskip("onnxruntime")
    import onnx

    import dfine.exporter as exporter_module

    checkpoint = tiny_checkpoint if task == "detect" else tiny_segment_checkpoint
    model = DFINE(checkpoint, task=task, device="cpu", verbose=False)
    fixed = model.export(
        format="onnx", imgsz=640, simplify=True, output=tmp_path / "fixed.onnx", verbose=False
    )
    proto = onnx.load(str(fixed))
    assert len(proto.graph.value_info) == 0
    onnx.checker.check_model(proto, full_check=True)
    assert read_onnx_meta(proto) is not None  # the meta survives

    original = exporter_module.drop_value_info
    exporter_module.drop_value_info = lambda _model: 0
    try:
        legacy = model.export(
            format="onnx", imgsz=640, simplify=True, output=tmp_path / "legacy.onnx", verbose=False
        )
    finally:
        exporter_module.drop_value_info = original
    assert len(onnx.load(str(legacy)).graph.value_info) > 0

    image = np.random.default_rng(0).random((1, 3, 640, 640), dtype=np.float32)
    providers = ["CPUExecutionProvider"]
    new = ort.InferenceSession(str(fixed), providers=providers).run(None, {"images": image})
    old = ort.InferenceSession(str(legacy), providers=providers).run(None, {"images": image})
    assert len(new) == len(old)
    for a, b in zip(new, old):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
