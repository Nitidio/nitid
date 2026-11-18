"""Unit tests for the ONNX export meta (dfine.export_meta)."""

from __future__ import annotations

import json

import pytest

from dfine.export_meta import (
    META_KEY,
    META_SCHEMA_VERSION,
    build_onnx_meta,
    embed_onnx_meta,
    encode_onnx_meta,
    read_onnx_meta,
)

NAMES = {0: "person", 1: "car"}


def _build(**overrides):
    kwargs = {
        "task": "detect",
        "imgsz": 640,
        "names": NAMES,
        "topk": 300,
        "nitid_version": "0.9.6",
        "model": "nitid1s",
    }
    kwargs.update(overrides)
    return build_onnx_meta(**kwargs)


def test_constants():
    assert META_KEY == "nitid_meta"
    assert META_SCHEMA_VERSION == 1


def test_detect_shape_is_canonical():
    meta = _build()
    assert meta == {
        "schema_version": 1,
        "task": "detect",
        "model": "nitid1s",
        "imgsz": 640,
        "input": {
            "name": "images",
            "shape": [1, 3, 640, 640],
            "dtype": "float32",
            "range": "0-1",
            "color": "RGB",
            "resize": "squash",
        },
        "outputs": [
            {"name": "labels", "dtype": "int64"},
            {"name": "boxes", "dtype": "float32", "format": "xyxy_input_px"},
            {"name": "scores", "dtype": "float32"},
        ],
        "names": {"0": "person", "1": "car"},
        "topk": 300,
        "default_iou": 0.85,
        "nitid_version": "0.9.6",
    }


def test_segment_adds_masks_with_stride():
    meta = _build(task="segment")
    assert [o["name"] for o in meta["outputs"]] == ["labels", "boxes", "scores", "masks"]
    assert meta["outputs"][-1] == {"name": "masks", "dtype": "float32", "stride": 4}


def test_postprocess_false_reflects_raw_outputs():
    detect = _build(postprocess=False)
    assert [o["name"] for o in detect["outputs"]] == ["pred_logits", "pred_boxes"]
    assert detect["outputs"][1]["format"] == "cxcywh_norm"
    segment = _build(task="segment", postprocess=False)
    assert [o["name"] for o in segment["outputs"]] == ["pred_logits", "pred_boxes", "pred_masks"]


def test_model_omitted_when_unknown():
    assert "model" not in _build(model=None)
    with pytest.raises(ValueError, match="model"):
        _build(model="")


def test_names_keys_are_strings_and_accept_digit_strings():
    meta = _build(names={0: "a", "1": "b", 10: "c"})
    assert meta["names"] == {"0": "a", "1": "b", "10": "c"}


def test_batch_and_imgsz_in_input_shape():
    meta = _build(imgsz=320, batch=2)
    assert meta["imgsz"] == 320
    assert meta["input"]["shape"] == [2, 3, 320, 320]


def test_determinism_byte_for_byte():
    a = encode_onnx_meta(_build(names={1: "b", 0: "a"}))
    b = encode_onnx_meta(_build(names={0: "a", 1: "b"}))
    assert a == b
    assert a.encode("utf-8") == b.encode("utf-8")
    assert " " not in a.replace("person", "")  # compact separators
    assert json.loads(a) == _build(names={0: "a", 1: "b"})
    keys = list(json.loads(a))
    assert keys == sorted(keys)


def test_non_ascii_names_roundtrip_utf8():
    text = encode_onnx_meta(_build(names={0: "señal", 1: "車"}))
    assert "señal" in text and "車" in text
    assert json.loads(text)["names"]["1"] == "車"


@pytest.mark.parametrize(
    "overrides",
    [
        {"task": "semantic"},
        {"task": "bogus"},
        {"imgsz": 0},
        {"imgsz": 64.0},
        {"imgsz": True},
        {"batch": 0},
        {"topk": 0},
        {"default_iou": 1.5},
        {"default_iou": -0.1},
        {"default_iou": "0.5"},
        {"nitid_version": ""},
        {"names": {}},
        {"names": None},
        {"names": {-1: "x"}},
        {"names": {"a": "x"}},
        {"names": {"01": "x"}},
        {"names": {True: "x"}},
        {"names": {0: ""}},
        {"names": {0: 5}},
        {"names": {0: "a", "0": "b"}},
        {"task": "segment", "imgsz": 62},
    ],
)
def test_invalid_inputs_raise(overrides):
    with pytest.raises(ValueError):
        _build(**overrides)


def test_embed_and_read_roundtrip_without_onnx_graph_changes():
    from onnx import TensorProto, helper

    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    proto = helper.make_model(graph)
    other = proto.metadata_props.add()
    other.key, other.value = "keep", "me"
    before_graph = proto.graph.SerializeToString()

    assert read_onnx_meta(proto) is None
    meta = _build()
    embed_onnx_meta(proto, meta)
    embed_onnx_meta(proto, meta)  # idempotent: replaces, does not duplicate
    keys = [p.key for p in proto.metadata_props]
    assert keys.count(META_KEY) == 1 and "keep" in keys
    assert proto.graph.SerializeToString() == before_graph
    assert read_onnx_meta(proto) == meta


def test_read_from_path_and_errors(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    proto = helper.make_model(graph)
    path = tmp_path / "m.onnx"
    onnx.save(proto, str(path))
    assert read_onnx_meta(path) is None
    assert read_onnx_meta(str(path)) is None

    entry = proto.metadata_props.add()
    entry.key, entry.value = META_KEY, "not json"
    with pytest.raises(ValueError, match="valid JSON"):
        read_onnx_meta(proto)
    entry.value = "[1]"
    with pytest.raises(ValueError, match="JSON object"):
        read_onnx_meta(proto)
