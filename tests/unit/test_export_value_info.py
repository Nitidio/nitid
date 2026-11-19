"""Unit tests for ``dfine.exporter.drop_value_info`` (hand-built ONNX graphs)."""

from __future__ import annotations

import pytest

onnx = pytest.importorskip("onnx")

from onnx import TensorProto, helper  # noqa: E402

from dfine.exporter import drop_value_info  # noqa: E402


def _model_with_stale_value_info() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 256])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 256])
    # Rank 0 declared for a [4, 256] tensor: the inconsistency onnxsim leaves behind.
    stale = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [])
    then_out = helper.make_tensor_value_info("t", TensorProto.FLOAT, [4, 256])
    then_graph = helper.make_graph(
        [helper.make_node("Identity", ["mid"], ["t"])],
        "then",
        [],
        [then_out],
        value_info=[helper.make_tensor_value_info("inner", TensorProto.FLOAT, [])],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Neg", ["mid"], ["e"])],
        "else",
        [],
        [helper.make_tensor_value_info("e", TensorProto.FLOAT, [4, 256])],
    )
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["x"], ["mid"]),
            helper.make_node("If", ["cond"], ["y"], then_branch=then_graph, else_branch=else_graph),
        ],
        "g",
        [x, cond],
        [y],
        value_info=[stale],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_removes_value_info_in_main_graph_and_subgraphs():
    model = _model_with_stale_value_info()
    assert drop_value_info(model) == 2
    assert len(model.graph.value_info) == 0
    then_branch = model.graph.node[1].attribute[0].g
    assert len(then_branch.value_info) == 0


def test_keeps_io_nodes_and_metadata():
    model = _model_with_stale_value_info()
    entry = model.metadata_props.add()
    entry.key, entry.value = "k", "v"
    drop_value_info(model)
    assert [i.name for i in model.graph.input] == ["x", "cond"]
    assert [o.name for o in model.graph.output] == ["y"]
    assert [n.op_type for n in model.graph.node] == ["Relu", "If"]
    assert {e.key: e.value for e in model.metadata_props} == {"k": "v"}
    onnx.checker.check_model(model, full_check=True)


def test_is_idempotent_and_zero_when_nothing_to_remove():
    model = _model_with_stale_value_info()
    drop_value_info(model)
    assert drop_value_info(model) == 0
