"""Tests for the DETRPose integration contract and audited upstream baseline."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from dfine.pose_contract import (
    COCO17_SCHEMA,
    CROWDPOSE14_SCHEMA,
    DETRPOSE_UPSTREAM_COMMIT,
    POSE_MODEL_NAMES,
    POSE_MODEL_SPECS,
    PoseCheckpointSpec,
    PoseSchema,
    get_pose_checkpoint,
    get_pose_model_spec,
    get_pose_schema,
    normalize_pose_model,
    validate_pose_model_task,
)

_GOLDEN = Path(__file__).parents[1] / "data" / "detrpose_n_zero_input_golden.json"


def test_official_pose_model_catalog_is_complete_and_consistent():
    assert POSE_MODEL_NAMES == (
        "detrpose_n",
        "detrpose_s",
        "detrpose_m",
        "detrpose_l",
        "detrpose_x",
    )
    assert set(POSE_MODEL_SPECS) == set(POSE_MODEL_NAMES)
    assert get_pose_model_spec("DETRPose-N").num_feature_levels == 2
    assert get_pose_model_spec("detrpose_s").num_feature_levels == 3
    assert get_pose_model_spec("detrpose_x").hidden_dim == 384
    assert get_pose_model_spec("detrpose_x").reg_scale == 8.0
    assert all(spec.num_queries == 60 for spec in POSE_MODEL_SPECS.values())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("detrpose_n", "detrpose_n"),
        ("DETRPose-N", "detrpose_n"),
        ("detrposen", "detrpose_n"),
        ("  detrpose_x  ", "detrpose_x"),
    ],
)
def test_pose_model_normalization_is_explicit_and_unambiguous(value, expected):
    assert normalize_pose_model(value) == expected


def test_pose_model_normalization_rejects_size_only_and_wrong_family():
    with pytest.raises(ValueError, match="Unsupported DETRPose model"):
        normalize_pose_model("n")
    with pytest.raises(ValueError, match="Unsupported DETRPose model"):
        normalize_pose_model("dfine_n")
    with pytest.raises(TypeError, match="model must be a string"):
        normalize_pose_model(1)  # type: ignore[arg-type]


def test_pose_model_task_pair_is_pose_only():
    assert validate_pose_model_task("detrpose_s", " pose ") == "detrpose_s"
    with pytest.raises(ValueError, match="task='pose' only"):
        validate_pose_model_task("detrpose_s", "detect")
    with pytest.raises(TypeError, match="task must be a string"):
        validate_pose_model_task("detrpose_s", True)  # type: ignore[arg-type]


def test_official_pose_schemas_have_complete_geometry_metadata():
    assert get_pose_schema("coco17") is COCO17_SCHEMA
    assert get_pose_schema("CrowdPose-14") is CROWDPOSE14_SCHEMA
    assert COCO17_SCHEMA.num_keypoints == 17
    assert CROWDPOSE14_SCHEMA.num_keypoints == 14
    for schema in (COCO17_SCHEMA, CROWDPOSE14_SCHEMA):
        assert len(schema.flip_idx) == schema.num_keypoints
        assert len(schema.oks_sigmas) == schema.num_keypoints
        assert all(schema.flip_idx[schema.flip_idx[i]] == i for i in range(schema.num_keypoints))
        assert all(max(edge) < schema.num_keypoints for edge in schema.skeleton)


def test_pose_schema_validation_rejects_incomplete_metadata():
    with pytest.raises(ValueError, match="flip_idx"):
        PoseSchema(
            name="coco",
            keypoint_names=("left", "right"),
            flip_idx=(0,),
            skeleton=((0, 1),),
            oks_sigmas=(0.1, 0.1),
        )
    with pytest.raises(ValueError, match="self-inverse"):
        PoseSchema(
            name="coco",
            keypoint_names=("a", "b", "c"),
            flip_idx=(1, 2, 0),
            skeleton=((0, 1),),
            oks_sigmas=(0.1, 0.1, 0.1),
        )


def test_pose_contract_values_are_immutable():
    with pytest.raises(FrozenInstanceError):
        COCO17_SCHEMA.annotation_dim = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        POSE_MODEL_SPECS["detrpose_n"] = get_pose_model_spec("detrpose_n")  # type: ignore[index]


def test_official_checkpoint_manifest_uses_dataset_specific_names():
    coco = get_pose_checkpoint("detrpose_n", "coco")
    crowdpose = get_pose_checkpoint("detrpose_x", "crowdpose")

    assert coco.filename == "detrpose_hgnetv2_n.pth"
    assert coco.url.endswith("/detrpose_hgnetv2_n.pth")
    assert coco.sha256 == "802014f3929b67d0ea7de068e66f11fc836310ecaf814dd3845b703d16256fac"
    assert crowdpose.filename == "detrpose_hgnetv2_x_crowdpose.pth"
    assert crowdpose.url.endswith("/detrpose_hgnetv2_x_crowdpose.pth")
    assert crowdpose.sha256 is None


def test_checkpoint_manifest_rejects_non_official_location():
    with pytest.raises(ValueError, match="official DETRPose release"):
        PoseCheckpointSpec(
            model="detrpose_n",
            dataset="coco",
            filename="detrpose_hgnetv2_n.pth",
            url="https://example.com/detrpose_hgnetv2_n.pth",
        )


def test_audited_zero_input_golden_fixture_has_stable_schema():
    fixture = json.loads(_GOLDEN.read_text())

    assert fixture["upstream"]["commit"] == DETRPOSE_UPSTREAM_COMMIT
    assert fixture["model"] == "detrpose_n"
    assert fixture["checkpoint"]["sha256"] == get_pose_checkpoint("detrpose_n").sha256
    assert fixture["input"] == {"kind": "constant", "value": 0.0, "shape": [1, 3, 640, 640]}
    assert fixture["outputs"]["logits_shape"] == [1, 60, 2]
    assert fixture["outputs"]["keypoints_shape"] == [1, 60, 34]
    assert set(fixture["outputs"]["samples"]) == {"0", "1", "30", "59"}
    numeric_values = [
        fixture["outputs"]["logits_sum"],
        fixture["outputs"]["keypoints_sum"],
        *(
            value
            for sample in fixture["outputs"]["samples"].values()
            for values in sample.values()
            for value in values
        ),
    ]
    assert all(math.isfinite(value) for value in numeric_values)
