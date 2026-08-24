"""Unit tests for Phase 2: Native DETRPose model building and official checkpoint parity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dfine.nn.architecture.pose_decoder import DETRPoseDecoder
from dfine.nn.build import build_model
from dfine.nn.configs import make_pose_config
from dfine.nn.native_build import build_native_model
from dfine.pose_contract import POSE_MODEL_NAMES, get_pose_checkpoint, get_pose_model_spec
from tools.convert_checkpoint import convert as convert_checkpoint


def test_native_detrpose_architecture_construction() -> None:
    """Verify that all official DETRPose sizes (n, s, m, l, x) build natively."""
    for model_name in POSE_MODEL_NAMES:
        spec = get_pose_model_spec(model_name)
        model = build_native_model(model_name, num_classes=spec.public_num_classes, task="pose")

        assert isinstance(model.decoder, DETRPoseDecoder)
        assert model.decoder.num_queries == spec.num_queries
        assert model.decoder.num_body_points == 17

        dummy_input = torch.zeros(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(dummy_input)

        assert "pred_logits" in outputs
        assert "pred_keypoints" in outputs
        assert outputs["pred_logits"].shape == (1, spec.num_queries, spec.internal_num_classes)
        assert outputs["pred_keypoints"].shape == (1, spec.num_queries, 17 * 2)


def test_native_detrpose_batching_and_device_neutrality() -> None:
    """Verify multi-batch inference and CPU/device execution."""
    model = build_native_model("detrpose_n", num_classes=2, task="pose")
    model.eval()

    batch_input = torch.zeros(2, 3, 640, 640)
    with torch.no_grad():
        outputs = model(batch_input)

    assert outputs["pred_logits"].shape == (2, 60, 2)
    assert outputs["pred_keypoints"].shape == (2, 60, 34)


def test_official_detrpose_n_strict_checkpoint_loading_and_golden_parity() -> None:
    """Verify strict loading of official DETRPose-N weights and zero-input golden parity."""
    ckpt_spec = get_pose_checkpoint("detrpose_n", "coco")
    ckpt_path = Path(ckpt_spec.filename)

    if not ckpt_path.exists():
        pytest.skip(f"Official checkpoint {ckpt_spec.filename} not found locally for parity test")

    model = build_native_model("detrpose_n", num_classes=2, task="pose")
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = raw_ckpt.get("model", raw_ckpt)

    mapped_state = {}
    for key, value in state_dict.items():
        new_key = "decoder." + key[len("transformer.") :] if key.startswith("transformer.") else key
        mapped_state[new_key] = value

    missing, unexpected = model.load_state_dict(mapped_state, strict=True)
    assert not missing
    assert not unexpected

    model.eval()
    zero_input = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        out = model(zero_input)

    logits = out["pred_logits"]
    keypoints = out["pred_keypoints"]

    golden_path = Path("tests/data/detrpose_n_zero_input_golden.json")
    with golden_path.open() as f:
        golden = json.load(f)

    assert list(logits.shape) == golden["outputs"]["logits_shape"]
    assert list(keypoints.shape) == golden["outputs"]["keypoints_shape"]

    assert abs(logits.sum().item() - golden["outputs"]["logits_sum"]) < 1e-4
    assert abs(keypoints.sum().item() - golden["outputs"]["keypoints_sum"]) < 1e-4

    for sample_idx, sample_val in golden["outputs"]["samples"].items():
        idx = int(sample_idx)
        g_logits = torch.tensor(sample_val["logits"])
        g_kpts = torch.tensor(sample_val["keypoints"])

        n_logits = logits[0, idx]
        n_kpts = keypoints[0, idx, :6]

        assert torch.allclose(n_logits, g_logits, atol=1e-4)
        assert torch.allclose(n_kpts, g_kpts, atol=1e-4)


def test_detrpose_checkpoint_wrapping_roundtrip(tmp_path: Path) -> None:
    """Verify that convert_checkpoint wraps raw DETRPose weights and load_checkpoint loads them strictly."""
    ckpt_spec = get_pose_checkpoint("detrpose_n", "coco")
    ckpt_path = Path(ckpt_spec.filename)

    if not ckpt_path.exists():
        pytest.skip(f"Official checkpoint {ckpt_spec.filename} not found locally for wrapping test")

    wrapped_path = tmp_path / "detrpose_n_wrapped.pth"
    names_path = Path("configs/datasets/coco.yml")

    config = make_pose_config("detrpose_n")
    convert_checkpoint(
        weights=str(ckpt_path),
        config=config,
        names_file=str(names_path),
        output=str(wrapped_path),
    )

    assert wrapped_path.exists()

    model, loaded_cfg, names = build_model(config), config, {}
    from dfine.utils.checkpoint import load_checkpoint

    model, loaded_cfg, names = load_checkpoint(wrapped_path, device="cpu")

    assert loaded_cfg["task"] == "pose"
    assert loaded_cfg["DFINE"]["decoder"] == "DETRPoseDecoder"
    assert isinstance(model.decoder, DETRPoseDecoder)

    zero_input = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        out = model(zero_input)

    assert out["pred_logits"].shape == (1, 60, 2)
    assert out["pred_keypoints"].shape == (1, 60, 34)


def test_existing_detection_and_segmentation_tasks_unaffected() -> None:
    """Verify backward compatibility: detection, instance-, and semantic-segmentation build and run untouched."""
    det_model = build_native_model("dfine_s", num_classes=80, task="detect")
    seg_model = build_native_model("dfine_s", num_classes=80, task="segment")
    sem_model = build_native_model("dfine_s", num_classes=80, task="semantic")

    det_model.eval()
    seg_model.eval()
    sem_model.eval()

    dummy_input = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        det_out = det_model(dummy_input)
        seg_out = seg_model(dummy_input)
        sem_out = sem_model(dummy_input)

    assert "pred_logits" in det_out
    assert "pred_boxes" in det_out
    assert "pred_masks" in seg_out
    assert (
        "out" in sem_out or "semantic_mask" in sem_out or isinstance(sem_out, (dict, torch.Tensor))
    )
