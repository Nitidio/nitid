"""
Unit tests for DETRPose / D-FINE pose loss functions, matcher, and criterion.
"""

from __future__ import annotations

import torch

from dfine.nn.configs import make_pose_config
from dfine.nn.criterion import build_criterion
from dfine.nn.losses.keypoint_loss import OKSLoss, oks_overlaps
from dfine.nn.losses.matcher import HungarianMatcher


def test_oks_loss_identical_keypoints() -> None:
    num_kpts = 17
    # 2 instances, 17 keypoints each
    kpt_preds = torch.tensor(
        [[i * 10.0 for i in range(num_kpts * 2)], [i * 5.0 for i in range(num_kpts * 2)]],
        dtype=torch.float32,
    )
    kpt_gts = kpt_preds.clone()
    valids = torch.ones((2, num_kpts), dtype=torch.float32)
    areas = torch.tensor([10000.0, 5000.0], dtype=torch.float32)

    loss_fn = OKSLoss(linear=True, num_keypoints=17)
    loss = loss_fn(kpt_preds, kpt_gts, valids, areas)

    # Identical keypoints -> OKS overlap is 1.0
    assert torch.allclose(loss, torch.tensor([1.0, 1.0]), atol=1e-4)


def test_oks_overlaps_offset_keypoints() -> None:
    num_kpts = 17
    kpt_gts = torch.zeros((1, num_kpts * 2), dtype=torch.float32)
    kpt_preds = torch.ones((1, num_kpts * 2), dtype=torch.float32) * 5.0
    valids = torch.ones((1, num_kpts), dtype=torch.float32)
    areas = torch.tensor([100.0], dtype=torch.float32)

    from dfine.nn.losses.keypoint_loss import COCO_SIGMAS

    overlaps = oks_overlaps(kpt_preds, kpt_gts, valids, areas, COCO_SIGMAS)

    assert overlaps.shape == (1,)
    assert 0.0 <= overlaps.item() < 1.0


def test_hungarian_matcher_pose() -> None:
    matcher = HungarianMatcher(
        weight_dict={"cost_class": 2.0, "cost_keypoints": 5.0, "cost_oks": 2.0},
        num_body_points=17,
        use_focal_loss=True,
    )

    outputs = {
        "pred_logits": torch.randn(2, 10, 1),
        "pred_keypoints": torch.rand(2, 10, 34),
    }
    targets = [
        {
            "labels": torch.tensor([0], dtype=torch.int64),
            "keypoints": torch.rand(1, 51),  # 17 * 3
            "area": torch.tensor([500.0]),
        },
        {
            "labels": torch.tensor([0, 0], dtype=torch.int64),
            "keypoints": torch.rand(2, 51),
            "area": torch.tensor([400.0, 600.0]),
        },
    ]

    result = matcher(outputs, targets)
    assert "indices" in result
    indices = result["indices"]
    assert len(indices) == 2
    # Batch 0 has 1 target
    assert indices[0][0].shape[0] == 1
    assert indices[0][1].shape[0] == 1
    # Batch 1 has 2 targets
    assert indices[1][0].shape[0] == 2
    assert indices[1][1].shape[0] == 2


def test_pose_criterion_forward() -> None:
    cfg = make_pose_config("detrpose_n")
    criterion = build_criterion(cfg)

    bs = 2
    num_queries = 10
    outputs = {
        "pred_logits": torch.randn(bs, num_queries, 1),
        "pred_keypoints": torch.rand(bs, num_queries, 34),
    }
    targets = [
        {
            "labels": torch.tensor([0], dtype=torch.int64),
            "keypoints": torch.rand(1, 51),
            "area": torch.tensor([500.0]),
        },
        {
            "labels": torch.tensor([0], dtype=torch.int64),
            "keypoints": torch.rand(1, 51),
            "area": torch.tensor([400.0]),
        },
    ]

    losses = criterion(outputs, targets)
    assert isinstance(losses, dict)
    assert "loss_vfl" in losses or "loss_keypoints" in losses
    for k, v in losses.items():
        assert isinstance(v, torch.Tensor)
        assert torch.isfinite(v)
