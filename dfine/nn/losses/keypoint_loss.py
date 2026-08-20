"""
Object Keypoint Similarity (OKS) loss and overlap calculations for DETRPose / D-FINE.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["OKSLoss", "oks_overlaps"]

COCO_SIGMAS = (
    np.array(
        [
            0.26,
            0.25,
            0.25,
            0.35,
            0.35,
            0.79,
            0.79,
            0.72,
            0.72,
            0.62,
            0.62,
            1.07,
            1.07,
            0.87,
            0.87,
            0.89,
            0.89,
        ],
        dtype=np.float32,
    )
    / 10.0
)

CROWDPOSE_SIGMAS = (
    np.array(
        [
            0.79,
            0.79,
            0.72,
            0.72,
            0.62,
            0.62,
            1.07,
            1.07,
            0.87,
            0.87,
            0.89,
            0.89,
            0.79,
            0.79,
        ],
        dtype=np.float32,
    )
    / 10.0
)


def oks_overlaps(
    kpt_preds: torch.Tensor,
    kpt_gts: torch.Tensor,
    kpt_valids: torch.Tensor,
    kpt_areas: torch.Tensor,
    sigmas: np.ndarray,
) -> torch.Tensor:
    """
    Computes Object Keypoint Similarity (OKS) overlaps between predicted and GT keypoints.

    Args:
        kpt_preds: [N, K * 2] or [N, K, 2] predicted keypoint coordinates.
        kpt_gts: [N, K * 2] or [N, K, 2] target keypoint coordinates.
        kpt_valids: [N, K] or [N, K, 1] target keypoint visibility flags.
        kpt_areas: [N] target object areas.
        sigmas: Per-keypoint standard deviation scaling array.

    Returns:
        oks: [N] OKS similarity scores in range [0, 1].
    """
    sigmas_t = kpt_preds.new_tensor(sigmas)
    variances = (sigmas_t * 2) ** 2

    kpt_preds_res = kpt_preds.reshape(-1, kpt_preds.size(-1) // 2, 2)
    kpt_gts_res = kpt_gts.reshape(-1, kpt_gts.size(-1) // 2, 2)
    valids_res = kpt_valids.reshape(-1, kpt_valids.size(-1))

    squared_distance = (kpt_preds_res[:, :, 0] - kpt_gts_res[:, :, 0]) ** 2 + (
        kpt_preds_res[:, :, 1] - kpt_gts_res[:, :, 1]
    ) ** 2
    squared_distance0 = squared_distance / (
        kpt_areas[:, None].clamp(min=1e-6) * variances[None, :] * 2
    )
    squared_distance1 = torch.exp(-squared_distance0) * valids_res
    oks = squared_distance1.sum(dim=1) / (valids_res.sum(dim=1) + 1e-6)
    return oks


class OKSLoss(nn.Module):
    """
    OKS loss module for multi-person pose estimation.
    """

    def __init__(
        self,
        linear: bool = True,
        num_keypoints: int = 17,
        eps: float = 1e-6,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.eps = eps
        self.loss_weight = loss_weight
        self.num_keypoints = num_keypoints

        if num_keypoints == 17:
            self.sigmas = COCO_SIGMAS
        elif num_keypoints == 14:
            self.sigmas = CROWDPOSE_SIGMAS
        else:
            self.sigmas = np.ones(num_keypoints, dtype=np.float32) * 0.05

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        area: torch.Tensor,
    ) -> torch.Tensor:
        oks = oks_overlaps(pred, target, valid, area, self.sigmas).clamp(min=self.eps)
        if self.linear:
            loss = oks
        else:
            loss = -oks.log()
        return self.loss_weight * loss
