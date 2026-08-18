"""Losses for dense semantic segmentation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemSegCriterion(nn.Module):
    """Cross-entropy, multiclass soft-Dice, and auxiliary semantic loss."""

    def __init__(
        self,
        weight_dict: Mapping[str, float],
        num_classes: int,
        ignore_index: int = 255,
        class_weights: Sequence[float] | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1], got {label_smoothing}")
        required_weights = {"loss_ce", "loss_dice", "loss_aux"}
        missing_weights = required_weights.difference(weight_dict)
        if missing_weights:
            missing = ", ".join(sorted(missing_weights))
            raise ValueError(f"Semantic criterion is missing loss weights: {missing}")
        if class_weights is not None and len(class_weights) != num_classes:
            raise ValueError(
                "class_weights must contain one value per class; "
                f"expected {num_classes}, got {len(class_weights)}"
            )

        self.weight_dict = {name: float(value) for name, value in weight_dict.items()}
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        weights: torch.Tensor | None = (
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        )
        self.class_weights: torch.Tensor | None
        self.register_buffer("class_weights", weights)

    def _dice_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = logits.softmax(dim=1)
        safe_target = torch.where(valid, target, 0)
        one_hot = F.one_hot(safe_target, self.num_classes).permute(0, 3, 1, 2)
        one_hot = one_hot.to(dtype=probabilities.dtype)
        valid_pixels = valid.unsqueeze(1).to(dtype=probabilities.dtype)
        probabilities = probabilities * valid_pixels
        one_hot = one_hot * valid_pixels
        intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
        denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
        return 1.0 - dice.mean()

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        if "sem_seg_logits" not in outputs:
            raise KeyError("Semantic model output is missing 'sem_seg_logits'")
        if not targets or any("sem_mask" not in target for target in targets):
            raise ValueError("Semantic targets must contain one 'sem_mask' tensor per image")

        logits = outputs["sem_seg_logits"].float()
        target = torch.stack([item["sem_mask"] for item in targets]).to(
            device=logits.device,
            dtype=torch.long,
        )
        if logits.ndim != 4 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"sem_seg_logits must have shape [B, num_classes, H, W], got {tuple(logits.shape)}"
            )
        if target.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
            raise ValueError(
                "Semantic targets must match the logits batch and spatial shape; "
                f"got {tuple(target.shape)} and {tuple(logits.shape)}"
            )

        valid = target != self.ignore_index
        if valid.any():
            valid_labels = target[valid]
            if valid_labels.min() < 0 or valid_labels.max() >= self.num_classes:
                raise ValueError(
                    f"Semantic targets must use class IDs in [0, {self.num_classes - 1}] "
                    f"or ignore_index={self.ignore_index}"
                )
            losses = {
                "loss_ce": F.cross_entropy(
                    logits,
                    target,
                    weight=self.class_weights,
                    ignore_index=self.ignore_index,
                    label_smoothing=self.label_smoothing,
                ),
                "loss_dice": self._dice_loss(logits, target, valid),
            }
            auxiliary = outputs.get("sem_seg_logits_aux")
            if auxiliary is not None:
                losses["loss_aux"] = F.cross_entropy(
                    auxiliary.float(),
                    target,
                    ignore_index=self.ignore_index,
                )
        else:
            zero = logits.sum() * 0.0
            losses = {"loss_ce": zero, "loss_dice": zero}
            auxiliary = outputs.get("sem_seg_logits_aux")
            if auxiliary is not None:
                losses["loss_aux"] = auxiliary.float().sum() * 0.0

        return {name: loss * self.weight_dict[name] for name, loss in losses.items()}
