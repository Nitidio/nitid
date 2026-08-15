"""Native D-FINE detection postprocessor.

Derived from D-FINE/RT-DETR and modified for integration into nitid in 2026.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert


class DFINEPostProcessor(nn.Module):
    """Convert raw D-FINE query outputs into per-image pixel-space detections."""

    def __init__(
        self,
        *,
        num_classes: int,
        use_focal_loss: bool = True,
        num_top_queries: int = 300,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if num_top_queries < 1:
            raise ValueError(f"num_top_queries must be positive, got {num_top_queries}")
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.deploy_mode = False

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        original_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = outputs["pred_logits"]
        normalized_boxes = outputs["pred_boxes"]
        boxes = box_convert(normalized_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        boxes = boxes * original_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            probabilities = torch.sigmoid(logits)
            scores, indices = torch.topk(probabilities.flatten(1), self.num_top_queries, dim=-1)
            labels = indices % self.num_classes
            query_indices = indices // self.num_classes
            boxes = boxes.gather(
                dim=1,
                index=query_indices.unsqueeze(-1).expand(-1, -1, boxes.shape[-1]),
            )
        else:
            probabilities = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = probabilities.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, indices = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = labels.gather(dim=1, index=indices)
                boxes = boxes.gather(
                    dim=1,
                    index=indices.unsqueeze(-1).expand(-1, -1, boxes.shape[-1]),
                )

        if self.deploy_mode:
            return labels, boxes, scores

        return [
            {"labels": image_labels, "boxes": image_boxes, "scores": image_scores}
            for image_labels, image_boxes, image_scores in zip(labels, boxes, scores)
        ]

    def deploy(self) -> DFINEPostProcessor:
        """Switch to the tuple output contract used by model export."""
        self.eval()
        self.deploy_mode = True
        return self

    def extra_repr(self) -> str:
        values: dict[str, Any] = {
            "num_classes": self.num_classes,
            "use_focal_loss": self.use_focal_loss,
            "num_top_queries": self.num_top_queries,
        }
        return ", ".join(f"{key}={value}" for key, value in values.items())
