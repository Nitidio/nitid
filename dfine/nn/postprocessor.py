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
    ) -> (
        list[dict[str, torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        logits = outputs["pred_logits"]
        normalized_boxes = outputs["pred_boxes"]
        boxes = box_convert(normalized_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        boxes = boxes * original_sizes.repeat(1, 2).unsqueeze(1)
        query_indices: torch.Tensor

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
            query_indices = (
                torch.arange(scores.shape[1], device=scores.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(scores.shape[0], -1)
            )
            if scores.shape[1] > self.num_top_queries:
                scores, indices = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = labels.gather(dim=1, index=indices)
                query_indices = query_indices.gather(dim=1, index=indices)
                boxes = boxes.gather(
                    dim=1,
                    index=indices.unsqueeze(-1).expand(-1, -1, boxes.shape[-1]),
                )

        masks = self._process_masks(outputs.get("pred_masks"), query_indices)

        if self.deploy_mode and masks is not None:
            return labels, boxes, scores, masks
        if self.deploy_mode:
            return labels, boxes, scores

        results = []
        for batch_index, (image_labels, image_boxes, image_scores) in enumerate(
            zip(labels, boxes, scores)
        ):
            result = {"labels": image_labels, "boxes": image_boxes, "scores": image_scores}
            if masks is not None:
                result["masks"] = masks[batch_index]
            results.append(result)
        return results

    @staticmethod
    def _process_masks(
        pred_masks: torch.Tensor | None,
        query_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        """Gather the low-resolution masks belonging to selected queries."""
        if pred_masks is None:
            return None
        return pred_masks.gather(
            1,
            query_indices[:, :, None, None].expand(
                -1, -1, pred_masks.shape[-2], pred_masks.shape[-1]
            ),
        )

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
