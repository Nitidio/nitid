"""Task-independent D-FINE backbone, encoder, and decoder composition."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from .backbone import HGNetv2
from .encoder import HybridEncoder


class DFINEModel(nn.Module):
    """Compose the native HGNetv2 backbone, hybrid encoder, and D-FINE decoder."""

    def __init__(
        self,
        backbone: HGNetv2,
        encoder: HybridEncoder,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        images: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, Any]:
        features = self.backbone(images)

        # Nano detection uses stride 16/32 encoder inputs. Its segmentation
        # model additionally exposes the stride-8 backbone feature to the mask head.
        low_level_feature = None
        if len(features) > len(self.encoder.in_channels):
            low_level_feature = features[0]
            features = features[1:]

        encoded = self.encoder(features)
        return self.decoder(encoded, targets, low_level_feat=low_level_feature)

    def deploy(self) -> DFINEModel:
        """Convert supported layers to their inference representation in place."""
        self.eval()
        for module in self.modules():
            if module is not self and hasattr(module, "convert_to_deploy"):
                cast(Any, module).convert_to_deploy()
        return self
