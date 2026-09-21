# Adapted from RiO-DETR / RT-DETRv2-OBB (Apache-2.0).
# Source: https://github.com/RicePasteM/RiO-DETR
# Modified for native integration into nitid in 2026.
# Upstream copyright notices are retained below; see THIRD_PARTY_NOTICES.md.
"""RiO-DETR OBB backbone/encoder/decoder composition."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dfine.nn.architecture.backbone import HGNetv2
from dfine.nn.architecture.encoder import HybridEncoder


class RioOBBModel(nn.Module):
    """Compose HGNetv2, HybridEncoder, and RT-DETRv2-OBB decoder."""

    def __init__(self, backbone: HGNetv2, encoder: HybridEncoder, decoder: nn.Module) -> None:
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
        encoded = self.encoder(features)
        return self.decoder(encoded, targets)

    def deploy(self) -> "RioOBBModel":
        """Convert supported backbone/encoder layers to deploy mode in place."""
        self.eval()
        for module in self.modules():
            deploy = getattr(module, "convert_to_deploy", None)
            if module is not self and callable(deploy):
                deploy()
        return self
