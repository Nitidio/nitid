# Adapted from RiO-DETR / RT-DETRv2-OBB (Apache-2.0).
# Source: https://github.com/RicePasteM/RiO-DETR
# Modified for native integration into nitid in 2026.
# Upstream copyright notices are retained below; see THIRD_PARTY_NOTICES.md.
"""Native RiO-DETR / RT-DETRv2-OBB components adapted for nitid."""

from dfine.nn.rio.configs import (
    DOTA_OBB_NAMES,
    RIO_OBB_MODEL_SIZES,
    get_rio_obb_config,
    make_rio_obb_config,
)
from dfine.nn.rio.model import RioOBBModel
from dfine.nn.rio.obb_matcher import HungarianMatcherOBB
from dfine.nn.rio.postprocessor import PostProcessorOBB
from dfine.nn.rio.rtdetrv2_obb_decoder import RTDETRTransformerv2OBB
from dfine.nn.rio.rtv4_obb_criterion import RTv4OBBCriterion

__all__ = [
    "HungarianMatcherOBB",
    "DOTA_OBB_NAMES",
    "PostProcessorOBB",
    "RIO_OBB_MODEL_SIZES",
    "RTDETRTransformerv2OBB",
    "RTv4OBBCriterion",
    "RioOBBModel",
    "get_rio_obb_config",
    "make_rio_obb_config",
]
