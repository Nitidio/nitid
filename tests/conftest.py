"""
Session-scoped fixture that builds a tiny D-FINE model with random weights
and saves it as a wrapped nitid checkpoint.  No pretrained download needed.
"""
from pathlib import Path
import pytest
import torch

_DFINE_CONFIGS = Path(__file__).parents[1] / "extern" / "dfine" / "configs"


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """
    Returns the path to a small wrapped .pth checkpoint built from the S
    config with decoder/encoder overrides to keep the file fast to build.
    The backbone (HGNetv2 B0) uses random weights (pretrained=False).
    """
    from dfine.nn.build import _ensure_dfine_on_path, build_model
    from dfine.utils.checkpoint import save_checkpoint

    _ensure_dfine_on_path()
    from src.core.yaml_utils import load_config

    cfg = load_config(str(_DFINE_CONFIGS / "dfine" / "dfine_hgnetv2_s_coco.yml"))

    # Shrink the decoder/encoder so the checkpoint builds quickly
    cfg["DFINETransformer"]["num_layers"] = 1
    cfg["DFINETransformer"]["num_queries"] = 10
    cfg["DFINETransformer"]["num_denoising"] = 0
    cfg["HybridEncoder"]["depth_mult"] = 0.1

    model = build_model(cfg)
    model.eval()

    names = {i: f"class_{i}" for i in range(80)}

    ckpt_dir = tmp_path_factory.mktemp("checkpoints")
    ckpt_path = ckpt_dir / "tiny_dfine.pth"
    save_checkpoint(str(ckpt_path), model, cfg, names)
    return str(ckpt_path)
