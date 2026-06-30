"""
build_model() — constructs a D-FINE nn.Module from a config dict.

Requires extern/dfine to be present as a git submodule:
    git submodule add https://github.com/Peterande/D-FINE extern/dfine
    git submodule update --init

The D-FINE repo root is added to sys.path so its src/ package is importable.
Importing src triggers all @register() decorators that populate GLOBAL_CONFIG.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

# extern/dfine repo root — one level up from this file's package root
_DFINE_ROOT = Path(__file__).parents[2] / "extern" / "dfine"


def _ensure_dfine_on_path() -> None:
    """
    Add extern/dfine to sys.path and pre-register stub modules to prevent
    src/__init__.py from eagerly importing training-only dependencies
    (faster_coco_eval, calflops, loguru) before we can control the import order.

    Must be called before any `import src.*` statement.
    """
    import types

    if not _DFINE_ROOT.exists():
        raise RuntimeError(
            "extern/dfine not found. Run:\n"
            "  git submodule update --init"
        )

    root = str(_DFINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    # Pre-register stubs so that when sub-packages do relative imports like
    # `from ..data import DataLoader` they get our stubs, not the real
    # __init__.py which pulls in coco_dataset → faster_coco_eval.

    if "src" not in sys.modules:
        stub = types.ModuleType("src")
        stub.__path__ = [str(_DFINE_ROOT / "src")]
        stub.__package__ = "src"
        sys.modules["src"] = stub

    # src.data: only expose DataLoader (all dist_utils needs from it)
    if "src.data" not in sys.modules:
        from torch.utils.data import DataLoader as _DL
        data_stub = types.ModuleType("src.data")
        data_stub.DataLoader = _DL
        sys.modules["src.data"] = data_stub

    # src.misc: namespace stub so sub-modules (box_ops, dist_utils) import
    # as real .py files without running __init__ (calflops, loguru)
    if "src.misc" not in sys.modules:
        misc_stub = types.ModuleType("src.misc")
        misc_stub.__path__ = [str(_DFINE_ROOT / "src" / "misc")]
        misc_stub.__package__ = "src.misc"
        sys.modules["src.misc"] = misc_stub


def _register_dfine_components() -> None:
    """Trigger all @register() decorators so GLOBAL_CONFIG is populated."""
    import src.nn  # noqa: F401 — backbone, encoder, decoder, postprocessor
    import src.optim  # noqa: F401 — EMA, optimizer wrappers
    import src.zoo  # noqa: F401 — DFINE, DFINECriterion, DFINETransformer …


def build_postprocessor(cfg: dict) -> nn.Module:
    """
    Instantiate DFINEPostProcessor from a config dict.

    The postprocessor converts raw model outputs (pred_logits, pred_boxes)
    to per-image lists of {labels, boxes, scores} with boxes scaled to the
    original image size.  It must NOT be put into deploy mode — that changes
    the output format to a tuple (used only for ONNX export).
    """
    _ensure_dfine_on_path()
    _register_dfine_components()
    from src.core.workspace import create
    from src.core.yaml_utils import merge_config

    cfg = dict(cfg)
    # Class names come from the checkpoint's names dict, not COCO ID remapping.
    # Force this off to avoid a src.data.dataset import we've stubbed out.
    cfg["remap_mscoco_category"] = False
    global_cfg = merge_config(cfg, inplace=False, overwrite=False)
    postprocessor = create(cfg["postprocessor"], global_cfg)
    postprocessor.eval()
    return postprocessor


def build_model(cfg: dict) -> nn.Module:
    """
    Instantiate the D-FINE backbone+encoder+decoder from a config dict.

    cfg is the dict embedded in a nitid checkpoint (originally loaded from
    one of the D-FINE YAML configs via load_config()).

    The backbone's pretrained flag is forced to False because weights are
    loaded from the checkpoint state_dict, not downloaded.
    """
    _ensure_dfine_on_path()

    _register_dfine_components()
    from src.core.workspace import create
    from src.core.yaml_utils import merge_config

    # Shallow-copy so we don't mutate the checkpoint's cfg dict
    cfg = dict(cfg)

    # Disable pretrained backbone download — weights come from our .pth
    if "HGNetv2" in cfg:
        cfg["HGNetv2"] = {**cfg["HGNetv2"], "pretrained": False}

    global_cfg = merge_config(cfg, inplace=False, overwrite=False)
    return create(cfg["model"], global_cfg)
