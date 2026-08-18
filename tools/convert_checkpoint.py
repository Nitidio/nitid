"""
convert_checkpoint.py

Migrates a raw D-FINE .pth (weights only) into the dfine-wrap format
(weights + embedded config + class names) so users only need one file.

Usage:
    python tools/convert_checkpoint.py \
        --weights dfine_l.pth \
        --model   dfine_l \
        --task    detect \
        --names   configs/datasets/coco.yml \
        --output  dfine_l_wrapped.pth
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml


def _load_config(file_path: str | Path) -> dict:
    """
    Load a D-FINE YAML config, recursively resolving ``__include__`` directives.
    Includes inherited YAML files recursively and merges nested mappings.
    """
    file_path = Path(file_path).resolve()
    with open(file_path) as f:
        file_cfg = yaml.safe_load(f) or {}

    merged: dict = {}
    for base_yaml in file_cfg.pop("__include__", []):
        base_path = Path(base_yaml)
        if not base_path.is_absolute():
            base_path = file_path.parent / base_yaml
        _merge(merged, _load_config(base_path))

    _merge(merged, file_cfg)
    return merged


def _merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def convert(
    weights: str,
    config: str | Path | Mapping[str, Any],
    names_file: str,
    output: str,
) -> None:
    print(f"Loading weights from {weights}")
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)

    # Raw D-FINE checkpoints store EMA weights at ckpt["ema"]["module"] when present
    # (same priority as D-FINE's own inference scripts).  Fall back to ckpt["model"].
    if "ema" in ckpt:
        state_dict = ckpt["ema"]["module"]
        print("Using EMA weights (ckpt['ema']['module'])")
    else:
        state_dict = ckpt.get("model", ckpt)
        print("Using model weights (ckpt['model'])")

    cfg = copy.deepcopy(dict(config)) if isinstance(config, Mapping) else _load_config(config)
    from dfine.tasks import normalize_task
    from dfine.utils.checkpoint import CHECKPOINT_FORMAT_VERSION

    task = normalize_task(str(cfg.get("task", "detect")))
    cfg["task"] = task

    with open(names_file) as f:
        names_cfg = yaml.safe_load(f)
    # Support both {names: [a, b, c]} and {names: {0: a, 1: b}} formats
    raw_names = names_cfg.get("names", [])
    if isinstance(raw_names, list):
        names = {i: n for i, n in enumerate(raw_names)}
    else:
        names = {int(k): v for k, v in raw_names.items()}

    out_ckpt = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "task": task,
        "model": state_dict,
        "config": cfg,
        "names": names,
        "epoch": ckpt.get("epoch", 0),
        "metrics": ckpt.get("metrics", {}),
        "training_state": {},
    }
    torch.save(out_ckpt, output)
    print(f"Saved wrapped checkpoint to {output}  ({len(names)} classes)")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert raw D-FINE checkpoint to dfine-wrap format")
    p.add_argument("--weights", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", choices=["dfine_n", "dfine_s", "dfine_m", "dfine_l", "dfine_x"])
    source.add_argument("--config", help="Path to a self-contained YAML config")
    p.add_argument("--task", choices=["detect", "segment"], default="detect")
    p.add_argument("--names", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.model:
        from dfine.nn.configs import make_model_config

        config = make_model_config(args.model, task=args.task)
    else:
        config = args.config
    convert(args.weights, config, args.names, args.output)


if __name__ == "__main__":
    main()
