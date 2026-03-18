"""
convert_checkpoint.py

Migrates a raw D-FINE .pth (weights only) into the dfine-wrap format
(weights + embedded config + class names) so users only need one file.

Usage:
    python tools/convert_checkpoint.py \
        --weights dfine_l.pth \
        --config  configs/models/dfine_l.yml \
        --names   configs/datasets/coco.yml \
        --output  dfine_l_wrapped.pth
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import yaml


def convert(weights: str, config: str, names_file: str, output: str) -> None:
    print(f"Loading weights from {weights}")
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)

    # Raw D-FINE checkpoints may store weights under "model" or at top level
    state_dict = ckpt.get("model", ckpt)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    with open(names_file) as f:
        names_cfg = yaml.safe_load(f)
    # Support both {names: [a, b, c]} and {names: {0: a, 1: b}} formats
    raw_names = names_cfg.get("names", [])
    if isinstance(raw_names, list):
        names = {i: n for i, n in enumerate(raw_names)}
    else:
        names = {int(k): v for k, v in raw_names.items()}

    out_ckpt = {
        "model":   state_dict,
        "config":  cfg,
        "names":   names,
        "epoch":   ckpt.get("epoch", 0),
        "metrics": ckpt.get("metrics", {}),
    }
    torch.save(out_ckpt, output)
    print(f"Saved wrapped checkpoint to {output}  ({len(names)} classes)")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert raw D-FINE checkpoint to dfine-wrap format")
    p.add_argument("--weights", required=True)
    p.add_argument("--config",  required=True)
    p.add_argument("--names",   required=True)
    p.add_argument("--output",  required=True)
    args = p.parse_args()
    convert(args.weights, args.config, args.names, args.output)


if __name__ == "__main__":
    main()
