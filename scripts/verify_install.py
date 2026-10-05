"""Exercise detection and segmentation through an installed nitid package."""

from __future__ import annotations

import tempfile
from importlib.metadata import version
from importlib.resources import as_file
from pathlib import Path

import torch
import yaml

import dfine
from dfine import DFINE, __version__
from dfine.data import COCO_NAMES
from dfine.nn.build import build_model
from dfine.nn.configs import make_model_config


def _checkpoint(root: Path, task: str) -> Path:
    config = make_model_config(
        "dfine_s",
        task=task,
        num_classes=2,
        image_size=(64, 64),
    )
    config["DFINETransformer"]["num_layers"] = 1
    config["DFINETransformer"]["num_queries"] = 10
    config["DFINETransformer"]["num_denoising"] = 0
    config["HybridEncoder"]["depth_mult"] = 0.1
    model = build_model(config)
    path = root / f"{task}.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "names": {0: "class_0", 1: "class_1"},
            "epoch": -1,
            "metrics": {},
        },
        path,
    )
    return path


def _verify_coco_names() -> None:
    """Pretrained downloads embed these names; a missing resource breaks NITID("model1s")."""
    with as_file(COCO_NAMES) as path:
        names = yaml.safe_load(path.read_text()).get("names", {})
    if len(names) != 80:
        raise RuntimeError(f"Bundled COCO names are incomplete: {len(names)} of 80 classes")


def main() -> None:
    package_path = Path(dfine.__file__).resolve()
    if "site-packages" not in package_path.parts:
        raise RuntimeError(f"Expected an installed package, imported {package_path}")
    if version("nitid") != __version__:
        raise RuntimeError("Installed distribution version does not match the release candidate")
    _verify_coco_names()

    with tempfile.TemporaryDirectory(prefix="nitid-wheel-") as directory:
        root = Path(directory)
        detector = DFINE(_checkpoint(root, "detect"), task="detect", device="cpu", verbose=False)
        segmenter = DFINE(_checkpoint(root, "segment"), task="segment", device="cpu", verbose=False)

        if detector.task != "detect" or segmenter.task != "segment":
            raise RuntimeError("Installed task API returned an unexpected task")
        if any(key.startswith("decoder.mask_") for key in detector._model.state_dict()):
            raise RuntimeError("Installed detection model unexpectedly contains a mask head")
        if not any(key.startswith("decoder.mask_") for key in segmenter._model.state_dict()):
            raise RuntimeError("Installed segmentation model is missing its mask head")

    print(f"Installed detection and segmentation APIs verified: {package_path}")


if __name__ == "__main__":
    main()
