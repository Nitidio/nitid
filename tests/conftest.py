"""
Session-scoped fixtures for integration tests.

tiny_checkpoint — small wrapped .pth built from random weights (no download).
tiny_dataset    — minimal synthetic COCO dataset (blank images + JSON anns)
                  with a data YAML ready for train/val calls.
tiny_yolo_dataset — minimal synthetic YOLO dataset using images/train + labels/train.
tiny_yolo_splitfirst_dataset — minimal synthetic YOLO dataset using train/images + train/labels.
"""

from pathlib import Path

import pytest

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


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    """
    Returns the path to a data YAML backed by a minimal synthetic COCO dataset:
      - 4 train images (64×64 black JPEG), 2 val images
      - One annotation per image (category_id=1, bbox=[10,10,20,20])
      - Two categories: 1=person, 2=car
    """
    import json

    import numpy as np
    import yaml
    from PIL import Image as _PILImage

    root = tmp_path_factory.mktemp("coco_dataset")

    categories = [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}]
    ann_dir = root / "annotations"
    ann_dir.mkdir()

    for split, n_imgs in (("train", 4), ("val", 2)):
        img_dir = root / "images" / split
        img_dir.mkdir(parents=True)

        images, annotations = [], []
        for i in range(1, n_imgs + 1):
            fname = f"{i:06d}.jpg"
            _PILImage.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)).save(
                img_dir / fname
            )
            images.append({"id": i, "file_name": fname, "width": 64, "height": 64})
            annotations.append(
                {
                    "id": i,
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [10, 10, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                }
            )

        with open(ann_dir / f"instances_{split}.json", "w") as f:
            json.dump({"images": images, "annotations": annotations, "categories": categories}, f)

    data_yaml = root / "data.yml"
    with open(data_yaml, "w") as f:
        yaml.dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "nc": 2,
                "names": {0: "person", 1: "car"},
            },
            f,
        )

    return str(data_yaml)


def _write_yolo_split(root, split_rel: str, n_imgs: int, split_first: bool) -> None:
    import numpy as np
    from PIL import Image as _PILImage

    if split_first:
        img_dir = root / split_rel / "images"
        label_dir = root / split_rel / "labels"
    else:
        img_dir = root / "images" / split_rel
        label_dir = root / "labels" / split_rel

    img_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    for i in range(1, n_imgs + 1):
        fname = f"{i:06d}.jpg"
        _PILImage.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)).save(
            img_dir / fname
        )

        label_path = label_dir / f"{i:06d}.txt"
        if i == 1:
            label_path.write_text("0 0.5 0.5 0.3125 0.3125\n")
        elif i == 2:
            label_path.write_text("1 0.4 0.4 0.2 0.2\n")
        elif i == 3:
            label_path.write_text("")
        # i == 4 intentionally has no label file


@pytest.fixture(scope="session")
def tiny_yolo_dataset(tmp_path_factory):
    """Minimal YOLO txt dataset using the standard images/train + labels/train layout."""
    import yaml

    root = tmp_path_factory.mktemp("yolo_dataset")
    _write_yolo_split(root, "train", 4, split_first=False)
    _write_yolo_split(root, "val", 2, split_first=False)

    data_yaml = root / "data.yml"
    with open(data_yaml, "w") as f:
        yaml.dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "nc": 2,
                "names": {0: "person", 1: "car"},
            },
            f,
        )

    return str(data_yaml)


@pytest.fixture(scope="session")
def tiny_yolo_splitfirst_dataset(tmp_path_factory):
    """Minimal YOLO txt dataset using the split-first train/images + train/labels layout."""
    import yaml

    root = tmp_path_factory.mktemp("yolo_splitfirst_dataset")
    _write_yolo_split(root, "train", 4, split_first=True)
    _write_yolo_split(root, "val", 2, split_first=True)

    data_yaml = root / "data.yml"
    with open(data_yaml, "w") as f:
        yaml.dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "val/images",
                "nc": 2,
                "names": {0: "person", 1: "car"},
            },
            f,
        )

    return str(data_yaml)
