"""
Dataset utilities for fine-tuning — COCO JSON format.

Data YAML (ultralytics-style):
    path: /data/my_dataset
    train: images/train         # image directory, relative to path
    val:   images/val
    nc: 3
    names:
      0: person
      1: car
      2: bicycle

Annotation files are discovered at:
    <path>/annotations/instances_<split_dirname>.json

or specified explicitly with train_ann / val_ann keys.

Category IDs in the annotation file are mapped to 0-based label indices
by sorting them; override with a cat_ids: {cat_id: label} mapping in the YAML.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def load_data_yaml(path: str | Path) -> dict:
    """Load an ultralytics-style data YAML and return it as a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


class CocoFinetuneDataset(Dataset):
    """
    COCO-format dataset compatible with DFINECriterion.

    Each item returns ``(tensor [3,H,W], target)`` where target is::

        {
            "labels": LongTensor  [N]     — 0-based class indices,
            "boxes":  FloatTensor [N, 4]  — cxcywh normalised to [0, 1],
            "image_id": LongTensor [1],
        }

    Boxes are normalised *before* the resize transform so the resize is
    transparent — no box coordinates need updating after resizing.
    """

    def __init__(
        self,
        img_dir: str | Path,
        ann_file: str | Path,
        imgsz: int,
        cat_id_to_label: dict[int, int] | None = None,
    ) -> None:
        from pycocotools.coco import COCO

        self.coco = COCO(str(ann_file))
        self.img_dir = Path(img_dir)
        # only keep images that have at least one non-crowd annotation
        self.ids = [
            img_id for img_id in self.coco.imgs if self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        ]
        self.transform = T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor()])

        if cat_id_to_label is None:
            sorted_cat_ids = sorted(self.coco.cats)
            cat_id_to_label = {c: i for i, c in enumerate(sorted_cat_ids)}
        self.cat_id_to_label = cat_id_to_label

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.coco.imgs[img_id]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        W, H = img.size

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([(x + w / 2) / W, (y + h / 2) / H, w / W, h / H])
            labels.append(self.cat_id_to_label.get(ann["category_id"], 0))

        target = {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "image_id": torch.tensor([img_id], dtype=torch.long),
        }
        return self.transform(img), target


def _collate(batch):
    """Stack images into [B,C,H,W]; keep targets as a list of dicts."""
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def build_coco_dataloader(
    data: str | Path,
    split: str,
    imgsz: int,
    batch_size: int,
) -> DataLoader:
    """
    Build a DataLoader from an ultralytics-style data YAML.

    Args:
        data:       Path to the data YAML file.
        split:      ``"train"`` or ``"val"``.
        imgsz:      Resize target (square).
        batch_size: Batch size.
    """
    cfg = load_data_yaml(data)
    root = Path(cfg["path"])
    img_dir = root / cfg[split]

    ann_key = f"{split}_ann"
    if ann_key in cfg:
        ann_file = root / cfg[ann_key]
    else:
        split_name = Path(cfg[split]).name  # "train" from "images/train"
        ann_file = root / "annotations" / f"instances_{split_name}.json"

    cat_ids_cfg = cfg.get("cat_ids")
    cat_id_to_label = {int(k): int(v) for k, v in cat_ids_cfg.items()} if cat_ids_cfg else None

    dataset = CocoFinetuneDataset(
        img_dir=img_dir,
        ann_file=ann_file,
        imgsz=imgsz,
        cat_id_to_label=cat_id_to_label,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        collate_fn=_collate,
        drop_last=(split == "train"),
    )
