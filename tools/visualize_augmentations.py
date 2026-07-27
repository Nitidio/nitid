"""Save annotated samples for visually auditing training augmentations."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from dfine.utils.augmentations import AugmentationConfig
from dfine.utils.data import build_detection_dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset YAML")
    parser.add_argument("--output", default="augmentation_preview.jpg")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    args = parser.parse_args()

    config = AugmentationConfig(mosaic=args.mosaic, mixup=args.mixup)
    loader = build_detection_dataloader(
        args.data,
        "train",
        args.imgsz,
        batch_size=4,
        seed=args.seed,
        augment=config,
    )
    images, targets = next(iter(loader))
    preview = Image.new("RGB", (args.imgsz * len(images), args.imgsz))
    for index, (tensor, target) in enumerate(zip(images, targets)):
        image = Image.fromarray(tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy(), mode="RGB")
        draw = ImageDraw.Draw(image)
        for cx, cy, width, height in target["boxes"] * args.imgsz:
            box = (
                float(cx - width / 2),
                float(cy - height / 2),
                float(cx + width / 2),
                float(cy + height / 2),
            )
            draw.rectangle(box, outline=(255, 0, 0), width=2)
        preview.paste(image, (index * args.imgsz, 0))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output)
    print(f"Saved augmentation preview to {output.resolve()}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
