"""Train nitid on COCO-mini.

COCO-mini is a small COCO-format dataset (128 train and 32 val images) built
from COCO 2017 images with redistributable licences. It serves detection and
instance segmentation, and is useful for checking the full training pipeline,
not for judging final model quality. It downloads on first use.

    uv run python examples/train_coco_mini.py --task detect
    uv run python examples/train_coco_mini.py --task segment
"""

import argparse

from nitid import NITID


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nitid on COCO-mini.")
    parser.add_argument("--task", choices=("detect", "segment"), default="detect")
    args = parser.parse_args()

    model = NITID("model1s", task=args.task)
    metrics = model.train(
        data="configs/datasets/coco-mini.yml",
        epochs=5,
        imgsz=640,
        batch=8,
        project="runs/train",
        name=f"coco-mini-{args.task}",
    )
    print(metrics)


if __name__ == "__main__":
    main()
