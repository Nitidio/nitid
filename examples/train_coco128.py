"""Train nitid on COCO128.

COCO128 is a tiny Ultralytics dataset with 128 images. It is useful for checking
the full training pipeline, not for judging final model quality.
"""

from nitid import NITID


def main() -> None:
    model = NITID("model1s", task="detect")
    metrics = model.train(
        data="configs/datasets/coco128.yml",
        epochs=5,
        imgsz=640,
        batch=8,
        project="runs/train",
        name="coco128",
    )
    print(metrics)


if __name__ == "__main__":
    main()
