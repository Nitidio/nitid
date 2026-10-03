"""Train nitid instance segmentation on COCO128-seg.

COCO128-seg is a tiny Ultralytics dataset with 128 images and polygon instance
annotations. It is useful for checking the full segmentation training pipeline,
not for judging final model quality.
"""

from nitid import NITID


def main() -> None:
    model = NITID("model1s", task="segment")
    metrics = model.train(
        data="configs/datasets/coco128-seg.yml",
        epochs=5,
        imgsz=640,
        batch=8,
        project="runs/train",
        name="coco128-seg",
    )
    print(metrics)


if __name__ == "__main__":
    main()
