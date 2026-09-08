"""Fine-tune oriented bounding boxes on a YOLO-OBB, DOTA, or COCO-polygon dataset."""

from dfine import NITID

model = NITID("nitid1s", task="obb")

metrics = model.train(
    data="obb_dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="runs/train",
    name="obb_experiment",
)
print(metrics)
