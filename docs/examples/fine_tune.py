"""Fine-tune a nitid model on a custom dataset."""

from nitid import NITID

model = NITID("nitid1s", task="detect")

metrics = model.train(
    data="my_dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    recipe="default",
    lr0=1e-4,
    project="runs/train",
    name="my_experiment",
)
print(metrics)
