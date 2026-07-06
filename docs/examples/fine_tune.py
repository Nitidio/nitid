"""Fine-tune D-FINE on a custom dataset."""

from dfine import DFINE

model = DFINE("dfine_l.pth")

metrics = model.train(
    data="my_dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=1e-4,
    project="runs/train",
    name="my_experiment",
)
print(metrics)
