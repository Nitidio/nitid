# Quickstart

This page is the canonical quickstart for nitid. It covers the shortest path to
first inference, the common train/val/export workflow, and the main differences
for users coming from raw D-FINE or Ultralytics YOLO.

## Installation

```bash
git clone https://github.com/Vaelsys/nitid.git && cd nitid
git submodule update --init        # pulls extern/dfine
uv sync --extra train
```

Add extras only when you need them:

```bash
uv sync --extra dev   # for developement
uv sync --extra web     # web application
```

## First Run

Create a model using any official D-FINE model name. nitid will automatically
download, wrap, and load the corresponding checkpoint on first use.

```python
from dfine import DFINE

model = DFINE("dfine_s")
results = model.predict("image.jpg", conf=0.5)
results[0].save("out.jpg")
```

`results[0].boxes.xyxy` is already in absolute pixel coordinates, so no manual
rescaling is needed.

## Common Workflow

Use the same model object for inference, training, validation, and export:

```python
from dfine import DFINE

model = DFINE("dfine_s")

# Inference
results = model.predict("image.jpg", conf=0.5)
results[0].save("out.jpg")

# Training
model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=50,
    batch=8,
    lr0=1e-4,
    optimizer="AdamW",
)

# Validation
metrics = model.val(data="configs/datasets/my_dataset.yml")

# Export
model.export(format="onnx")
model.export(format="torchscript")
model.export(format="tensorrt")
```

Detections can also be exported to tabular data for analysis:

```python
result = results[0]
df = result.pandas()
same_df = result.to_df()
result.to_csv("out.csv")
```

See [fine_tuning.md](fine_tuning.md) for dataset format, optimizers, AMP, EMA,
and validation details. See [export.md](export.md) for TensorRT, FP16, and
advanced export options.

## Coming from D-FINE

If you already have a raw D-FINE checkpoint, nitid wraps it in a self-contained
`.pth` that embeds the model config and class names, so you only ever deal with
one file.

### Convert a checkpoint

```bash
uv run python tools/convert_checkpoint.py \
    --weights dfine_l.pth \
    --config  extern/dfine/configs/dfine/dfine_hgnetv2_l_coco.yml \
    --names   configs/datasets/coco.yml \
    --output  dfine_l_wrapped.pth
```

- `--config` must be one of the canonical D-FINE configs from `extern/dfine/configs/`.
- `--names` must point to a YAML file with a `names:` mapping.
- EMA weights are used automatically when present, matching D-FINE's own inference scripts.

Load the wrapped checkpoint directly afterward:

```python
from dfine import DFINE

model = DFINE("dfine_l_wrapped.pth")
results = model.predict("image.jpg", conf=0.5)
```

Epoch checkpoints from `model.train(...)` are also saved as wrapped `.pth`
files, so they can be loaded directly with `DFINE("epoch50.pth")`.

## Coming from Ultralytics

nitid mirrors the `ultralytics.YOLO` interface closely, so most migrations are
a one-line import swap.

### Drop-in replacement

```python
# Before
from ultralytics import YOLO
model = YOLO("yolo11n.pt")

# After
from dfine import DFINE
model = DFINE("dfine_l")
```

### Results access

```python
results = model("image.jpg", conf=0.5)

for i in range(len(results[0].boxes)):
    x1, y1, x2, y2 = results[0].boxes.xyxy[i].tolist()
    conf = results[0].boxes.conf[i].item()
    name = results[0].names[int(results[0].boxes.cls[i])]

boxes_n = results[0].boxes.xyxyn
boxes_wh = results[0].boxes.xywh
boxes_whn = results[0].boxes.xywhn

json_data = results[0].to_json()
df = results[0].pandas()
```

### Streaming, training, and export

```python
for r in model.predict("video.mp4", stream=True):
    annotated = r.plot()

model.train(data="my_dataset.yml", epochs=50, batch=16)
model.train(data="my_dataset.yml", epochs=50, amp=True, ema=True)

metrics = model.val(data="my_dataset.yml")
model.export(format="onnx")
model.export(format="torchscript")
model.export(format="tensorrt", half=True)
```

### Key differences from Ultralytics YOLO

| Feature | Ultralytics YOLO | nitid DFINE |
|---------|-----------------|-------------|
| Checkpoint format | `.pt` (architecture inferred from filename) | `.pth` (config embedded inside) |
| Raw weights | Download directly | Downloaded and wrapped automatically |
| `model.info()` | Returns param/FLOP stats | Supported — params, GFLOPs, size on disk |
| TensorRT export | Supported | Supported (see [export.md](export.md)) |
| AMP / EMA training | Supported | Supported (`amp=True`, `ema=True`) |
| `model.task` | `"detect"`, `"segment"`, ... | Always `"detect"` |

## CLI

Use the CLI when you want to run nitid from the terminal instead of Python.

Run prediction and save the annotated image:

```bash
uv run dfine predict model=dfine_s source=image.jpg save=true conf=0.5
```

This automatically downloads and wraps `dfine_s` on first use, runs detection
on `image.jpg`, and saves the annotated image to:

```text
runs/detect/exp/image.jpg
```

Choose your own output folder name:

```bash
uv run dfine predict \
    model=dfine_s \
    source=image.jpg \
    save=true \
    project=runs/detect \
    name=street-test
```

The saved image will be:

```text
runs/detect/street-test/image.jpg
```

Other common CLI commands:

```bash
uv run dfine train model=dfine_l data=my_dataset.yml epochs=50
uv run dfine val model=dfine_l data=my_dataset.yml
uv run dfine export model=dfine_l format=onnx
```

- `train`: fine-tune a model on a COCO-format dataset YAML.
- `val`: run COCO-style evaluation on the validation or test split.
- `export`: convert a wrapped checkpoint to ONNX, TorchScript, or TensorRT.

By default, `train` writes wrapped epoch checkpoints to `runs/train/exp/`, while `val` prints metrics to the terminal without creating a run directory.

Show all prediction options:

```bash
uv run dfine predict --help
```
