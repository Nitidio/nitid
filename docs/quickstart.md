# Quickstart

## Installation

```bash
git clone https://github.com/Vaelsys/nitid.git && cd nitid
git submodule update --init        # pulls extern/dfine
uv sync --extra dev
```

---

## For D-FINE users

If you already have a raw D-FINE checkpoint, nitid wraps it in a self-contained
`.pth` that embeds the model config and class names — so you only ever deal with
one file.

### 1. Convert your checkpoint

```bash
uv run python tools/convert_checkpoint.py \
    --weights dfine_l.pth \
    --config  extern/dfine/configs/dfine/dfine_hgnetv2_l_coco.yml \
    --names   configs/datasets/coco.yml \
    --output  dfine_l_wrapped.pth
```

- `--config` must be one of the canonical D-FINE configs from `extern/dfine/configs/`.
  The converter resolves all `__include__` directives and embeds the full model definition.
- `--names` must be a file with a `names:` mapping. Use `configs/datasets/coco.yml` for
  COCO models, or your own dataset config.
- When the raw checkpoint contains EMA weights (`ckpt["ema"]["module"]`), the converter
  uses them automatically — this matches D-FINE's own inference scripts and gives better
  accuracy than the non-EMA weights.

### 2. Run inference

```python
from dfine import DFINE

model = DFINE("dfine_l")
results = model.predict("image.jpg", conf=0.5)
results[0].save("out.jpg")
```

The interface is intentionally close to D-FINE's own inference scripts, but with
pre/post-processing handled for you. `results[0].boxes.xyxy` is in absolute pixel
coordinates; no manual rescaling needed.

### 3. Fine-tune on your data

Prepare a COCO-format dataset and a data YAML (see [fine_tuning.md](fine_tuning.md)):

```python
metrics = model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=50,
    batch=8,
    lr0=1e-4,
    optimizer="AdamW",
)
# final summary lives at the top level; per-epoch rows are in metrics["history"]
```

Epoch checkpoints are saved as wrapped `.pth` files and can be loaded directly with
`DFINE("epoch50.pth")` — config and names travel with the weights.

---

## For Ultralytics users

nitid mirrors the `ultralytics.YOLO` interface. If you already use YOLO, the
switch is mostly a one-line change.

### Drop-in replacement

```python
# Before
from ultralytics import YOLO
model = YOLO("yolo11n.pt")

# After
from dfine import DFINE
model = DFINE("dfine_l")
```

All the patterns you already know work the same way:

```python
# Inference
results = model("image.jpg", conf=0.5)
results = model.predict("image.jpg", conf=0.5, classes=[0, 2])

# Streaming (memory-efficient for video)
for r in model.predict("video.mp4", stream=True):
    annotated = r.plot()

# Iterate boxes
for i in range(len(results[0].boxes)):
    x1, y1, x2, y2 = results[0].boxes.xyxy[i].tolist()
    conf = results[0].boxes.conf[i].item()
    name = results[0].names[int(results[0].boxes.cls[i])]

# Normalised coords (same as YOLO)
boxes_n = results[0].boxes.xyxyn   # 0–1 range
boxes_wh = results[0].boxes.xywh   # cx cy w h absolute
boxes_whn = results[0].boxes.xywhn # cx cy w h normalised

# Save / show / serialise
results[0].save("out.jpg")
results[0].show()
json_data = results[0].to_json()   # list of dicts

# Fine-tune
model.train(data="my_dataset.yml", epochs=50, batch=16)
model.train(data="my_dataset.yml", epochs=50, amp=True, ema=True)  # with AMP + EMA

# Evaluate
metrics = model.val(data="my_dataset.yml")
# {"mAP50-95": ..., "mAP50": ..., "AR1": ..., "AR100": ..., "precision": ..., "recall": ...}

# Model info
model.info()
# [D-FINE] 31.4M params (31.4M trainable)  120.3 GFLOPs  98.6 MB

# Export
model.export(format="onnx")
model.export(format="torchscript")
model.export(format="tensorrt")          # requires: uv sync --extra tensorrt
model.export(format="tensorrt", half=True)   # FP16
```

### Key differences from Ultralytics YOLO

| Feature | Ultralytics YOLO | nitid DFINE |
|---------|-----------------|-------------|
| Checkpoint format | `.pt` (architecture inferred from filename) | `.pth` (config embedded inside) |
| Raw weights | Download directly | Downloaded and wrapped automatically |
| `model.info()` | Returns param/FLOP stats | Supported — params, GFLOPs, size on disk |
| TensorRT export | Supported | Supported (`see Installation in README`) |
| AMP / EMA training | Supported | Supported (`amp=True`, `ema=True`) |
| `model.task` | `"detect"`, `"segment"`, … | Always `"detect"` |

### CLI

```bash
uv run dfine predict model=dfine_l source=image.jpg conf=0.5
uv run dfine train  model=dfine_l data=my_dataset.yml epochs=50
uv run dfine val    model=dfine_l data=my_dataset.yml
uv run dfine export model=dfine_l format=onnx
```
