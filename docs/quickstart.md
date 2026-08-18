# Quickstart

This page is the canonical quickstart for nitid. It covers the shortest path to
first inference, the common train/val/export workflow, and the main differences
for users coming from raw D-FINE or Ultralytics YOLO.

## Installation

```bash
git clone https://github.com/Vaelsys/nitid.git && cd nitid
uv sync --extra train
```

Add extras only when you need them:

```bash
uv sync --extra dev   # for developement
uv sync --extra web     # web application
uv sync --extra openvino # OpenVINO IR export and runtime
uv sync --extra track    # ByteTrack, BoT-SORT, and OC-SORT tracking
```

## First Run

Create a model using any official D-FINE model name. nitid will automatically
download, wrap, and load the corresponding checkpoint on first use.

```python
from dfine import DFINE

model = DFINE("dfine_s")
# Default: Objects365 → COCO. Use weights="coco" for COCO-only weights.
results = model.predict("image.jpg", conf=0.5)
results[0].save("out.jpg")
```

`results[0].boxes.xyxy` is already in absolute pixel coordinates, so no manual
rescaling is needed.

For instance segmentation, select the task when constructing the model:

```python
model = DFINE("dfine_s", task="segment")
result = model.predict("image.jpg", conf=0.5)[0]

boxes = result.boxes.xyxy
masks = result.masks.data       # uint8 [N, H, W]
polygons = result.masks.xyn     # normalized polygons
result.save("segmented.jpg")
```

Segmentation defaults to the official COCO-pretrained mask weights. Detection
defaults to Objects365→COCO for S/M/L/X and supports `weights="coco"` as well.

For dense semantic segmentation, load a trained semantic checkpoint and read
the original-resolution class map from `result.semantic.mask`:

```python
semantic = DFINE("semantic_best.pth", task="semantic")
result = semantic.predict("image.jpg", save=True, return_probs=True)[0]

class_map = result.semantic.mask       # int64 [H, W]
probabilities = result.semantic.probs  # float [C, H, W]
result.save_semantic("class_ids.png")
```

`save=True` writes an annotated overlay and a lossless class-ID PNG in the
run's `masks/` directory.

## Common Workflow

Use the same model object for inference, training, validation, and export:

```python
from dfine import DFINE

model = DFINE("dfine_s")

# Inference
results = model.predict("image.jpg", conf=0.5)
results[0].save("out.jpg")

# Tracking (requires: uv sync --extra track)
for result in model.track("video.mp4", conf=0.5, stream=True, save=True):
    track_ids = result.boxes.id

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
model.export(format="openvino")
model.export(format="torchscript")
model.export(format="tensorrt")
```

Detections can also be exported to tabular data for analysis:

```python
result = results[0]
df = result.pandas()
same_df = result.to_df()
result.to_csv("out.csv")
crops = result.crop()          # cropped objects as numpy arrays + metadata
result.crop(save_dir="crops")  # save crops into class-name folders
```

See [fine_tuning.md](fine_tuning.md) for dataset format, optimizers, AMP, EMA,
and validation details. See [export.md](export.md) for TensorRT, FP16, and
advanced export options.

Tracking processes frames in source order and returns the normal `Results`
objects with persistent IDs in `result.boxes.id`. With `save=True`, the
annotated video defaults to `runs/track/exp/video.mp4`. Prefer `stream=True`
for video and live sources so results are not retained in memory.

For a live RTSP camera, select GStreamer explicitly and enable reconnection:

```python
for result in model.track(
    "rtsp://camera/live",
    backend="gstreamer",
    reconnect=True,
    conf=0.5,
    stream=True,
):
    track_ids = result.boxes.id
```

This requires an OpenCV build compiled with GStreamer. See
[GStreamer and RTSP](gstreamer.md) for verification and pipeline examples.

Write annotated one-minute segments from the CLI:

```bash
uv run dfine track model=dfine_s source=video.mp4 \
    output=runs/segments segment_duration=60 conf=0.5
```

Discover an ONVIF camera, select a media profile, and hand it directly to the
tracking pipeline:

```python
from dfine import DFINE, ONVIFCamera, discover_onvif_devices

device = discover_onvif_devices(timeout=3)[0]
camera = ONVIFCamera(device.service_url, username="operator", password="secret")
source = camera.gstreamer_source("Main Stream", hardware_profile="vaapi")

model = DFINE("dfine_s")
for result in model.track(source, stream=True, conf=0.5):
    ...
```

See [ONVIF cameras](onvif.md) for CLI credential handling, network discovery,
profile selection, and clock troubleshooting.

## Converting a raw checkpoint

If you already have a raw D-FINE checkpoint, nitid wraps it in a self-contained
`.pth` that embeds the model config and class names, so you only ever deal with
one file.

### Convert a checkpoint

```bash
uv run python tools/convert_checkpoint.py \
    --weights dfine_l.pth \
    --model   dfine_l \
    --task    detect \
    --names   configs/datasets/coco.yml \
    --output  dfine_l_wrapped.pth
```

- `--model` selects the integrated N/S/M/L/X architecture.
- `--task` selects detection or instance segmentation.
- `--names` must point to a YAML file with a `names:` mapping.
- EMA weights are used automatically when present, matching D-FINE's own inference scripts.

Load the wrapped checkpoint directly afterward:

```python
from dfine import DFINE

model = DFINE("dfine_l_wrapped.pth", task="detect")
results = model.predict("image.jpg", conf=0.5)
results[0].save("out.jpg")
```

The interface is intentionally close to D-FINE's own inference scripts, but with
pre/post-processing handled for you. `results[0].boxes.xyxy` is in absolute pixel
coordinates; no manual rescaling needed.

You can also export detections directly into tabular data for analysis:

```python
result = results[0]
df = result.pandas()       # pandas DataFrame
same_df = result.to_df()   # alias for pandas()
result.to_csv("out.csv")   # write detections to CSV
```

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
crops = results[0].crop()
```

### Streaming, training, and export

```python
for r in model.predict("video.mp4", stream=True):
    annotated = r.plot()

for r in model.track("video.mp4", stream=True, conf=0.5):
    track_ids = r.boxes.id

model.train(data="my_dataset.yml", epochs=50, batch=16)
model.train(data="my_dataset.yml", epochs=50, amp=True, ema=True)

metrics = model.val(data="my_dataset.yml")
# {"mAP50-95": ..., "mAP50": ..., "AR1": ..., "AR100": ..., "precision": ..., "recall": ...}

# Model info
model.info()
# [D-FINE] 31.4M params (31.4M trainable)  120.3 GFLOPs  98.6 MB

# Export
model.export(format="onnx")
model.export(format="openvino")
model.export(format="torchscript")
model.export(format="tensorrt", half=True)
```

### Capture a Python API bug report

```python
from dfine import DFINE, bugreport

with bugreport("prediction") as report:
    model = DFINE("dfine_s")
    results = model.predict("image.jpg")

print(report.path)
```

The report contains the shared environment snapshot and all stdout/stderr. A
failure traceback is added to the same log before the original exception is
re-raised.

### Key differences from Ultralytics YOLO

| Feature | Ultralytics YOLO | nitid DFINE |
|---------|-----------------|-------------|
| Checkpoint format | `.pt` (architecture inferred from filename) | `.pth` (config embedded inside) |
| Raw weights | Download directly | Downloaded and wrapped automatically |
| `model.info()` | Returns param/FLOP stats | Supported — params, GFLOPs, size on disk |
| TensorRT export | Supported | Supported (see [export.md](export.md)) |
| AMP / EMA training | Supported | Supported (`amp=True`, `ema=True`) |
| `model.task` | `"detect"`, `"segment"`, ... | `"detect"` or `"segment"` |

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

Track a video and save annotations with persistent IDs:

```bash
uv sync --extra track
uv run dfine track model=dfine_s source=video.mp4 conf=0.5 save=true
```

The tracked video defaults to `runs/track/exp/video.mp4`.

Other common CLI commands:

```bash
uv run dfine train model=dfine_l data=my_dataset.yml epochs=50
uv run dfine val model=dfine_l data=my_dataset.yml
uv run dfine export model=dfine_l format=onnx
```

- `train`: fine-tune a model on a COCO-format dataset YAML.
- `val`: run COCO-style evaluation on the validation or test split.
- `export`: convert a wrapped checkpoint to ONNX, OpenVINO IR, TorchScript, or TensorRT.

By default, `train` writes wrapped epoch checkpoints to `runs/train/exp/`, while `val` prints metrics to the terminal without creating a run directory.

Show all prediction or tracking options:

```bash
uv run dfine predict --help
uv run dfine track --help
```
