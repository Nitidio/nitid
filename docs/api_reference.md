# API Reference

## `DFINE`

```python
from dfine import DFINE
```

The single public class. Instantiate with a path to a nitid-wrapped `.pth` checkpoint, or a registry model name (e.g. `"dfine_s"`, `"dfine_l"`) to automatically download, wrap, and load the official weights.

```python
model = DFINE("dfine_s", device="cuda:0")
```

| Argument  | Type  | Default | Description |
|-----------|-------|---------|-------------|
| `model`   | `str` | `"dfine_l.pth"` | Path to wrapped checkpoint or a registry model name (`"dfine_s"`, `"dfine_m"`, `"dfine_l"`, `"dfine_x"`) |
| `device`  | `str \| int \| None` | `None` | PyTorch device selector. Omit it to auto-select `"cuda:0"` when available, otherwise `"cpu"`. |
| `verbose` | `bool`| `True`  | Print load summary |

If you pass an explicit device string, it is used as-is after normalization. Omitting `device` gives the Ultralytics-style smart default.

---

### `predict()`

Run inference on any source.

```python
results = model.predict(
    source,           # path, dir, URL, ndarray, int (webcam), or list
    conf=0.5,         # confidence threshold
    imgsz=640,        # inference size (square)
    classes=None,     # filter to these class indices, e.g. [0, 2]
    stream=False,     # return generator instead of list
    augment=False,    # run test-time augmentation (horizontal flip)
    save=False,       # save annotated images to project/name
    project="runs/detect",
    name="exp",
    iou=0.85,          # IoU threshold for TTA NMS
)
```

Returns `list[Results]` (or a generator when `stream=True`).

When `save=True`, nitid also writes annotated images to `project/name` while
still returning the normal `Results` objects:

```python
results = model.predict("image.jpg", save=True)
# saved image: runs/detect/exp/image.jpg

results = model.predict("image.jpg", save=True, project="runs/detect", name="street-test")
# saved image: runs/detect/street-test/image.jpg
```

#### `Results`

| Attribute  | Type            | Description |
|------------|-----------------|-------------|
| `orig_img` | `np.ndarray`    | Original image HWC BGR |
| `path`     | `str`           | Source path or descriptor |
| `names`    | `dict[int,str]` | Class index → name |
| `boxes`    | `Boxes \| None` | Detection boxes |
| `save_path` | `str \| None`  | Saved annotated image path when `save=True` |

```python
r = results[0]
r.plot()            # → HWC BGR ndarray with boxes drawn
r.save("out.jpg")   # write plotted image to disk
r.show()            # display in a window (blocks until key press)
r.to_json()         # → list[dict] with box/score/class per detection
len(r)              # number of detections
```

Stream predictions with `stream=True` to avoid buffering all frames in memory:

```python
for r in model.predict("video.mp4", stream=True, conf=0.3):
    annotated = r.plot()   # process one frame at a time
```

#### `Boxes`

| Property | Shape     | Description |
|----------|-----------|-------------|
| `xyxy`   | `[N, 4]`  | Absolute pixel coords x1 y1 x2 y2 |
| `xyxyn`  | `[N, 4]`  | Normalised 0–1 coords |
| `xywh`   | `[N, 4]`  | cx cy w h absolute pixels |
| `xywhn`  | `[N, 4]`  | cx cy w h normalised |
| `conf`   | `[N]`     | Confidence scores |
| `cls`    | `[N]`     | Class indices (int) |
| `data`   | `[N, 6]`  | Raw tensor: xyxy + conf + cls |

Iterate detections by index:

```python
for i in range(len(results[0].boxes)):
    x1, y1, x2, y2 = results[0].boxes.xyxy[i].tolist()
    conf = results[0].boxes.conf[i].item()
    cls  = int(results[0].boxes.cls[i].item())
```

---

### `train()`

Fine-tune on a custom COCO-format dataset. See [fine_tuning.md](fine_tuning.md).

```python
metrics = model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=1e-4,
    lrf=0.01,
    optimizer="AdamW",   # or "SGD"
    resume=False,        # reserved for future checkpoint resuming
    amp=False,           # FP16 mixed precision (CUDA only)
    ema=False,           # EMA weight averaging
    ema_decay=0.9999,
    device=None,         # override training device
    project="runs/train",
    name="exp",
    verbose=True,
)
# metrics = {
#   "loss": ...,
#   "fitness": ...,
#   "mAP50": ...,
#   "mAP50-95": ...,
#   "history": [{...}, ...],
# }
```

The top-level values summarize the final epoch. `metrics["history"]` contains
one row per epoch with training loss terms and validation metrics.

---

### `val()`

Evaluate with COCO mAP metrics. See [fine_tuning.md](fine_tuning.md).

```python
metrics = model.val(
    data="configs/datasets/my_dataset.yml",
    imgsz=640,
    batch=16,
    conf=0.001,
    split="val",         # "val" or "test"
    project="runs/val",
    name="exp",
    plots=True,
    verbose=True,
)
# metrics = {
#   "mAP50-95": ...,
#   "mAP50": ...,
#   "AR1": ...,
#   "AR100": ...,
#   "precision": ...,
#   "recall": ...,
#   "f1": ...,
#   "per_class": [{...}, ...],
# }
```

---

### `export()`

Export to ONNX, TorchScript, or TensorRT. See [export.md](export.md).

```python
model.export(format="onnx")        # → dfine_640.onnx
model.export(format="torchscript") # → dfine_640.torchscript
model.export(format="tensorrt")    # → dfine_640.engine  (requires tensorrt installation)
```

| Argument    | Default  | Description |
|-------------|----------|-------------|
| `format`    | `"onnx"` | `"onnx"`, `"torchscript"`, or `"tensorrt"` |
| `imgsz`     | 640      | Must match model's `eval_spatial_size` |
| `batch`     | 1        | Static batch size |
| `dynamic`   | `False`  | Dynamic batch axis (ONNX and TensorRT) |
| `simplify`  | `True`   | Run onnxsim after export (ONNX only) |
| `opset`     | 17       | ONNX opset version (ONNX only) |
| `half`      | `False`  | FP16 precision (TensorRT only) |
| `device`    | `None`   | Override export device |
| `verbose`   | `True`   | Print export progress |

---

### Properties

```python
model.names   # {0: "person", 1: "bicycle", ...}  — class index → name
model.device  # "cpu" or "cuda:0"                 — device the model lives on
model.task    # "detect"                           — always "detect"
```

`names` is the class mapping embedded in the checkpoint.

---

### `model.info()`

```python
info = model.info(detailed=False, verbose=True)
# [D-FINE] 31.4M params (31.4M trainable)  120.3 GFLOPs  98.6 MB
# {
#   "params":           31_400_000,
#   "params_trainable": 31_400_000,
#   "gflops":           120.3,       # None if profiling unavailable
#   "size_mb":          98.6,        # None if checkpoint file moved
# }
```

`detailed=True` adds a `"layers"` key mapping each parameter name to its element count.

GFLOPs are measured via `torch.profiler` (counts conv, linear, and matmul ops).
The figure covers a single `[1, 3, 640, 640]` forward pass and is `None` on
the rare case profiling raises an exception.

---

## Error handling

| Situation | Exception |
|-----------|-----------|
| Checkpoint file not found | `FileNotFoundError` |
| File exists but has no embedded config (raw D-FINE .pth) | `KeyError` — run `tools/convert_checkpoint.py` first |
| `export(format=…)` with unsupported format | `ValueError` |
| `train(optimizer=…)` with unknown name | `ValueError` |

```python
try:
    model = DFINE("my_model.pth")
except FileNotFoundError:
    print("Checkpoint not found — check the path")
```
