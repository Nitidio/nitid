# API Reference

## `DFINE`

```python
from dfine import DFINE
```

The single public class. Instantiate with a path to a nitid-wrapped `.pth`
checkpoint; all methods are available immediately.

```python
model = DFINE("dfine_l_wrapped.pth", device="cuda:0")
```

| Argument  | Type  | Default | Description |
|-----------|-------|---------|-------------|
| `model`   | `str` | —       | Path to wrapped `.pth` checkpoint |
| `device`  | `str` | `"cpu"` | PyTorch device string (`"cpu"`, `"cuda:0"`, …) |
| `verbose` | `bool`| `True`  | Print load summary |

---

### `predict()`

Run inference on any source.

```python
results = model.predict(
    source,           # path, dir, URL, ndarray, int (webcam), or list
    conf=0.25,        # confidence threshold
    imgsz=640,        # inference size (square)
    classes=None,     # filter to these class indices, e.g. [0, 2]
    stream=False,     # return generator instead of list
    augment=False,    # run test-time augmentation (horizontal flip)
    iou=0.85,          # IoU threshold for TTA NMS
)
```

Returns `list[Results]` (or a generator when `stream=True`).

#### `Results`

| Attribute  | Type            | Description |
|------------|-----------------|-------------|
| `orig_img` | `np.ndarray`    | Original image HWC BGR |
| `path`     | `str`           | Source path or descriptor |
| `names`    | `dict[int,str]` | Class index → name |
| `boxes`    | `Boxes \| None` | Detection boxes |

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
    batch=16,
    lr0=1e-4,
    lrf=0.01,
    optimizer="AdamW",   # or "SGD"
    amp=True,            # FP16 mixed precision (CUDA only)
    ema=True,            # EMA weight averaging
    ema_decay=0.9999,
    project="runs/train",
    name="exp",
)
# metrics = {"loss": <final_epoch_loss>}
```

---

### `val()`

Evaluate with COCO mAP metrics. See [fine_tuning.md](fine_tuning.md).

```python
metrics = model.val(
    data="configs/datasets/my_dataset.yml",
    split="val",         # "val" or "test"
    batch=16,
    conf=0.001,
    verbose=True,
)
# metrics = {"mAP50-95": ..., "mAP50": ..., "AR1": ..., "AR100": ...}
```

---

### `export()`

Export to ONNX, TorchScript, or TensorRT. See [export.md](export.md).

```python
model.export(format="onnx")        # → dfine_640.onnx
model.export(format="torchscript") # → dfine_640.torchscript
model.export(format="tensorrt")    # → dfine_640.engine  (requires tensorrt extra)
```

| Argument    | Default  | Description |
|-------------|----------|-------------|
| `format`    | `"onnx"` | `"onnx"`, `"torchscript"`, or `"tensorrt"` |
| `imgsz`     | 640      | Must match model's `eval_spatial_size` |
| `batch`     | 1        | Static batch size |
| `dynamic`   | `False`  | Dynamic batch axis (ONNX and TensorRT) |
| `half`      | `False`  | FP16 precision (TensorRT only) |
| `simplify`  | `True`   | Run onnxsim after export (ONNX only) |
| `opset`     | 17       | ONNX opset version (ONNX only) |

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
