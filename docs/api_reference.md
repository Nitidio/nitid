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
r.save("out.jpg")   # save plotted image
r.to_json()         # → list[dict] with box/score/class per detection
len(r)              # number of detections
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

Export to ONNX or TorchScript. See [export.md](export.md).

```python
model.export(format="onnx")        # saves dfine_l_wrapped.onnx
model.export(format="torchscript") # saves dfine_l_wrapped.torchscript
```

| Argument    | Default | Description |
|-------------|---------|-------------|
| `format`    | `"onnx"`| `"onnx"` or `"torchscript"` |
| `imgsz`     | 640     | Must match model's `eval_spatial_size` |
| `batch`     | 1       | Static batch size |
| `dynamic`   | `False` | Dynamic batch/spatial axes (ONNX only) |
| `simplify`  | `True`  | Run onnxsim after export |
| `opset`     | 17      | ONNX opset version |

---

### `model.names`

```python
model.names  # {0: "person", 1: "bicycle", ...}
```

Class index → name mapping embedded in the checkpoint.
