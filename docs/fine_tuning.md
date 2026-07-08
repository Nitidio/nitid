# Fine-tuning

## Requirements

Fine-tuning and validation require `pycocotools`. Install the `train` extras:

```bash
uv sync --extra train
```

## Dataset format

nitid accepts either **COCO JSON** annotations or **YOLO `.txt`** labels.

### COCO JSON

Your dataset directory can look like:

```
my_dataset/
  images/
    train/
      000001.jpg
      ...
    val/
      000100.jpg
      ...
  annotations/
    instances_train.json
    instances_val.json
```

Annotation files follow the standard [COCO detection format](https://cocodataset.org/#format-data).

### YOLO `.txt`

Standard Ultralytics-style layout is also supported:

```text
my_dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Split-first layouts are supported too:

```text
my_dataset/
  train/
    images/
    labels/
  val/
    images/
    labels/
```

nitid detects the layout automatically and converts YOLO labels to cached COCO
JSON internally for training and validation.

### Data YAML

Point to your dataset with an ultralytics-style YAML:

```yaml
# configs/datasets/my_dataset.yml
path: /data/my_dataset
train: images/train
val:   images/val

nc: 3
names:
  0: person
  1: car
  2: bicycle
```

nitid derives annotation file paths automatically:
`<path>/annotations/instances_<split_dirname>.json`

Override with explicit keys if your layout differs:

```yaml
train_ann: annotations/my_train.json
val_ann:   annotations/my_val.json
```

For YOLO datasets, use the same YAML shape and point `train:` / `val:` at the
image directories:

```yaml
path: /data/my_dataset
train: images/train
val:   images/val

names:
  0: person
  1: car
  2: bicycle
```

or, for split-first layouts:

```yaml
path: /data/my_dataset
train: train/images
val:   val/images

names:
  0: person
  1: car
  2: bicycle
```

#### Category ID mapping

COCO annotation files use arbitrary `category_id` integers (often starting at
1, or non-contiguous). nitid needs 0-based label indices for the model head.
By default it sorts the `category_id` values found in the annotation file and
maps them to `0, 1, 2, …` in that order.

If your IDs are non-contiguous or in a custom order, supply an explicit mapping:

```yaml
# annotation file has category_id 1 and 3 (no 2)
cat_ids:
  1: 0   # category_id 1  →  label index 0  (person)
  3: 1   # category_id 3  →  label index 1  (car)
```

Without this override, the auto-mapping would produce the same result for IDs
`{1, 3}` because they sort to `[1, 3]` → `[0, 1]`. The override is only
needed when you want a *different* ordering than sorted order.

## Fine-tuning

### Python API

```python
from dfine import DFINE

model = DFINE("dfine_l")
metrics = model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=50,
    batch=16,
    lr0=1e-4,
    lrf=0.01,       # final lr = lr0 * lrf (linear decay)
    optimizer="AdamW",
    project="runs/train",
    name="my_experiment",
)
print(metrics)  # {"loss": <final_epoch_loss>}
```

Checkpoints are saved after every epoch to `runs/train/my_experiment/epoch{N}.pth`.
Each checkpoint is a full nitid-wrapped `.pth` (config + names embedded) and can
be loaded directly with `DFINE("epoch50.pth")`.

### CLI

```bash
uv run dfine train \
    model=dfine_l \
    data=configs/datasets/my_dataset.yml \
    epochs=50 \
    batch=16
```

### Key parameters

| Parameter    | Default      | Description |
|--------------|--------------|-------------|
| `epochs`     | 50           | Number of training epochs |
| `batch`      | 16           | Batch size |
| `imgsz`      | 640          | Input resolution (square) |
| `lr0`        | 1e-4         | Initial learning rate |
| `lrf`        | 0.01         | Final LR factor (linear decay: ends at `lr0 * lrf`) |
| `optimizer`  | `"AdamW"`    | `"AdamW"` or `"SGD"` |
| `amp`        | `False`      | Enable AMP mixed-precision (CUDA only) |
| `ema`        | `False`      | Enable EMA weight averaging |
| `ema_decay`  | 0.9999       | EMA decay factor (ignored when `ema=False`) |
| `project`    | `runs/train` | Output root directory |
| `name`       | `exp`        | Run name |
| `resume`     | `False`      | Reserved — not yet implemented |

## AMP — mixed-precision training

AMP uses `torch.amp.autocast` and `GradScaler` to run the forward pass in
FP16 while keeping the master weights in FP32. It typically cuts GPU memory
usage by ~40 % and speeds up training on modern NVIDIA GPUs.

```python
metrics = model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=50,
    batch=32,        # larger batch fits in GPU memory with AMP
    amp=True,
)
```

> **CPU fallback:** `amp=True` is silently ignored on CPU devices (a warning
> is logged). Training continues in full precision.

## EMA — exponential moving average

EMA maintains a shadow copy of the model whose weights are updated after
every optimiser step:

```
ema_weight = decay × ema_weight + (1 − decay) × model_weight
```

The EMA weights are what gets saved to the epoch checkpoint, so loading
`DFINE("epoch50.pth")` gives you the more stable EMA model directly.
Integer parameters (e.g. anchor indices) are copied verbatim rather than
blended.

```python
metrics = model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=50,
    ema=True,
    ema_decay=0.9999,   # standard value; lower = faster adaptation
)
```

### AMP + EMA together

```python
metrics = model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=50,
    batch=32,
    amp=True,
    ema=True,
)
```

## Validation

### Python API

```python
metrics = model.val(
    data="configs/datasets/my_dataset.yml",
    imgsz=640,
    split="val",
    batch=16,
    conf=0.001,   # low threshold — include all detections in mAP computation
    verbose=True,
)
print(metrics)
# {
#   "mAP50-95": 0.412,   # AP averaged over IoU 0.50:0.05:0.95
#   "mAP50":    0.623,   # AP at IoU=0.50
#   "AR1":      0.341,   # Average Recall at max 1 detection per image
#   "AR100":    0.512,   # Average Recall at max 100 detections per image
# }
```

### CLI

```bash
uv run dfine val \
    model=dfine_l \
    data=configs/datasets/my_dataset.yml \
    conf=0.001
```

## Notes

- The model is **not** put into deploy mode during training (BN fusion would
  prevent further training). Deploy happens automatically on the first
  `predict()` call after training.
- Loss weighting (`weight_dict`) comes from the checkpoint's embedded D-FINE
  config so it stays consistent with the original pre-training setup.
- When `ema=True` the saved checkpoint contains EMA weights. Loading it with
  `DFINE(path)` gives you the EMA model directly — no extra step needed.
- AMP is only active on CUDA; on CPU it degrades gracefully to full precision.
