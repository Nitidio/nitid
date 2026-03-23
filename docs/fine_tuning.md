# Fine-tuning

## Requirements

Fine-tuning and validation require `pycocotools`. Install the `train` extras:

```bash
uv sync --extra train
```

## Dataset format

nitid expects **COCO JSON** annotations. Your dataset directory should look like:

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

#### Category ID mapping

By default nitid sorts the category IDs found in the annotation file and maps
them to 0-based label indices. If your IDs are not contiguous or follow a
custom order, override with a `cat_ids` block:

```yaml
cat_ids:
  1: 0   # category_id 1 → label 0 (person)
  3: 1   # category_id 3 → label 1 (car)
```

## Fine-tuning

### Python API

```python
from dfine import DFINE

model = DFINE("dfine_l_wrapped.pth")
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
    model=dfine_l_wrapped.pth \
    data=configs/datasets/my_dataset.yml \
    epochs=50 \
    batch=16
```

### Key parameters

| Parameter   | Default      | Description |
|-------------|--------------|-------------|
| `epochs`    | 50           | Number of training epochs |
| `batch`     | 16           | Batch size |
| `imgsz`     | 640          | Input resolution (square) |
| `lr0`       | 1e-4         | Initial learning rate |
| `lrf`       | 0.01         | Final LR factor (linear decay) |
| `optimizer` | AdamW        | `"AdamW"` or `"SGD"` |
| `project`   | `runs/train` | Output root directory |
| `name`      | `exp`        | Run name |

## Validation

### Python API

```python
metrics = model.val(
    data="configs/datasets/my_dataset.yml",
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
    model=dfine_l_wrapped.pth \
    data=configs/datasets/my_dataset.yml \
    conf=0.001
```

## Notes

- The model is **not** put into deploy mode during training (BN fusion would
  prevent further training). Deploy happens automatically on the first
  `predict()` call after training.
- Loss weighting (`weight_dict`) comes from the checkpoint's embedded D-FINE
  config so it stays consistent with the original pre-training setup.
- AMP (mixed precision) and EMA are not yet enabled; planned for a future phase.
