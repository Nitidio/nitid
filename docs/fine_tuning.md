# Fine-tuning

## Requirements

Fine-tuning and validation require `pycocotools`. Install the `train` extras:

```bash
uv sync --extra train
```

## Dataset format

nitid accepts either **COCO JSON** annotations or **YOLO `.txt`** labels for
detection and instance segmentation. Pose training uses COCO keypoint JSON.

Semantic segmentation instead uses one dense class-ID PNG mask per image.

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
For `task="segment"`, every object must also contain a COCO polygon or RLE
`segmentation` field.

For `task="pose"`, annotations must follow the COCO keypoints convention:
`bbox`, `area`, `num_keypoints`, and flattened `keypoints` values
`[x1, y1, v1, ..., xK, yK, vK]`. The built-in DETRPose models expose the
single public class `person` and use the COCO-17 keypoint order.

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
JSON internally for training and validation. The generated cache is stored in
nitid's user cache directory rather than inside the dataset tree.

Detection rows use the usual box representation:

```text
class x_center y_center width height
```

Instance-segmentation rows use normalized polygon points:

```text
class x1 y1 x2 y2 x3 y3 ...
```

Bounding-box-only annotations are rejected when training a segmentation model,
preventing an accidental all-zero mask training run.

### Dense semantic masks

For `task="semantic"`, image and mask paths mirror one another:

```text
my_semantic_dataset/
  images/
    train/example.jpg
    val/example.jpg
  labels/
    train/example.png
    val/example.png
```

Each PNG must be a single-channel integer class map with the same dimensions as
its image. Pixel values are contiguous class IDs from `0` through `nc - 1`.
The configured `ignore_index` (255 by default) marks pixels excluded from loss
and validation metrics. Dense masks are always resized with nearest-neighbor
sampling; padding introduced by geometric augmentation is filled with the
ignore index.

```yaml
path: /data/my_semantic_dataset
train: images/train
val: images/val
nc: 3
names:
  0: background
  1: road
  2: vehicle
ignore_index: 255
```

When image and mask directories do not follow the mirrored `images`/`labels`
layout, set `train_masks:` and `val_masks:` explicitly. Semantic training does
not accept instance-only `mosaic`, `mixup`, `classes`, or `single_cls` options.

### Pose keypoints

A compact COCO-keypoints dataset can use the same image split layout:

```text
my_pose_dataset/
  images/
    train/
    val/
  annotations/
    person_keypoints_train.json
    person_keypoints_val.json
```

```yaml
path: /data/my_pose_dataset
train: images/train
val: images/val
train_ann: annotations/person_keypoints_train.json
val_ann: annotations/person_keypoints_val.json

nc: 1
names:
  0: person
cat_ids:
  1: 0

kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
```

Validation reports both the compatibility box metrics and keypoint metrics:
`pose_mAP50` and `pose_mAP50-95`.

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

When `model.train(...)` starts, nitid reads this taxonomy and automatically
rebuilds the class head when it differs from the checkpoint. Backbone and
localization weights are retained, the pretrained encoder scorer is converted
to generic objectness for proposal selection, and only taxonomy-specific class
heads are initialized from scratch. No manual head replacement is required.

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

model = DFINE("dfine_l", task="detect")

# Instance segmentation uses the same training API and mask-aware annotations.
segmenter = DFINE("dfine_s", task="segment")
segment_metrics = segmenter.train(
    data="configs/datasets/my_segment_dataset.yml",
    epochs=50,
)
# segment_metrics["mask_mAP50"], segment_metrics["mask_mAP50-95"]

# Semantic models initialize shared features from the matching COCO
# instance-segmentation checkpoint. The dense classifiers train on your taxonomy.
semantic = DFINE("dfine_s", task="semantic")
semantic_metrics = semantic.train(
    data="configs/datasets/my_semantic_dataset.yml",
    epochs=50,
)
# semantic_metrics["mIoU"], semantic_metrics["pixel_accuracy"]

# Pose models use DETRPose checkpoints and COCO-keypoint annotations.
pose = DFINE("detrpose_n", task="pose")
pose_metrics = pose.train(
    data="configs/datasets/my_pose_dataset.yml",
    epochs=50,
)
# pose_metrics["pose_mAP50"], pose_metrics["pose_mAP50-95"]

def print_epoch_end(trainer):
    row = trainer.current_row
    if row is not None:
        print(
            f"epoch={row['epoch']} loss={row['loss']:.4f} "
            f"mAP50-95={row['mAP50-95']:.4f}"
        )

metrics = model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=50,
    batch=16,
    lr0=1e-4,
    lrf=0.01,       # final lr = lr0 * lrf
    cos_lr=True,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    optimizer="AdamW",
    project="runs/train",
    name="my_experiment",
    callbacks={"on_train_epoch_end": print_epoch_end},
)
print(metrics)
# {
#   "loss": 1.234,
#   "fitness": 0.567,
#   "mAP50": 0.612,
#   "mAP50-95": 0.401,
#   "history": [
#     {
#       "epoch": 1,
#       "loss": 2.345,
#       "loss_bbox": 0.321,
#       "loss_giou": 0.654,
#       "loss_vfl": 0.712,
#       "loss_fgl": 0.889,
#       "precision": 0.51,
#       "recall": 0.47,
#       "mAP50": 0.28,
#       "mAP50-95": 0.14,
#       ...
#     },
#     ...
#   ],
# }
```

With the default `save_period=1`, checkpoints are saved after every epoch to
`runs/train/my_experiment/epoch{N}.pth`. Use a larger interval or `-1` to reduce
periodic checkpoint files; `last.pth` and `best.pth` remain available when
`save=True`.
Each checkpoint is a full nitid-wrapped `.pth` (config + names embedded) and can
be loaded directly with `DFINE("epoch50.pth")`.

The top-level values are the final epoch summary for backward compatibility.
Use `metrics["history"]` to inspect per-epoch training and validation metrics,
including `mAP50` and `mAP50-95`, from within Python.

### Trainer callbacks

Training accepts an optional `callbacks=` argument for extending trainer
behavior without patching the core loop. Pass either:

- an object with one or more lifecycle-hook methods, each taking `trainer`
- a mapping from hook name to callable or list of callables, each taking `trainer`

You can also register persistent callbacks on the model in an Ultralytics-like
way:

```python
def stop_after_first_epoch(trainer):
    if trainer.current_epoch == 1:
        trainer.stop = True

model.add_callback("on_train_epoch_end", stop_after_first_epoch)
```

Callbacks receive only the active `DFINETrainer` instance, for consistency
with the Ultralytics style. The most useful callback attributes are:

| Attribute | Type | Meaning | Available |
|-----------|------|---------|-----------|
| `trainer.stop` | `bool` | Set to `True` to request a clean stop | all hooks |
| `trainer.current_epoch` | `int` | 1-based epoch number currently in progress | epoch / val / end hooks |
| `trainer.current_val_metrics` | `dict \| None` | Latest validation metrics | `on_val_end`, `on_train_epoch_end`, `on_train_end` |
| `trainer.current_row` | `dict \| None` | Latest finalized per-epoch metrics row | `on_train_epoch_end`, `on_train_end` |
| `trainer.current_fitness` | `float` | Latest epoch fitness | `on_train_epoch_end`, `on_train_end` |
| `trainer.metrics` | `dict \| None` | Final return value from training | `on_train_end` |
| `trainer.error` | `BaseException \| None` | Exception that interrupted training | `on_train_error` |
| `trainer.history` | `list[dict]` | Live per-epoch history accumulated so far | all hooks |
| `trainer.tracking_state` | `dict[str, object]` | Tracker metadata persisted in `last.pth` | all hooks |
| `trainer.save_dir` | `Path \| None` | Run artifact directory | all hooks |
| `trainer.results_path` | `Path \| None` | CSV metrics file path | all hooks |
| `trainer.optimizer` / `trainer.scheduler` / `trainer.criterion` / `trainer.scaler` / `trainer.ema_model` | runtime objects | Active training components | all hooks after setup |
| `trainer.dataloader` | dataloader | Active train dataloader | all hooks after setup |
| `trainer.train_args` | `dict[str, object]` | Resolved train arguments for this run | all hooks |
| `trainer.start_epoch` | `int` | Resume start epoch | all hooks |

These are live objects, not deep-copied snapshots. Reading them is safe and
expected. Mutating `history`, `optimizer`, `scheduler`, or similar attributes
changes the active training run.

Supported hooks:

- `on_train_start`
- `on_train_epoch_start`
- `on_val_end`
- `on_train_epoch_end`
- `on_train_end`
- `on_train_error`

`on_train_error` runs when training exits with an exception. Cleanup callbacks
should use this hook; the original training exception is always re-raised.

### Weights & Biases

Install the optional integration dependency:

```bash
pip install "nitid[wandb]"
# or, from a source checkout
uv sync --extra wandb
```

Enable logging directly from `train()`, in the same style as Ultralytics:

```python
from dfine import DFINE

model = DFINE("dfine_s.pth")
model.train(
    data="data.yaml",
    epochs=50,
    project="runs/train",
    name="dfine-s-baseline",
    wandb=True,
)
```

`wandb=True` uses the WandB project `nitid` and uses the training `name` as the
WandB run name. For additional WandB configuration, pass a mapping instead:

```python
model.train(
    data="data.yaml",
    epochs=50,
    name="dfine-s-baseline",
    wandb={
        "project": "nitid-detection",
        "entity": "my-team",
        "tags": ["dfine-s", "coco"],
    },
)
```

The integration records the resolved training hyperparameters at run start and
logs the complete per-epoch row (total and component losses, learning rate,
validation metrics, timing, and memory). At successful completion it uploads
`last.pth` and `best.pth` once as versioned model artifacts and writes final
scalar metrics to the run summary. Set `log_checkpoints=False` if checkpoint
artifacts are not needed.

Periodic epoch artifacts are opt-in to avoid uploading several full model files
after every epoch:

```python
model.train(
    data="data.yaml",
    wandb={"checkpoint_interval": 10},  # also upload epoch10.pth, epoch20.pth, ...
)
```

The WandB run ID is stored in `last.pth`. Calling `train(resume=True,
wandb=True)` reconnects to that run with `resume="allow"`, keeping the metrics
in one continuous WandB run. If training fails, nitid finishes the WandB run
with a failed exit status before re-raising the original exception.

For local testing without an account or network connection, use offline mode:

```python
model.train(
    data="data.yaml",
    epochs=1,
    wandb={"project": "nitid-local", "mode": "offline"},
)
```

Offline runs are stored in the local `wandb/` directory and can be uploaded
later with `wandb sync`.

The callback API remains available when direct lifecycle control is useful:

```python
from dfine.integrations import WandbCallback

tracker = WandbCallback(project="nitid-detection")
model.train(data="data.yaml", callbacks=tracker)
```

### MLflow

Install the optional dependency:

```bash
pip install "nitid[mlflow]"
# or, from a source checkout
uv sync --extra mlflow
```

Enable MLflow directly on training:

```python
from dfine import DFINE

model = DFINE("dfine_s.pth")
model.train(
    data="data.yaml",
    epochs=50,
    project="runs/train",
    name="dfine-s-baseline",
    mlflow=True,
)
```

This follows the Ultralytics MLflow conventions:

- the tracking URI defaults to `runs/mlflow`
- the experiment defaults to the training `project`
- the MLflow run name defaults to the training `name`
- an already-active MLflow run is reused and is not closed by nitid
- resolved training parameters are logged when training starts
- losses, learning rate, validation metrics, timing, and memory are logged each epoch
- checkpoints, CSV results, YAML files, and generated plots are logged at training end
- initialization and logging failures warn and disable tracking instead of stopping training

The same environment variables supported by Ultralytics take precedence over
the defaults and Python options:

| Variable | Purpose |
|----------|---------|
| `MLFLOW_TRACKING_URI` | Local store or remote tracking-server URI |
| `MLFLOW_EXPERIMENT_NAME` | Experiment name |
| `MLFLOW_RUN` | Run name |
| `MLFLOW_KEEP_RUN_ACTIVE` | Keep a nitid-created run open when set to `1`, `true`, `yes`, `on`, `y`, or `t` (case-insensitive) |

For a fully local workflow, no server is required:

```python
model.train(data="data.yaml", epochs=2, mlflow=True)
```

Inspect those results through the MLflow UI:

```bash
mlflow server --backend-store-uri runs/mlflow
```

Then open `http://127.0.0.1:5000`. To use a different local store without
environment variables, pass an options mapping:

```python
model.train(
    data="data.yaml",
    mlflow={
        "tracking_uri": "runs/custom-mlflow",
        "experiment_name": "nitid-detection",
        "run_name": "dfine-s-baseline",
        "keep_run_active": False,
        "autolog": True,
    },
)
```

The MLflow run ID is persisted in `last.pth`, so `resume=True, mlflow=True`
continues the same run. A training exception marks a nitid-created run as
failed. Advanced users may also pass `MLflowCallback` through `callbacks=`.

Example using a mapping:

```python
def log_to_tracker(trainer):
    row = trainer.current_row
    if row is None:
        return
    tracker.log(
        {
            "epoch": row["epoch"],
            "loss": row["loss"],
            "mAP50": row["mAP50"],
            "mAP50-95": row["mAP50-95"],
        }
    )

model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=50,
    callbacks={"on_train_epoch_end": log_to_tracker},
)
```

To stop training from a callback, set `trainer.stop = True`. The trainer checks
this flag at safe lifecycle boundaries and finalizes the run cleanly.

### CLI

```bash
uv run dfine train \
    model=dfine_l \
    data=configs/datasets/my_dataset.yml \
    epochs=50 \
    batch=16
```

### Hyperparameter reference

The table below documents the full public `model.train(...)` surface as it
exists today. Defaults match [`DFINE.train()`](https://github.com/Vaelsys/nitid/blob/develop/dfine/model.py).
Use this table as the authoritative reference for train-time arguments.

| Parameter | Type | Default | Valid range / values | Description |
|-----------|------|---------|----------------------|-------------|
| `data` | `str` | required | path to a dataset YAML | Ultralytics-style dataset config describing `path`, split locations, class count, and names. |
| `epochs` | `int` | `50` | `>= 1` | Number of full passes over the training set. |
| `imgsz` | `int` | `640` | `>= 1` | Square training resolution applied during preprocessing. |
| `batch` | `int` | `16` | positive integer | Images per batch. Choose this explicitly for the available device memory. |
| `lr0` | `float` | `1e-4` | `> 0` | Initial learning rate passed to the optimizer. |
| `lrf` | `float` | `0.01` | `> 0` | Final learning-rate multiplier. Both linear decay and cosine decay end at `lr0 * lrf`. |
| `cos_lr` | `bool` | `False` | `True`, `False` | Switches the main schedule from linear decay to cosine decay. |
| `warmup_epochs` | `float` | `0.0` | `>= 0` | Number of warmup epochs before the main LR schedule begins. Fractional values are allowed. |
| `warmup_momentum` | `float` | `0.8` | typically `0 <= x <= 1` | Starting momentum or Adam/AdamW beta1 used during warmup. It linearly ramps to the optimizer's target value. |
| `warmup_bias_lr` | `float` | `0.1` | `>= 0` | Starting learning rate for bias parameters during warmup. Non-bias parameters warm up from `0.0`. |
| `optimizer` | `str` | `"AdamW"` | `"Auto"`, `"Adam"`, `"AdamW"`, `"SGD"`, `"RAdam"`, `"NAdam"`, `"RMSprop"` | Optimizer choice. The predictable `Auto` policy always selects AdamW. |
| `momentum` | `float` | `0.9` | usually `0 <= x < 1` | SGD momentum or beta1 for Adam-family optimizers. |
| `weight_decay` | `float` | `1e-4` | `>= 0` | Weight decay applied to non-bias parameters. |
| `clip_grad` | `float` | `0.1` | `>= 0` | Maximum gradient norm; `0` disables clipping. |
| `resume` | `bool` | `False` | `True`, `False` | Restore the latest run state from `project/name/last.pth`. |
| `amp` | `bool` | `False` | `True`, `False` | Enables mixed-precision training through `torch.amp.autocast` and `GradScaler` on CUDA devices. |
| `ema` | `bool` | `False` | `True`, `False` | Maintains an exponential moving average copy of the model and saves EMA weights in checkpoints. |
| `ema_decay` | `float` | `0.9999` | usually `0 < x < 1` | Target EMA smoothing factor. The effective decay ramps over the first 1,000 optimizer updates so newly initialized custom heads are not stale during early validation. |
| `device` | `str \| None` | `None` | e.g. `"cpu"`, `"cuda"`, `"cuda:0"` | Optional override for the training device. If omitted, training uses the device selected when the `DFINE` object was created. |
| `project` | `str` | `"runs/train"` | any writable path | Root directory for run artifacts such as checkpoints and metrics. |
| `name` | `str` | `"exp"` | any filesystem-friendly name | Run subdirectory created under `project`. |
| `save_dir` | `str \| Path \| None` | `None` | writable directory | Exact requested run directory; non-resume training still increments if it exists. |
| `exist_ok` | `bool` | `False` | `True`, `False` | Existing training directories are reused only by `resume=True`. |
| `patience` | `int` | `100` | `>= 0` | Stop after this many validated epochs without improvement; `0` disables early stopping. Final metrics report `best_epoch`. |
| `save` | `bool` | `True` | `True`, `False` | Save last, best, and enabled periodic checkpoints. |
| `save_period` | `int` | `1` | `-1` or `>= 1` | Save `epochN.pth` every N epochs; `-1` disables periodic files. |
| `val` | `bool` | `True` | `True`, `False` | Enable validation during training. |
| `plots` | `bool` | `True` | `True`, `False` | Save training-history and validation plots. |
| `val_period` | `int` | `1` | `>= 1` | Validate every N epochs and at the final/time-limited epoch. |
| `workers` | `int` | `0` | `>= 0` | DataLoader worker processes. |
| `cache` | `bool \| str` | `False` | `False`, `True`, `"ram"` | Cache decoded training images in memory. |
| `seed` | `int` | `0` | any integer | Seed Python, NumPy, PyTorch, dataset sampling, and DataLoader generators. |
| `deterministic` | `bool` | `True` | `True`, `False` | Request deterministic PyTorch/cuDNN behavior where available. |
| `freeze` | `int \| str \| list \| None` | `None` | layer count, stage, substring, or glob | Freeze matching parameters before optimizer creation. |
| `classes` | `list[int] \| None` | `None` | valid class IDs | Keep annotations only for selected classes. |
| `single_cls` | `bool` | `False` | `True`, `False` | Remap all retained annotations to class 0. |
| `fraction` | `float` | `1.0` | `(0, 1]` | Deterministically sample this fraction of training images. |
| `accumulate` | `int` | `1` | `>= 1` | Accumulate gradients across batches before optimizer and EMA steps. |
| `multi_scale` | `bool` | `False` | `True`, `False` | Randomly resize batches from roughly 0.5× to 1.5× `imgsz`, in multiples of 32. |
| `augment` | `bool` | `True` | `True`, `False` | Enable box-aware training augmentation. Letterbox preprocessing remains active when disabled. |
| `fliplr` | `float` | `0.5` | `[0, 1]` | Probability of a horizontal flip. |
| `scale` | `float` | `0.5` | `[0, 1)` | Maximum random isotropic scale gain. |
| `translate` | `float` | `0.1` | `[0, 1]` | Maximum translation as a fraction of image width/height. |
| `crop` | `float` | `0.0` | `[0, 1]` | Crop probability and maximum fraction sampled independently from each edge. |
| `hsv_h` | `float` | `0.015` | `[0, 0.5]` | Hue jitter gain. |
| `hsv_s` | `float` | `0.7` | `[0, 1]` | Saturation jitter gain. |
| `hsv_v` | `float` | `0.4` | `[0, 1]` | Brightness/value jitter gain. |
| `mosaic` | `float` | `0.0` | `[0, 1]` | Mosaic probability. Experimental and disabled until a D-FINE benchmark demonstrates a gain. |
| `mixup` | `float` | `0.0` | `[0, 1]` | MixUp probability. Experimental and disabled until benchmarked. |
| `close_mosaic` | `int` | `10` | `>= 0` | Disable mosaic for the final N epochs; zero keeps it active. |
| `time` | `float \| None` | `None` | positive hours or `None` | Training duration in hours. When supplied, this overrides `epochs` as the loop's stopping limit. |
| `verbose` | `bool` | `True` | `True`, `False` | Enables per-epoch console logging during training. |
| `callbacks` | `object \| dict \| None` | `None` | callback object or hook mapping | Optional lifecycle hooks for custom logging, experiment tracking, or other training-time integrations. |
| `wandb` | `bool \| dict` | `False` | `True`, `False`, or WandB options | Enables the optional Weights & Biases integration. |
| `mlflow` | `bool \| dict` | `False` | `True`, `False`, or MLflow options | Enables the optional Ultralytics-style MLflow integration. |

Images are first resized with aspect ratio preserved and padded to `imgsz`. Every
geometric transform operates on absolute `xyxy` boxes, clips them to the visible
image, removes empty boxes and their labels, and only then converts targets to the
normalized `cxcywh` format expected by D-FINE. The sample/epoch seed makes transform
choices independent of DataLoader worker scheduling. All resolved values above are
written to `args.yaml` and restored from `last.pth` on resume.

Mosaic and MixUp are implemented as opt-in benchmark candidates rather than
selected defaults. Validate them against the unaugmented baseline on the target
dataset before enabling them by default.

## Resume training

nitid saves a `last.pth` checkpoint after every epoch. That checkpoint now
contains the full training state needed to continue an interrupted run:

- model weights used for training
- optimizer state
- scheduler state
- AMP scaler state when AMP is enabled
- EMA weights and decay when EMA is enabled
- per-epoch metrics history

To resume, keep the same `project` and `name` and set `resume=True`:

```python
metrics = model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=100,            # new total target epoch count
    resume=True,
    project="runs/train",
    name="my_experiment",
)
```

nitid restores state from `runs/train/my_experiment/last.pth` and continues at
the next epoch. All data, optimization, stopping, reproducibility, freezing,
filtering, accumulation, multi-scale, validation, plotting, and saving controls
in the table above are restored; conflicting values produce a warning.
`epochs` is deliberately controlled by the new invocation so a run can be
extended. Run location, verbosity, callbacks, and tracker enablement also
belong to the new invocation. If the checkpoint already reached or
exceeded the requested `epochs`, training does not run again and the saved
history is returned.

## Cosine LR with warmup

By default, nitid uses linear LR decay. To match the more common Ultralytics
fine-tuning setup, enable cosine decay and a warmup phase:

```python
metrics = model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=50,
    lr0=1e-4,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
)
```

With this configuration, nitid warms up per batch during the first
`warmup_epochs` epochs, using `warmup_bias_lr` for bias parameters and
`0.0` for non-bias parameters, while momentum or Adam/AdamW beta1 ramps from
`warmup_momentum` to the optimizer's target value. After warmup, cosine decay
reduces the learning rate down to `lr0 * lrf`.

### Interaction notes

- `amp=True` is only active on CUDA. On CPU, nitid logs a warning and
  continues in FP32.
- `ema_decay` only matters when `ema=True`.
- `warmup_epochs=0` disables warmup entirely.
- `warmup_momentum` affects SGD momentum and Adam/AdamW beta1 during warmup.
- `device` in `train()` overrides the device selected in `DFINE(...)` for that
  training run only.
- `momentum`, `weight_decay`, and `clip_grad` explicitly control optimizer
  momentum/beta1, non-bias regularization, and maximum gradient norm.

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
    project="runs/val",
    name="exp",
    plots=True,
    verbose=True,
)
print(metrics)
# {
#   "mAP50-95": 0.412,   # AP averaged over IoU 0.50:0.05:0.95
#   "mAP50":    0.623,   # AP at IoU=0.50
#   "AR1":      0.341,   # Average Recall at max 1 detection per image
#   "AR100":    0.512,   # Average Recall at max 100 detections per image
#   "precision": 0.701,
#   "recall":    0.655,
#   "f1":        0.677,
#   "per_class": [...],
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

- The trainable model is **not** put into deploy mode during training or
  inference. `predict()` and `export()` use a separate deployed copy, and that
  cache is rebuilt automatically after training or weight loading.
- Loss weighting (`weight_dict`) comes from the checkpoint's embedded D-FINE
  config so it stays consistent with the original pre-training setup.
- When `ema=True` the saved checkpoint contains EMA weights. Loading it with
  `DFINE(path)` gives you the EMA model directly — no extra step needed.
- AMP is only active on CUDA; on CPU it degrades gracefully to full precision.
