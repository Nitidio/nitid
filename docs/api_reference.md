# API Reference

## Run output behavior

Artifact-producing calls allocate unique directories by default: `exp`, `exp2`,
`exp3`, and so on. Pass `save_dir` to choose the requested directory directly and
`exist_ok=True` to deliberately reuse it for prediction, tracking, validation, or export.
Training never reuses an existing directory unless `resume=True`; non-resume
training increments even when `exist_ok=True`. Every run stores `args.yaml` and
`environment.yaml` alongside its artifacts.

Exports default to `runs/export/exp/dfine_640.onnx` (with the appropriate format
suffix). Use `output="path/model.onnx"` for an exact artifact path. Checkpoints and
export artifacts are published atomically.

## Python bug reports

Use the public `bugreport()` context manager to capture Python API operations in
the same single-file format as the CLI's `--report` flag:

```python
from nitid import NITID, bugreport

with bugreport("training") as report:
    model = NITID("nitid1s", task="detect")
    model.train(data="data.yaml")

print(report.path)
```

Put model construction inside the context when download, checkpoint, device, or
import failures also need to be captured. The context tees stdout and stderr to
the terminal and report, includes the shared `environment.yaml` snapshot, and
always prints the saved path. If an exception occurs, its traceback is appended
to the report and the original exception is re-raised unchanged.

One report can cover multiple operations:

```python
with bugreport("full-experiment") as report:
    model = NITID("nitid1s", task="detect")
    model.train(data="data.yaml")
    model.val(data="data.yaml")
    model.export(format="onnx")
```

By default reports are written to `runs/bugreports`. Choose another directory
with `bugreport("training", report_dir="reports")`.

## `NITID`

```python
from nitid import NITID
```

The recommended public class. Instantiate with a path to a nitid-wrapped `.pth`
checkpoint, or use a registry model name and task.

```python
detector = NITID("nitid1s", task="detect", device="cuda:0")
segmenter = NITID("nitid1s", task="segment", device="cuda:0")
semantic = NITID("nitid1s", task="semantic", device="cuda:0")
pose = NITID("nitid1s", task="pose", device="cuda:0")
obb = NITID("nitid1s", task="obb", device="cuda:0")
```

| Argument  | Type  | Default | Description |
|-----------|-------|---------|-------------|
| `model`   | `str \| Path` | `"nitid1l"` | Checkpoint path or registry model name such as `nitid1n`, `nitid1s`, `nitid1m`, `nitid1l`, or `nitid1x` |
| `task` | `str` | `"detect"` | `"detect"`, `"segment"`, `"semantic"` (`"sem_seg"` alias), `"pose"`, or `"obb"`. Must match an explicit checkpoint's embedded task. |
| `weights` | `str` | `"default"` | Official weight variant for the selected model/task. Do not combine a non-default value with a checkpoint path. |
| `backend` | `str` | `"torch"` | `"torch"` (PyTorch, CPU/CUDA) or `"openvino"` (OpenVINO Runtime, CPU/Intel iGPU/Intel NPU). Changes what `device` means — see below. |
| `device`  | `str \| int \| None` | `None` | For `backend="torch"`: a PyTorch device selector (`"cuda"`, `"cpu"`, `"cuda:N"`); omit it to auto-select `"cuda:0"` when available, otherwise `"cpu"`. For `backend="openvino"`: an OpenVINO device string (`"CPU"`, `"GPU"` for Intel integrated GPU, `"NPU"`), or `"auto"`/omit to prefer NPU, then GPU, then CPU. |
| `verbose` | `bool`| `True`  | Print load summary |

If you pass an explicit device string, it is used as-is after normalization. Omitting `device` gives the Ultralytics-style smart default.

`DFINE(...)` remains available for existing code and D-FINE-family model names,
but new code should prefer `NITID(...)`.

### Intel NPU / integrated GPU inference (OpenVINO backend)

```python
from nitid import NITID

model = NITID("nitid1s", backend="openvino", device="NPU")   # Intel NPU
model = NITID("nitid1s", backend="openvino", device="GPU")   # Intel integrated GPU
model = NITID("nitid1s", backend="openvino", device="AUTO")  # prefer NPU, then GPU, then CPU
results = model.predict("image.jpg", conf=0.5)
```

Requires the OpenVINO extra: `uv sync --extra openvino`. Under the hood, the
deployed model is traced to a raw (non-postprocessed) OpenVINO IR and compiled
for the requested device the first time `predict()`/`track()` is called at a
given `imgsz`; the compiled model is cached per `imgsz` on the instance, so
only the first call at a new `imgsz` pays the compile cost. As with the torch
backend, `imgsz` must match the checkpoint's `eval_spatial_size` (usually
`640`) for tasks other than `"obb"` — the decoder's anchors are static for a
given spatial size.

`backend="openvino"` is inference-only: `train()` and `val()` raise a clear
error, and every other method (`export()`, `info()`) is unaffected by it —
they keep running on PyTorch/CPU regardless of which backend `predict()`
uses. Requesting a `device` OpenVINO doesn't report as available (check
`dfine.nn.openvino_runtime.list_openvino_devices()`) raises a `ValueError`
naming what's actually available, instead of an unexplained backend error.

Note the naming overlap: this constructor-level `backend` selects the
*compute* backend (PyTorch vs. OpenVINO). It is unrelated to `predict()`'s
own `backend` argument below, which selects the *video source* backend
(`"opencv"` vs. `"gstreamer"`) — the two are independent and can be combined
freely, e.g. `NITID(..., backend="openvino", device="NPU").predict(source, backend="gstreamer")`.

---

### `predict()`

Run inference on any source.

```python
results = model.predict(
    source,           # path, dir, URL, ndarray, int (webcam), or list
    conf=0.5,         # confidence threshold
    mask_threshold=0.5, # segment: mask probability threshold
    imgsz=640,        # inference size (square)
    classes=None,     # filter to these class indices, e.g. [0, 2]
    stream=False,     # return generator instead of list
    vid_stride=1,     # process every Nth frame for video/webcam/stream sources
    augment=False,    # run test-time augmentation (horizontal flip)
    save=False,       # save annotated outputs to project/name
    project=None,     # defaults to runs/detect or runs/segment
    name="exp",
    backend="opencv", # or "gstreamer" for video/live sources
    gst_pipeline=None, # optional explicit GStreamer pipeline
    rtsp_latency=200,
    rtsp_transport="tcp",
    rtsp_username=None,
    rtsp_password=None, # Python only; CLI reads passwords from an environment variable
    iou=0.85,          # IoU threshold for TTA NMS
    sink=None,         # optional FrameSink receiving annotated frames
    return_probs=False, # semantic only: retain float [C, H, W] probabilities
)
```

Returns `list[Results]` (or a generator when `stream=True`).

`predict()` preserves the trainable model. Internally, nitid creates and caches
a separate deployed inference copy, so workflows such as `predict()` → `train()`
continue training the original full architecture.

Semantic prediction uses per-pixel argmax, so `conf` and `mask_threshold` do
not alter its output. `classes` filtering is rejected because removing dense
classes would leave undefined pixels.

For video, webcam, and stream sources, `vid_stride=N` keeps every Nth frame in
source order while skipping the intermediate frames.

When `save=True`, nitid also writes annotated outputs to `project/name` while
still returning the normal `Results` objects:

```python
results = model.predict("image.jpg", save=True)
# saved image: runs/detect/exp/image.jpg

results = model.predict("image.jpg", save=True, project="runs/detect", name="street-test")
# saved image: runs/detect/street-test/image.jpg

results = model.predict("video.mp4", save=True, vid_stride=2)
# saved video: runs/detect/exp/video.mp4
```

For video-file sources, nitid saves an annotated `.mp4`. When `vid_stride=N`,
the saved video contains the processed frames and its FPS is reduced by `N` so
playback stays close to the original duration.

#### `Results`

| Attribute  | Type            | Description |
|------------|-----------------|-------------|
| `orig_img` | `np.ndarray`    | Original image HWC BGR |
| `path`     | `str`           | Source path or descriptor |
| `names`    | `dict[int,str]` | Class index → name |
| `boxes`    | `Boxes \| None` | Detection boxes |
| `obb`      | `OBB \| None` | Oriented boxes for `task="obb"` |
| `masks`    | `Masks \| None` | Full-resolution instance masks for `task="segment"` |
| `keypoints` | `Keypoints \| None` | Per-instance keypoints for `task="pose"` |
| `semantic` | `SemanticMask \| None` | Original-resolution class map for `task="semantic"`; alias of `semantic_mask` |
| `semantic_save_path` | `str \| None` | Lossless class-ID PNG written under `masks/` when semantic prediction uses `save=True` |
| `save_path` | `str \| None`  | Saved annotated image or video path when `save=True` |
| `speed` | `dict[str, float]` | Timing in milliseconds for `preprocess`, `inference`, and `postprocess` |
| `frame_metadata` | `FrameMetadata \| None` | Source ID, zero-based frame index, timestamp, FPS, stride, and discontinuity flag |

```python
r = results[0]
r.plot()            # → HWC BGR ndarray with boxes drawn
r.save("out.jpg")   # write plotted image to disk
r.show()            # display in a window (blocks until key press)
r.to_json()         # → list[dict] with box/score/class per detection
r.pandas()          # → pandas.DataFrame with xyxy/conf/class/name columns
r.to_df()           # → same DataFrame as r.pandas()
r.to_csv("out.csv") # write detections to CSV
r.speed             # → {"preprocess": 4.2, "inference": 18.7, "postprocess": 2.1}
r.save_json("predictions.json")  # write detections to disk
r.save_txt("predictions.txt")  # write YOLO-format labels to disk
r.crop()            # → list[dict] with cropped object images and metadata
r.crop(save_dir="crops")  # save crops into class-name folders
r.masks.data       # uint8 [N, H, W], aligned with r.boxes
r.masks.xy         # absolute polygon coordinates
r.masks.xyn        # normalized polygon coordinates
r.keypoints.xy     # float [N, K, 2], pose only
r.keypoints.conf   # optional float [N, K], pose only
r.obb.xywhr        # float [N, 5], OBB only: cx, cy, w, h, angle
r.obb.xyxyxyxy     # float [N, 8], OBB only: four polygon corners
r.obb.conf         # float [N], OBB confidence
r.obb.cls          # int [N], OBB class IDs
len(r)              # number of detections
```

Semantic results retain full probabilities only when requested:

```python
r = semantic.predict("image.jpg", return_probs=True)[0]
r.semantic.mask             # int64 [H, W]
r.semantic.probs            # float [C, H, W]
r.semantic.colorize()       # HWC BGR preview
r.save_semantic("ids.png") # lossless class-ID map
r.save_semantic("color.png", colorize=True)
```

`crop()` returns one dictionary per detection:

```python
crop = r.crop()[0]
crop["im"]          # cropped HWC BGR ndarray
crop["box"]         # {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
crop["confidence"]  # detection confidence
crop["class"]       # class id
crop["name"]        # class name
crop["save_path"]   # saved image path, or None when not saving
crop.get("track_id") # persistent ID for tracked results
```

When `save_dir` is provided, crops are written in a YOLO-like class-folder layout:

```text
crops/
  person/
    image.jpg
  bus/
    image.jpg
```

`save_txt()` writes one detection per line:

```text
class_id x_center y_center width height
```

For detection, box values are normalized from `0` to `1`. For instance
segmentation, each line contains the class followed by normalized polygon
coordinates. For OBB, each line contains class plus normalized four-corner
YOLO-OBB coordinates. Use `save_conf=True` to append the confidence score:

```python
r.save_txt("predictions.txt", save_conf=True)
```

The tabular export helpers use these columns:

`x1`, `y1`, `x2`, `y2`, `confidence`, `class`, `name`

Tracked results append a `track_id` column. JSON and crop dictionaries also
include `track_id`, and `save_txt()` appends it to each tracked row.

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
| `id`     | `[N] \| None` | Persistent IDs for tracking results; `None` for detections |
| `is_track` | `bool` | Whether the container holds tracking data |
| `data`   | `[N, 6]` or `[N, 7]` | Detection layout: xyxy + conf + cls. Tracking layout: xyxy + track ID + conf + cls. |

Iterate detections by index:

```python
for i in range(len(results[0].boxes)):
    x1, y1, x2, y2 = results[0].boxes.xyxy[i].tolist()
    conf = results[0].boxes.conf[i].item()
    cls  = int(results[0].boxes.cls[i].item())
```

#### `Masks`

| Property | Type | Description |
|----------|------|-------------|
| `data` | tensor `[N,H,W]` | Binary full-resolution instance masks |
| `xy` | `list[np.ndarray]` | Largest external contour per instance in pixels |
| `xyn` | `list[np.ndarray]` | Contours normalized to `[0,1]` |

---

### `track()`

Run D-FINE detection followed by ByteTrack, BoT-SORT, or OC-SORT association.
ByteTrack is the default. Install the optional dependency first:

```bash
uv sync --extra track
```

```python
results = model.track(
    source,
    conf=0.1,
    imgsz=640,
    classes=None,
    stream=False,
    vid_stride=1,
    augment=False,
    save=False,
    project="runs/track",
    name="exp",
    save_dir=None,
    exist_ok=False,
    verbose=True,
    backend="opencv",
    gst_pipeline=None,
    rtsp_latency=200,
    rtsp_transport="tcp",
    rtsp_username=None,
    rtsp_password=None, # Python only; CLI uses rtsp_password_env
    iou=0.85,
    tracker="bytetrack",
    tracker_kwargs=None,
    sink=None,
)
```

The return type matches `predict()`: `list[Results]`, or a generator when
`stream=True`. Tracking results use seven-column `Boxes` data and expose IDs
through `result.boxes.id`:

```python
for result in model.track("video.mp4", conf=0.5, stream=True):
    boxes = result.boxes.xyxy
    track_ids = result.boxes.id
```

Use streaming for long videos and live sources. With `save=True`, tracking is
applied before rendering, so the annotated MP4 contains persistent IDs:

```python
for result in model.track("video.mp4", conf=0.5, stream=True, save=True):
    output_path = result.save_path

# runs/track/exp/video.mp4
```

`conf` is the D-FINE detection filter. Select `tracker="bytetrack"`,
`tracker="botsort"`, or `tracker="ocsort"`; tracker-specific settings belong
in `tracker_kwargs`:

```python
results = model.track(
    "video.mp4",
    conf=0.5,
    tracker_kwargs={
        "track_activation_threshold": 0.4,
        "high_conf_det_threshold": 0.6,
        "minimum_iou_threshold": 0.1,
        "minimum_consecutive_frames": 2,
        "lost_track_buffer": 60,
    },
)
```

OC-SORT uses the same result and lifecycle contracts:

```python
results = model.track(
    "video.mp4",
    conf=0.5,
    tracker="ocsort",
    tracker_kwargs={
        "high_conf_det_threshold": 0.6,
        "minimum_iou_threshold": 0.3,
        "minimum_consecutive_frames": 3,
        "lost_track_buffer": 30,
        "direction_consistency_weight": 0.2,
        "delta_t": 3,
    },
)
```

| ByteTrack option | Default | Description |
|---|---:|---|
| `track_activation_threshold` | `0.25` | Minimum score for starting a candidate track. |
| `high_conf_det_threshold` | `0.6` | High-score cutoff used during association. |
| `minimum_iou_threshold` | `0.1` | Minimum IoU used to associate detections and tracks. |
| `minimum_consecutive_frames` | `1` | Consecutive observations required to confirm a track. |
| `lost_track_buffer` | `30` | Number of processed frames for which a lost track is retained. |
| `frame_rate` | automatic | Override the effective FPS used by ByteTrack. By default nitid uses source FPS divided by `vid_stride`. |

BoT-SORT uses the original frame for camera-motion compensation. The current
integration is motion-only and does not use appearance embeddings or a ReID
model.

```python
for result in model.track(
    "video.mp4",
    tracker="botsort",
    tracker_kwargs={"enable_cmc": True, "cmc_method": "sparseOptFlow"},
    stream=True,
):
    track_ids = result.boxes.id
```

| BoT-SORT option | Default | Description |
|---|---:|---|
| `track_activation_threshold` | `0.7` | Minimum score for starting a candidate track. |
| `high_conf_det_threshold` | `0.6` | High-score cutoff used during association. |
| `minimum_consecutive_frames` | `2` | Consecutive observations required to confirm a track. |
| `lost_track_buffer` | `30` | Number of processed frames for which a lost track is retained. |
| `minimum_iou_threshold_first_assoc` | `0.2` | Minimum IoU for first-stage association. |
| `minimum_iou_threshold_second_assoc` | `0.5` | Minimum IoU for low-confidence recovery. |
| `minimum_iou_threshold_unconfirmed_assoc` | `0.3` | Minimum IoU for unconfirmed-track association. |
| `enable_cmc` | `True` | Enable camera-motion compensation. |
| `cmc_method` | `"sparseOptFlow"` | CMC method: `orb`, `sift`, `sparseOptFlow`, or `ecc`. |
| `cmc_downscale` | `2` | Downscale factor used during CMC estimation. |
| `instant_first_frame_activation` | `True` | Immediately confirm tracks created on the first frame. |
| `frame_rate` | automatic | Override the effective FPS derived from source metadata. |

| OC-SORT option | Default | Description |
|---|---:|---|
| `high_conf_det_threshold` | `0.6` | Minimum detection score used for association. |
| `minimum_iou_threshold` | `0.3` | Minimum IoU used to associate detections and tracks. |
| `minimum_consecutive_frames` | `3` | Consecutive observations required to confirm a track. |
| `lost_track_buffer` | `30` | Number of processed frames for which a lost track is retained. |
| `direction_consistency_weight` | `0.2` | Weight given to motion-direction consistency during association. |
| `delta_t` | `3` | Frame interval used to estimate an object's direction. |
| `frame_rate` | automatic | Override the effective FPS. By default nitid uses source FPS divided by `vid_stride`. |

New candidates may temporarily have ID `-1` until the tracker confirms them.
Non-negative IDs are persistent track identities.

Tracker state belongs to one `model.track()` invocation. It resets when:

- a new `model.track()` call starts;
- the source ID changes, such as when a source list advances to another video;
- `FrameMetadata.discontinuity` is true, which a custom `FrameSource` can set
  to prevent identities from leaking across a gap in its frames.

Passing a custom `FrameSink` through `sink=` writes annotated tracked frames
and closes the sink when iteration finishes or the generator is closed.

#### Media contracts

The public media types can be imported directly:

```python
from nitid import (
    Frame,
    FrameMetadata,
    FrameSink,
    FrameSource,
    GStreamerVideoSink,
)
```

A `FrameSource` yields ordered BGR frames with stable metadata. Both
`predict()` and `track()` accept a `FrameSource` anywhere they accept a file or
camera source. A `FrameSink` receives annotated frames through `sink=`. Sources
and sinks have explicit `close()` methods and support context-manager use.

`GStreamerFrameSource` is the built-in GStreamer/live-stream implementation:

```python
from nitid import GStreamerFrameSource

source = GStreamerFrameSource("rtsp://camera/live", rtsp_transport="tcp")
for result in model.track(source, stream=True):
    ...
```

Alternatively, pass the same settings directly to `predict()` or `track()`
using `backend="gstreamer"`. See [GStreamer and RTSP](gstreamer.md).

Use `GStreamerVideoSink` to send annotated results to a file, segmented files,
an RTSP publishing endpoint, or a custom appsrc pipeline. The sink opens on its
first frame, so width, height, and source FPS do not need to be known upfront:

```python
sink = GStreamerVideoSink("runs/segments/camera", segment_duration=60)
for result in model.track("video.mp4", stream=True, sink=sink):
    ...
```

```python
sink = GStreamerVideoSink("rtsp://media-server/nitid")
for result in model.track("rtsp://camera/input", stream=True, sink=sink):
    ...
```

The predictor sends `result.plot()` to the sink after tracking, so output
frames contain persistent IDs. The sink is closed when inference finishes or
when the streaming generator is explicitly closed.

Use `inspect_gstreamer_capabilities()` or `nitid gstreamer-info` to check that
the active OpenCV build has GStreamer enabled before deployment.

---

### `train()`

Fine-tune on a custom dataset. See [fine_tuning.md](fine_tuning.md).

```python
metrics = model.train(
    data="configs/datasets/my_dataset.yml",
    recipe="default",    # default task-native training, or "deim" for detection
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=1e-4,
    lrf=0.01,
    optimizer="AdamW",   # Auto, Adam, AdamW, SGD, RAdam, NAdam, or RMSprop
    resume=False,        # resume from project/name/last.pth
    amp=False,           # FP16 mixed precision (CUDA only)
    ema=False,           # EMA weight averaging; decay ramps for 1,000 updates
    ema_decay=0.9999,
    device=None,         # override training device
    project="runs/train",
    name="exp",
    verbose=True,
    wandb=False,         # True or a mapping of WandB options
    mlflow=False,        # True or a mapping of MLflow options
)
# metrics = {
#   "loss": ...,
#   "fitness": ...,
#   "mAP50": ...,
#   "mask_mAP50-95": ...,  # segment task
#   "mask_mAP50": ...,     # segment task
#   "pose_mAP50-95": ...,  # pose task
#   "pose_mAP50": ...,     # pose task
#   "mAP50-95": ...,
#   "history": [{...}, ...],
# }
```

The top-level values summarize the final epoch. `metrics["history"]` contains
one row per epoch with training loss terms and validation metrics.

Training controls added to the public API:

| Argument | Type / default | Meaning |
|---|---|---|
| `recipe` | `str = "default"` | Training recipe. Use `"default"` for the task-native path or `"deim"` for detection-only DEIM training. |
| `batch` | recipe/task dependent | Explicit positive batch size. |
| `epochs`, `lr0`, `lrf` | recipe/task dependent | Omitted detection values follow the selected recipe and model size. Explicit values override the recipe. |
| `optimizer` | `str = "AdamW"` | Auto, Adam, AdamW, SGD, RAdam, NAdam, or RMSprop; Auto predictably selects AdamW. |
| `momentum` | `float = 0.9` | SGD momentum or Adam-family beta1. |
| `weight_decay` | `float = 1e-4` | Non-bias weight decay. |
| `clip_grad` | `float = 0.1` | Maximum gradient norm; zero disables clipping. |
| `patience` | `int = 100` | Validated epochs without improvement before stopping; zero disables. |
| `save` | `bool = True` | Save checkpoint artifacts. |
| `save_period` | `int = 1` | Periodic `epochN.pth` interval; `-1` disables periodic files. |
| `val` | `bool = True` | Enable validation during training. |
| `val_period` | `int = 1` | Validation interval in epochs. |
| `plots` | `bool = True` | Save training and validation plots. |
| `workers` | `int = 0` | DataLoader worker processes. |
| `cache` | `bool \| str = False` | Cache decoded images in RAM with `True`/`"ram"`. |
| `seed` | `int = 0` | Python, NumPy, PyTorch, sampling, and worker seed. |
| `deterministic` | `bool = True` | Request deterministic PyTorch behavior. |
| `freeze` | `int \| str \| list \| None = None` | Freeze early stages or matching parameter names/globs. |
| `classes` | `list[int] \| None = None` | Keep only selected training class IDs. |
| `single_cls` | `bool = False` | Remap retained targets to class zero. |
| `fraction` | `float = 1.0` | Deterministically use a fraction in `(0, 1]`. |
| `accumulate` | `int = 1` | Batches accumulated per optimizer step. |
| `multi_scale` | `bool = False` | Random per-batch resizing around `imgsz`. |
| `augment` | `bool = True` | Enable recipe-aware box transforms for detection and task-native transforms for other tasks. |
| `fliplr` | `float = 0.5` | Horizontal-flip probability. |
| `scale` | `float = 0.5` | Random isotropic scale gain. |
| `translate` | `float = 0.1` | Random translation gain. |
| `crop` | `float = 0.0` | Crop probability and maximum edge fraction. |
| `hsv_h`, `hsv_s`, `hsv_v` | `0.015`, `0.7`, `0.4` | Hue, saturation, and brightness jitter gains. |
| `mosaic`, `mixup` | `float = 0.0` (`mosaic=0.5` in DEIM when omitted) | Detection probabilities. `recipe="deim"` also enables recipe-controlled batch MixUp. Not supported for semantic, pose, or OBB. |
| `close_mosaic` | `int = 10` | Turn mosaic off for the final N epochs. |
| `time` | `float \| None = None` | Training duration in hours; when set, it overrides `epochs`. |
| `save_dir` | `str \| Path \| None = None` | Exact requested run directory. |
| `exist_ok` | `bool = False` | Training still reuses an existing directory only with resume. |

When `resume=True`, nitid restores the latest checkpoint from `project/name/last.pth`,
including optimizer, scheduler, EMA, AMP scaler, metrics history, tracker IDs,
and every control above. Conflicts warn and restore the saved value. `epochs`,
run location, verbosity, callbacks, and tracker enablement are deliberate
new-invocation overrides. Final metrics include `best_epoch`.

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

Export to ONNX, OpenVINO IR, TorchScript, or TensorRT. See [export.md](export.md).

```python
model.export(format="onnx")        # → dfine_640.onnx
model.export(format="openvino")    # → dfine_640.xml + dfine_640.bin
model.export(format="torchscript") # → dfine_640.torchscript
model.export(format="tensorrt")    # → dfine_640.engine  (requires tensorrt installation)
```

`export()` uses the same deployed-copy path as inference and does not mutate the
trainable model. Repeated exports reuse the valid deployed copy when possible;
training or loading different weights invalidates that cache automatically.

Semantic ONNX/OpenVINO exports have one output named `semantic_logits` with
shape `[B, C, H, W]`. Apply softmax and argmax in the consuming runtime. Pose
ONNX/OpenVINO exports return `(labels, boxes, scores, keypoints)`. OBB exports
return `(labels, boxes, scores)`, with OBB boxes shaped `[B, topk, 5]` in
`cx, cy, w, h, angle` format.

| Argument    | Default  | Description |
|-------------|----------|-------------|
| `format`    | `"onnx"` | `"onnx"`, `"openvino"`, `"torchscript"`, or `"tensorrt"` |
| `imgsz`     | 640      | Must match model's `eval_spatial_size` |
| `batch`     | 1        | Static batch size |
| `dynamic`   | `False`  | Dynamic batch axis (ONNX, OpenVINO, and TensorRT) |
| `simplify`  | `True`   | Simplify the intermediate ONNX graph (ONNX and OpenVINO) |
| `opset`     | 17       | ONNX opset version (ONNX and OpenVINO) |
| `half`      | `False`  | FP16 precision/weight compression (OpenVINO and TensorRT) |
| `device`    | `None`   | Override export device |
| `verbose`   | `True`   | Print export progress |

---

### Properties

```python
model.names   # {0: "person", 1: "bicycle", ...}  — class index → name
model.device  # "cpu" or "cuda:0"                 — device the model lives on
model.task    # "detect", "segment", "semantic", "pose", or "obb"
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
| File exists but has no embedded config | `KeyError`; use a supported model name, a nitid training checkpoint, or the maintainer conversion tool |
| `export(format=…)` with unsupported format | `ValueError` |
| `train(optimizer=…)` with unknown name | `ValueError` |

```python
try:
    model = NITID("my_model.pth", task="detect")
except FileNotFoundError:
    print("Checkpoint not found — check the path")
```
