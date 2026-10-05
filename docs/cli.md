# Command-line guide

Nitid's terminal command is named `nitid`. Commands use `key=value`
arguments:

```bash
uv run nitid predict model=model1s task=detect source=image.jpg conf=0.5
uv run nitid predict model=model1s task=segment source=image.jpg conf=0.5
uv run nitid predict model=semantic_best.pth task=semantic source=image.jpg save=true
```

## Before using the command

Complete the repository installation first:

```bash
git clone https://github.com/Nitidio/nitid.git
cd nitid
uv sync
```

Run commands from the repository root, where `pyproject.toml` is located.

## Option 1: use `uv run` (recommended)

You do not need to activate the virtual environment:

```bash
uv run nitid --help
uv run nitid download --help
uv run nitid predict --help
uv run nitid track --help
```

`uv run` finds the project environment and runs the installed `nitid` command
inside it.

## Option 2: activate the virtual environment

On Linux or macOS:

```bash
source .venv/bin/activate
```

The shorter commands will then work:

```bash
nitid --help
nitid download --help
nitid predict --help
nitid track --help
```

Leave the environment when you are finished:

```bash
deactivate
```

If the terminal prints `nitid: command not found`, the environment is not
active or nitid has not been installed. Use `uv run nitid ...`, or activate
`.venv` first.

## Find available commands

Show the command list:

```bash
uv run nitid --help
```

Show the required arguments, optional settings, defaults, and examples for one
command:

```bash
uv run nitid download --help
uv run nitid predict --help
uv run nitid track --help
uv run nitid train --help
uv run nitid val --help
uv run nitid export --help
uv run nitid info --help
uv run nitid bugreport --help
```

The shorter `-h` flag also works:

```bash
uv run nitid predict -h
```

Use `nitid --help` or `nitid COMMAND --help`; `nitid help` is not a supported
form.

## Download a model

Download an official model checkpoint for local reuse:

```bash
uv run nitid download model=model1s task=detect
```

Supported nitid model names are `model1n`, `model1s`, `model1m`, `model1l`,
and `model1x`; the equivalent `dfine_*` names are also accepted. Select the
task explicitly with `task=detect`, `task=segment`, or `task=semantic`.
`model1n` has no detection checkpoint, so use it with `task=segment` or
`task=semantic`. Select a weight variant explicitly when needed:

```bash
uv run nitid download model=model1s task=detect weights=coco
```

Without `output=`, the wrapped checkpoint is saved in the current directory:

```text
model1s_detect_wrapped.pth
```

Save it in a model directory:

```bash
uv run nitid download model=model1s task=detect output=models
```

Download again and replace an existing checkpoint:

```bash
uv run nitid download model=model1s task=detect output=models force=true
```

## Common commands

Run prediction:

```bash
uv run nitid predict model=model1s task=detect source=image.jpg conf=0.5
```

## Track objects in video

Install the optional tracking dependencies:

```bash
uv sync --extra track
```

Track a video with ByteTrack, the default tracker:

```bash
uv run nitid track model=model1s task=detect source=video.mp4 conf=0.5
```

Select BoT-SORT for motion-only tracking with camera-motion compensation. This
is useful for moving, handheld, vehicle-mounted, or PTZ cameras:

```bash
uv run nitid track model=model1s task=detect source=video.mp4 tracker=botsort conf=0.5
```

Select OC-SORT when occlusions or non-linear motion make direction-aware
association useful:

```bash
uv run nitid track model=model1s task=detect source=video.mp4 tracker=ocsort conf=0.5
```

Save an annotated video containing class labels, confidence scores, and
persistent track IDs:

```bash
uv run nitid track model=model1s task=detect source=video.mp4 conf=0.5 save=true
```

The default output is `runs/track/exp/video.mp4`. The CLI processes tracking
results incrementally by default, so long videos are not accumulated in
memory. Pass `stream=false` only when a caller specifically needs a list.

Filter classes or sample every second frame:

```bash
uv run nitid track model=model1s task=detect source=video.mp4 classes=[0,2] vid_stride=2 save=true
```

Tracker settings are passed as flat `key=value` arguments. For ByteTrack:

```bash
uv run nitid track \
    model=model1s task=detect \
    source=video.mp4 \
    conf=0.5 \
    track_activation_threshold=0.4 \
    lost_track_buffer=60 \
    minimum_consecutive_frames=2
```

BoT-SORT enables camera-motion compensation by default. Its method and
downscale factor can be changed without affecting the other trackers:

```bash
uv run nitid track \
    model=model1s task=detect \
    source=video.mp4 \
    tracker=botsort \
    conf=0.5 \
    enable_cmc=true \
    cmc_method=sparseOptFlow \
    cmc_downscale=2
```

This integration is the motion-only BoT-SORT variant; it does not run an
appearance or ReID model.

For OC-SORT:

```bash
uv run nitid track \
    model=model1s task=detect \
    source=video.mp4 \
    tracker=ocsort \
    conf=0.5 \
    direction_consistency_weight=0.2 \
    delta_t=3 \
    lost_track_buffer=60
```

`conf` filters D-FINE detections before tracking. The tracker-specific
activation and association thresholds operate afterward. Run
`uv run nitid track --help` for every supported option and its defaults.

Track an RTSP or HTTP stream. Streams are read through OpenCV's default FFmpeg
backend:

```bash
uv run nitid track \
    model=model1s task=detect \
    source=rtsp://camera/live \
    conf=0.5
```

Tracking ends when the stream stops delivering frames.
FFmpeg options such as the RTSP transport are set through OpenCV's
`OPENCV_FFMPEG_CAPTURE_OPTIONS` environment variable, for example
`OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"`.

## Authenticated RTSP cameras

Pass the camera username as an option and the password through an environment
variable:

```bash
export CAMERA_RTSP_PASSWORD='camera password'
uv run nitid track \
    model=model1s task=detect \
    source=rtsp://192.0.2.10/Streaming/Channels/101 \
    rtsp_username=operator \
    rtsp_password_env=CAMERA_RTSP_PASSWORD
```

nitid percent-encodes both values and injects them into the URL that OpenCV
opens, so passwords with characters such as `@`, `:` or `/` work unchanged.
The source recorded in results and in `args.yaml` stays the credential-free URL
you passed; the password is never written there. Credentials are accepted only
for `rtsp://` and `rtsps://` sources that do not already contain a
`user:password@` part.

A direct `rtsp_password=` CLI argument is rejected because process arguments
may be visible to other users.

Fine-tune a model:

```bash
uv run nitid train model=model1s task=detect data=my_dataset.yml epochs=50
uv run nitid train model=model1s task=detect data=my_dataset.yml epochs=50 recipe=deim
```

By default, training saves wrapped epoch checkpoints under `runs/train/exp/`.
Add `wandb=true` to log the run to the default `nitid` WandB project:

```bash
uv run nitid train model=model1s task=detect data=my_dataset.yml epochs=50 wandb=true
```

Add `mlflow=true` for Ultralytics-style local MLflow tracking. Logs default to
a SQLite store in `runs/mlflow`:

```bash
uv run nitid train model=model1s task=detect data=my_dataset.yml epochs=50 mlflow=true
```

Validate a model:

```bash
uv run nitid val model=model1s task=detect data=my_dataset.yml
```

Validation reports COCO metrics to the terminal and does not create a run directory by default.

Export a model:

```bash
uv run nitid export model=model1s task=detect format=onnx
```

Display model information:

```bash
uv run nitid info model=model1s task=detect
```

## Convert detection datasets

Convert every split declared in a dataset YAML between YOLO text labels and
COCO JSON annotations:

```bash
uv run nitid convert data=data.yaml target=coco output=converted-coco
uv run nitid convert data=data.yaml target=yolo output=converted-yolo
```

The converter preserves `train`, `val`, and `test` splits, copies images and
labels into a self-contained output directory, and writes a directly trainable
configuration under `OUTPUT/configs/datasets/`. Bounding boxes and polygon
segmentations are retained. Existing output is protected unless
`exist_ok=true` is supplied explicitly.

## Create a bug-report log

Add `--report` to a training, prediction, tracking, validation, or export command:

```bash
uv run nitid predict model=model1s task=detect source=image.jpg --report
uv run nitid track model=model1s task=detect source=video.mp4 --report
uv run nitid train model=model1s task=detect data=my_dataset.yml epochs=50 --report
uv run nitid val model=model1s task=detect data=my_dataset.yml --report
uv run nitid export model=model1s task=detect format=onnx --report
```

The command continues printing normally while stdout and stderr are copied to a
single log under `runs/bugreports`. The log begins with the same environment
snapshot used for each run's `environment.yaml`, including OS, Python, package,
PyTorch, CUDA, cuDNN, and GPU information. Successful output and crash
tracebacks are captured, and the final log path is printed for attachment to a
GitHub issue.

When no model command can run, create an environment-only report:

```bash
uv run nitid bugreport
```

## Using the development Docker container

The Docker container does not automatically activate `.venv`. Enter the
container and run the executable from the project environment:

```bash
docker exec -it nitid_container bash
cd /app
.venv/bin/nitid download --help
```

Or run one command from the host terminal:

```bash
docker exec nitid_container /app/.venv/bin/nitid download --help
```

The full `.venv/bin/nitid` path is mainly useful for Docker automation. Normal
Linux users can use `uv run nitid ...` or activate `.venv` and use `nitid ...`.
