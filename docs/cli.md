# Command-line guide

nitid's terminal command is named `dfine`. Commands use `key=value`
arguments:

```bash
uv run dfine predict model=nitid1s task=detect source=image.jpg conf=0.5
uv run dfine predict model=nitid1s task=segment source=image.jpg conf=0.5
uv run dfine predict model=semantic_best.pth task=semantic source=image.jpg save=true
uv run dfine predict model=nitid1s task=obb source=aerial.jpg conf=0.25
```

## Before using the command

Complete the repository installation first:

```bash
git clone https://github.com/Vaelsys/nitid.git
cd nitid
uv sync
```

Run commands from the repository root, where `pyproject.toml` is located.

## Option 1: use `uv run` (recommended)

You do not need to activate the virtual environment:

```bash
uv run dfine --help
uv run dfine download --help
uv run dfine predict --help
uv run dfine track --help
```

`uv run` finds the project environment and runs the installed `dfine` command
inside it.

## Option 2: activate the virtual environment

On Linux or macOS:

```bash
source .venv/bin/activate
```

The shorter commands will then work:

```bash
dfine --help
dfine download --help
dfine predict --help
dfine track --help
```

Leave the environment when you are finished:

```bash
deactivate
```

If the terminal prints `dfine: command not found`, the environment is not
active or nitid has not been installed. Use `uv run dfine ...`, or activate
`.venv` first.

## Find available commands

Show the command list:

```bash
uv run dfine --help
```

Show the required arguments, optional settings, defaults, and examples for one
command:

```bash
uv run dfine download --help
uv run dfine predict --help
uv run dfine track --help
uv run dfine train --help
uv run dfine val --help
uv run dfine export --help
uv run dfine info --help
uv run dfine bugreport --help
```

The shorter `-h` flag also works:

```bash
uv run dfine predict -h
```

Use `dfine --help` or `dfine COMMAND --help`; `dfine help` is not a supported
form.

## Download a model

Download an official model checkpoint for local reuse:

```bash
uv run dfine download model=nitid1s task=detect
```

Supported nitid model names are `nitid1n`, `nitid1s`, `nitid1m`, `nitid1l`,
and `nitid1x`. Select the task explicitly with `task=detect`, `task=segment`,
`task=semantic`, `task=pose`, or `task=obb`. Select a weight variant explicitly
when needed:

```bash
uv run dfine download model=nitid1s task=detect weights=coco
```

Without `output=`, the wrapped checkpoint is saved in the current directory:

```text
nitid1s_detect_wrapped.pth
```

Save it in a model directory:

```bash
uv run dfine download model=nitid1s task=detect output=models
```

Download again and replace an existing checkpoint:

```bash
uv run dfine download model=nitid1s task=detect output=models force=true
```

## Common commands

Run prediction:

```bash
uv run dfine predict model=nitid1s task=detect source=image.jpg conf=0.5
```

## Track objects in video

Install the optional tracking dependencies:

```bash
uv sync --extra track
```

Track a video with ByteTrack, the default tracker:

```bash
uv run dfine track model=nitid1s task=detect source=video.mp4 conf=0.5
```

Select BoT-SORT for motion-only tracking with camera-motion compensation. This
is useful for moving, handheld, vehicle-mounted, or PTZ cameras:

```bash
uv run dfine track model=nitid1s task=detect source=video.mp4 tracker=botsort conf=0.5
```

Select OC-SORT when occlusions or non-linear motion make direction-aware
association useful:

```bash
uv run dfine track model=nitid1s task=detect source=video.mp4 tracker=ocsort conf=0.5
```

Save an annotated video containing class labels, confidence scores, and
persistent track IDs:

```bash
uv run dfine track model=nitid1s task=detect source=video.mp4 conf=0.5 save=true
```

The default output is `runs/track/exp/video.mp4`. The CLI processes tracking
results incrementally by default, so long videos are not accumulated in
memory. Pass `stream=false` only when a caller specifically needs a list.

Filter classes or sample every second frame:

```bash
uv run dfine track model=nitid1s task=detect source=video.mp4 classes=[0,2] vid_stride=2 save=true
```

Tracker settings are passed as flat `key=value` arguments. For ByteTrack:

```bash
uv run dfine track \
    model=nitid1s task=detect \
    source=video.mp4 \
    conf=0.5 \
    track_activation_threshold=0.4 \
    lost_track_buffer=60 \
    minimum_consecutive_frames=2
```

BoT-SORT enables camera-motion compensation by default. Its method and
downscale factor can be changed without affecting the other trackers:

```bash
uv run dfine track \
    model=nitid1s task=detect \
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
uv run dfine track \
    model=nitid1s task=detect \
    source=video.mp4 \
    tracker=ocsort \
    conf=0.5 \
    direction_consistency_weight=0.2 \
    delta_t=3 \
    lost_track_buffer=60
```

`conf` filters D-FINE detections before tracking. The tracker-specific
activation and association thresholds operate afterward. Run
`uv run dfine track --help` for every supported option and its defaults.

Use the GStreamer backend for a reconnecting RTSP source:

```bash
uv run dfine track \
    model=nitid1s task=detect \
    source=rtsp://camera/live \
    backend=gstreamer \
    reconnect=true \
    reconnect_max_delay=30 \
    conf=0.5
```

The first frame after a successful reconnect is marked as a discontinuity,
which resets the active tracker before it assigns IDs. `reconnect_attempts`
limits the number of attempts for each connection failure; omit it to keep
retrying until the process is stopped. See [GStreamer and RTSP](gstreamer.md)
for installation requirements and explicit pipelines.

Publish annotated tracking to an RTSP server that supports client publishing:

```bash
uv run dfine track \
    model=nitid1s task=detect \
    source=rtsp://camera/input \
    backend=gstreamer \
    reconnect=true \
    output=rtsp://media-server/nitid \
    output_rtsp_transport=tcp
```

Record annotated MP4 segments instead:

```bash
uv run dfine track \
    model=nitid1s task=detect \
    source=rtsp://camera/input \
    backend=gstreamer \
    reconnect=true \
    output=runs/segments/camera-1 \
    segment_duration=60
```

`output=` activates the GStreamer output sink and is separate from `save=true`.
Use `output_pipeline=` for a fully custom appsrc pipeline and `output_encoder=`
to select a platform encoder.

Inspect named hardware profiles on the current host:

```bash
uv run dfine gstreamer-info
```

The command reports input and output availability independently. Select a
validated H.264 RTSP decoder with `hardware_profile=vaapi` and a validated
output encoder with `output_hardware_profile=vaapi`. Missing elements are an
error; nitid does not silently switch to software.

## ONVIF cameras

Discover cameras on the local IPv4 network:

```bash
uv run dfine onvif action=discover timeout=3
```

List profiles and resolve a profile's RTSP URI:

```bash
export ONVIF_USERNAME=operator
export ONVIF_PASSWORD='camera password'

uv run dfine onvif action=profiles host=192.0.2.10
uv run dfine onvif action=uri host=192.0.2.10 profile='Main Stream'
```

The URI command does not insert credentials. Feed it to tracking with a
password environment variable:

```bash
export CAMERA_RTSP_PASSWORD='camera password'
uv run dfine track \
    model=nitid1s task=detect \
    source=rtsp://192.0.2.10/Streaming/Channels/101 \
    backend=gstreamer \
    rtsp_username=operator \
    rtsp_password_env=CAMERA_RTSP_PASSWORD \
    reconnect=true
```

Direct `password=` and `rtsp_password=` CLI arguments are rejected because
process arguments may be visible to other users. See [ONVIF cameras](onvif.md).

Fine-tune a model:

```bash
uv run dfine train model=nitid1s task=detect data=my_dataset.yml epochs=50
```

By default, training saves wrapped epoch checkpoints under `runs/train/exp/`.
Add `wandb=true` to log the run to the default `nitid` WandB project:

```bash
uv run dfine train model=nitid1s task=detect data=my_dataset.yml epochs=50 wandb=true
```

Add `mlflow=true` for Ultralytics-style local MLflow tracking. Logs default to
`runs/mlflow`:

```bash
uv run dfine train model=nitid1s task=detect data=my_dataset.yml epochs=50 mlflow=true
```

Validate a model:

```bash
uv run dfine val model=nitid1s task=detect data=my_dataset.yml
```

Validation reports COCO metrics to the terminal and does not create a run directory by default.

Export a model:

```bash
uv run dfine export model=nitid1s task=detect format=onnx
```

Display model information:

```bash
uv run dfine info model=nitid1s task=detect
```

## Convert detection datasets

Convert every split declared in a dataset YAML between YOLO text labels and
COCO JSON annotations:

```bash
uv run dfine convert data=data.yaml target=coco output=converted-coco
uv run dfine convert data=data.yaml target=yolo output=converted-yolo
```

The converter preserves `train`, `val`, and `test` splits, copies images and
labels into a self-contained output directory, and writes a directly trainable
configuration under `OUTPUT/configs/datasets/`. Bounding boxes and polygon
segmentations are retained. Existing output is protected unless
`exist_ok=true` is supplied explicitly.

## Create a bug-report log

Add `--report` to a training, prediction, tracking, validation, or export command:

```bash
uv run dfine predict model=nitid1s task=detect source=image.jpg --report
uv run dfine track model=nitid1s task=detect source=video.mp4 --report
uv run dfine train model=nitid1s task=detect data=my_dataset.yml epochs=50 --report
uv run dfine val model=nitid1s task=detect data=my_dataset.yml --report
uv run dfine export model=nitid1s task=detect format=onnx --report
```

The command continues printing normally while stdout and stderr are copied to a
single log under `runs/bugreports`. The log begins with the same environment
snapshot used for each run's `environment.yaml`, including OS, Python, package,
PyTorch, CUDA, cuDNN, and GPU information. Successful output and crash
tracebacks are captured, and the final log path is printed for attachment to a
GitHub issue.

When no model command can run, create an environment-only report:

```bash
uv run dfine bugreport
```

## Using the development Docker container

The Docker container does not automatically activate `.venv`. Enter the
container and run the executable from the project environment:

```bash
docker exec -it nitid_container bash
cd /app
.venv/bin/dfine download --help
```

Or run one command from the host terminal:

```bash
docker exec nitid_container /app/.venv/bin/dfine download --help
```

The full `.venv/bin/dfine` path is mainly useful for Docker automation. Normal
Linux users can use `uv run dfine ...` or activate `.venv` and use `dfine ...`.
