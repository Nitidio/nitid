# GStreamer and RTSP

nitid can decode video through a GStreamer pipeline while preserving the same
`Frame`, prediction, and tracking interfaces used by the default OpenCV path.
The backend is opt-in:

```python
from dfine import DFINE

model = DFINE("dfine_s")
for result in model.track(
    "rtsp://camera/live",
    backend="gstreamer",
    reconnect=True,
    conf=0.5,
    stream=True,
):
    print(result.boxes.id)
```

```bash
uv run dfine track \
    model=dfine_s \
    source=rtsp://camera/live \
    backend=gstreamer \
    reconnect=true \
    conf=0.5
```

OpenCV remains the default backend. Selecting GStreamer does not change the
model, result, or tracker APIs.

## System requirement

The Python package uses OpenCV's `CAP_GSTREAMER` integration. GStreamer and the
required plugins must be installed on the host, and OpenCV must have been
compiled with GStreamer enabled. This is a system capability, not a Python
extra.

Verify the active environment:

```bash
uv run python -c "import cv2; print(cv2.getBuildInformation())"
```

The `Video I/O` section must contain `GStreamer: YES`. The standard PyPI
OpenCV wheel may report `NO`; in that case use a distribution or container
OpenCV build compiled against the host GStreamer libraries. nitid raises a
clear runtime error instead of silently falling back to another backend.

A typical Debian/Ubuntu runtime needs GStreamer core plus the base, good, bad,
ugly, and libav plugin sets. The exact packages and hardware plugins depend on
the target distribution and accelerator.

## Input forms

### RTSP

An RTSP URL produces a TCP pipeline with a 200 ms jitter buffer by default:

```python
results = model.predict(
    "rtsp://camera/live",
    backend="gstreamer",
    rtsp_latency=300,
    rtsp_transport="tcp",  # or "udp"
    stream=True,
)
```

Credentials remain part of the source URL. Avoid printing URLs or committing
them to scripts when they contain passwords.

### Recorded video

```python
results = model.predict("video.mp4", backend="gstreamer", stream=True)
```

End-of-stream on a recorded file completes normally and is never reconnected.

### Webcam

Integer camera indexes select a platform pipeline (`v4l2src` on Linux,
`avfvideosrc` on macOS, and `ksvideosrc` on Windows):

```python
results = model.predict(0, backend="gstreamer", stream=True)
```

Device properties differ between GStreamer plugins. Use an explicit pipeline
when the automatic source does not match the camera.

### Explicit pipeline

Pass a pipeline through `gst_pipeline`. nitid appends BGR conversion and a
bounded, non-synchronizing appsink when the pipeline does not already contain
an appsink:

```python
pipeline = (
    "rtspsrc location=rtsp://camera/live latency=200 protocols=tcp "
    "! rtph264depay ! h264parse ! avdec_h264"
)

results = model.track(
    "camera-1",
    backend="gstreamer",
    gst_pipeline=pipeline,
    reconnect=True,
    stream=True,
)
```

If the supplied pipeline already contains `appsink`, it must emit three-channel
BGR frames. Pipelines are parsed by GStreamer directly; they are not executed
through a shell.

## Reconnection semantics

Reconnection is available for live stream and webcam modes:

| Option | Default | Meaning |
|---|---:|---|
| `reconnect` | `False` | Reopen the pipeline after an open/read failure. |
| `reconnect_initial_delay` | `1.0` | Delay before the first retry, in seconds. |
| `reconnect_max_delay` | `30.0` | Maximum exponential-backoff delay. |
| `reconnect_attempts` | unlimited | Retry limit per failure; `0` disables retries. |

Frame indexes remain increasing across a recovered connection. The first
emitted frame after recovery has `FrameMetadata.discontinuity=True`. ByteTrack
observes that marker and creates a fresh tracker backend, preventing IDs from
leaking across an unknown camera gap.

Closing the source or closing a streaming result generator releases the active
capture. A close request also interrupts reconnect backoff. Depending on the
platform plugin, an in-progress blocking network read may return only after the
plugin's own timeout.

## Annotated output

`GStreamerVideoSink` receives the rendered frame after detection or tracking.
It initializes lazily from the first frame and derives output FPS from source
FPS divided by `vid_stride`. Pass `fps=` to override that value.

### Segmented recording

```python
from dfine import GStreamerVideoSink

sink = GStreamerVideoSink(
    "runs/segments/camera-1",
    segment_duration=60,
)

for result in model.track(
    "rtsp://camera/input",
    backend="gstreamer",
    reconnect=True,
    stream=True,
    sink=sink,
):
    ...
```

This uses `splitmuxsink` and produces
`runs/segments/camera-1/segment_00000.mp4`, `segment_00001.mp4`, and so on. If
the destination ends in `.mp4`, its stem becomes the segment prefix instead:
`camera_00000.mp4`.

The equivalent CLI command is:

```bash
uv run dfine track \
    model=dfine_s \
    source=rtsp://camera/input \
    backend=gstreamer \
    reconnect=true \
    output=runs/segments/camera-1 \
    segment_duration=60
```

### RTSP publishing

```python
sink = GStreamerVideoSink(
    "rtsp://media-server/nitid",
    rtsp_transport="tcp",
)

for result in model.track("video.mp4", stream=True, sink=sink):
    ...
```

The target must be an RTSP server that accepts client publishing through
`rtspclientsink`. A camera's read-only RTSP input URL normally cannot be used
as the destination. Authentication and mount-point configuration belong to the
publishing server.

CLI:

```bash
uv run dfine track model=dfine_s source=video.mp4 \
    output=rtsp://media-server/nitid output_rtsp_transport=tcp
```

### Single MP4 and custom pipelines

Without `segment_duration`, a local destination creates one MP4:

```python
sink = GStreamerVideoSink("runs/annotated.mp4")
```

For another protocol, container, or hardware stack, supply the pipeline after
or including `appsrc`:

```python
sink = GStreamerVideoSink(
    pipeline=(
        "appsrc format=time ! videoconvert ! vaapih264enc "
        "! h264parse ! mpegtsmux ! udpsink host=127.0.0.1 port=5000"
    )
)
```

The CLI equivalents are `output_pipeline=...`, `output_fps=...`, and
`output_encoder=...`. Quote pipelines and encoder strings containing spaces so
the shell passes each `key=value` expression as one argument.

The built-in output pipelines use `x264enc`, `h264parse`, MP4/RTSP elements,
and a four-frame leaky queue. Required plugins must be installed on the target.
Closing the result generator closes the sink and finalizes the current MP4
fragment.

## Hardware decoding

nitid provides named H.264 codec profiles:

| Profile | Decode elements | Encode elements | Intended platform |
|---|---|---|---|
| `software` | `avdec_h264` | `x264enc` | Portable CPU baseline |
| `vaapi` | `vah264dec` or `vaapih264dec` | `vah264enc` or `vaapih264enc` | Intel/AMD VA-API |
| `v4l2` | `v4l2h264dec` | `v4l2h264enc` | Linux V4L2 M2M |
| `nvidia` | `nvh264dec` | `nvh264enc` | NVIDIA desktop GStreamer |
| `jetson` | `nvv4l2decoder` | `nvv4l2h264enc` | NVIDIA Jetson |

Inspect the active OpenCV build and installed elements:

```bash
dfine gstreamer-info
```

Decode and encode are reported separately because a host can support only one
direction. Selecting an unavailable profile raises an error naming the missing
elements; there is no implicit software fallback.

Apply a profile to automatic H.264 RTSP ingest:

```bash
dfine track \
    model=dfine_s \
    source=rtsp://camera/live \
    backend=gstreamer \
    hardware_profile=vaapi \
    reconnect=true
```

Apply a potentially different profile to output:

```bash
dfine track \
    model=dfine_s \
    source=rtsp://camera/live \
    backend=gstreamer \
    hardware_profile=vaapi \
    output=runs/segments/camera \
    segment_duration=60 \
    output_hardware_profile=vaapi
```

Named input profiles intentionally cover H.264 RTSP only. Containers, files,
H.265, unusual memory layouts, and vendor plugin variants should use an
explicit `gst_pipeline`. A custom pipeline cannot also select a named profile.

## GStreamer container

`Dockerfile.gstreamer` provides a Debian runtime
with:

- an OpenCV binding compiled with `CAP_GSTREAMER`;
- GStreamer base/good/bad/ugly/libav plugins;
- software H.264 decode and encode;
- VA-API plugins, Mesa drivers, and diagnostic tools;
- a non-root runtime user and locked Python dependencies.

The image constrains NumPy to the 1.x ABI after the locked sync because
Debian's `python3-opencv` extension is built against that ABI.

The image build verifies `GStreamer: YES`, `avdec_h264`, and `x264enc`. A broken
runtime fails during the build instead of failing on the first camera.

Build and inspect it:

```bash
docker build -f Dockerfile.gstreamer -t nitid-gstreamer .
docker run --rm nitid-gstreamer
```

For Intel/AMD VA-API on Linux:

```bash
docker run --rm \
    --device /dev/dri:/dev/dri \
    nitid-gstreamer \
    dfine gstreamer-info
```

The optional Compose service supplies `/dev/dri` and host video/render group
IDs:

```bash
VIDEO_GID=$(getent group video | cut -d: -f3) \
RENDER_GID=$(getent group render | cut -d: -f3) \
docker compose --profile gstreamer run --rm nitid-gstreamer
```

Add `/dev/video0` as a device when using a V4L2 camera. NVIDIA desktop and
Jetson profiles require the vendor container runtime and GStreamer plugins;
derive a target-specific image from the appropriate NVIDIA base rather than
assuming the generic Debian image contains them.

## Current boundary

This backend covers decode, RTSP reconnect, bounded buffering, timestamps,
discontinuity propagation, annotated RTSP publishing, and segmented recording.
It also provides validated codec profiles, a software/VA-API container
baseline, and ONVIF discovery/profile resolution. Vendor-specific
NVIDIA/Jetson images remain later deployment stages. `save=True` continues to use nitid's existing OpenCV output;
the GStreamer output options are independent and can be used alone.
