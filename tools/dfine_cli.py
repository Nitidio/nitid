"""
dfine CLI — mirrors the `yolo` command from Ultralytics.

Usage:
    dfine predict  model=dfine_l weights=obj2coco source=image.jpg conf=0.5
    dfine track    model=dfine_s source=video.mp4 conf=0.5 save=true
    dfine download model=dfine_l weights=coco
    dfine train    model=dfine_l data=coco.yaml epochs=50
    dfine val      model=dfine_l data=coco.yaml
    dfine export   model=dfine_l format=onnx
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

COMMANDS = {
    "predict",
    "track",
    "download",
    "train",
    "val",
    "export",
    "info",
    "gstreamer-info",
    "onvif",
    "bugreport",
}
REPORT_COMMANDS = {"predict", "track", "train", "val", "export"}
HELP_FLAGS = {"-h", "--help"}
TRACKER_OPTIONS = {
    "cmc_downscale",
    "cmc_method",
    "delta_t",
    "direction_consistency_weight",
    "enable_cmc",
    "frame_rate",
    "lost_track_buffer",
    "track_activation_threshold",
    "minimum_consecutive_frames",
    "minimum_iou_threshold",
    "minimum_iou_threshold_first_assoc",
    "minimum_iou_threshold_second_assoc",
    "minimum_iou_threshold_unconfirmed_assoc",
    "high_conf_det_threshold",
    "instant_first_frame_activation",
}

GENERAL_HELP = """\
nitid D-FINE CLI

Usage:
  dfine COMMAND [key=value ...] [--report]

Commands:
  predict  Run detection or instance segmentation on an image, video, or stream
  track    Detect and track objects with persistent IDs across video frames
  download Download and wrap an official D-FINE checkpoint
  train    Fine-tune a model on a COCO-format dataset
  val      Evaluate a model and report COCO metrics
  export   Export a model to ONNX, OpenVINO, TorchScript, or TensorRT
  info     Show model parameters, GFLOPs, and checkpoint size
  gstreamer-info  Show GStreamer and hardware codec profile availability
  onvif    Discover cameras, list media profiles, or resolve an RTSP URI
  bugreport Create an environment-only log for a GitHub issue

Run "dfine COMMAND --help" for command-specific options and examples.
Add --report to train, predict, track, val, or export to capture output and environment details.
"""

COMMAND_HELP = {
    "predict": """\
Usage:
  dfine predict model=MODEL source=SOURCE [key=value ...]

Required:
  source=SOURCE       Image, directory, video, URL, webcam index, or stream URL

Options:
  model=MODEL         Architecture name or wrapped checkpoint path (default: dfine_l)
  task=TASK           detect or segment (default: detect)
  weights=NAME        default, obj2coco, or coco (default: default)
  conf=FLOAT          Confidence threshold (default: 0.5)
  imgsz=INT           Square inference image size (default: 640)
  stream=BOOL         Return results as a generator (default: false)
  backend=NAME        Video backend: opencv or gstreamer (default: opencv)
  gst_pipeline=TEXT   Explicit GStreamer pipeline ending before or at appsink
  reconnect=BOOL      Reconnect a live GStreamer source after failure (default: false)
  reconnect_initial_delay=FLOAT  Initial reconnect delay in seconds (default: 1)
  reconnect_max_delay=FLOAT      Maximum reconnect delay in seconds (default: 30)
  reconnect_attempts=INT         Retry limit; omitted means unlimited
  rtsp_latency=INT    GStreamer RTSP jitter-buffer latency in ms (default: 200)
  rtsp_transport=NAME RTSP transport: tcp or udp (default: tcp)
  hardware_profile=NAME  H.264 RTSP decoder: software, vaapi, v4l2, nvidia, jetson
  rtsp_username=USER RTSP username passed as a GStreamer property
  rtsp_password_env=NAME  Environment variable containing the RTSP password
  output=DEST         Annotated MP4 path, segment directory, or RTSP publish URL
  output_pipeline=TEXT  Explicit GStreamer appsrc output pipeline
  output_fps=FLOAT    Override output FPS (default: source FPS)
  segment_duration=FLOAT  Split local output every N seconds
  output_encoder=TEXT GStreamer encoder element and properties (default: x264enc)
  output_hardware_profile=NAME  Encoder: software, vaapi, v4l2, nvidia, jetson
  output_rtsp_transport=NAME  RTSP publish transport: tcp or udp (default: tcp)
  augment=BOOL        Use test-time augmentation (default: false)
  iou=FLOAT            IoU threshold for augmented-view NMS (default: 0.85)
  save=BOOL           Save annotated images (default: false)
  project=PATH        Parent output directory when save=true (default: runs/detect)
  name=NAME           Run directory name when save=true (default: exp)
  save_dir=PATH       Exact output directory override
  exist_ok=BOOL       Reuse the requested directory (default: false)
  verbose=BOOL        Print prediction progress (default: true)
  --report            Tee stdout/stderr and environment details to a bug-report log

Examples:
  dfine predict model=dfine_l source=image.jpg
  dfine predict model=dfine_l weights=coco source=image.jpg save=true
  dfine predict model=dfine_l source=video.mp4 conf=0.3 stream=true
""",
    "track": """\
Usage:
  dfine track model=MODEL source=SOURCE [key=value ...]

Required:
  source=SOURCE       Video, webcam index, or stream URL

Options:
  model=MODEL         Architecture name or wrapped checkpoint path (default: dfine_l)
  weights=NAME        default, obj2coco, or coco (default: default)
  tracker=NAME        bytetrack (default), botsort, or ocsort
  conf=FLOAT          Detection confidence threshold (default: 0.1)
  imgsz=INT           Square inference image size (default: 640)
  classes=LIST        Track only selected class IDs, e.g. classes=[0,2]
  stream=BOOL         Process results incrementally (CLI default: true)
  vid_stride=INT      Process every Nth source frame (default: 1)
  backend=NAME        Video backend: opencv or gstreamer (default: opencv)
  gst_pipeline=TEXT   Explicit GStreamer pipeline ending before or at appsink
  reconnect=BOOL      Reconnect a live GStreamer source after failure (default: false)
  reconnect_initial_delay=FLOAT  Initial reconnect delay in seconds (default: 1)
  reconnect_max_delay=FLOAT      Maximum reconnect delay in seconds (default: 30)
  reconnect_attempts=INT         Retry limit; omitted means unlimited
  rtsp_latency=INT    GStreamer RTSP jitter-buffer latency in ms (default: 200)
  rtsp_transport=NAME RTSP transport: tcp or udp (default: tcp)
  hardware_profile=NAME  H.264 RTSP decoder: software, vaapi, v4l2, nvidia, jetson
  rtsp_username=USER RTSP username passed as a GStreamer property
  rtsp_password_env=NAME  Environment variable containing the RTSP password
  output=DEST         Annotated MP4 path, segment directory, or RTSP publish URL
  output_pipeline=TEXT  Explicit GStreamer appsrc output pipeline
  output_fps=FLOAT    Override output FPS (default: source FPS)
  segment_duration=FLOAT  Split local output every N seconds
  output_encoder=TEXT GStreamer encoder element and properties (default: x264enc)
  output_hardware_profile=NAME  Encoder: software, vaapi, v4l2, nvidia, jetson
  output_rtsp_transport=NAME  RTSP publish transport: tcp or udp (default: tcp)
  augment=BOOL        Use test-time augmentation (default: false)
  iou=FLOAT            IoU threshold for augmented-view NMS (default: 0.85)
  save=BOOL           Save annotated output with persistent IDs (default: false)
  project=PATH        Parent output directory when save=true (default: runs/track)
  name=NAME           Run directory name when save=true (default: exp)
  save_dir=PATH       Exact output directory override
  exist_ok=BOOL       Reuse the requested directory (default: false)
  verbose=BOOL        Print tracking progress (default: true)
  --report            Capture output and environment details in a log

Shared tracker options:
  high_conf_det_threshold=FLOAT     Detection threshold (default: 0.6)
  lost_track_buffer=INT             Frames to retain a lost track (default: 30)
  frame_rate=FLOAT                  Override effective source FPS (default: auto)

ByteTrack options:
  track_activation_threshold=FLOAT  New-track confidence threshold (default: 0.25)
  minimum_iou_threshold=FLOAT       Minimum association IoU (default: 0.1)
  minimum_consecutive_frames=INT    Frames required to confirm a track (default: 1)

BoT-SORT options:
  track_activation_threshold=FLOAT  New-track confidence threshold (default: 0.7)
  minimum_consecutive_frames=INT    Frames required to confirm a track (default: 2)
  minimum_iou_threshold_first_assoc=FLOAT  First-stage IoU (default: 0.2)
  minimum_iou_threshold_second_assoc=FLOAT Second-stage IoU (default: 0.5)
  minimum_iou_threshold_unconfirmed_assoc=FLOAT  Unconfirmed IoU (default: 0.3)
  enable_cmc=BOOL                   Compensate for camera motion (default: true)
  cmc_method=NAME                   orb, sift, sparseOptFlow, or ecc
  cmc_downscale=INT                 CMC image downscale factor (default: 2)
  instant_first_frame_activation=BOOL  Confirm first-frame tracks immediately (default: true)

OC-SORT options:
  minimum_iou_threshold=FLOAT       Minimum association IoU (default: 0.3)
  minimum_consecutive_frames=INT    Frames required to confirm a track (default: 3)
  direction_consistency_weight=FLOAT  Motion-direction cost weight (default: 0.2)
  delta_t=INT                       Frames used to estimate direction (default: 3)

Install tracking support first with: uv sync --extra track

Examples:
  dfine track model=dfine_s source=video.mp4
  dfine track model=dfine_s source=video.mp4 tracker=botsort
  dfine track model=dfine_s source=video.mp4 tracker=ocsort
  dfine track model=dfine_s source=video.mp4 conf=0.5 save=true
  dfine track model=dfine_s source=0 classes=[0] stream=true
  dfine track model=dfine_s source=rtsp://camera/stream lost_track_buffer=60
  dfine track model=dfine_s source=rtsp://camera/stream backend=gstreamer reconnect=true
  dfine track model=dfine_s source=video.mp4 output=runs/segments segment_duration=60
  dfine track model=dfine_s source=rtsp://camera/stream backend=gstreamer hardware_profile=vaapi
""",
    "download": """\
Usage:
  dfine download [model=MODEL] [key=value ...]

Options:
  model=NAME          dfine_s, dfine_m, dfine_l, or dfine_x (default: dfine_l)
  weights=NAME        default, obj2coco, or coco (default: default)
  output=PATH         Output directory or .pth file (default: current directory)
  force=BOOL          Overwrite an existing wrapped checkpoint (default: false)

The command downloads the official raw checkpoint and converts it to nitid's
wrapped .pth format. The filename includes the resolved weight variant.

Examples:
  dfine download model=dfine_s
  dfine download model=dfine_s weights=coco
  dfine download model=dfine_m output=models
  dfine download model=dfine_l output=models/custom.pth force=true
""",
    "train": """\
Usage:
  dfine train model=MODEL data=DATA [key=value ...]

Required:
  data=PATH           Dataset YAML file using COCO-format annotations

Options:
  model=MODEL         Architecture name or wrapped checkpoint path (default: dfine_l)
  weights=NAME        default, obj2coco, or coco (default: default)
  epochs=INT          Number of training epochs (default: 50)
  imgsz=INT           Square training image size (default: 640)
  batch=INT           Batch size (default: 16)
  lr0=FLOAT           Initial learning rate (default: 0.0001)
  lrf=FLOAT           Final learning-rate factor (default: 0.01)
  optimizer=NAME      Auto, Adam, AdamW, SGD, RAdam, NAdam, or RMSprop (default: AdamW)
  momentum=FLOAT      SGD momentum or Adam beta1 (default: 0.9)
  weight_decay=FLOAT  Weight decay (default: 0.0001)
  clip_grad=FLOAT     Maximum gradient norm; 0 disables (default: 0.1)
  patience=INT        Early-stopping patience; 0 disables (default: 100)
  time=FLOAT          Training duration in hours; overrides epochs
  resume=BOOL         Resume a previous run (default: false)
  amp=BOOL            Enable mixed precision on CUDA (default: false)
  ema=BOOL            Enable exponential moving average (default: false)
  ema_decay=FLOAT     EMA decay value (default: 0.9999)
  device=DEVICE       cpu, cuda, or cuda:N (default: model device)
  project=PATH        Parent output directory (default: runs/train)
  name=NAME           Run directory name (default: exp)
  save_dir=PATH       Exact run directory override
  exist_ok=BOOL       Accepted for API parity; non-resume training still increments
  save=BOOL           Save last/best checkpoints (default: true)
  save_period=INT     Periodic checkpoint interval; -1 disables (default: 1)
  val=BOOL            Run validation during training (default: true)
  val_period=INT      Validate every N epochs (default: 1)
  plots=BOOL          Save training/validation plots (default: true)
  workers=INT         Data-loading worker processes (default: 0)
  cache=BOOL          Cache resized training images in RAM (default: false)
  seed=INT            Random seed (default: 0)
  deterministic=BOOL  Request deterministic algorithms (default: true)
  freeze=VALUE        Layer count, stage, glob, or list to freeze
  classes=LIST        Train only selected class IDs, e.g. classes=[0,2]
  single_cls=BOOL     Treat all selected classes as one class (default: false)
  fraction=FLOAT      Fraction of training images to use (default: 1.0)
  accumulate=INT      Gradient accumulation steps (default: 1)
  multi_scale=BOOL    Randomly resize batches during training (default: false)
  augment=BOOL        Enable box-aware training augmentation (default: true)
  fliplr=FLOAT        Horizontal-flip probability (default: 0.5)
  scale=FLOAT         Random scale gain (default: 0.5)
  translate=FLOAT     Random translation gain (default: 0.1)
  crop=FLOAT          Random crop probability/maximum edge gain (default: 0.0)
  hsv_h=FLOAT         Hue jitter gain (default: 0.015)
  hsv_s=FLOAT         Saturation jitter gain (default: 0.7)
  hsv_v=FLOAT         Brightness jitter gain (default: 0.4)
  mosaic=FLOAT        Mosaic probability; experimental, off by default
  mixup=FLOAT         MixUp probability; experimental, off by default
  close_mosaic=INT    Disable mosaic for the final N epochs (default: 10)
  wandb=BOOL          Enable Weights & Biases logging (default: false)
  mlflow=BOOL         Enable MLflow logging (default: false)
  verbose=BOOL        Print training progress (default: true)
  --report            Tee stdout/stderr and environment details to a bug-report log

Example:
  dfine train model=dfine_l data=coco.yaml epochs=50 batch=16 mlflow=true
""",
    "val": """\
Usage:
  dfine val model=MODEL data=DATA [key=value ...]

Required:
  data=PATH           Dataset YAML file using COCO-format annotations

Options:
  model=MODEL         Architecture name or wrapped checkpoint path (default: dfine_l)
  weights=NAME        default, obj2coco, or coco (default: default)
  imgsz=INT           Square validation image size (default: 640)
  batch=INT           Batch size (default: 16)
  conf=FLOAT          Confidence threshold (default: 0.001)
  split=NAME          Dataset split: val or test (default: val)
  project=PATH        Output root for validation artifacts (default: runs/val)
  name=NAME           Validation run name (default: exp)
  save_dir=PATH       Exact output directory override
  exist_ok=BOOL       Reuse the requested directory (default: false)
  plots=BOOL          Save PR/confusion plots and results.png (default: true)
  verbose=BOOL        Print validation progress (default: true)
  --report            Tee stdout/stderr and environment details to a bug-report log

Example:
  dfine val model=dfine_l data=coco.yaml split=val batch=16
""",
    "export": """\
Usage:
  dfine export model=MODEL [key=value ...]

Options:
  model=MODEL         Architecture name or wrapped checkpoint path (default: dfine_l)
  weights=NAME        default, obj2coco, or coco (default: default)
  format=FORMAT       onnx, openvino, torchscript, or tensorrt (default: onnx)
  imgsz=INT           Square export image size (default: 640)
  batch=INT           Static batch size (default: 1)
  dynamic=BOOL        Enable a dynamic batch axis (default: false)
  simplify=BOOL       Simplify the ONNX graph (default: true)
  opset=INT           ONNX opset version (default: 17)
  half=BOOL           Enable FP16 OpenVINO/TensorRT export (default: false)
  device=DEVICE       cpu, cuda, or cuda:N (default: model device)
  project=PATH        Export run root (default: runs/export)
  name=NAME           Export run name (default: exp)
  save_dir=PATH       Exact export run directory override
  output=PATH         Exact exported artifact path
  exist_ok=BOOL       Reuse/replace an explicit destination (default: false)
  verbose=BOOL        Print export progress (default: true)
  --report            Tee stdout/stderr and environment details to a bug-report log

Examples:
  dfine export model=dfine_l format=onnx
  dfine export model=dfine_l weights=coco format=openvino
  dfine export model=dfine_l format=tensorrt half=true
""",
    "info": """\
Usage:
  dfine info model=MODEL [key=value ...]

Options:
  model=MODEL         Architecture name or wrapped checkpoint path (default: dfine_l)
  weights=NAME        default, obj2coco, or coco (default: default)
  detailed=BOOL       Include per-layer parameter counts (default: false)

Example:
  dfine info model=dfine_l weights=obj2coco detailed=true
""",
    "gstreamer-info": """\
Usage:
  dfine gstreamer-info

Reports whether OpenCV has GStreamer enabled, whether gst-inspect-1.0 is
available, and which named decode/encode profiles have all required elements.

Profiles:
  software  libav/x264 CPU path
  vaapi     Intel/AMD VA-API
  v4l2      Linux V4L2 memory-to-memory
  nvidia    NVIDIA desktop CUDA/NVENC
  jetson    NVIDIA Jetson NVMM/V4L2
""",
    "onvif": """\
Usage:
  dfine onvif action=discover [timeout=SECONDS] [interface=IP]
  dfine onvif action=profiles host=HOST [username=USER] [password_env=NAME]
  dfine onvif action=uri host=HOST [profile=TOKEN_OR_NAME] [username=USER]

Options:
  action=NAME         discover, profiles, or uri (default: discover)
  host=HOST           Camera host or complete ONVIF device-service URL
  port=INT            Override the ONVIF HTTP(S) port
  timeout=FLOAT       Discovery or SOAP timeout in seconds (default: 3 or 5)
  interface=IP        IPv4 interface address used for multicast discovery
  username=USER       ONVIF username (default: ONVIF_USERNAME environment variable)
  password_env=NAME   Environment variable containing the password (default: ONVIF_PASSWORD)
  profile=VALUE       Profile token or case-insensitive profile name
  verify_ssl=BOOL     Verify camera HTTPS certificates (default: true)
  time_offset=FLOAT   Camera clock correction for WS-Security, in seconds

Passwords are intentionally read from the environment instead of command-line
arguments, which may be visible to other local processes.
""",
    "bugreport": """\
Usage:
  dfine bugreport

Creates an environment-only log containing OS, Python, package, PyTorch,
CUDA, cuDNN, and GPU information. Attach the resulting file to a GitHub issue.
""",
}


def _print_help(command: str | None = None) -> None:
    """Print general help or help for one command."""
    print(COMMAND_HELP[command] if command else GENERAL_HELP)


def parse_args(argv: list[str]) -> tuple[str, dict]:
    """Parse 'command key=value ...' style args."""
    if len(argv) < 2:
        _print_help()
        sys.exit(1)

    command = argv[1].lower()
    if command in HELP_FLAGS:
        _print_help()
        sys.exit(0)
    if command not in COMMANDS:
        print(f"ERROR: unknown command '{argv[1]}'\n")
        _print_help()
        sys.exit(1)
    if any(token in HELP_FLAGS for token in argv[2:]):
        _print_help(command)
        sys.exit(0)

    kwargs = {}
    for token in argv[2:]:
        if "=" in token:
            k, v = token.split("=", 1)
            kwargs[k.strip()] = _coerce(v.strip())
    return command, kwargs


def _coerce(v: str):
    """Auto-cast strings to int/float/bool where obvious."""
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v.startswith("[") and v.endswith("]"):
        import yaml

        value = yaml.safe_load(v)
        if isinstance(value, list):
            return value
    return v


def _configure_output_sink(kwargs: dict) -> str | None:
    """Convert CLI output options into a lazy GStreamer frame sink."""
    destination = kwargs.pop("output", None)
    pipeline = kwargs.pop("output_pipeline", None)
    option_names = {
        "output_fps": "fps",
        "segment_duration": "segment_duration",
        "output_encoder": "encoder",
        "output_rtsp_transport": "rtsp_transport",
        "output_hardware_profile": "hardware_profile",
    }
    provided_options = [name for name in option_names if name in kwargs]
    sink_options = {
        sink_name: kwargs.pop(cli_name)
        for cli_name, sink_name in option_names.items()
        if cli_name in kwargs
    }
    if destination is None and pipeline is None:
        if sink_options:
            names = ", ".join(sorted(provided_options))
            raise ValueError(f"output options require output= or output_pipeline=: {names}")
        return None

    from dfine import GStreamerVideoSink

    kwargs["sink"] = GStreamerVideoSink(
        destination,
        pipeline=pipeline,
        **sink_options,
    )
    return str(destination) if destination is not None else "custom GStreamer pipeline"


def _configure_rtsp_credentials(kwargs: dict) -> None:
    """Resolve an RTSP password from the environment without exposing it in argv."""
    if "rtsp_password" in kwargs:
        print("ERROR: use rtsp_password_env=NAME instead of placing an RTSP password in argv")
        raise SystemExit(1)
    password_env = kwargs.pop("rtsp_password_env", None)
    if password_env is None:
        return
    password = os.environ.get(str(password_env))
    if password is None:
        print(f"ERROR: RTSP password environment variable '{password_env}' is not set")
        raise SystemExit(1)
    username = kwargs.get("rtsp_username") or os.environ.get("ONVIF_USERNAME")
    if username is None:
        print("ERROR: rtsp_username= or ONVIF_USERNAME is required with rtsp_password_env")
        raise SystemExit(1)
    kwargs["rtsp_username"] = username
    kwargs["rtsp_password"] = password


def _execute_onvif(kwargs: dict) -> None:
    """Execute ONVIF discovery and read-only Media1 operations."""
    from dfine.onvif import ONVIFCamera, discover_onvif_devices

    action = str(kwargs.pop("action", "discover")).lower()
    if "password" in kwargs:
        print(
            "ERROR: use password_env=NAME instead of placing an ONVIF password on the command line"
        )
        raise SystemExit(1)

    if action == "discover":
        timeout = float(kwargs.pop("timeout", 3.0))
        interface = kwargs.pop("interface", None)
        if kwargs:
            print(f"ERROR: unsupported ONVIF discovery options: {', '.join(sorted(kwargs))}")
            raise SystemExit(1)
        devices = discover_onvif_devices(timeout=timeout, interface=interface)
        print(f"Discovered {len(devices)} ONVIF device{'s' if len(devices) != 1 else ''}")
        for index, device in enumerate(devices):
            endpoint = device.endpoint_reference or "unknown endpoint"
            print(f"  [{index}] {device.service_url or 'no service URL'} ({endpoint})")
        return

    if action not in {"profiles", "uri"}:
        print("ERROR: ONVIF action must be discover, profiles, or uri")
        raise SystemExit(1)

    host = kwargs.pop("host", None)
    if host is None:
        print(f"ERROR: host= is required for ONVIF action={action}")
        raise SystemExit(1)
    username = kwargs.pop("username", os.environ.get("ONVIF_USERNAME"))
    password_env = str(kwargs.pop("password_env", "ONVIF_PASSWORD"))
    password = os.environ.get(password_env)
    port_value = kwargs.pop("port", None)
    port = int(port_value) if port_value is not None else None
    timeout = float(kwargs.pop("timeout", 5.0))
    verify_ssl = kwargs.pop("verify_ssl", True)
    time_offset = float(kwargs.pop("time_offset", 0.0))
    profile_selector = kwargs.pop("profile", None)
    if action == "profiles" and profile_selector is not None:
        print("ERROR: profile= is valid only for ONVIF action=uri")
        raise SystemExit(1)
    if kwargs:
        print(f"ERROR: unsupported ONVIF options: {', '.join(sorted(kwargs))}")
        raise SystemExit(1)

    camera = ONVIFCamera(
        str(host),
        username=username,
        password=password,
        port=port,
        timeout=timeout,
        verify_ssl=verify_ssl,
        time_offset=time_offset,
    )
    if action == "profiles":
        profiles = camera.get_profiles()
        print(f"Found {len(profiles)} media profile{'s' if len(profiles) != 1 else ''}")
        for profile in profiles:
            resolution = (
                f"{profile.width}x{profile.height}"
                if profile.width is not None and profile.height is not None
                else "unknown resolution"
            )
            fps = f" {profile.frame_rate:g}fps" if profile.frame_rate is not None else ""
            encoding = profile.encoding or "unknown codec"
            print(f"  {profile.token}: {profile.name} — {encoding} {resolution}{fps}")
        return

    print(camera.get_stream_uri(profile_selector))


def _execute(argv: list[str]) -> None:
    """Parse and execute one command without report lifecycle handling."""
    command, kwargs = parse_args(argv)

    if command == "bugreport":
        from dfine.utils.reporting import write_standalone_report

        path = write_standalone_report()
        print(f"Bug report saved to {path}")
        return

    if command == "download":
        model_name = kwargs.pop("model", "dfine_l")
        task = kwargs.pop("task", "detect")
        weights = kwargs.pop("weights", "default")
        output = kwargs.pop("output", None)
        force = kwargs.pop("force", False)
        from dfine.utils.downloads import download_model

        path = download_model(
            model=model_name,
            task=task,
            weights=weights,
            output=output,
            force=force,
        )
        print(f"Downloaded wrapped checkpoint to {path}")
        return

    if command == "gstreamer-info":
        from dfine.gstreamer import inspect_gstreamer_capabilities

        capabilities = inspect_gstreamer_capabilities()
        print("OpenCV GStreamer: " + ("yes" if capabilities["opencv_gstreamer"] else "no"))
        print("gst-inspect-1.0: " + ("yes" if capabilities["gst_inspect"] else "no"))
        print("Profiles:")
        profiles = capabilities["profiles"]
        assert isinstance(profiles, dict)
        for name, status in profiles.items():
            assert isinstance(status, dict)
            decode = "yes" if status["decode"] else "no"
            encode = "yes" if status["encode"] else "no"
            print(f"  {name:<8} decode={decode:<3} encode={encode:<3} {status['description']}")
        return

    if command == "onvif":
        from dfine.onvif import ONVIFError

        try:
            _execute_onvif(kwargs)
        except ONVIFError as exc:
            print(f"ERROR: {exc}")
            raise SystemExit(1) from None
        return

    model_path = kwargs.pop("model", "dfine_l")
    task = kwargs.pop("task", "detect")
    weights = kwargs.pop("weights", "default")

    from dfine import DFINE

    model = DFINE(model_path, task=task, weights=weights)

    if command in {"predict", "track"}:
        source = kwargs.pop("source", None)
        if source is None:
            print(f"ERROR: source= is required for {command}")
            sys.exit(1)
        _configure_rtsp_credentials(kwargs)
        output_label = _configure_output_sink(kwargs)
        if command == "predict":
            results = model.predict(source, **kwargs)
            for r in results:
                print(r)
                if getattr(r, "save_path", None):
                    print(f"Saved {r.save_path}")
            if output_label is not None:
                print(f"Wrote annotated output to {output_label}")
        else:
            tracker_kwargs = {key: kwargs.pop(key) for key in TRACKER_OPTIONS if key in kwargs}
            if tracker_kwargs:
                kwargs["tracker_kwargs"] = tracker_kwargs
            kwargs.setdefault("stream", True)
            results = model.track(source, **kwargs)
            frame_count = 0
            save_paths: list[str] = []
            for frame_count, result in enumerate(results, start=1):
                save_path = getattr(result, "save_path", None)
                if save_path and save_path not in save_paths:
                    save_paths.append(save_path)
            print(f"Tracked {frame_count} frame{'s' if frame_count != 1 else ''}")
            for save_path in save_paths:
                print(f"Saved {save_path}")
            if output_label is not None:
                print(f"Wrote annotated output to {output_label}")
    elif command == "train":
        metrics = model.train(**kwargs)
        print(metrics)
    elif command == "val":
        metrics = model.val(**kwargs)
        print(metrics)
    elif command == "export":
        path = model.export(**kwargs)
        print(f"Exported to {path}")
    elif command == "info":
        model.info(detailed=kwargs.get("detailed", False))


def main(argv: list[str] | None = None, report_dir: str | Path = "runs/bugreports") -> None:
    argv = list(argv or sys.argv)
    report_requested = "--report" in argv[2:]
    if report_requested:
        argv = [token for token in argv if token != "--report"]

    command = argv[1].lower() if len(argv) > 1 else ""
    if not report_requested:
        if command == "bugreport":
            from dfine.utils.reporting import write_standalone_report

            parse_args(argv)
            path = write_standalone_report(report_dir)
            print(f"Bug report saved to {path}")
            return
        _execute(argv)
        return

    if command not in REPORT_COMMANDS:
        print("ERROR: --report is supported only for train, predict, track, val, and export")
        raise SystemExit(1)

    from dfine.utils.reporting import capture_command_report

    failure: BaseException | None = None
    command_exit: SystemExit | None = None
    with capture_command_report(command, report_dir) as path:
        try:
            _execute(argv)
        except SystemExit as error:
            command_exit = error
        except BaseException as error:
            failure = error
            traceback.print_exc()

    print(f"Bug report saved to {path}")
    if command_exit is not None:
        raise command_exit
    if failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
