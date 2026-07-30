"""
dfine CLI — mirrors the `yolo` command from Ultralytics.

Usage:
    dfine predict  model=dfine_l weights=obj2coco source=image.jpg conf=0.5
    dfine download model=dfine_l weights=coco
    dfine train    model=dfine_l data=coco.yaml epochs=50
    dfine val      model=dfine_l data=coco.yaml
    dfine export   model=dfine_l format=onnx
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

COMMANDS = {"predict", "download", "train", "val", "export", "info", "bugreport"}
REPORT_COMMANDS = {"predict", "train", "val", "export"}
HELP_FLAGS = {"-h", "--help"}

GENERAL_HELP = """\
nitid D-FINE CLI

Usage:
  dfine COMMAND [key=value ...] [--report]

Commands:
  predict  Run object detection on an image, directory, video, URL, or webcam
  download Download and wrap an official D-FINE checkpoint
  train    Fine-tune a model on a COCO-format dataset
  val      Evaluate a model and report COCO metrics
  export   Export a model to ONNX, OpenVINO, TorchScript, or TensorRT
  info     Show model parameters, GFLOPs, and checkpoint size
  bugreport Create an environment-only log for a GitHub issue

Run "dfine COMMAND --help" for command-specific options and examples.
Add --report to train, predict, val, or export to capture output and environment details.
"""

COMMAND_HELP = {
    "predict": """\
Usage:
  dfine predict model=MODEL source=SOURCE [key=value ...]

Required:
  source=SOURCE       Image, directory, video, URL, webcam index, or stream URL

Options:
  model=MODEL         Architecture name or wrapped checkpoint path (default: dfine_l)
  weights=NAME        default, obj2coco, or coco (default: default)
  conf=FLOAT          Confidence threshold (default: 0.5)
  imgsz=INT           Square inference image size (default: 640)
  stream=BOOL         Return results as a generator (default: false)
  augment=BOOL        Use test-time augmentation (default: false)
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
        weights = kwargs.pop("weights", "default")
        output = kwargs.pop("output", None)
        force = kwargs.pop("force", False)
        from dfine.utils.downloads import download_model

        path = download_model(model=model_name, weights=weights, output=output, force=force)
        print(f"Downloaded wrapped checkpoint to {path}")
        return

    model_path = kwargs.pop("model", "dfine_l")
    weights = kwargs.pop("weights", "default")

    from dfine import DFINE

    model = DFINE(model_path, weights=weights)

    if command == "predict":
        source = kwargs.pop("source", None)
        if source is None:
            print("ERROR: source= is required for predict")
            sys.exit(1)
        results = model.predict(source, **kwargs)
        for r in results:
            print(r)
            if getattr(r, "save_path", None):
                print(f"Saved {r.save_path}")
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
        print("ERROR: --report is supported only for train, predict, val, and export")
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
