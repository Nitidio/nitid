"""
dfine CLI — mirrors the `yolo` command from Ultralytics.

Usage:
    dfine predict  model=dfine_l.pth  source=image.jpg  conf=0.5
    dfine download model=dfine_l
    dfine train    model=dfine_l.pth  data=coco.yaml    epochs=50
    dfine val      model=dfine_l.pth  data=coco.yaml
    dfine export   model=dfine_l.pth  format=onnx
"""

from __future__ import annotations

import sys

COMMANDS = {"predict", "download", "train", "val", "export", "info"}
HELP_FLAGS = {"-h", "--help"}

GENERAL_HELP = """\
nitid D-FINE CLI

Usage:
  dfine COMMAND [key=value ...]

Commands:
  predict  Run object detection on an image, directory, video, URL, or webcam
  download Download and wrap an official D-FINE checkpoint
  train    Fine-tune a model on a COCO-format dataset
  val      Evaluate a model and report COCO metrics
  export   Export a model to ONNX, TorchScript, or TensorRT
  info     Show model parameters, GFLOPs, and checkpoint size

Run "dfine COMMAND --help" for command-specific options and examples.
"""

COMMAND_HELP = {
    "predict": """\
Usage:
  dfine predict model=MODEL source=SOURCE [key=value ...]

Required:
  source=SOURCE       Image, directory, video, URL, webcam index, or stream URL

Options:
  model=PATH          Wrapped checkpoint path (default: dfine_l.pth)
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

Examples:
  dfine predict model=dfine_l.pth source=image.jpg
  dfine predict model=dfine_l.pth source=image.jpg save=true
  dfine predict model=dfine_l.pth source=video.mp4 conf=0.3 stream=true
""",
    "download": """\
Usage:
  dfine download [model=MODEL] [key=value ...]

Options:
  model=NAME          dfine_s, dfine_m, dfine_l, or dfine_x (default: dfine_l)
  output=PATH         Output directory or .pth file (default: current directory)
  force=BOOL          Overwrite an existing wrapped checkpoint (default: false)

The command downloads the official raw checkpoint and converts it to nitid's
wrapped .pth format. The default filename is MODEL_wrapped.pth.

Examples:
  dfine download model=dfine_s
  dfine download model=dfine_m output=models
  dfine download model=dfine_l output=models/custom.pth force=true
""",
    "train": """\
Usage:
  dfine train model=MODEL data=DATA [key=value ...]

Required:
  data=PATH           Dataset YAML file using COCO-format annotations

Options:
  model=PATH          Wrapped checkpoint path (default: dfine_l.pth)
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
  wandb=BOOL          Enable Weights & Biases logging (default: false)
  mlflow=BOOL         Enable MLflow logging (default: false)
  verbose=BOOL        Print training progress (default: true)

Example:
  dfine train model=dfine_l.pth data=coco.yaml epochs=50 batch=16 mlflow=true
""",
    "val": """\
Usage:
  dfine val model=MODEL data=DATA [key=value ...]

Required:
  data=PATH           Dataset YAML file using COCO-format annotations

Options:
  model=PATH          Wrapped checkpoint path (default: dfine_l.pth)
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

Example:
  dfine val model=dfine_l.pth data=coco.yaml split=val batch=16
""",
    "export": """\
Usage:
  dfine export model=MODEL [key=value ...]

Options:
  model=PATH          Wrapped checkpoint path (default: dfine_l.pth)
  format=FORMAT       onnx, torchscript, or tensorrt (default: onnx)
  imgsz=INT           Square export image size (default: 640)
  batch=INT           Static batch size (default: 1)
  dynamic=BOOL        Enable a dynamic batch axis (default: false)
  simplify=BOOL       Simplify the ONNX graph (default: true)
  opset=INT           ONNX opset version (default: 17)
  half=BOOL           Enable FP16 TensorRT export (default: false)
  device=DEVICE       cpu, cuda, or cuda:N (default: model device)
  project=PATH        Export run root (default: runs/export)
  name=NAME           Export run name (default: exp)
  save_dir=PATH       Exact export run directory override
  output=PATH         Exact exported artifact path
  exist_ok=BOOL       Reuse/replace an explicit destination (default: false)
  verbose=BOOL        Print export progress (default: true)

Examples:
  dfine export model=dfine_l.pth format=onnx
  dfine export model=dfine_l.pth format=tensorrt half=true
""",
    "info": """\
Usage:
  dfine info model=MODEL [key=value ...]

Options:
  model=PATH          Wrapped checkpoint path (default: dfine_l.pth)
  detailed=BOOL       Include per-layer parameter counts (default: false)

Example:
  dfine info model=dfine_l.pth detailed=true
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


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv
    command, kwargs = parse_args(argv)

    if command == "download":
        model_name = kwargs.pop("model", "dfine_l")
        output = kwargs.pop("output", None)
        force = kwargs.pop("force", False)
        from dfine.utils.downloads import download_model

        path = download_model(model=model_name, output=output, force=force)
        print(f"Downloaded wrapped checkpoint to {path}")
        return

    model_path = kwargs.pop("model", "dfine_l.pth")

    from dfine import DFINE

    model = DFINE(model_path)

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


if __name__ == "__main__":
    main()
