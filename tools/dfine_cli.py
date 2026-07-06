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


def parse_args(argv: list[str]) -> tuple[str, dict]:
    """Parse 'command key=value ...' style args."""
    if len(argv) < 2 or argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    command = argv[1]
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
