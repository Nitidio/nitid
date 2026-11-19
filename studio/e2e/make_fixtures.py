"""Build the inputs of the Studio end-to-end tests: tiny checkpoints and a test image.

Usage::

    PYTHONPATH=<repo> python studio/e2e/make_fixtures.py OUT_DIR

Writes ``OUT_DIR/models/tiny-det.pth``, ``OUT_DIR/models/tiny-seg.pth`` and
``OUT_DIR/image.png``. Nothing is downloaded and nothing is committed: the models are
random-weight networks (compact decoder, 2 classes, 10 queries) anchored at 640 px, which is
what the server and the ONNX export use when the caller does not pass ``imgsz``. A few head
parameters are perturbed exactly as ``studio/tools/make_parity_fixtures.py`` does, so scores,
boxes and masks are varied instead of degenerate. The tiny model is meaningless as a
detector; it exercises the plumbing and the arithmetic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
IMGSZ = 640
IMAGE_SIZE = (320, 240)  # width, height: not square, smaller than IMGSZ (upscaled)


def _load_parity_tool():
    spec = importlib.util.spec_from_file_location(
        "make_parity_fixtures", HERE.parent / "tools" / "make_parity_fixtures.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.IMGSZ = IMGSZ  # make_checkpoint reads this module-level constant
    return module


def build(out_dir: Path) -> dict[str, Path]:
    tool = _load_parity_tool()
    models = out_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for task, name in (("detect", "tiny-det"), ("segment", "tiny-seg")):
        built = tool.make_checkpoint(models, task)  # writes models/<task>.pth
        target = models / f"{name}.pth"
        built.replace(target)
        paths[task] = target
    rgb = tool.synth_image(*IMAGE_SIZE)
    image = out_dir / "image.png"
    if not cv2.imwrite(str(image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("could not write the test image")
    paths["image"] = image
    return paths


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    for name, path in build(Path(argv[1])).items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
