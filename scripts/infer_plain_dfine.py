"""
Inference using plain D-FINE (the upstream submodule API).

Usage:
    uv run python scripts/infer_plain_dfine.py \
        --config extern/dfine/configs/dfine/dfine_hgnetv2_l_coco.yml \
        --weights path/to/raw_dfine_l.pth \
        --image   path/to/image.jpg \
        --conf    0.5 \
        --device  cpu

Notes:
    - Expects a *raw* D-FINE checkpoint (keys: model or ema.module),
      NOT a nitid-wrapped checkpoint.
    - Output image saved to plain_dfine_result.jpg
"""
import argparse
import os
import sys
import types

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw

# Make extern/dfine importable
_DFINE_ROOT = os.path.join(os.path.dirname(__file__), "..", "extern", "dfine")
sys.path.insert(0, _DFINE_ROOT)

# Block src/__init__.py from eagerly importing src.data (→ faster_coco_eval)
# before we can control the import order — same trick as dfine/nn/build.py.
if "src" not in sys.modules:
    stub = types.ModuleType("src")
    stub.__path__ = [os.path.join(_DFINE_ROOT, "src")]
    stub.__package__ = "src"
    sys.modules["src"] = stub

if "src.data" not in sys.modules:
    from torch.utils.data import DataLoader as _DL
    data_stub = types.ModuleType("src.data")
    data_stub.DataLoader = _DL
    sys.modules["src.data"] = data_stub

if "src.misc" not in sys.modules:
    misc_stub = types.ModuleType("src.misc")
    misc_stub.__path__ = [os.path.join(_DFINE_ROOT, "src", "misc")]
    misc_stub.__package__ = "src.misc"
    sys.modules["src.misc"] = misc_stub

from src.core import YAMLConfig  # noqa: E402

# Trigger @register() decorators so the component registry is populated
import src.nn    # noqa: F401, E402
import src.optim  # noqa: F401, E402
import src.zoo    # noqa: F401, E402


def build_model(config_path: str, weights_path: str, device: str) -> nn.Module:
    cfg = YAMLConfig(config_path, resume=weights_path)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    return _Model().to(device).eval()


def infer(model: nn.Module, image_path: str, conf: float, device: str) -> None:
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(device)

    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    tensor = transform(im).unsqueeze(0).to(device)

    with torch.no_grad():
        labels, boxes, scores = model(tensor, orig_size)

    labels = labels[0]
    boxes  = boxes[0]
    scores = scores[0]

    mask = scores > conf
    labels, boxes, scores = labels[mask], boxes[mask], scores[mask]

    print(f"Detections above {conf}: {mask.sum().item()}")
    draw = ImageDraw.Draw(im)
    for label, box, score in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"cls={int(label)} {score:.2f}", fill="blue")

    out_path = "plain_dfine_result.jpg"
    im.save(out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",  required=True, help="D-FINE YAML config path")
    parser.add_argument("-w", "--weights", required=True, help="Raw D-FINE .pth checkpoint")
    parser.add_argument("-i", "--image",   required=True, help="Input image path")
    parser.add_argument("--conf",   type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = build_model(args.config, args.weights, args.device)
    infer(model, args.image, args.conf, args.device)
