"""
Inference using the nitid DFINE wrapper.

Usage:
    uv run python scripts/infer_nitid.py \
        --model  path/to/wrapped_dfine_l.pth \
        --image  path/to/image.jpg \
        --conf   0.5 \
        --device cpu

Notes:
    - Expects a *nitid-wrapped* checkpoint (keys: model, config, names).
      Convert a raw D-FINE checkpoint first if needed:
        uv run python tools/convert_checkpoint.py \
            --weights dfine_l.pth \
            --model   dfine_l \
            --task    detect \
            --names   configs/datasets/coco.yml \
            --output  dfine_l_wrapped.pth
    - Output image saved to nitid_result.jpg
"""

import argparse
from typing import Any, cast

from PIL import Image, ImageDraw

from dfine import DFINE


def infer(model_path: str, image_path: str, conf: float, device: str) -> None:
    model = DFINE(model_path, device=device)

    results = cast(list[Any], model.predict(image_path, conf=conf, stream=False))
    result = results[0]

    print(f"Detections above {conf}: {len(result.boxes)}")

    im = Image.fromarray(result.orig_img[..., ::-1])  # BGR → RGB
    draw = ImageDraw.Draw(im)
    boxes = result.boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        score = boxes.conf[i].item()
        cls = int(boxes.cls[i].item())
        name = model.names.get(cls, str(cls))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{name} {score:.2f}", fill="blue")

    out_path = "nitid_result.jpg"
    im.save(out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Nitid-wrapped .pth checkpoint")
    parser.add_argument("-i", "--image", required=True, help="Input image path")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    infer(args.model, args.image, args.conf, args.device)
