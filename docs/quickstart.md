# Quickstart

## Installation

```bash
git clone <repo> && cd nitid
git submodule update --init        # pulls extern/dfine
uv sync --extra dev
```

## Convert a raw D-FINE checkpoint

Raw D-FINE checkpoints (weights only) must be converted to the nitid self-contained format
before use. The converter embeds the model config and class names into a single `.pth` file.

```bash
uv run python tools/convert_checkpoint.py \
    --weights dfine_l.pth \
    --config  extern/dfine/configs/dfine/dfine_hgnetv2_l_coco.yml \
    --names   configs/datasets/coco.yml \
    --output  dfine_l_wrapped.pth
```

**Important notes:**
- `--config` must be one of the canonical D-FINE configs from `extern/dfine/configs/`. The
  converter resolves all `__include__` directives so the full model definition is embedded.
- `--names` must be a file with a `names:` mapping (`{int: str}` or list). Use
  `configs/datasets/coco.yml` for COCO models, or your own dataset config.
- When the raw checkpoint contains EMA weights (`ckpt["ema"]["module"]`), the converter
  automatically uses them — this matches D-FINE's own inference scripts and gives better
  accuracy than the non-EMA weights.

## Predict

```python
from dfine import DFINE

model = DFINE("dfine_l_wrapped.pth")
results = model.predict("image.jpg", conf=0.5)

for r in results:
    print(r)          # Results(path='image.jpg', detections=17)
    r.save("out.jpg") # draws boxes and saves
```

Or via the CLI:

```bash
uv run dfine predict model=dfine_l_wrapped.pth source=image.jpg conf=0.5
```

## Iterate detections

```python
result = results[0]
boxes = result.boxes          # Boxes object

for i in range(len(boxes)):
    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
    conf = boxes.conf[i].item()
    cls  = int(boxes.cls[i].item())
    name = result.names[cls]
    print(f"{name} {conf:.2f}  [{x1:.0f} {y1:.0f} {x2:.0f} {y2:.0f}]")
```

## Export

```bash
uv run dfine export model=dfine_l_wrapped.pth format=onnx
uv run dfine export model=dfine_l_wrapped.pth format=torchscript
```

Export `imgsz` must match the model's `eval_spatial_size` (default 640). See `docs/export.md`.
