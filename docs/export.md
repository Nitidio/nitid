# Export

Exports are written to `runs/export/exp`, then `exp2`, `exp3`, and so on by
default. Use `project` and `name`, an exact `save_dir`, or `output` for an exact
artifact filename. Set `exist_ok=True` only when replacing/reusing an explicit
destination is intentional. Export files are atomically published, and each run
includes `args.yaml` and `environment.yaml`.

Export a nitid-wrapped checkpoint to ONNX, TorchScript, or TensorRT for deployment.

## ONNX

```python
from dfine import DFINE

model = DFINE("dfine_l")
model.export(format="onnx")
# → dfine_640.onnx
```

```bash
uv run dfine export model=dfine_l format=onnx
```

The exported model takes a single input `images [B, 3, H, W]` and returns
`(labels, boxes, scores)` with the postprocessor in deploy mode baked in.

### Options

| Argument   | Default | Description |
|------------|---------|-------------|
| `imgsz`    | 640     | Input spatial size — **must match** the model's `eval_spatial_size` |
| `batch`    | 1       | Static batch size (use `dynamic=True` for variable batch) |
| `dynamic`  | `False` | Export with dynamic batch and spatial axes |
| `simplify` | `True`  | Run `onnxsim` to fold constants and simplify the graph |
| `opset`    | 17      | ONNX opset version |

### Dynamic export

```python
model.export(format="onnx", dynamic=True, simplify=False)
```

Dynamic axes: `batch` (axis 0) and spatial dimensions (axes 2, 3).

## TorchScript

```python
model.export(format="torchscript")
# → dfine_640.torchscript
```

TorchScript export uses `torch.jit.trace` with `strict=False` because
D-FINE's forward pass returns a dict.

## TensorRT

```bash
pip install --extra-index-url https://pypi.nvidia.com tensorrt>=8.6
```

```python
model.export(format="tensorrt")
# → dfine_640.engine
```

```bash
uv run dfine export model=dfine_l format=tensorrt
```

The workflow is: trace model → temporary ONNX → TensorRT engine (the intermediate ONNX is removed automatically). Requires a CUDA-capable GPU.

### Options

| Argument   | Default | Description |
|------------|---------|-------------|
| `imgsz`    | 640     | Input spatial size — **must match** `eval_spatial_size` |
| `batch`    | 1       | Static batch size (use `dynamic=True` for variable batch) |
| `dynamic`  | `False` | Build engine with dynamic batch axis (min=1, opt=batch, max=batch×4) |
| `half`     | `False` | Enable FP16 precision |

### FP16

```python
model.export(format="tensorrt", half=True)
```

FP16 halves memory usage and typically improves throughput on modern GPUs. nitid warns if the GPU reports no native FP16 support but still builds the engine.

### Dynamic batch

```python
model.export(format="tensorrt", dynamic=True, batch=4)
# min=1, opt=4, max=16
```

## Constraint: `imgsz` must match `eval_spatial_size`

D-FINE pre-computes positional anchors for a fixed spatial size
(`eval_spatial_size`, default `[640, 640]`). Exporting at a different
`imgsz` will raise a shape mismatch error at trace time. Check the value stored in
the checkpoint config before exporting:

```python
print(model._cfg["eval_spatial_size"])  # [640, 640]
model.export(format="onnx", imgsz=640)  # must match
```
