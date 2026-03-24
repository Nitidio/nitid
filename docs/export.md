# Export

Export a nitid-wrapped checkpoint to ONNX or TorchScript for deployment.

## ONNX

```python
from dfine import DFINE

model = DFINE("dfine_l_wrapped.pth")
model.export(format="onnx")
# → dfine_l_wrapped.onnx
```

```bash
uv run dfine export model=dfine_l_wrapped.pth format=onnx
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
# → dfine_l_wrapped.torchscript
```

TorchScript export uses `torch.jit.trace` with `strict=False` because
D-FINE's forward pass returns a dict.

## TensorRT

TensorRT export is not yet implemented. Calling `model.export(format="tensorrt")` raises `NotImplementedError`.

The intended workflow when implemented will be: export to ONNX first, then convert with `trtexec` or the TensorRT Python API.

## Constraint: `imgsz` must match `eval_spatial_size`

D-FINE pre-computes positional anchors for a fixed spatial size
(`eval_spatial_size`, default `[640, 640]`). Exporting at a different
`imgsz` will raise a shape mismatch error at trace time. Check the value stored in
the checkpoint config before exporting:

```python
print(model._cfg["eval_spatial_size"])  # [640, 640]
model.export(format="onnx", imgsz=640)  # must match
```
