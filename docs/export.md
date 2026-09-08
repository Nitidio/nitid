# Export

Exports are written to `runs/export/exp`, then `exp2`, `exp3`, and so on by
default. Use `project` and `name`, an exact `save_dir`, or `output` for an exact
artifact filename. Set `exist_ok=True` only when replacing/reusing an explicit
destination is intentional. Export files are atomically published, and each run
includes `args.yaml` and `environment.yaml`.

Export a nitid checkpoint or supported model name to ONNX, OpenVINO IR,
TorchScript, or TensorRT for deployment. Format availability depends on the
task.

| Task | ONNX | OpenVINO | TorchScript | TensorRT |
|---|---:|---:|---:|---:|
| Detection | yes | yes | yes | yes |
| Instance segmentation | yes | yes | yes | yes |
| Semantic segmentation | yes | yes | no | no |
| Pose | yes | yes | no | no |
| Oriented bounding boxes | yes | yes | no | no |

## ONNX

```python
from dfine import NITID

model = NITID("nitid1l", task="detect")
model.export(format="onnx")
# → dfine_640.onnx
```

```bash
uv run dfine export model=nitid1l task=detect format=onnx
```

The exported model takes a single input `images [B, 3, H, W]`. Detection returns
`(labels, boxes, scores)` with the postprocessor in deploy mode baked in.
Instance segmentation returns `(labels, boxes, scores, masks)`; `masks`
contains selected mask probabilities aligned with the first three outputs. Pose
returns `(labels, boxes, scores, keypoints)`, where boxes are derived from the
selected keypoints for compatibility with common downstream consumers.

OBB exports return `(labels, boxes, scores)`. OBB `boxes` have shape
`[B, topk, 5]` and use `cx, cy, w, h, angle` in input-image pixels.

Semantic segmentation exports return one output, `semantic_logits [B, C, H,
W]`, at model-input resolution. Apply softmax for per-class probabilities and
argmax over axis 1 for the class-ID map. The integration test executes this
graph with ONNX Runtime and checks numerical parity with PyTorch.

```python
semantic = NITID("semantic_best.pth", task="semantic")
semantic.export(format="onnx", output="semantic.onnx")
```

Semantic ONNX and OpenVINO exports are intended for dense-map deployment. Apply
the same softmax/argmax post-processing after either backend.

### Options

| Argument   | Default | Description |
|------------|---------|-------------|
| `imgsz`    | 640     | Input spatial size. Use the checkpoint's trained/eval size unless you intentionally trained another size. |
| `batch`    | 1       | Static batch size (use `dynamic=True` for variable batch) |
| `dynamic`  | `False` | Export with dynamic batch and spatial axes |
| `simplify` | `True`  | Run `onnxsim` to fold constants and simplify the graph |
| `opset`    | 17      | ONNX opset version |

### Dynamic export

```python
model.export(format="onnx", dynamic=True, simplify=False)
```

Dynamic export currently marks the batch axis as dynamic.

## OpenVINO

Install the optional OpenVINO dependency:

```bash
uv sync --extra openvino
```

Export through the corrected ONNX graph to OpenVINO Intermediate Representation
(IR):

```python
from dfine import NITID

model = NITID("nitid1l", task="detect")
xml_path = model.export(format="openvino")
# → runs/export/exp/dfine_640.xml
# → runs/export/exp/dfine_640.bin
```

```bash
uv run dfine export model=nitid1l task=detect format=openvino
```

The `.xml` file stores the graph and the matching `.bin` file stores its weights.
`export()` returns the `.xml` path. nitid first exports its deployment graph to a
temporary ONNX file, converts it with `openvino.convert_model()`, verifies that
OpenVINO can load and compile the IR on CPU, and then publishes the `.bin` and
`.xml` files. The temporary ONNX file is removed automatically.

Load the result for inference with OpenVINO Runtime:

```python
import openvino as ov

core = ov.Core()
compiled_model = core.compile_model(xml_path, "AUTO")
outputs = compiled_model([images])
```

OpenVINO export uses the ONNX-related `imgsz`, `batch`, `dynamic`, `simplify`,
and `opset` arguments. Set `half=True` to compress IR weights to FP16; the
default preserves FP32 weights to make numerical comparison with ONNX easier.
Detection, instance segmentation, semantic segmentation, pose, and OBB all
route through this ONNX-to-IR path.

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
uv run dfine export model=nitid1l task=detect format=tensorrt
```

The workflow is: trace model → temporary ONNX → TensorRT engine (the intermediate ONNX is removed automatically). Requires a CUDA-capable GPU.

### Options

| Argument   | Default | Description |
|------------|---------|-------------|
| `imgsz`    | 640     | Input spatial size |
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

## Export spatial size

Most checkpoints are calibrated for a fixed evaluation spatial size
(`eval_spatial_size`, usually `[640, 640]`). Use the checkpoint's configured
size unless you have intentionally trained/evaluated another size. OBB export
aligns its internal RiO-DETR anchors to the requested export size.

```python
print(model._cfg["eval_spatial_size"])  # [640, 640]
model.export(format="onnx", imgsz=640)
```
