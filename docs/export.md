# Export

Exports are written to `runs/export/exp`, then `exp2`, `exp3`, and so on by
default. Use `project` and `name`, an exact `save_dir`, or `output` for an exact
artifact filename. Set `exist_ok=True` only when replacing/reusing an explicit
destination is intentional. Export files are atomically published, and each run
includes `args.yaml` and `environment.yaml`.

Export a nitid checkpoint or supported model name to ONNX, OpenVINO IR,
TorchScript, or TensorRT for deployment. Format availability depends on the
task. To run the exported artifact outside nitid, see
[Deployment](deployment.md).

| Task | ONNX | OpenVINO | TorchScript | TensorRT |
|---|---:|---:|---:|---:|
| Detection | yes | yes | yes | yes |
| Instance segmentation | yes | yes | yes | yes |
| Semantic segmentation | yes | yes | no | no |

## ONNX

```python
from nitid import NITID

model = NITID("nitid1l", task="detect")
model.export(format="onnx")
# → dfine_640.onnx
```

```bash
uv run nitid export model=nitid1l task=detect format=onnx
```

The exported model takes a single input `images [B, 3, H, W]`. Detection returns
`(labels, boxes, scores)` with the postprocessor in deploy mode baked in.
Instance segmentation returns `(labels, boxes, scores, masks)`; `masks`
contains selected mask probabilities aligned with the first three outputs.

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

### Embedded metadata (`nitid_meta`)

Detection and instance-segmentation ONNX files carry a self-describing JSON
document in the ONNX `metadata_props` under the key `nitid_meta`. Without it, a
client such as ONNX Runtime Web has no class names and cannot know how to
prepare the input. The meta is additive: the graph, input/output names and
computed values are identical to an export without it, and ONNX files written
by earlier nitid versions (which have no `nitid_meta`) keep loading. It is
written after `onnxsim`, so `simplify=True` and `simplify=False` both keep it,
and it is written with `postprocess=True` and `postprocess=False`. The value is
compact UTF-8 JSON with sorted keys, so two identical exports carry
byte-identical meta. Semantic-segmentation ONNX exports and OpenVINO/TensorRT/
TorchScript artifacts do not carry it (OpenVINO is converted from an
intermediate ONNX file that is discarded).

Read it with `nitid` installed:

```python
from dfine.export_meta import read_onnx_meta

meta = read_onnx_meta("dfine_640.onnx")  # dict, or None for older files
print(meta["names"]["0"], meta["input"]["resize"])
```

or with only `onnx` / `onnxruntime`:

```python
import json
import onnx

proto = onnx.load("dfine_640.onnx", load_external_data=False)
meta = json.loads({p.key: p.value for p in proto.metadata_props}["nitid_meta"])

import onnxruntime as ort

session = ort.InferenceSession("dfine_640.onnx")
meta = json.loads(session.get_modelmeta().custom_metadata_map["nitid_meta"])
```

Example (`nitid1s`, detection, `postprocess=True`; `names` shortened):

```json
{
  "schema_version": 1,
  "task": "detect",
  "model": "nitid1s",
  "imgsz": 640,
  "input": {"name": "images", "shape": [1, 3, 640, 640], "dtype": "float32",
            "range": "0-1", "color": "RGB", "resize": "squash"},
  "outputs": [
    {"name": "labels", "dtype": "int64"},
    {"name": "boxes", "dtype": "float32", "format": "xyxy_input_px"},
    {"name": "scores", "dtype": "float32"}
  ],
  "names": {"0": "person", "1": "bicycle"},
  "topk": 300,
  "default_iou": 0.85,
  "nitid_version": "0.9.6"
}
```

| Field | Meaning |
|---|---|
| `schema_version` | Shape of this document (currently `1`). |
| `task` | `detect` or `segment`. |
| `model` | Public model name (`nitid1s`, ...) inferred from the checkpoint architecture. Omitted when it matches no supported size; it is never an empty string. |
| `imgsz` | Square input size in pixels. |
| `input` | Name, shape, dtype and preprocessing. `resize: "squash"` means a plain resize to `imgsz x imgsz` that does **not** keep the aspect ratio (no letterbox); `range: "0-1"` means pixel values divided by 255 with no mean/std normalisation; `color: "RGB"`. This is what `nitid` itself does (`PIL` RGB, `T.Resize((imgsz, imgsz))`, `T.ToTensor()`). The first shape entry is the `batch` used for the export (with `dynamic=True` that axis is dynamic). |
| `outputs` | Real output names and dtypes, in graph order. With `postprocess=True`: `labels`, `boxes`, `scores` (+ `masks` for segment). `boxes` are `xyxy` pixels in the *model input* (`imgsz`) space, so divide by `imgsz` and multiply by the original width/height to map back to the source image. `masks` carries `stride: 4`: the mask grid is `imgsz / 4` square and holds sigmoid probabilities. With `postprocess=False` the outputs are `pred_logits`, `pred_boxes` (`cxcywh_norm`: normalised centre/size) and, for segment, `pred_masks`. |
| `names` | Class id (as a string, because JSON keys are strings) to class name. Do not rely on key order. |
| `topk` | Rows returned per image: the post-processor keeps `min(300, num_queries x num_classes)` (focal-loss models), which can be below 300 for small models. With `postprocess=False` it is the number a client should keep. |
| `default_iou` | nitid's default IoU threshold (`0.85`). The exported graph does not run NMS; nitid uses it only to merge flip-augmentation views. |
| `nitid_version` | Version of the nitid library that wrote the file. |

The file's own `sha256` is intentionally not in the meta, because a file cannot
contain its own hash; whoever serves the file can publish the hash alongside it
(Nitid Studio does: `GET /api/v1/onnx/{model}/meta` returns this document plus
`sha256`, see [Studio](studio.md)). The document is built by
`dfine.export_meta.build_onnx_meta` and the field names above are the canonical
`schema_version` 1 contract.

### Why `simplify=True` drops `graph.value_info`

`value_info` is optional documentation of intermediate tensor shapes and types. It
is not needed to run the model: every runtime infers shapes when it loads the graph.
`onnxsim` rewrites the graph but keeps these annotations, and some of them end up
wrong. In the segmentation graph, for example, the output of `Gemm_1` (`_v_2148`)
was declared with rank 0 while its real shape is `[4, 256]`. The effects:

| Consumer | Result with the stale entry |
|---|---|
| `onnx.checker.check_model(model, full_check=True)` | `ShapeInferenceError ... Inferred shape and existing shape differ in rank: (2) vs (0)` |
| onnxruntime (Python) | Loads, with a warning ("Error merging shape info ... Falling back to lenient merge") |
| onnxruntime-web 1.31.1 | Refuses to load: `ShapeInferenceError ... inferred=2 declared=0` |

Since the fix, the exporter removes every `value_info` entry (main graph and
sub-graphs) right after `onnxsim`, before the `nitid_meta` is written. Graph
inputs, outputs, initializers, nodes and metadata are untouched, and the outputs
are bit-identical: with a tiny segmentation checkpoint, onnxruntime returned
exactly the same arrays before and after, and the full ONNX checker now passes.
The file also gets a little smaller (about 40 KB on that checkpoint).
`simplify=False` never had the problem (torch writes no `value_info`).
Files exported by earlier versions can still contain it; the browser engine
keeps a fallback (`stripValueInfo`) that retries without it, and you can clear
an old file yourself:

```python
import onnx
from dfine.exporter import drop_value_info

proto = onnx.load("old_segment.onnx")
drop_value_info(proto)
onnx.save(proto, "old_segment_fixed.onnx")
```

## OpenVINO

Install the optional OpenVINO dependency:

```bash
uv sync --extra openvino
```

Export through the corrected ONNX graph to OpenVINO Intermediate Representation
(IR):

```python
from nitid import NITID

model = NITID("nitid1l", task="detect")
xml_path = model.export(format="openvino")
# → runs/export/exp/dfine_640.xml
# → runs/export/exp/dfine_640.bin
```

```bash
uv run nitid export model=nitid1l task=detect format=openvino
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
Detection, instance segmentation, and semantic segmentation all route through
this ONNX-to-IR path.

Run exported models with OpenVINO 2026.4.1 or later, which the `openvino`
extra installs. On an Intel GPU with FP16 inference precision, OpenVINO
2026.2.1 returns correct detections for `nitid1m` only on the first inference
after `compile_model()`; every later inference returns the same low score for
all 300 queries, which evaluates to 0.0 mAP. The bug is in the GPU plugin, so
an IR exported with any nitid version works once the runtime is upgraded.
Keeping the model on FP32 (`INFERENCE_PRECISION_HINT: "f32"`) also avoids it,
at about 3.4 times the latency. See
[#52](https://github.com/Nitidio/nitid/issues/52).

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
uv run nitid export model=nitid1l task=detect format=tensorrt
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

## Raw export (no postprocessor)

By default the exported graph has the postprocessor baked in, so it returns the
decoded `(labels, boxes, scores, ...)` contract described above. Pass
`postprocess=False` to export the decoder's raw outputs instead and do the
decoding in the consuming application:

```python
model.export(format="openvino", postprocess=False)
model.export(format="onnx", postprocess=False)
```

```bash
uv run nitid export model=nitid1l task=detect format=openvino postprocess=false
```

This is the same graph the `backend="openvino"` inference path compiles in
memory, only written to disk as a regular export artifact. Use it when the
consumer already has its own (often batched or fused) postprocessing, or when it
needs every decoder query rather than the top-k the postprocessor keeps.

Output names per task:

| Task | Raw outputs |
|---|---|
| Detection | `pred_logits [B, Q, C]`, `pred_boxes [B, Q, 4]` |
| Instance segmentation | `pred_logits`, `pred_boxes`, `pred_masks` |
| Semantic segmentation | `sem_seg_logits [B, C, H, W]` |

`Q` is the number of decoder queries (300 for the official checkpoints) — all of
them, with no score threshold or top-k applied. `pred_logits` are unactivated,
and `pred_boxes` are `cxcywh` normalised to `[0, 1]`.

The consumer has to reproduce what the postprocessor would have done:

1. `scores = sigmoid(pred_logits)` (focal-loss models; softmax otherwise)
2. `topk` over the flattened `[Q * C]` scores, then `label = index % C` and
   `query = index // C`
3. convert the gathered boxes from `cxcywh` to `xyxy`
4. scale by the input `[w, h, w, h]`

No NMS is involved — D-FINE is NMS-free. The reference implementation is
`DFINEPostProcessor.forward()` in
[`dfine/nn/postprocessor.py`](https://github.com/Nitidio/nitid/blob/main/dfine/nn/postprocessor.py),
which also decodes the instance masks.

For semantic segmentation the raw export returns the same dense logit map, only
named `sem_seg_logits` instead of `semantic_logits`; apply softmax/argmax as
usual.

`postprocess=False` applies to ONNX, OpenVINO, and TensorRT. TorchScript always
traces the raw model (it has no postprocessor to drop), so the flag makes no
difference there.

## Export spatial size

Most checkpoints are calibrated for a fixed evaluation spatial size
(`eval_spatial_size`, usually `[640, 640]`). Use the checkpoint's configured
size unless you have intentionally trained/evaluated another size.

```python
print(model._cfg["eval_spatial_size"])  # [640, 640]
model.export(format="onnx", imgsz=640)
```
