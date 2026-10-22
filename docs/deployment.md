# Deployment

This guide starts from an artifact created by `nitid export` and shows how to
run it outside nitid: with ONNX Runtime, OpenVINO Runtime, LibTorch (C++) and
TensorRT under Triton. [Export](export.md) covers how to create each artifact
and its options. Export at the image size the checkpoint was trained or
validated at, usually 640.

## Export contract

Every example feeds the same input: one `float32` tensor named `images`, RGB,
`NCHW`, resized to the export size (a plain resize, no letterbox) and scaled to
`[0, 1]`, with no mean/std normalisation. This matches nitid's own
preprocessing.

ONNX, OpenVINO and TensorRT exports include the postprocessor, so detection
returns:

| Output | Shape | Type | Meaning |
|---|---|---|---|
| `labels` | `[batch, 300]` | `int64` | zero-based class IDs |
| `boxes` | `[batch, 300, 4]` | `float32` | `x1, y1, x2, y2` in pixels of the network input |
| `scores` | `[batch, 300]` | `float32` | confidence scores |

`300` is the number of kept detections, sorted by score; filter them with your
own threshold. Boxes are in the coordinates of the resized input, so scale them
back by `original_width / export_size` and `original_height / export_size`.

Instance segmentation adds `masks`, and semantic segmentation returns
`semantic_logits [batch, classes, height, width]` instead. To export the
decoder's raw outputs and decode them yourself, see
[Raw export](export.md#raw-export-no-postprocessor). TorchScript always exports
the raw outputs.

## ONNX Runtime

Install ONNX Runtime and export a detection model:

```bash
pip install onnxruntime
nitid export model=model1l task=detect format=onnx output=model.onnx
```

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

SIZE = 640

image = Image.open("image.jpg").convert("RGB")
array = np.asarray(image.resize((SIZE, SIZE)), dtype=np.float32) / 255.0
images = np.transpose(array, (2, 0, 1))[None]

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
labels, boxes, scores = session.run(None, {"images": images})

keep = scores[0] >= 0.5
scale = np.array([image.width, image.height, image.width, image.height]) / SIZE
for label, box, score in zip(labels[0][keep], boxes[0][keep] * scale, scores[0][keep]):
    print(int(label), box.round(1), round(float(score), 3))
```

Use `CUDAExecutionProvider` when the installed ONNX Runtime package and CUDA
versions match. Check `ort.get_available_providers()` rather than assuming that
a provider is usable.

## OpenVINO Runtime

For Intel CPUs, integrated GPUs and NPUs. Install the extra and export:

```bash
pip install "nitid[openvino]"
nitid export model=model1l task=detect format=openvino output=model.xml
```

The exported IR has the same inputs and outputs as the ONNX model, so the
preprocessing and the box scaling above apply unchanged:

```python
import openvino as ov

core = ov.Core()
compiled = core.compile_model("model.xml", "AUTO")  # or "CPU", "GPU", "NPU"
result = compiled({"images": images})
labels, boxes, scores = (result[name] for name in ("labels", "boxes", "scores"))
```

nitid can also run the IR itself: `NITID(..., backend="openvino")` predicts with
OpenVINO Runtime through the usual API. The
[CUDA and OpenVINO guide](training_cuda_openvino.md) runs the whole path, from
training on a GPU to inference on an Intel machine.

### Measuring latency and accuracy on Intel hardware

`tools/benchmark_openvino.py` measures the exported detection models on every
OpenVINO device of the machine. It lives in the repository, not in the wheel,
so run it from a clone. On Linux, one script does the whole run:

```bash
scripts/benchmark_openvino.sh                 # latency, then COCO mAP
scripts/benchmark_openvino.sh --latency-only
```

It checks the hardware, the GPU and NPU drivers, the power profile, the load
and the free disk space, and prints the `sudo` command for anything it cannot
fix itself. It then installs `uv` if needed, sets up the environment,
downloads COCO val2017 into `datasets/coco`, benchmarks every device that
OpenVINO sees and packs the reports into a `.tar.gz`. To run the steps by hand:

```bash
uv sync --extra openvino --extra train
uv run python -m tools.benchmark_openvino --coco datasets/coco --output runs/benchmark/openvino
```

`--coco` points at a COCO root with `val2017/` and
`annotations/instances_val2017.json`. By default the script exports `model1s`
to `model1x`, then runs CPU in FP32 and the integrated GPU and NPU in FP16
(`--models` and `--matrix GPU:FP32 ...` change that). For each combination it
reports:

- latency: 1000 warm-up runs, then one timed run per COCO image, batch 1 at
  640x640 with the postprocessor included. This is the method behind the
  README's T4 TensorRT figures, so the numbers can be read side by side;
- COCO mAP50-95 of the same IR on the same device and precision;
- compile time, first-run time, package temperature and, where readable,
  package power (RAPL usually needs root);
- how much CPU other processes used during the run. A run where they averaged
  more than one core (`--max-background-cpu`) is marked ⚠ and listed under
  "Unreliable runs": something else was competing for the machine, so rerun it
  when the machine is idle.

Results go to `results.json`, stamped with the CPU, devices, driver packages,
library versions and weight hashes, and to a `results.md` summary.
`benchmark_app.sh` holds the equivalent `benchmark_app` commands for a
cross-check. Expect `benchmark_app` to report lower latency on GPUs: it
repeats one input, and D-FINE's data-dependent sampling is faster when every
run reads the same memory.

## TorchScript in C++

Export the PyTorch graph:

```bash
nitid export model=model1l task=detect format=torchscript output=model.torchscript
```

Link against LibTorch and load the module with the same preprocessing:

```cpp
#include <torch/script.h>

#include <iostream>

int main() {
  torch::jit::script::Module model = torch::jit::load("model.torchscript");
  model.eval();

  torch::Tensor images = torch::zeros({1, 3, 640, 640}, torch::kFloat32);
  c10::IValue value = model.forward({images});
  c10::impl::GenericDict outputs = value.toGenericDict();

  auto logits = outputs.at("pred_logits").toTensor();
  auto boxes = outputs.at("pred_boxes").toTensor();
  std::cout << logits.sizes() << " " << boxes.sizes() << "\n";
}
```

TorchScript returns the raw decoder outputs: `pred_logits` are unactivated and
`pred_boxes` are normalised `cx, cy, w, h`, for every decoder query. Decode them
as described in [Raw export](export.md#raw-export-no-postprocessor) before
presenting results.

## TensorRT with Triton

The export needs a CUDA-capable system with TensorRT installed (see
[Export › TensorRT](export.md#tensorrt)):

```bash
nitid export model=model1l task=detect format=tensorrt batch=1 output=model.engine
```

Create a Triton model repository:

```text
model_repository/
└── nitid/
    ├── config.pbtxt
    └── 1/
        └── model.plan
```

Copy `model.engine` to `nitid/1/model.plan`. A fixed-batch detection
configuration is:

```protobuf
name: "nitid"
platform: "tensorrt_plan"
max_batch_size: 0
input [{ name: "images" data_type: TYPE_FP32 dims: [1, 3, 640, 640] }]
output [
  { name: "labels" data_type: TYPE_INT64 dims: [1, 300] },
  { name: "boxes" data_type: TYPE_FP32 dims: [1, 300, 4] },
  { name: "scores" data_type: TYPE_FP32 dims: [1, 300] }
]
instance_group [{ kind: KIND_GPU count: 1 }]
```

TensorRT engines are tied to the GPU model and TensorRT version they were built
with, so build the engine on the deployment machine or an identical one.
Confirm the tensor names and types with `trtexec --loadEngine=model.plan`
before starting Triton, because other tasks add outputs. Start the server with:

```bash
tritonserver --model-repository=/models/model_repository
```

## Batching and dynamic shapes

- A static export accepts exactly the exported batch and spatial size.
- `dynamic=true` makes the ONNX batch dimension dynamic.
- A dynamic TensorRT export builds a batch profile with minimum `1`, optimum
  `batch` and maximum `batch * 4`.
- The spatial size stays fixed: the decoder's anchors are computed for the
  export size.
- Measure warm, steady-state latency. Model loading, CUDA context creation and
  TensorRT engine initialisation should not count towards per-request latency.

Example dynamic export:

```bash
nitid export model=model1l task=detect format=onnx batch=4 dynamic=true
```
