# Deployment

This guide starts with an artifact created by `dfine export` and shows how to
run it outside nitid. Export at the same image size used for training unless the
checkpoint was validated at another size.

## Export contracts

All examples use normalized RGB `float32` tensors in `NCHW` order. Detection
ONNX and TensorRT exports include post-processing and return:

| Output | Shape | Meaning |
|---|---|---|
| `labels` | `[batch, topk]` | zero-based class IDs |
| `boxes` | `[batch, topk, 4]` | `x1, y1, x2, y2` in input pixels |
| `scores` | `[batch, topk]` | confidence scores |

Instance segmentation adds `masks`; pose adds `keypoints`; OBB uses five-value
`cx, cy, width, height, angle` boxes. Semantic exports return
`semantic_logits [batch, classes, height, width]`.

TorchScript preserves the model's raw dictionary output. Apply the matching
nitid postprocessor when decoded boxes are required.

## ONNX Runtime

Install ONNX Runtime and export a detection model:

```bash
python -m pip install onnxruntime
dfine export model=nitid1l format=onnx imgsz=640 output=model.onnx
```

```python
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def prepare_image(path: str, size: int = 640) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((size, size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None]


session = ort.InferenceSession(
    str(Path("model.onnx")), providers=["CPUExecutionProvider"]
)
labels, boxes, scores = session.run(None, {"images": prepare_image("image.jpg")})
keep = scores[0] >= 0.5
print(labels[0][keep], boxes[0][keep], scores[0][keep])
```

Use `CUDAExecutionProvider` when the installed ONNX Runtime package and CUDA
versions match. Check `ort.get_available_providers()` rather than assuming that
a provider is usable.

## TorchScript in C++

Export the raw PyTorch graph:

```bash
dfine export model=nitid1l format=torchscript imgsz=640 output=model.torchscript
```

Link against LibTorch and load the module with the same preprocessing used
above:

```cpp
#include <torch/script.h>

#include <iostream>
#include <vector>

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

`pred_boxes` are normalized model outputs, not the decoded pixel boxes from the
ONNX contract. Use nitid's task postprocessor or implement the documented
decoder before presenting results.

## TensorRT with Triton

Export requires a CUDA-capable system with TensorRT installed:

```bash
dfine export model=nitid1l format=tensorrt imgsz=640 batch=1 output=model.engine
```

Create this Triton model repository:

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
  { name: "labels" data_type: TYPE_INT64 dims: [1, -1] },
  { name: "boxes" data_type: TYPE_FP32 dims: [1, -1, 4] },
  { name: "scores" data_type: TYPE_FP32 dims: [1, -1] }
]
instance_group [{ kind: KIND_GPU count: 1 }]
```

Confirm the exact tensor names and data types with `trtexec --loadEngine` before
starting Triton because task-specific exports add or change outputs. Start the
server with:

```bash
tritonserver --model-repository=/models/model_repository
```

## Batching and dynamic shapes

- A static export accepts exactly the exported batch and spatial dimensions.
- `dynamic=true` makes the ONNX batch dimension dynamic.
- TensorRT dynamic export builds a batch profile with minimum `1`, optimum
  `batch`, and maximum `batch * 4`.
- Group requests with the same image size. Padding mixed sizes wastes compute
  and can change latency enough to invalidate a benchmark.
- Measure warm and steady-state performance. Model loading, CUDA context
  creation, and TensorRT engine initialization should not be included in
  per-request latency.

Example dynamic export:

```bash
dfine export model=nitid1l format=onnx imgsz=640 batch=4 dynamic=true
```

## Production web application

Run one API worker because the in-process model cache is not safe for
concurrent inference:

```bash
NITID_SECRET_KEY="$(openssl rand -hex 32)" \
  uv run uvicorn web.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Build the frontend with its API URL configured for the public origin, then
serve the generated `web/frontend/dist` directory through the same reverse
proxy used for the API.

```nginx
server {
    listen 443 ssl http2;
    server_name vision.example.com;

    root /srv/nitid/web/frontend/dist;
    location / { try_files $uri /index.html; }

    location ~ ^/(auth|models|runs|files|devices)(/|$) {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 2g;
        proxy_read_timeout 3600s;
    }
}
```

Before exposing the service:

1. Set a unique `NITID_SECRET_KEY`; changing it invalidates existing tokens.
2. Terminate TLS at the reverse proxy and redirect HTTP to HTTPS.
3. Restrict upload size and access at the proxy according to the deployment.
4. Persist the database, model, upload, and result directories.
5. Keep the API bound to a private interface and back up the database.

See [Web App](web_app.md) for all environment variables and storage paths.

## Benchmark exported formats

Use the benchmark command to export and compare formats on the target machine.
Warm-up calls are excluded from the reported mean latency, throughput in
batches per second, and image FPS:

```bash
dfine benchmark model=nitid1s formats=[onnx,torchscript,tensorrt] \
  imgsz=640 batch=1 warmup=10 iterations=100
```

Each run writes its artifacts, arguments, environment details, and
`benchmark.json` under `runs/benchmark/`. A missing runtime is shown as skipped
without hiding results from the formats that are installed. TensorRT requires
`device=cuda` and a CUDA-capable machine.

Use `formats=[onnx,torchscript]` on CPU-only systems. Always compare on the
final production hardware because laptop results do not predict accelerator or
server performance.
