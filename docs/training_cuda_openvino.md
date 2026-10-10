# Training on CUDA and OpenVINO machines

This guide takes a freshly installed machine to a finished end-to-end run:
install, download a small dataset, fine-tune, validate, predict, export, and
run the exported model. It covers two machines:

- **Path A, NVIDIA GPU.** Training and inference run on the GPU through PyTorch
  and CUDA.
- **Path B, Intel machine with OpenVINO.** Training runs on the CPU through
  PyTorch. The trained model then runs through OpenVINO Runtime on the Intel
  CPU, integrated GPU, or NPU.

The runs are deliberately tiny (160 images, 1–2 epochs). They check that every
step works on the machine; they do not produce a useful model.

## What OpenVINO does in nitid

OpenVINO is an inference runtime. In nitid it is used in two places:

| Operation | Where it runs |
|-----------|---------------|
| `train()` | PyTorch: an NVIDIA GPU (`device="cuda"`) or the CPU (`device="cpu"`) |
| `val()` | PyTorch, same devices as training |
| `export(format="openvino")` | PyTorch traces the model, OpenVINO converts it to IR (`.xml` + `.bin`) |
| `predict()` / `track()` with `backend="openvino"` | OpenVINO Runtime on `"CPU"`, `"GPU"` (Intel integrated GPU), or `"NPU"` |

Training on an Intel GPU or NPU is not supported. The PyTorch build that nitid
installs has no Intel GPU (`xpu`) support, and the trainer only selects CUDA or
the CPU. On an Intel machine, training therefore runs on the CPU and is slow.
Keep the model small (`model1s`) and the dataset small.

`backend="openvino"` is inference-only: `train()` and `val()` raise an error on
such a model.

## Before you start

**Repository access.** This guide works from a git clone of the repository,
which is private. The machine needs read access to `Nitidio/nitid` on GitHub,
either through an SSH key added to your GitHub account or through the GitHub
CLI (`gh auth login`).

While the repository is private, its release files cannot be downloaded
anonymously, so the automatic dataset download in step 2 fails with
`HTTP Error 404`. Download the archive with the GitHub CLI instead, from the
repository root:

```bash
gh release download datasets-v1 --repo Nitidio/nitid --pattern coco-mini.zip --dir datasets
unzip -q datasets/coco-mini.zip -d datasets && rm datasets/coco-mini.zip
```

**Pretrained weights.** These are not stored in the nitid repository.
`NITID("model1s", task="detect")` downloads the original D-FINE weights from a
public GitHub release (`github.com/Peterande/storage`). Instance segmentation
weights come from a public Hugging Face repository (`huggingface.co/ArgoSA/D-FINE-seg`).
No account or token is needed. The machine needs outbound HTTPS to `github.com`,
`objects.githubusercontent.com`, `huggingface.co`, `pypi.org`,
and `download.pytorch.org`. The dataset in step 2 also downloads from
`github.com`. The wrapped checkpoint is saved in the current directory, so run
every command from the repository root.

**Operating system.** The commands are for Linux on x86_64. They were run on
Ubuntu 24.04. The lock file also resolves on Windows x86_64, but the driver
steps below are Linux-only.

**Disk and memory.** Allow about 10 GB of free disk: the virtual environment
is about 6 GB, because Linux x86_64 installs the CUDA build of PyTorch even on
machines without an NVIDIA GPU. CPU training of `model1s` with `batch=4` peaked
at about 5 GB of RAM.

## Step 1: Install nitid (both machines)

Install the system tools and `uv`:

```bash
sudo apt-get update
sudo apt-get install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"   # or open a new terminal
uv --version
```

Clone the repository. With the GitHub CLI:

```bash
sudo apt-get install -y gh
gh auth login
gh repo clone Nitidio/nitid
cd nitid
```

or with an SSH key already registered on GitHub:

```bash
git clone git@github.com:Nitidio/nitid.git
cd nitid
```

Create the environment. The `train` extra adds `pycocotools`, needed for
training and validation. The `openvino` extra adds OpenVINO Runtime.

```bash
# Path A, NVIDIA machine
uv sync --extra train

# Path B, Intel machine
uv sync --extra train --extra openvino
```

`uv` installs the exact versions in `uv.lock`. It also downloads a suitable
Python (3.10 to 3.13) if the system one is not supported. On Linux x86_64
with Python 3.10 to 3.12 the lock pins PyTorch `2.5.1+cu121` and torchvision
`0.20.1+cu121` from the PyTorch CUDA 12.1 index; Python 3.13 takes PyTorch
`2.13.0` and torchvision `0.28.0` from PyPI, whose Linux wheels are CUDA builds
as well; the OpenVINO extra resolves to OpenVINO 2026.2.

Check the installation:

```bash
uv run nitid --help
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Step 2: Get a small dataset (both machines)

The guide uses COCO-mini, the example dataset declared in
`configs/datasets/coco-mini.yml`. It has 128 training and 32 validation images
from COCO val2017, in COCO JSON format with boxes and instance masks. Nothing
needs to be run in this step: the first `train()` or `val()` call downloads the
archive (about 26 MB) from the `datasets-v1` release, checks its sha256, and
extracts it to `datasets/coco-mini/`:

```text
datasets/coco-mini/
  ATTRIBUTION.md
  images/train/*.jpg   (128)
  images/val/*.jpg     (32)
  annotations/instances_train.json
  annotations/instances_val.json
```

Only images licensed CC BY 2.0, "No known copyright restrictions" or "United
States Government Work" are included; `ATTRIBUTION.md` lists each image's
Flickr source and licence, and the annotations are CC BY 4.0.

The class list is the same as that of the pretrained COCO checkpoints, so
fine-tuning keeps the pretrained class head and predictions make sense straight
away. With your own classes, nitid rebuilds the class head automatically; see
[Fine-tuning](fine_tuning.md#data-yaml). YOLO `.txt` datasets work the same way.

Because the images come from COCO val2017, which the pretrained models never
trained on, the metrics are reasonable, but 160 images and two epochs say
nothing about real accuracy.

## Path A: NVIDIA GPU with CUDA

### A1. Prerequisites

| Requirement | Value |
|-------------|-------|
| GPU | NVIDIA, compute capability 5.0 to 9.0 (Maxwell to Hopper, for example RTX 20/30/40 series, A100, H100) |
| Driver | 525.60.13 or newer, the minimum for CUDA 12.x |
| CUDA toolkit | Not needed. The PyTorch wheel includes the CUDA 12.1 runtime and cuDNN |
| GPU memory | Start with `batch=8` for `model1s`; lower it if you hit out-of-memory errors |

> **RTX 50 series and other Blackwell GPUs** (compute capability 10.0 or 12.0)
> are not supported by the locked `cu121` PyTorch build. Training fails with
> `no kernel image is available for execution on the device`. See
> [Troubleshooting](#cuda-error-no-kernel-image-is-available-for-execution-on-the-device).

Install the driver on Ubuntu if `nvidia-smi` is not found:

```bash
sudo ubuntu-drivers install
sudo reboot
```

### A2. Check the GPU

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The second command must print `True` and the GPU name. `nvidia-smi` shows the
highest CUDA version that the driver supports; it must be 12.1 or higher.

### A3. Train

Python API:

```bash
cat > train_cuda.py <<'EOF'
from nitid import NITID

data = "configs/datasets/coco-mini.yml"

model = NITID("model1s", task="detect", device="cuda:0")
metrics = model.train(
    data=data,
    epochs=2,
    imgsz=640,
    batch=8,
    workers=4,
    project="runs/train",
    name="cuda-smoke",
)
print({key: value for key, value in metrics.items() if key != "history"})
EOF
uv run python train_cuda.py
```

The same run from the command line:

```bash
uv run nitid train model=model1s task=detect \
    data=configs/datasets/coco-mini.yml \
    epochs=2 imgsz=640 batch=8 workers=4 device=cuda:0 name=cuda-smoke-cli
```

On the first run, nitid downloads the `model1s` detection weights (about 40 MB)
and saves `dfine_s_obj2coco_wrapped.pth` in the current directory. AMP (mixed
precision) and EMA are on by default for detection on CUDA.

Each epoch prints one summary line:

```text
[dfine] Epoch 1/2  loss=...  P=...  R=...  mAP50=...  mAP50-95=...  fitness=...  mem=...MB
[dfine] Training complete: best epoch 1, best fitness ...
```

### A4. Validate, predict, export

```bash
cat > after_cuda.py <<'EOF'
from nitid import NITID

data = "configs/datasets/coco-mini.yml"
image = "datasets/coco-mini/images/val/000000347930.jpg"

model = NITID("runs/train/cuda-smoke/best.pth", task="detect", device="cuda:0")

metrics = model.val(data=data, batch=8, project="runs/val", name="cuda-smoke")
print("mAP50-95:", metrics["mAP50-95"], "mAP50:", metrics["mAP50"])

results = model.predict(image, conf=0.5, save=True, project="runs/predict", name="cuda-smoke")
print(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf)

print(model.export(format="onnx", project="runs/export", name="cuda-smoke-onnx"))
EOF
uv run python after_cuda.py
```

TensorRT export is optional and needs NVIDIA's package; see
[Export](export.md#tensorrt).

### A5. Expected run time

The CUDA path was not timed for this guide. On any supported GPU the two
epochs over 128 images should take on the order of a minute, plus the
first-run weight download.

## Path B: Intel machine with OpenVINO

### B1. Prerequisites

| Requirement | Value |
|-------------|-------|
| CPU | x86_64 with AVX2. Training uses every core; more cores train faster |
| RAM | 8 GB minimum, 16 GB recommended |
| Intel GPU (optional) | Integrated or Arc GPU, with Intel's compute runtime (Level Zero / OpenCL) installed |
| Intel NPU (optional) | Core Ultra processor (Meteor Lake or newer), Linux kernel 6.6 or newer, Intel NPU driver installed |

OpenVINO's `"CPU"` device needs no extra drivers. The `"GPU"` and `"NPU"`
devices need user-space drivers that a fresh Ubuntu does not include.

Intel GPU on Ubuntu 24.04, using Intel's graphics PPA:

```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt-get install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo
sudo usermod -aG render "$USER"
```

Intel NPU: install the `.deb` packages from the latest
[intel/linux-npu-driver release](https://github.com/intel/linux-npu-driver/releases)
for your Ubuntu version, following the installation section of that release.
The packages are `intel-driver-compiler-npu`, `intel-fw-npu`, and
`intel-level-zero-npu`. Then:

```bash
sudo usermod -aG render "$USER"
sudo reboot
```

After the reboot, check that the device nodes exist:

```bash
ls -l /dev/dri/renderD*      # Intel GPU
ls -l /dev/accel/accel0      # Intel NPU
groups                       # must include "render"
```

For other distributions and for Windows, follow Intel's
[OpenVINO GPU](https://docs.openvino.ai/2025/get-started/install-openvino/configurations/configurations-intel-gpu.html)
and
[NPU](https://docs.openvino.ai/2025/get-started/install-openvino/configurations/configurations-intel-npu.html)
configuration pages.

### B2. Check the OpenVINO devices

```bash
uv run python -c "
import openvino as ov
core = ov.Core()
for device in core.available_devices:
    print(device, core.get_property(device, 'FULL_DEVICE_NAME'))
"
```

On a Core Ultra 5 225H with both drivers installed, this prints:

```text
CPU Intel(R) Core(TM) Ultra 5 225H
GPU Intel(R) Arc(TM) Graphics (iGPU)
NPU Intel(R) AI Boost
```

`CPU` is always listed. If `GPU` or `NPU` is missing, see
[Troubleshooting](#openvino-device-gpu-or-npu-is-not-available).

### B3. Train on the CPU

Python API:

```bash
cat > train_cpu.py <<'EOF'
from nitid import NITID

data = "configs/datasets/coco-mini.yml"

model = NITID("model1s", task="detect", device="cpu")
metrics = model.train(
    data=data,
    epochs=2,
    imgsz=640,
    batch=4,
    workers=2,
    amp=False,   # AMP is CUDA-only; this avoids a warning on the CPU
    project="runs/train",
    name="cpu-smoke",
)
print({key: value for key, value in metrics.items() if key != "history"})
EOF
uv run python train_cpu.py
```

Command line:

```bash
uv run nitid train model=model1s task=detect \
    data=configs/datasets/coco-mini.yml \
    epochs=2 imgsz=640 batch=4 workers=2 amp=false device=cpu name=cpu-smoke-cli
```

Keep `imgsz=640`. Export and OpenVINO inference use the checkpoint's evaluation
size, 640 for the pretrained models.

On an Intel Core Ultra 5 225H (14 cores), two epochs over 64 images took about
6 minutes: about 5 seconds per batch of 4 images, plus validation after each
epoch. Peak memory was about 5 GB. COCO-mini has 128 training images, so expect
about twice that. For a quicker first check, add `fraction=0.25` to train on a
quarter of the images.

### B4. Validate and predict with PyTorch

```bash
cat > after_cpu.py <<'EOF'
from nitid import NITID

data = "configs/datasets/coco-mini.yml"
image = "datasets/coco-mini/images/val/000000347930.jpg"

model = NITID("runs/train/cpu-smoke/best.pth", task="detect", device="cpu")

metrics = model.val(data=data, batch=4, project="runs/val", name="cpu-smoke")
print("mAP50-95:", metrics["mAP50-95"], "mAP50:", metrics["mAP50"])

results = model.predict(image, conf=0.5, save=True, project="runs/predict", name="cpu-smoke")
print(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf)
EOF
uv run python after_cpu.py
```

Validation of 16 images took about 30 seconds on the CPU, so the 32 COCO-mini
validation images take about a minute.

### B5. Export to OpenVINO IR

```bash
cat > export_openvino.py <<'EOF'
from nitid import NITID

model = NITID("runs/train/cpu-smoke/best.pth", task="detect", device="cpu")
print(model.export(format="openvino", project="runs/export", name="cpu-smoke-openvino"))
print(model.export(format="onnx", project="runs/export", name="cpu-smoke-onnx"))
EOF
uv run python export_openvino.py
```

This writes `runs/export/cpu-smoke-openvino/dfine_640.xml` and `dfine_640.bin`
(graph and weights), plus `runs/export/cpu-smoke-onnx/dfine_640.onnx`. Each
export takes about 10 seconds. Add `half=True` to store the IR weights in FP16.

### B6. Predict with OpenVINO on the CPU, GPU, or NPU

`backend="openvino"` runs `predict()` through OpenVINO Runtime and returns the
same `Results` objects as the PyTorch backend:

```bash
cat > predict_openvino.py <<'EOF'
import openvino as ov

from nitid import NITID

image = "datasets/coco-mini/images/val/000000347930.jpg"

for device in ov.Core().available_devices:       # e.g. ['CPU', 'GPU', 'NPU']
    model = NITID("runs/train/cpu-smoke/best.pth", task="detect",
                  backend="openvino", device=device)
    model.predict(image, conf=0.5, verbose=False)   # first call compiles the model
    results = model.predict(image, conf=0.5, save=True,
                            project="runs/predict", name=f"openvino-{device.lower()}")
    print(device, len(results[0].boxes), "objects,",
          f"{results[0].speed['inference']:.1f} ms inference")
EOF
uv run python predict_openvino.py
```

`backend="openvino"` does not read the exported `.xml`. It converts the
checkpoint in memory and compiles it for the device on the first `predict()`
call, then reuses it for later calls at the same `imgsz`. The `nitid` command
has no backend option, so OpenVINO inference is Python-only.

Measured on the Core Ultra 5 225H with the checkpoint trained above, one
640×480 image:

| Backend, device | First call (compile + infer) | Inference after that |
|-----------------|-----------------------------:|---------------------:|
| PyTorch, CPU | about 4 s | — |
| OpenVINO, `"CPU"` | 8 s | 55–65 ms |
| OpenVINO, `"GPU"` (Arc iGPU) | 7 s | 15–20 ms |
| OpenVINO, `"NPU"` (AI Boost) | 22 s | 50–60 ms |

PyTorch and OpenVINO on the CPU and GPU returned the same objects. The NPU
computes in reduced precision and can drop a detection scored just above
`conf` (in one run, 0.51 against `conf=0.5`), so expect small differences
between devices near the threshold. On this machine the integrated GPU
is the fastest device. `device="auto"` prefers the NPU, then the GPU, then the
CPU, so pass `device="GPU"` explicitly when the GPU is faster on your hardware.

### B7. Run the exported IR without nitid

For deployment, the exported `.xml` needs only OpenVINO, OpenCV, and NumPy.
The input is one RGB image resized to 640×640, scaled to `[0, 1]`, in
`[1, 3, 640, 640]` layout. The outputs are `labels`, `boxes` (`xyxy`, in
640×640 input pixels), and `scores`, with post-processing included:

```bash
cat > run_ir.py <<'EOF'
import sys

import cv2
import numpy as np
import openvino as ov

xml_path, image_path, device = sys.argv[1], sys.argv[2], sys.argv[3]
compiled = ov.Core().compile_model(xml_path, device)

bgr = cv2.imread(image_path)
h, w = bgr.shape[:2]
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
images = cv2.resize(rgb, (640, 640)).astype(np.float32).transpose(2, 0, 1)[None] / 255.0

labels, boxes, scores = compiled([images]).values()
keep = scores[0] > 0.5
boxes = boxes[0][keep] * np.array([w / 640, h / 640, w / 640, h / 640])
for label, box, score in zip(labels[0][keep], boxes, scores[0][keep]):
    print(int(label), round(float(score), 3), box.round(1).tolist())
EOF
uv run python run_ir.py runs/export/cpu-smoke-openvino/dfine_640.xml \
    datasets/coco-mini/images/val/000000347930.jpg GPU
```

Labels are 0-based indices into the `names:` list of
`configs/datasets/coco-mini.yml`. The boxes
match the PyTorch predictions to within a pixel; nitid's own preprocessing
resizes with PIL rather than OpenCV, so scores can differ slightly.

## Where the outputs go

| Directory | Contents |
|-----------|----------|
| `runs/train/<name>/` | `best.pth`, `last.pth`, `epoch1.pth`, …, `results.csv`, `results.png`, PR/F1 curves, confusion matrices, `args.yaml`, `environment.yaml` |
| `runs/val/<name>/` | Validation plots, `args.yaml`, `environment.yaml` |
| `runs/predict/<name>/` | Images with drawn predictions (`save=True`) |
| `runs/export/<name>/` | `dfine_640.onnx`, or `dfine_640.xml` + `dfine_640.bin` |

If `<name>` already exists, nitid appends a number (`cpu-smoke2`, …) instead
of overwriting. Each checkpoint is self-contained: model configuration, weights
and class names travel together. `best.pth` is the epoch with the highest
fitness, `last.pth` the latest one, and both hold EMA weights when EMA is on.

## Instance segmentation

The dataset from step 2 also contains polygon masks, so the same steps work for
instance segmentation. Use `task="segment"` everywhere: `NITID("model1s",
task="segment")` downloads the segmentation weights from Hugging Face, and
`results[0].masks` holds the predicted masks. OpenVINO export and
`backend="openvino"` work for segmentation models too. On the CPU, one epoch over
16 images (`fraction=0.25`) took about 75 seconds and 6.3 GB of RAM, and the
validation summary adds `mask_mAP50` and `mask_mAP50-95`. Semantic segmentation needs dense
PNG masks instead and is not covered here; see
[Fine-tuning](fine_tuning.md#dense-semantic-masks).

## Troubleshooting

### `torch.cuda.is_available()` returns `False`

- `nvidia-smi` fails: the NVIDIA driver is missing or not loaded. Install it
  (`sudo ubuntu-drivers install`) and reboot.
- `nvidia-smi` works but PyTorch still reports `False`: check that
  `torch.version.cuda` prints `12.1`. If it prints `None`, a CPU-only PyTorch
  has been installed over the locked one; run `uv sync --extra train` again.
- Inside a container, start it with GPU access (`docker run --gpus all ...`
  with the NVIDIA Container Toolkit installed).

### `CUDA driver version is insufficient for CUDA runtime version`

The driver is older than CUDA 12.1 requires. Update it to 525.60.13 or newer.

### `CUDA error: no kernel image is available for execution on the device`

The GPU is newer than the locked PyTorch build supports, typically an RTX 50
series or other Blackwell GPU. Supporting them needs a PyTorch build for CUDA
12.8 or newer, which nitid's lock file does not pin yet. As an untested
stopgap, you can replace PyTorch inside the environment and then run commands
with `--no-sync`, so that `uv` does not restore the locked version:

```bash
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv run --no-sync python train_cuda.py
```

### `CUDA out of memory`

Lower `batch` (8 → 4 → 2). To keep the effective batch size, add
`accumulate=2` or `accumulate=4`. Keep `imgsz=640`. Close other processes on
the GPU (`nvidia-smi` lists them).

### CPU training is very slow or the machine runs out of memory

CPU training is expected to be slow. Use `fraction=0.25` or fewer epochs. If memory runs out, lower `batch` to 2 and
`workers` to 0. Other heavy processes on the same CPU slow training
considerably.

### `ImportError: libGL.so.1`

The GUI build of OpenCV has replaced the headless one, usually because another
package pulled in `opencv-python`. See
[Troubleshooting](troubleshooting.md#importerror-libglso1-when-importing-nitid).

### `OpenVINO is not installed`

Install the extra: `uv sync --extra train --extra openvino`. Running
`uv sync --extra train` alone later removes OpenVINO again, so always pass both
extras on the Intel machine.

### OpenVINO device `"GPU"` or `"NPU"` is not available

nitid raises `ValueError: OpenVINO device 'NPU' is not available on this
machine. Available devices: ['CPU']`. Check, in order:

1. The hardware exists: `lspci | grep -iE "vga|display"` for the GPU;
   `lspci | grep -i "processing accelerator"` or `lsmod | grep intel_vpu` for
   the NPU.
2. The device node exists: `/dev/dri/renderD128` (GPU), `/dev/accel/accel0`
   (NPU). A missing NPU node usually means the kernel is older than 6.6.
3. Your user is in the `render` group (`groups`). After `usermod`, log out and
   back in, or reboot.
4. The user-space driver is installed: `clinfo | grep "Device Name"` lists the
   GPU; for the NPU, `dpkg -l | grep npu` shows the three driver packages.

Then check again with the command in [B2](#b2-check-the-openvino-devices).

### The first OpenVINO prediction is slow

The first `predict()` call converts and compiles the model for the device:
about 7–8 seconds on the CPU and GPU and over 20 seconds on the NPU in the
measurements above. Later calls on the same model object reuse the compiled
model.

### Warnings during export

`TracerWarning: torch.as_tensor results are registered as constants` during
export, and `nanobind: leaked ... instances!` when Python exits after an
OpenVINO export, are expected and harmless. The export is complete when nitid
prints `OpenVINO IR export saved to ...`.

## Verification status

| Step | Status |
|------|--------|
| Install with `uv sync`, PyTorch 2.5.1+cu121, OpenVINO 2026.2 | Run on Ubuntu 24.04 |
| COCO-mini dataset and checkpoint paths in `NITID(...)` | Steps B3 to B7 re-run as written on the Core Ultra 5 225H (2026-10-05): two epochs, mAP50-95 0.60 on the 32 validation images |
| CPU training, Python API and CLI | Run on an Intel Core Ultra 5 225H |
| Validation, prediction, ONNX and OpenVINO export | Run on the CPU |
| `backend="openvino"` on `"CPU"`, `"GPU"`, `"NPU"`, and the standalone IR script | Run on the Core Ultra 5 225H (Arc iGPU, AI Boost NPU) |
| Instance segmentation: CPU training, OpenVINO export, `backend="openvino"` on `"GPU"` | Run, one epoch on 16 images |
| CUDA training and inference | Not run: no NVIDIA GPU was available. Commands follow the same API |
| Driver installation on a fresh machine (NVIDIA, Intel GPU, Intel NPU) | Not run: the test machine already had the drivers |
