# Troubleshooting

Use this page when nitid does not start, a command is missing, a model does not load, or Docker behaves differently than expected.

Most nitid problems come from one missing setup step. The usual flow is:

1. You run a command.
2. That command looks for the project environment or a model checkpoint.
3. If one of those pieces is missing, Python prints an error.
4. The fix is usually to install dependencies, download a model, or use the right command form.

## `dfine: command not found`

### What it means

The `dfine` command exists inside nitid's virtual environment, but your terminal cannot see it directly.

This usually happens when `.venv` is not activated.

### Fix

From the repository root, use `uv run`:

```bash
uv run dfine --help
uv run dfine predict --help
```

Or activate the environment first:

```bash
source .venv/bin/activate
dfine --help
```

Inside the Docker container, you can also call the executable directly:

```bash
.venv/bin/dfine --help
```

See the [command-line guide](cli.md) for the full CLI workflow.

## `uv sync` fails with Python 3.13 or `torchvision`

### What it means

nitid currently supports Python `>=3.10,<3.13`.

PyTorch and torchvision wheels are not always available for every new Python version immediately. If your system uses Python 3.13, dependency installation can fail before nitid is even installed.

### Fix

Use Python 3.10, 3.11, or 3.12.

Then recreate the environment:

```bash
uv sync
```

For development tools:

```bash
uv sync --extra dev
```

On Apple Silicon Macs, use the Docker setup if native dependency installation is not suitable:

```bash
docker compose up -d
docker compose exec -it nitid-dev bash
uv sync --frozen --extra dev
```

See the [macOS Docker setup](macos_docker_setup.md) for details.

## Model download fails

### What it means

When you use a supported model name like `dfine_s` or `detrpose_n`, nitid
resolves the selected task, downloads the matching checkpoint when needed, and
prepares it for the integrated runtime.

The workflow is:

```text
DFINE("dfine_s")
-> nitid checks the model registry
-> resolves weights for the selected task
-> downloads the official checkpoint when needed
-> builds the matching integrated model configuration
-> writes a local prepared checkpoint
```

If the internet connection or class names file is missing, the download or wrapping step can fail.

### Fix

Try a direct download command:

```bash
uv run dfine download model=dfine_s
```

To save the model in a folder:

```bash
uv run dfine download model=dfine_s output=models
```

To replace an existing wrapped checkpoint:

```bash
uv run dfine download model=dfine_s output=models force=true
```

Detection supports pretrained S/M/L/X models and the `obj2coco` (default) and
`coco` variants. Instance segmentation supports pretrained N/S/M/L/X models
with `task=segment` and COCO weights. Semantic models initialize from the
matching segmentation checkpoint. Pose uses `detrpose_n` through `detrpose_x`
with `task=pose`.

## Checkpoint gives `KeyError: 'config'`

### What it means

nitid expects a runtime checkpoint with embedded config and class names. Training
runs save this format automatically.

Some external research checkpoints contain only weights. That is why nitid
cannot find `config`.

### Fix

If you are using an official model, use the model name and let nitid prepare it:

```python
from dfine import DFINE

model = DFINE("dfine_s")
```

Or use the CLI:

```bash
uv run dfine download model=dfine_s
```

If you are maintaining support for a new upstream checkpoint, use the conversion
tool explicitly:

```bash
uv run python tools/convert_checkpoint.py \
    --weights dfine_l.pth \
    --model dfine_l \
    --task detect \
    --names configs/datasets/coco.yml \
    --output dfine_l_wrapped.pth
```

Most users should not need this path; use supported model names or checkpoints
created by `model.train(...)`.

## CUDA is not available

### What it means

CUDA is NVIDIA's GPU runtime. It is only available when your machine has a compatible NVIDIA GPU, driver, and PyTorch CUDA build.

If you are on a Mac or a CPU-only machine, `torch.cuda.is_available()` returns `False`. That is expected.

### Fix

For CPU testing, let nitid choose the device automatically or use CPU:

```python
from dfine import DFINE

model = DFINE("dfine_s", device="cpu")
```

For CLI usage:

```bash
uv run dfine predict model=dfine_s source=image.jpg
```

For GPU usage, run on a Linux or Windows machine with an NVIDIA GPU and working drivers.

## TensorRT export fails

### What it means

TensorRT export requires NVIDIA-specific packages and a CUDA-capable GPU. It is not expected to work on macOS or CPU-only machines.

### Fix

Install TensorRT manually on an NVIDIA machine:

```bash
pip install --extra-index-url https://pypi.nvidia.com tensorrt>=8.6
```

Then export:

```bash
uv run dfine export model=dfine_s format=tensorrt
```

If you do not need TensorRT, export to ONNX instead:

```bash
uv run dfine export model=dfine_s format=onnx
```

See the [export guide](export.md) for export options and constraints.

## Docker container works but `dfine` command does not

### What it means

The Docker container starts a Linux shell, but it does not automatically activate `.venv`.

So `dfine` may not be found even though nitid is installed inside the project environment.

### Fix

Inside the container:

```bash
cd /app
.venv/bin/dfine --help
```

Or use `uv run`:

```bash
uv run dfine --help
```

From your host terminal:

```bash
docker exec nitid_container /app/.venv/bin/dfine --help
```

See the [macOS Docker setup](macos_docker_setup.md) for the full container workflow.

## Web app shows no models

### What it means

The web app lists checkpoints from the `models/` directory. If that folder is empty, the UI has no model to run.

### Fix

Create the folder and download a model into it:

```bash
mkdir -p models
uv run dfine download model=dfine_s output=models
```

Then start the backend:

```bash
uv run uvicorn web.api.main:app --workers 1
```

Use one worker only. The model cache is process-local, and inference is designed for a single API worker.

See the [web app guide](web_app.md) for the full setup.

## Still stuck?

Run these checks from the repository root and include the output when asking for help:

```bash
git status
python --version
uv run dfine --help
uv run pytest tests/unit
```

If you are using Docker, also include:

```bash
docker compose ps
docker exec nitid_container /app/.venv/bin/dfine --help
```
