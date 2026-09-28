# Troubleshooting

Use this page when nitid does not start, a command is missing, a model does not load, or Docker behaves differently than expected.

Most nitid problems come from one missing setup step. The usual flow is:

1. You run a command.
2. That command looks for the project environment or a model checkpoint.
3. If one of those pieces is missing, Python prints an error.
4. The fix is usually to install dependencies, download a model, or use the right command form.

## `nitid: command not found`

### What it means

The `nitid` command exists inside Nitid's virtual environment, but your terminal cannot see it directly.

This usually happens when `.venv` is not activated.

### Fix

From the repository root, use `uv run`:

```bash
uv run nitid --help
uv run nitid predict --help
```

Or activate the environment first:

```bash
source .venv/bin/activate
nitid --help
```

Inside the Docker container, you can also call the executable directly:

```bash
.venv/bin/nitid --help
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

When you use a supported model name like `nitid1s`, nitid
resolves the selected task, downloads the matching checkpoint when needed, and
prepares it for the integrated runtime.

The workflow is:

```text
NITID("nitid1s", task="detect")
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
uv run nitid download model=nitid1s task=detect
```

To save the model in a folder:

```bash
uv run nitid download model=nitid1s task=detect output=models
```

To replace an existing wrapped checkpoint:

```bash
uv run nitid download model=nitid1s task=detect output=models force=true
```

Detection, instance segmentation, and semantic segmentation all use the same
public model names (`nitid1n`, `nitid1s`, `nitid1m`, `nitid1l`,
`nitid1x`) with an explicit `task=...`.

## Checkpoint gives `KeyError: 'config'`

### What it means

nitid expects a runtime checkpoint with embedded config and class names. Training
runs save this format automatically.

Some external research checkpoints contain only weights. That is why nitid
cannot find `config`.

### Fix

If you are using an official model, use the model name and let nitid prepare it:

```python
from nitid import NITID

model = NITID("nitid1s", task="detect")
```

Or use the CLI:

```bash
uv run nitid download model=nitid1s task=detect
```

If you are maintaining support for a new upstream checkpoint, see the maintainer
tools in this repository. Most users should use supported model names or
checkpoints created by `model.train(...)`.

## CUDA is not available

### What it means

CUDA is NVIDIA's GPU runtime. It is only available when your machine has a compatible NVIDIA GPU, driver, and PyTorch CUDA build.

If you are on a Mac or a CPU-only machine, `torch.cuda.is_available()` returns `False`. That is expected.

### Fix

For CPU testing, let nitid choose the device automatically or use CPU:

```python
from nitid import NITID

model = NITID("nitid1s", task="detect", device="cpu")
```

For CLI usage:

```bash
uv run nitid predict model=nitid1s task=detect source=image.jpg
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
uv run nitid export model=nitid1s task=detect format=tensorrt
```

If you do not need TensorRT, export to ONNX instead:

```bash
uv run nitid export model=nitid1s task=detect format=onnx
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
uv run nitid --help
```

From your host terminal:

```bash
docker exec nitid_container /app/.venv/bin/dfine --help
```

See the [macOS Docker setup](macos_docker_setup.md) for the full container workflow.

## `Results.show()` raises `RuntimeError: Results.show() needs a display`

### What it means

`show()` opens a window, and there is nowhere to open one: a server, a container or an SSH session
without X11 or Wayland. nitid installs the headless OpenCV build so that `import nitid` works on
those machines, and `show()` then falls back to matplotlib. It raises this error only when matplotlib
cannot display anything either.

### Fix

Write the image to disk, or take it as an array:

```python
result.save("out.jpg")
image = result.plot()  # HWC BGR numpy array
```

In Jupyter, `show()` renders inline through matplotlib.

## `ImportError: libGL.so.1` when importing nitid

### What it means

The GUI build of OpenCV (`opencv-python`) is installed, and it needs system graphics libraries that
slim or server images do not have. nitid itself depends on `opencv-python-headless`, which does not
need them. `opencv-python` usually arrives with another package; the `track` extra's `trackers`
dependency requires it, for example. When both builds are installed, they share the `cv2` module,
and whichever was installed last wins.

### Fix

Make the headless build the one that provides `cv2`:

```bash
pip uninstall -y opencv-python
pip install --force-reinstall --no-deps opencv-python-headless
```

Or install the system library instead: `apt-get install -y libgl1 libglib2.0-0`.

## Still stuck?

Run these checks from the repository root and include the output when asking for help:

```bash
git status
python --version
uv run nitid --help
uv run pytest tests/unit
```

If you are using Docker, also include:

```bash
docker compose ps
docker exec nitid_container /app/.venv/bin/dfine --help
```
