# Command-line guide

nitid's terminal command is named `dfine`. Commands use `key=value`
arguments:

```bash
uv run dfine predict model=dfine_s source=image.jpg conf=0.5
```

## Before using the command

Complete the repository installation first:

```bash
git clone https://github.com/Vaelsys/nitid.git
cd nitid
git submodule update --init
uv sync
```

Run commands from the repository root, where `pyproject.toml` is located.

## Option 1: use `uv run` (recommended)

You do not need to activate the virtual environment:

```bash
uv run dfine --help
uv run dfine download --help
uv run dfine predict --help
```

`uv run` finds the project environment and runs the installed `dfine` command
inside it.

## Option 2: activate the virtual environment

On Linux or macOS:

```bash
source .venv/bin/activate
```

The shorter commands will then work:

```bash
dfine --help
dfine download --help
dfine predict --help
```

Leave the environment when you are finished:

```bash
deactivate
```

If the terminal prints `dfine: command not found`, the environment is not
active or nitid has not been installed. Use `uv run dfine ...`, or activate
`.venv` first.

## Find available commands

Show the command list:

```bash
uv run dfine --help
```

Show the required arguments, optional settings, defaults, and examples for one
command:

```bash
uv run dfine download --help
uv run dfine predict --help
uv run dfine train --help
uv run dfine val --help
uv run dfine export --help
uv run dfine info --help
```

The shorter `-h` flag also works:

```bash
uv run dfine predict -h
```

Use `dfine --help` or `dfine COMMAND --help`; `dfine help` is not a supported
form.

## Download a model

Download an official D-FINE model and convert it to nitid's wrapped checkpoint
format:

```bash
uv run dfine download model=dfine_s
```

Supported model names are `dfine_s`, `dfine_m`, `dfine_l`, and `dfine_x`.
Without `output=`, the wrapped checkpoint is saved in the current directory:

```text
dfine_s_wrapped.pth
```

Save it in a model directory:

```bash
uv run dfine download model=dfine_s output=models
```

Download again and replace an existing checkpoint:

```bash
uv run dfine download model=dfine_s output=models force=true
```

## Common commands

Run prediction:

```bash
uv run dfine predict model=dfine_s source=image.jpg conf=0.5
```

Fine-tune a model:

```bash
uv run dfine train model=dfine_s data=my_dataset.yml epochs=50
```

By default, training saves wrapped epoch checkpoints under `runs/train/exp/`.
Add `wandb=true` to log the run to the default `nitid` WandB project:

```bash
uv run dfine train model=dfine_s data=my_dataset.yml epochs=50 wandb=true
```

Add `mlflow=true` for Ultralytics-style local MLflow tracking. Logs default to
`runs/mlflow`:

```bash
uv run dfine train model=dfine_s data=my_dataset.yml epochs=50 mlflow=true
```

Validate a model:

```bash
uv run dfine val model=dfine_s data=my_dataset.yml
```

Validation reports COCO metrics to the terminal and does not create a run directory by default.

Export a model:

```bash
uv run dfine export model=dfine_s format=onnx
```

Display model information:

```bash
uv run dfine info model=dfine_s
```

## Using the development Docker container

The Docker container does not automatically activate `.venv`. Enter the
container and run the executable from the project environment:

```bash
docker exec -it nitid_container bash
cd /app
.venv/bin/dfine download --help
```

Or run one command from the host terminal:

```bash
docker exec nitid_container /app/.venv/bin/dfine download --help
```

The full `.venv/bin/dfine` path is mainly useful for Docker automation. Normal
Linux users can use `uv run dfine ...` or activate `.venv` and use `dfine ...`.
