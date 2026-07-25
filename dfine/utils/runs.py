"""Run-directory allocation, metadata snapshots, and atomic file publishing."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

import torch
import yaml


def increment_path(path: str | Path, exist_ok: bool = False) -> Path:
    """Atomically claim ``path`` or the next free ``path2``, ``path3``, ... directory."""
    path = Path(path)
    if exist_ok:
        path.mkdir(parents=True, exist_ok=True)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = path if index == 1 else path.with_name(f"{path.name}{index}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            index += 1


def resolve_run_dir(
    *,
    project: str | Path,
    name: str,
    save_dir: str | Path | None = None,
    exist_ok: bool = False,
    resume: bool = False,
) -> Path:
    """Resolve and create a run directory, preserving an existing directory only to resume."""
    requested = Path(save_dir) if save_dir is not None else Path(project) / name
    if resume:
        if not requested.is_dir():
            raise FileNotFoundError(
                f"resume=True requested but run directory was not found: '{requested}'"
            )
        return requested
    return increment_path(requested, exist_ok=exist_ok)


@contextmanager
def atomic_output_path(destination: str | Path) -> Iterator[Path]:
    """Yield a sibling temporary path and atomically publish it on success."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        yield temporary_path
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_yaml(path: str | Path, values: Mapping[str, object]) -> None:
    """Serialize YAML and publish it atomically."""
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(_yaml_safe(dict(values)), stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())


def write_run_metadata(save_dir: str | Path, args: Mapping[str, object]) -> None:
    """Write resolved arguments and a reproducibility-oriented environment snapshot."""
    save_dir = Path(save_dir)
    atomic_write_yaml(save_dir / "args.yaml", args)
    packages: dict[str, str | None] = {}
    for package in (
        "nitid",
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "onnx",
        "openvino",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "packages": packages,
    }
    atomic_write_yaml(save_dir / "environment.yaml", environment)


def _yaml_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    if value is None:
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return repr(value)
