"""Validate the contents and metadata of a nitid wheel."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REQUIRED_FILES = {
    "dfine/data/coco_names.yml",
    "dfine/model.py",
    "dfine/nn/architecture/decoder.py",
    "dfine/nn/losses/criterion.py",
    "dfine/utils/downloads.py",
    "nitid/cli.py",
    "nitid/convert_checkpoint.py",
}
FORBIDDEN_PARTS = {"extern", "D-FINE-seg-main"}
ALLOWED_TOP_LEVEL_PACKAGES = {"dfine", "nitid"}

# Nitid Studio's compiled UI (see docs/studio-packaging.md).
STUDIO_STATIC_PREFIX = "nitid/studio/static/"
STUDIO_INDEX = f"{STUDIO_STATIC_PREFIX}index.html"
# The only place a WebAssembly file may live (ONNX Runtime Web, if the UI ships it).
STUDIO_WASM_PREFIX = f"{STUDIO_STATIC_PREFIX}ort/"
# Directory names that must never be packaged (frontend tooling).
FORBIDDEN_DIRECTORIES = {"node_modules"}
# Model weights are downloaded or exported at run time, never shipped in the wheel.
# Source maps and TypeScript sources are build inputs, not distributable files.
FORBIDDEN_SUFFIXES = {".onnx", ".pth", ".map", ".ts", ".tsx"}
WASM_SUFFIX = ".wasm"
# Total size budget of the wheel file (as uploaded to PyPI), in bytes.
MAX_WHEEL_BYTES = 15 * 1024 * 1024


def _mb(size: int) -> str:
    return f"{size / (1024 * 1024):.2f} MB"


def _check_studio_payload(
    names: set[str], *, require_ui: bool, wheel_size: int, max_bytes: int
) -> None:
    """Rules for the Studio UI and for anything that must not ride along with it."""
    for name in sorted(names):
        parts = Path(name).parts
        if FORBIDDEN_DIRECTORIES.intersection(parts):
            raise RuntimeError(f"Wheel contains frontend tooling (node_modules): {name}")
        suffix = Path(name).suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            raise RuntimeError(f"Wheel contains a forbidden {suffix} file: {name}")
        if suffix == WASM_SUFFIX and not name.startswith(STUDIO_WASM_PREFIX):
            raise RuntimeError(f"Wheel contains a .wasm file outside {STUDIO_WASM_PREFIX}: {name}")
    if require_ui and STUDIO_INDEX not in names:
        raise RuntimeError(
            f"Wheel is missing the Studio UI ({STUDIO_INDEX}). Build it first with "
            "`python scripts/build_studio.py`, or pass --allow-missing-ui for a development check."
        )
    if wheel_size > max_bytes:
        raise RuntimeError(
            f"Wheel is {_mb(wheel_size)}, over the {_mb(max_bytes)} budget "
            f"(MAX_WHEEL_BYTES). Find what grew with `unzip -lv <wheel> | sort -k1 -n`."
        )


def verify_wheel(path: Path, *, require_ui: bool = True, max_bytes: int = MAX_WHEEL_BYTES) -> None:
    """Require the complete model core, notices, the Studio UI, and no forbidden files.

    ``require_ui=False`` exists only for development checks without a built frontend;
    the command line never enables it unless ``--allow-missing-ui`` is given.
    """
    if not path.is_file() or path.suffix != ".whl":
        raise ValueError(f"Expected a wheel path, got {path}")

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        top_level_files = [name for name in names if name.endswith(".dist-info/top_level.txt")]
        if len(top_level_files) != 1:
            raise RuntimeError("Wheel must contain exactly one top_level.txt declaration")
        declared_packages = {
            line.strip()
            for line in archive.read(top_level_files[0]).decode("utf-8").splitlines()
            if line.strip()
        }

    missing = REQUIRED_FILES - names
    if missing:
        raise RuntimeError(f"Wheel is missing required files: {sorted(missing)}")
    if not any(name.endswith(".dist-info/licenses/LICENSE") for name in names):
        raise RuntimeError("Wheel is missing the Apache-2.0 license")
    if not any(name.endswith(".dist-info/licenses/THIRD_PARTY_NOTICES.md") for name in names):
        raise RuntimeError("Wheel is missing third-party notices")
    if any(FORBIDDEN_PARTS.intersection(Path(name).parts) for name in names):
        raise RuntimeError("Wheel contains a reference source tree")
    importable_roots = {
        name.split("/", 1)[0]
        for name in names
        if name.count("/") == 1 and name.endswith("/__init__.py")
    }
    unexpected_packages = (declared_packages | importable_roots) - ALLOWED_TOP_LEVEL_PACKAGES
    if unexpected_packages:
        raise RuntimeError(
            f"Wheel exposes unexpected top-level packages: {sorted(unexpected_packages)}"
        )
    _check_studio_payload(
        names, require_ui=require_ui, wheel_size=path.stat().st_size, max_bytes=max_bytes
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--allow-missing-ui",
        action="store_true",
        help="development only: do not require nitid/studio/static/index.html (never use in CI/release)",
    )
    args = parser.parse_args()
    verify_wheel(args.wheel, require_ui=not args.allow_missing_ui)
    print(f"Wheel contents verified: {args.wheel}")


if __name__ == "__main__":
    main()
