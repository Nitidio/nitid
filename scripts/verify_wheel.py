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


def verify_wheel(path: Path) -> None:
    """Require the complete model core, notices, and no reference source trees."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    verify_wheel(args.wheel)
    print(f"Wheel contents verified: {args.wheel}")


if __name__ == "__main__":
    main()
