"""Validate the contents and metadata of a nitid wheel."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REQUIRED_FILES = {
    "dfine/model.py",
    "dfine/nn/architecture/decoder.py",
    "dfine/nn/losses/criterion.py",
    "dfine/utils/downloads.py",
}
FORBIDDEN_PARTS = {"extern", "D-FINE-seg-main"}


def verify_wheel(path: Path) -> None:
    """Require the complete model core, notices, and no reference source trees."""
    if not path.is_file() or path.suffix != ".whl":
        raise ValueError(f"Expected a wheel path, got {path}")

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())

    missing = REQUIRED_FILES - names
    if missing:
        raise RuntimeError(f"Wheel is missing required files: {sorted(missing)}")
    if not any(name.endswith(".dist-info/licenses/LICENSE") for name in names):
        raise RuntimeError("Wheel is missing the Apache-2.0 license")
    if not any(name.endswith(".dist-info/licenses/THIRD_PARTY_NOTICES.md") for name in names):
        raise RuntimeError("Wheel is missing third-party notices")
    if any(FORBIDDEN_PARTS.intersection(Path(name).parts) for name in names):
        raise RuntimeError("Wheel contains a reference source tree")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    verify_wheel(args.wheel)
    print(f"Wheel contents verified: {args.wheel}")


if __name__ == "__main__":
    main()
