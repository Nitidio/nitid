"""Pins how the compiled Studio UI enters the wheel and the sdist.

Resolved by building for real (see docs/studio-packaging.md): ``package-data`` with
``static/**`` is enough; no MANIFEST.in is needed. These tests build a tiny tree with the
repository's real ``pyproject.toml`` through setuptools, offline (no build isolation).
"""

from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("setuptools")
pytest.importorskip("wheel")

REPO_ROOT = Path(__file__).resolve().parents[2]

STATIC_FILES = {
    "nitid/studio/static/index.html": '<div id="root"></div>',
    "nitid/studio/static/assets/app-abc123.js": "console.log(1)",
    "nitid/studio/static/ort/ort-wasm.wasm": "wasm",
}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copy(REPO_ROOT / name, root / name)
    for package in ("nitid", "nitid/studio", "dfine", "dfine/data"):
        (root / package).mkdir(parents=True, exist_ok=True)
        (root / package / "__init__.py").write_text("")
    (root / "dfine/data/coco_names.yml").write_text("names: {}\n")
    for name, content in STATIC_FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def _build(project: Path, kind: str) -> Path:
    from setuptools import build_meta

    out = project / "out"
    out.mkdir(exist_ok=True)
    previous = Path.cwd()
    os.chdir(project)
    try:
        name = getattr(build_meta, f"build_{kind}")(str(out))
    finally:
        os.chdir(previous)
    return out / name


def test_wheel_contains_the_static_tree(project: Path):
    wheel = _build(project, "wheel")
    names = set(zipfile.ZipFile(wheel).namelist())
    assert set(STATIC_FILES) <= names


def test_sdist_contains_the_static_tree_when_it_was_built(project: Path):
    sdist = _build(project, "sdist")
    with tarfile.open(sdist) as archive:
        names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
    assert set(STATIC_FILES) <= names


def test_wheel_without_a_built_ui_has_no_static_files(project: Path):
    shutil.rmtree(project / "nitid/studio/static")
    wheel = _build(project, "wheel")
    names = zipfile.ZipFile(wheel).namelist()
    assert not [name for name in names if "/static/" in name]
