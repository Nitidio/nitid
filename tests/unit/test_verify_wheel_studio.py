"""Studio rules of scripts/verify_wheel.py, exercised on synthetic wheels (zip files)."""

from __future__ import annotations

import importlib.util
import os
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "verify_wheel", REPO_ROOT / "scripts/verify_wheel.py"
)
assert _spec is not None and _spec.loader is not None
verify_wheel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_wheel)

DIST_INFO = "nitid-0.0.0.dist-info"
STATIC = "nitid/studio/static"


def base_files() -> dict[str, bytes]:
    files = {name: b"x" for name in verify_wheel.REQUIRED_FILES}
    files.update(
        {
            f"{DIST_INFO}/top_level.txt": b"dfine\nnitid\n",
            f"{DIST_INFO}/licenses/LICENSE": b"license",
            f"{DIST_INFO}/licenses/THIRD_PARTY_NOTICES.md": b"notices",
            f"{STATIC}/index.html": b'<div id="root"></div>',
            f"{STATIC}/assets/app-abc123.js": b"console.log(1)",
        }
    )
    return files


def make_wheel(tmp_path: Path, extra: dict[str, bytes] | None = None, drop=()) -> Path:
    files = base_files()
    for name in drop:
        files.pop(name)
    files.update(extra or {})
    path = tmp_path / "nitid-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def test_valid_wheel_passes(tmp_path):
    verify_wheel.verify_wheel(make_wheel(tmp_path))


def test_wasm_under_ort_is_allowed(tmp_path):
    wheel = make_wheel(tmp_path, {f"{STATIC}/ort/ort-wasm.wasm": b"\0asm"})
    verify_wheel.verify_wheel(wheel)


def test_missing_index_fails_by_default(tmp_path):
    wheel = make_wheel(tmp_path, drop=[f"{STATIC}/index.html"])
    with pytest.raises(RuntimeError, match="missing the Studio UI.*build_studio.py"):
        verify_wheel.verify_wheel(wheel)


def test_missing_index_allowed_only_when_explicit(tmp_path):
    wheel = make_wheel(tmp_path, drop=[f"{STATIC}/index.html"])
    verify_wheel.verify_wheel(wheel, require_ui=False)


def test_cli_requires_ui_unless_flag(tmp_path, monkeypatch, capsys):
    wheel = make_wheel(tmp_path, drop=[f"{STATIC}/index.html"])
    monkeypatch.setattr("sys.argv", ["verify_wheel.py", str(wheel)])
    with pytest.raises(RuntimeError, match="Studio UI"):
        verify_wheel.main()
    monkeypatch.setattr("sys.argv", ["verify_wheel.py", str(wheel), "--allow-missing-ui"])
    verify_wheel.main()
    assert "Wheel contents verified" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("name", "message"),
    [
        (f"{STATIC}/node_modules/react/index.js", "node_modules"),
        (f"{STATIC}/model.onnx", r"\.onnx"),
        ("nitid/weights/m.pth", r"\.pth"),
        (f"{STATIC}/assets/app.js.map", r"\.map"),
        ("nitid/studio/src/App.tsx", r"\.tsx"),
        ("nitid/studio/src/main.ts", r"\.ts"),
        (f"{STATIC}/MODEL.ONNX", r"\.onnx"),
        (f"{STATIC}/assets/ort.wasm", r"\.wasm file outside"),
        ("nitid/other/engine.wasm", r"\.wasm file outside"),
        (f"{STATIC}/ortx/engine.wasm", r"\.wasm file outside"),
    ],
)
def test_forbidden_files_fail(tmp_path, name, message):
    wheel = make_wheel(tmp_path, {name: b"data"})
    with pytest.raises(RuntimeError, match=message):
        verify_wheel.verify_wheel(wheel)


def test_size_budget_is_15_mb():
    assert verify_wheel.MAX_WHEEL_BYTES == 15 * 1024 * 1024


def test_oversized_wheel_fails_with_clear_message(tmp_path):
    # Random bytes do not compress, so the wheel really is larger than the budget.
    wheel = make_wheel(tmp_path, {f"{STATIC}/assets/big.js": os.urandom(1_200_000)})
    with pytest.raises(RuntimeError, match=r"over the .* budget \(MAX_WHEEL_BYTES\)"):
        verify_wheel.verify_wheel(wheel, max_bytes=1_000_000)


def test_wheel_at_budget_passes(tmp_path):
    wheel = make_wheel(tmp_path)
    verify_wheel.verify_wheel(wheel, max_bytes=wheel.stat().st_size)
    with pytest.raises(RuntimeError, match="budget"):
        verify_wheel.verify_wheel(wheel, max_bytes=wheel.stat().st_size - 1)


def test_existing_rules_still_apply(tmp_path):
    with pytest.raises(RuntimeError, match="unexpected top-level"):
        verify_wheel.verify_wheel(make_wheel(tmp_path, {"evil/__init__.py": b""}))
    with pytest.raises(RuntimeError, match="missing required"):
        verify_wheel.verify_wheel(make_wheel(tmp_path, drop=["nitid/cli.py"]))
