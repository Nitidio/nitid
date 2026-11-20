"""Unit tests for scripts/assemble_hosted.py (the gate before a hosted deploy)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "assemble_hosted", REPO_ROOT / "scripts/assemble_hosted.py"
)
assert _spec is not None and _spec.loader is not None
assemble_hosted = importlib.util.module_from_spec(_spec)
sys.modules["assemble_hosted"] = assemble_hosted  # dataclasses need the module registered
_spec.loader.exec_module(assemble_hosted)

ONNX = b"not really onnx, but bytes are bytes"
SHA = hashlib.sha256(ONNX).hexdigest()


def make_ui(root: Path) -> Path:
    (root / "assets").mkdir(parents=True)
    (root / "ort").mkdir()
    (root / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (root / "assets" / "index-abc.js").write_text("1", encoding="utf-8")
    (root / "ort" / "ort-wasm-simd-threaded.wasm").write_bytes(b"\0asm")
    return root


def make_bundle(root: Path, *, sha: str = SHA, entries: list | None = None) -> Path:
    root.mkdir(parents=True)
    (root / "nitid1s-detect.onnx").write_bytes(ONNX)
    catalog = {
        "models": entries
        or [{"id": "nitid1s", "task": "detect", "url": "nitid1s-detect.onnx", "sha256": sha}]
    }
    (root / "models.json").write_text(json.dumps(catalog), encoding="utf-8")
    (root / "SHA256SUMS").write_text(f"{SHA}  nitid1s-detect.onnx\n", encoding="utf-8")
    (root / "LICENSE").write_text("Apache-2.0", encoding="utf-8")
    return root


@pytest.fixture
def parts(tmp_path):
    headers = tmp_path / "_headers"
    headers.write_text("/*\n  X-Test: 1\n", encoding="utf-8")
    return {
        "ui": make_ui(tmp_path / "ui"),
        "bundle": make_bundle(tmp_path / "bundle"),
        "headers": headers,
        "out": tmp_path / "site",
    }


def run(parts, **kwargs):
    return assemble_hosted.assemble(
        parts["ui"], parts["bundle"], parts["out"], parts["headers"], **kwargs
    )


def problems_of(parts) -> list[str]:
    with pytest.raises(assemble_hosted.AssembleError) as info:
        run(parts)
    return info.value.problems


def test_assembles_ui_bundle_and_headers_into_one_folder(parts):
    report = run(parts)
    out = parts["out"]
    assert (out / "index.html").is_file()
    assert (out / "ort" / "ort-wasm-simd-threaded.wasm").is_file()
    assert (out / "_headers").read_text(encoding="utf-8") == "/*\n  X-Test: 1\n"
    assert (out / "models" / "models.json").is_file()
    assert (out / "models" / "nitid1s-detect.onnx").read_bytes() == ONNX
    assert (out / "models" / "LICENSE").is_file()  # licence files travel with the models
    assert report.models == ["nitid1s"]
    assert report.files == 8
    assert report.warnings == []
    # The inputs are never modified.
    assert not (parts["ui"] / "models").exists()
    # No staging leftovers next to the output.
    assert [p.name for p in out.parent.iterdir() if p.name.startswith(".site-")] == []


def test_missing_models_json(parts):
    (parts["bundle"] / "models.json").unlink()
    assert any("no models.json" in p for p in problems_of(parts))
    assert not parts["out"].exists()


@pytest.mark.parametrize("body", ["not json", "[]", '{"models": []}', '{"models": {}}'])
def test_unusable_catalogue(parts, body):
    (parts["bundle"] / "models.json").write_text(body, encoding="utf-8")
    assert problems_of(parts)


def test_sha256_mismatch_in_catalogue(parts):
    make_bundle_again = parts["bundle"] / "models.json"
    make_bundle_again.write_text(
        json.dumps(
            {
                "models": [
                    {"id": "a", "task": "detect", "url": "nitid1s-detect.onnx", "sha256": "0" * 64}
                ]
            }
        ),
        encoding="utf-8",
    )
    assert any("does not match models.json" in p for p in problems_of(parts))


def test_sha256_mismatch_in_sums(parts):
    (parts["bundle"] / "SHA256SUMS").write_text(
        f"{'1' * 64}  nitid1s-detect.onnx\n", encoding="utf-8"
    )
    assert any("SHA256SUMS: sha256 of nitid1s-detect.onnx" in p for p in problems_of(parts))


def test_sums_may_name_missing_or_unparseable_files(parts):
    (parts["bundle"] / "SHA256SUMS").write_text(f"{SHA}  gone.onnx\nnonsense\n", encoding="utf-8")
    problems = problems_of(parts)
    assert any("'gone.onnx' is not a file" in p for p in problems)
    assert any("cannot parse" in p for p in problems)


def test_binary_mode_marker_in_sums_is_accepted(parts):
    (parts["bundle"] / "SHA256SUMS").write_text(f"{SHA} *nitid1s-detect.onnx\n", encoding="utf-8")
    run(parts)


def test_referenced_file_must_exist(parts):
    (parts["bundle"] / "nitid1s-detect.onnx").unlink()
    (parts["bundle"] / "SHA256SUMS").unlink()
    assert any("file does not exist" in p for p in problems_of(parts))


@pytest.mark.parametrize(
    "url", ["../secret.onnx", "/etc/passwd", "https://example.com/m.onnx", "a\\b.onnx", ""]
)
def test_urls_must_stay_inside_the_bundle(parts, url):
    entries = [{"id": "a", "task": "detect", "url": url, "sha256": SHA}]
    (parts["bundle"] / "models.json").write_text(json.dumps({"models": entries}), encoding="utf-8")
    assert any(".url" in p for p in problems_of(parts))


@pytest.mark.parametrize("name", ["w.pth", "w.PT", "w.ckpt", "w.safetensors", "w.pkl"])
def test_training_checkpoints_are_refused_anywhere_in_the_bundle(parts, name):
    sub = parts["bundle"] / "extra"
    sub.mkdir()
    (sub / name).write_bytes(b"weights")
    assert any(name in p and "only ONNX" in p for p in problems_of(parts))
    assert not parts["out"].exists()


def test_symlinks_are_refused(parts):
    (parts["bundle"] / "link.onnx").symlink_to(parts["bundle"] / "nitid1s-detect.onnx")
    assert any("symlink" in p for p in problems_of(parts))


def test_cloudflare_file_size_limit(parts, monkeypatch):
    monkeypatch.setattr(assemble_hosted, "MAX_FILE_BYTES", 10)
    problems = problems_of(parts)
    assert any("allows 25 MiB per file" in p for p in problems)


def test_cloudflare_file_count_limit(parts, monkeypatch):
    monkeypatch.setattr(assemble_hosted, "MAX_FILES", 2)
    assert any("allows 2 per version" in p for p in problems_of(parts))


def test_missing_sha256_and_sums_are_warnings_not_errors(parts):
    entries = [{"id": "a", "task": "detect", "url": "nitid1s-detect.onnx"}]
    (parts["bundle"] / "models.json").write_text(json.dumps({"models": entries}), encoding="utf-8")
    (parts["bundle"] / "SHA256SUMS").unlink()
    (parts["bundle"] / "stray.onnx").write_bytes(b"x")
    report = run(parts)
    text = " | ".join(report.warnings)
    assert "no sha256" in text and "no SHA256SUMS" in text and "stray.onnx is not listed" in text


def test_duplicate_and_malformed_entries(parts):
    entry = {"id": "a", "task": "detect", "url": "nitid1s-detect.onnx", "sha256": SHA}
    entries = [
        entry,
        entry,
        {"id": "b", "task": "pose", "url": "nitid1s-detect.onnx"},
        "x",
        {"id": 3},
    ]
    (parts["bundle"] / "models.json").write_text(json.dumps({"models": entries}), encoding="utf-8")
    problems = " | ".join(problems_of(parts))
    for fragment in (
        "duplicate model",
        '.task must be "detect"',
        "is not an object",
        ".id must be",
    ):
        assert fragment in problems


def test_ui_must_be_built(parts):
    (parts["ui"] / "index.html").unlink()
    (parts["ui"] / "ort" / "ort-wasm-simd-threaded.wasm").unlink()
    problems = " | ".join(problems_of(parts))
    assert "no index.html" in problems and "no ort/*.wasm" in problems


def test_ui_must_not_already_have_models(parts):
    (parts["ui"] / "models").mkdir()
    assert any("reserved for the bundle" in p for p in problems_of(parts))


def test_headers_file_must_exist(parts):
    parts["headers"].unlink()
    assert any("headers file" in p for p in problems_of(parts))


def test_output_must_be_empty_or_cleaned(parts):
    parts["out"].mkdir()
    (parts["out"] / "old.txt").write_text("old", encoding="utf-8")
    assert any("not empty" in p for p in problems_of(parts))
    assert (parts["out"] / "old.txt").exists()  # a refused run touches nothing
    run(parts, clean=True)
    assert not (parts["out"] / "old.txt").exists()
    assert (parts["out"] / "index.html").is_file()


def test_output_must_not_overlap_inputs(parts):
    parts["out"] = parts["bundle"]
    assert any("overlaps" in p for p in problems_of(parts))
    parts["out"] = parts["ui"] / "sub"
    assert any("overlaps" in p for p in problems_of(parts))


def test_failed_run_leaves_no_staging_folder(parts, monkeypatch):
    monkeypatch.setattr(
        assemble_hosted.shutil, "copyfile", lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))
    )
    with pytest.raises(OSError):
        run(parts)
    assert not parts["out"].exists()
    assert [p.name for p in parts["out"].parent.iterdir() if p.name.startswith(".site-")] == []


def test_main_exit_codes_and_messages(parts, capsys):
    argv = [
        "--ui",
        str(parts["ui"]),
        "--bundle",
        str(parts["bundle"]),
        "--headers",
        str(parts["headers"]),
        "--out",
        str(parts["out"]),
    ]
    assert assemble_hosted.main(argv) == 0
    assert "assembled" in capsys.readouterr().out
    assert assemble_hosted.main([*argv, "--clean"]) == 0
    assert assemble_hosted.main(argv) == 1  # not empty any more, no --clean
    assert "not empty" in capsys.readouterr().err


def test_real_headers_file_is_the_default():
    assert assemble_hosted.DEFAULT_HEADERS.is_file()
    assert assemble_hosted.DEFAULT_HEADERS == REPO_ROOT / "studio/hosting/cloudflare/_headers"
