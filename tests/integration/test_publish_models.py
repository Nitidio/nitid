"""End-to-end: tiny random checkpoints -> real export -> bundle -> validator (no network).

The lock below is synthetic: its sha256 values are those of the tiny wrapped checkpoints from
``tests/conftest.py``, so the whole real path runs (checksum check, wrapping detection, the
exporter Studio's local server uses, ``nitid_meta``, sidecars, catalogue, checksums).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("jsonschema")
pytest.importorskip("onnx")
pytest.importorskip("onnxsim")

from tests.unit.test_publish_models import entry, make_lock  # noqa: E402
from tools import check_hosted_bundle as chk  # noqa: E402
from tools import publish_models as pm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STUDIO = ROOT / "studio"


@pytest.fixture(scope="module")
def weights(tmp_path_factory, tiny_checkpoint, tiny_segment_checkpoint) -> Path:
    folder = tmp_path_factory.mktemp("offline-weights")
    shutil.copyfile(tiny_checkpoint, folder / "dfine-s-coco.pth")
    shutil.copyfile(tiny_segment_checkpoint, folder / "dfine-seg-s-coco.pth")
    return folder


@pytest.fixture(scope="module")
def lock(weights) -> dict[str, Any]:
    detect = entry("nitid1s", "detect", sha=pm.sha256_file(weights / "dfine-s-coco.pth"))
    segment = entry("nitid1s", "segment", sha=pm.sha256_file(weights / "dfine-seg-s-coco.pth"))
    refused = entry("nitid1s", variant="obj2coco", hostable=False)
    return make_lock(detect, segment, refused)


@pytest.fixture(scope="module")
def notices(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("notices")
    (base / "Peterande-D-FINE").mkdir()
    (base / "Peterande-D-FINE" / "LICENSE").write_text("Apache License 2.0", encoding="utf-8")
    return base


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, lock, weights, notices) -> Path:
    out = tmp_path_factory.mktemp("bundle")
    built = pm.publish(lock, out, offline_from=weights, notices_dir=notices, nitid_version="9.9.9")
    assert [m.file for m in built] == ["nitid1s-detect.onnx", "nitid1s-segment.onnx"]
    return out


def test_bundle_passes_the_validator(bundle, lock):
    report = chk.check_bundle(bundle, lock)
    assert report.ok, report.problems


def test_onnx_files_carry_meta_with_names_and_the_public_model_name(bundle):
    detect = pm.read_meta(bundle / "nitid1s-detect.onnx")
    segment = pm.read_meta(bundle / "nitid1s-segment.onnx")
    assert (detect["task"], detect["model"]) == ("detect", "nitid1s")
    assert (segment["task"], segment["model"]) == ("segment", "nitid1s")
    assert detect["names"]["0"] == "class_0" and len(detect["names"]) == 80
    assert [o["name"] for o in segment["outputs"]] == ["labels", "boxes", "scores", "masks"]


def test_onnx_is_static_and_runs_in_onnxruntime(bundle):
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    session = ort.InferenceSession(
        str(bundle / "nitid1s-detect.onnx"), providers=["CPUExecutionProvider"]
    )
    shape = session.get_inputs()[0].shape
    assert shape[:2] == [1, 3] and shape[2] == shape[3] == 640
    outputs = session.run(None, {"images": np.zeros(shape, dtype=np.float32)})
    assert [o.shape[0] for o in outputs] == [1, 1, 1]


def test_catalogue_sha256_matches_and_urls_are_relative(bundle):
    catalogue = json.loads((bundle / "models.json").read_text(encoding="utf-8"))
    assert [(m["id"], m["task"]) for m in catalogue["models"]] == [
        ("nitid1s", "detect"),
        ("nitid1s", "segment"),
    ]
    for model in catalogue["models"]:
        assert "/" not in model["url"]
        assert model["sha256"] == pm.sha256_file(bundle / model["url"])


def test_rebuilding_is_idempotent_and_byte_identical(bundle, lock, weights, notices):
    before = {p.name: p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
    pm.publish(lock, bundle, offline_from=weights, notices_dir=notices, nitid_version="9.9.9")
    assert before == {p.name: p.read_bytes() for p in bundle.rglob("*") if p.is_file()}


def test_a_forced_rebuild_gives_the_same_onnx_bytes(bundle, lock, weights, notices, tmp_path):
    """The export is deterministic here, so the catalogue hash survives a rebuild."""
    out = tmp_path / "second"
    pm.publish(lock, out, offline_from=weights, notices_dir=notices, nitid_version="9.9.9")
    assert (out / "models.json").read_bytes() == (bundle / "models.json").read_bytes()


def test_cli_end_to_end_and_validator_cli(tmp_path, lock, weights, notices, capsys):
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    out = tmp_path / "cli-bundle"
    code = pm.main(
        [
            "--lock", str(lock_path),
            "--out", str(out),
            "--offline-from", str(weights),
            "--notices-dir", str(notices),
            "--tasks", "segment",
        ]
    )  # fmt: skip
    assert code == 0
    assert (out / "nitid1s-segment.onnx").is_file()
    assert not (out / "nitid1s-detect.onnx").exists()
    assert chk.main([str(out), "--lock", str(lock_path)]) == 0
    assert "hosted bundle OK" in capsys.readouterr().out


def test_cli_refuses_the_non_hostable_entry_and_writes_nothing(tmp_path, lock, weights, notices):
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    out = tmp_path / "refused"
    code = pm.main(
        ["--lock", str(lock_path), "--out", str(out), "--models", "dfine-s-obj2coco",
         "--offline-from", str(weights), "--notices-dir", str(notices)]
    )  # fmt: skip
    assert code == 1
    assert not out.exists()


def test_the_typescript_parser_accepts_the_generated_catalogue(bundle):
    """Runs studio's vitest against the real ``models.json`` of this bundle.

    Skipped unless ``studio/node_modules`` is installed (``pnpm install`` in studio/).
    """
    vitest = STUDIO / "node_modules" / ".bin" / "vitest"
    node = shutil.which("node")
    if not vitest.exists() or node is None:
        pytest.skip("studio/node_modules not installed (run `pnpm install` in studio/)")
    result = subprocess.run(
        [str(vitest), "run", "src/browser/hosted-bundle.test.ts"],
        cwd=STUDIO,
        env={**__import__("os").environ, "NITID_HOSTED_CATALOG": str(bundle / "models.json")},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
