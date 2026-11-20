"""Unit tests for ``tools/check_hosted_bundle.py``: every rule has a passing and a failing case."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("jsonschema")

from tests.unit.test_publish_models import (  # noqa: E402
    FIXTURE,
    Fakes,
    entry,
    fake_onnx,
    make_lock,
)
from tools import check_hosted_bundle as chk  # noqa: E402
from tools import publish_models as pm  # noqa: E402


@pytest.fixture
def lock() -> dict[str, Any]:
    return make_lock(entry("nitid1n"), entry("nitid1s"), entry("nitid1s", "segment"))


@pytest.fixture
def bundle(tmp_path, lock) -> Path:
    notices = tmp_path / "notices-src" / "Peterande-D-FINE"
    notices.mkdir(parents=True)
    (notices / "LICENSE").write_text("Apache License", encoding="utf-8")
    out = tmp_path / "bundle"
    fakes = Fakes()
    pm.publish(
        lock,
        out,
        notices_dir=tmp_path / "notices-src",
        nitid_version="9.9.9",
        fetch=fakes.fetch,
        prepare=fakes.prepare,
        export=fakes.export,
    )
    return out


def rewrite_sums(bundle: Path) -> None:
    """Make SHA256SUMS agree with the files again, so a test isolates one other fault."""
    hashes = {
        p.relative_to(bundle).as_posix(): pm.sha256_file(p)
        for p in sorted(bundle.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS"
    }
    (bundle / "SHA256SUMS").write_text(pm.format_sha256sums(hashes), encoding="utf-8")


def problems(bundle: Path, lock: dict[str, Any], **kwargs: Any) -> list[str]:
    return chk.check_bundle(bundle, lock, **kwargs).problems


def edit_catalogue(bundle: Path, change) -> None:
    path = bundle / "models.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(pm.dump_json(document), encoding="utf-8")
    rewrite_sums(bundle)


# --------------------------------------------------------------------------- happy path


def test_a_fresh_bundle_is_valid(bundle, lock):
    report = chk.check_bundle(bundle, lock)
    assert report.ok, report.problems
    assert report.total_bytes == sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())


def test_size_report_lists_files_largest_first_and_the_total(bundle, lock):
    report = chk.check_bundle(bundle, lock)
    text = chk.format_size_report(report)
    lines = text.splitlines()
    assert lines[-1].startswith("total")
    assert "nitid1s-detect.onnx" in text and "MiB" in text
    sizes = [int(line.split()[-4].replace(",", "")) for line in lines[:-1]]
    assert sizes == sorted(sizes, reverse=True)


def test_size_limit(bundle, lock):
    assert any("exceeds the limit" in p for p in problems(bundle, lock, max_total_bytes=10))
    assert not problems(bundle, lock, max_total_bytes=10_000_000)


# -------------------------------------------------------------------- catalogue schema


def test_the_committed_typescript_fixture_passes_the_python_mirror_of_catalog_ts():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert chk.catalogue_errors(document) == []


@pytest.mark.parametrize(
    ("document", "needle"),
    [
        ([], r'{"models"'),
        ({"models": {}}, r'{"models"'),
        ({"models": []}, "no models"),
        ({"models": ["x"]}, "not an object"),
        ({"models": [{"task": "detect", "url": "a.onnx", "sha256": "a" * 64}]}, "id must be"),
        ({"models": [{"id": "a", "task": "pose", "url": "a.onnx", "sha256": "a" * 64}]}, "task"),
        ({"models": [{"id": "a", "task": "detect", "sha256": "a" * 64}]}, "url must be"),
        ({"models": [{"id": "a", "task": "detect", "url": "a.onnx"}]}, "sha256"),
        (
            {"models": [{"id": "a", "task": "detect", "url": "a.onnx", "sha256": "A" * 64}]},
            "sha256",
        ),
        (
            {
                "models": [
                    {"id": "a", "task": "detect", "url": "a.onnx", "sha256": "a" * 64, "label": 1}
                ]
            },
            "label",
        ),
        (
            {"models": [{"id": "a", "task": "detect", "url": "a.onnx", "sha256": "a" * 64}] * 2},
            "duplicate",
        ),
    ],
)
def test_catalogue_errors(document, needle):
    assert any(needle in p for p in chk.catalogue_errors(document)), chk.catalogue_errors(document)


@pytest.mark.parametrize(
    "url",
    ["https://cdn.test/a.onnx", "//cdn.test/a.onnx", "/abs/a.onnx", "../a.onnx", "d/a.onnx",
     "a.pth", "a\\b.onnx"],
)  # fmt: skip
def test_catalogue_urls_must_be_plain_relative_onnx_names(url):
    document = {"models": [{"id": "a", "task": "detect", "url": url, "sha256": "a" * 64}]}
    assert any("url" in p for p in chk.catalogue_errors(document))


# -------------------------------------------------------------------------- failures


def test_missing_catalogue(bundle, lock):
    (bundle / "models.json").unlink()
    assert any("models.json is missing" in p for p in problems(bundle, lock))


def test_catalogue_that_is_not_json(bundle, lock):
    (bundle / "models.json").write_text("{nope", encoding="utf-8")
    assert any("cannot read as JSON" in p for p in problems(bundle, lock))


def test_not_a_directory(tmp_path, lock):
    assert any("not a directory" in p for p in problems(tmp_path / "nope", lock))


def test_sha256_in_catalogue_must_match_the_file(bundle, lock):
    edit_catalogue(bundle, lambda d: d["models"][0].update(sha256="0" * 64))
    assert any("sha256 of nitid1n-detect.onnx does not match" in p for p in problems(bundle, lock))


def test_catalogue_pointing_at_a_missing_file(bundle, lock):
    (bundle / "nitid1n-detect.onnx").unlink()
    rewrite_sums(bundle)
    assert any("missing" in p and "nitid1n-detect.onnx" in p for p in problems(bundle, lock))


def test_onnx_file_not_in_the_catalogue(bundle, lock):
    fake_onnx(bundle / "extra.onnx")
    rewrite_sums(bundle)
    assert any("extra.onnx: ONNX file is not in the catalogue" in p for p in problems(bundle, lock))


def test_onnx_without_nitid_meta(bundle, lock):
    import onnx

    path = bundle / "nitid1n-detect.onnx"
    proto = onnx.load(str(path))
    del proto.metadata_props[:]
    onnx.save(proto, str(path))
    edit_catalogue(bundle, lambda d: d["models"][0].update(sha256=pm.sha256_file(path)))
    assert any("no nitid_meta" in p for p in problems(bundle, lock))


def test_onnx_with_invalid_meta_schema(bundle, lock):
    import onnx

    from dfine.export_meta import META_KEY

    path = bundle / "nitid1n-detect.onnx"
    proto = onnx.load(str(path))
    meta = json.loads(proto.metadata_props[0].value)
    assert proto.metadata_props[0].key == META_KEY
    meta["input"]["range"] = "0-255"
    proto.metadata_props[0].value = json.dumps(meta)
    onnx.save(proto, str(path))
    edit_catalogue(bundle, lambda d: d["models"][0].update(sha256=pm.sha256_file(path)))
    assert any("nitid_meta" in p for p in problems(bundle, lock))


def test_meta_task_must_match_the_catalogue(bundle, lock):
    edit_catalogue(bundle, lambda d: d["models"][0].update(task="segment"))
    assert any("differs from catalogue" in p for p in problems(bundle, lock))


def test_meta_errors_semantic_shape_check():
    from dfine.export_meta import build_onnx_meta

    meta = build_onnx_meta(
        task="detect", imgsz=640, names={0: "a"}, topk=300, nitid_version="1.0.0", model="nitid1s"
    )
    assert chk.meta_errors(meta) == []
    meta["imgsz"] = 320
    assert any("does not match imgsz" in p for p in chk.meta_errors(meta))


def test_missing_sidecar(bundle, lock):
    (bundle / "nitid1n-detect.onnx.meta.json").unlink()
    rewrite_sums(bundle)
    assert any("sidecar" in p and "missing" in p for p in problems(bundle, lock))


def test_sidecar_naming_a_non_hostable_lock_entry(bundle, lock):
    lock["entries"][0]["hostable"] = False
    assert any("hostable: false" in p for p in problems(bundle, lock))


def test_sidecar_naming_an_unknown_lock_entry(bundle, lock):
    lock["entries"].pop(0)
    assert any("does not exist" in p for p in problems(bundle, lock))


def test_sidecar_source_hash_must_equal_the_lock(bundle, lock):
    lock["entries"][0]["sha256"] = "e" * 64
    assert any("source sha256 differs" in p for p in problems(bundle, lock))


def test_lock_entry_without_verified_sha(bundle, lock):
    lock["entries"][0]["sha256_status"] = "unverified"
    assert any("no verified sha256" in p for p in problems(bundle, lock))


def test_sidecar_meta_must_equal_embedded_meta(bundle, lock):
    path = bundle / "nitid1n-detect.onnx.meta.json"
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    sidecar["nitid_meta"]["topk"] = 1
    path.write_text(pm.dump_json(sidecar), encoding="utf-8")
    rewrite_sums(bundle)
    assert any("sidecar nitid_meta differs" in p for p in problems(bundle, lock))


def test_missing_licence_file(bundle, lock):
    (bundle / "notices" / "Peterande-D-FINE" / "LICENSE").unlink()
    rewrite_sums(bundle)
    assert any("licence file" in p for p in problems(bundle, lock))


def test_missing_notice_txt(bundle, lock):
    (bundle / "NOTICE.txt").unlink()
    rewrite_sums(bundle)
    assert any("NOTICE.txt is missing" in p for p in problems(bundle, lock))


@pytest.mark.parametrize("name", ["leak.pth", "leak.pt", "sub/leak.safetensors", "x.ckpt"])
def test_checkpoints_must_not_be_in_the_bundle(bundle, lock, name):
    target = bundle / name
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(b"weights")
    rewrite_sums(bundle)
    assert any("must never be hosted" in p for p in problems(bundle, lock))


def test_missing_sha256sums(bundle, lock):
    (bundle / "SHA256SUMS").unlink()
    assert any("SHA256SUMS is missing" in p for p in problems(bundle, lock))


def test_sha256sums_detects_a_changed_file(bundle, lock):
    (bundle / "NOTICE.txt").write_text("edited", encoding="utf-8")
    assert any("hash of NOTICE.txt does not match" in p for p in problems(bundle, lock))


def test_sha256sums_detects_unlisted_and_phantom_files(bundle, lock):
    (bundle / "stray.txt").write_text("x", encoding="utf-8")
    sums = bundle / "SHA256SUMS"
    sums.write_text(sums.read_text(encoding="utf-8") + f"{'a' * 64}  ghost.txt\n", encoding="utf-8")
    found = problems(bundle, lock)
    assert any("stray.txt is not listed" in p for p in found)
    assert any("ghost.txt, which does not exist" in p for p in found)


def test_parse_sha256sums_reports_malformed_and_duplicate_lines():
    hashes, found = chk.parse_sha256sums(f"{'a' * 64}  f\nbad line\n{'b' * 64}  f\n")
    assert hashes == {"f": "b" * 64}
    assert any("malformed" in p for p in found) and any("twice" in p for p in found)


# ------------------------------------------------------------------------------------ CLI


def test_cli_ok_and_failure_exit_codes(bundle, lock, tmp_path, capsys):
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    assert chk.main([str(bundle), "--lock", str(lock_path)]) == 0
    assert "hosted bundle OK" in capsys.readouterr().out
    (bundle / "NOTICE.txt").unlink()
    assert chk.main([str(bundle), "--lock", str(lock_path)]) == 1
    assert "PROBLEM:" in capsys.readouterr().err
    assert chk.main([str(bundle), "--lock", str(tmp_path / "nope.json")]) == 2
