from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("jsonschema")

from tools.check_models_lock import (  # noqa: E402
    ALLOWED_LICENSES,
    check_lock,
    main,
    rule_errors,
    schema_errors,
)

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "studio" / "models.lock.json"
SCHEMA_PATH = ROOT / "studio" / "models.lock.schema.json"
SCHEMA: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
SHA = "a" * 64


def entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "dfine-s-coco",
        "task": "detect",
        "nitid_model": "nitid1s",
        "weights_variant": "coco",
        "format": "pytorch-checkpoint",
        "upstream_repo": "https://github.com/Peterande/D-FINE",
        "revision": "dfinev1.0",
        "revision_kind": "release-tag",
        "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth",
        "size_bytes": 41841422,
        "sha256": SHA,
        "sha256_status": "verified",
        "license": "Apache-2.0",
        "license_source": "LICENSE at a pinned commit",
        "training_data": ["COCO train2017"],
        "open_caveats": [],
        "notice_required": True,
        "notice_files": ["LICENSE"],
        "hostable": True,
        "hostable_reason": "Passes the gate.",
    }
    base.update(overrides)
    return base


def lock(*entries: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_on": "2026-11-18",
        "entries": list(entries or [entry()]),
    }


def test_committed_lock_is_valid() -> None:
    committed = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    assert check_lock(committed, SCHEMA) == []


def test_committed_lock_hostable_entries_have_verified_hashes() -> None:
    committed = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    for item in committed["entries"]:
        if item["hostable"]:
            assert item["sha256_status"] == "verified"
            assert not item["open_caveats"]


def test_valid_entry_passes() -> None:
    assert check_lock(lock(), SCHEMA) == []


@pytest.mark.parametrize("licence", sorted(ALLOWED_LICENSES))
def test_every_allowlisted_licence_passes(licence: str) -> None:
    assert check_lock(lock(entry(license=licence)), SCHEMA) == []


@pytest.mark.parametrize(
    "licence",
    ["CC-BY-NC-4.0", "AGPL-3.0-only", "GPL-3.0-only", "MIT OR GPL-3.0-only", "NOASSERTION"],
)
def test_licence_outside_allowlist_is_rejected(licence: str) -> None:
    problems = check_lock(lock(entry(license=licence, hostable=False)), SCHEMA)

    assert any("not in the allowlist" in p for p in problems)


@pytest.mark.parametrize("revision", ["main", "master", "latest", "HEAD", " Main "])
def test_floating_revision_is_rejected(revision: str) -> None:
    problems = rule_errors(lock(entry(revision=revision)))

    assert any("not pinned" in p for p in problems)


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/ArgoSA/D-FINE-seg/resolve/main/dfine_seg_s_coco.pt",
        "https://github.com/o/r/releases/latest/download/w.pth",
        "https://example.com/w.onnx?rev=main",
    ],
)
def test_url_with_floating_revision_is_rejected(url: str) -> None:
    problems = rule_errors(lock(entry(url=url)))

    assert any("floating revision" in p for p in problems)


def test_pinned_url_containing_the_word_main_elsewhere_is_accepted() -> None:
    url = "https://example.com/domain/mainframe/w.onnx"

    assert rule_errors(lock(entry(url=url))) == []


def test_duplicate_ids_are_rejected() -> None:
    problems = rule_errors(lock(entry(), entry()))

    assert any("duplicate id" in p for p in problems)


def test_hostable_requires_verified_sha256() -> None:
    unverified = entry(sha256=None, sha256_status="unverified")

    problems = rule_errors(lock(unverified))

    assert any("verified sha256" in p for p in problems)


def test_unverified_entry_may_exist_when_not_hostable() -> None:
    unverified = entry(sha256=None, sha256_status="unverified", hostable=False)

    assert check_lock(lock(unverified), SCHEMA) == []


def test_hostable_requires_notice_coverage() -> None:
    problems = rule_errors(lock(entry(notice_files=[])))

    assert any("notice_files is empty" in p for p in problems)
    assert any("notice_required to be covered" in p for p in problems)


def test_notice_not_required_means_empty_notice_files_are_fine() -> None:
    assert check_lock(lock(entry(notice_required=False, notice_files=[])), SCHEMA) == []


def test_open_caveats_block_hostable() -> None:
    problems = rule_errors(lock(entry(open_caveats=["Objects365 terms unresolved"])))

    assert any("open_caveats" in p for p in problems)


def test_open_caveats_with_hostable_false_is_valid() -> None:
    caveat = entry(open_caveats=["Objects365 terms unresolved"], hostable=False)

    assert check_lock(lock(caveat), SCHEMA) == []


def test_schema_rejects_verified_status_without_hash() -> None:
    problems = schema_errors(lock(entry(sha256=None)), SCHEMA)

    assert problems


def test_schema_rejects_hash_with_unverified_status() -> None:
    problems = schema_errors(lock(entry(sha256_status="unverified", hostable=False)), SCHEMA)

    assert problems


def test_schema_rejects_malformed_sha256() -> None:
    assert schema_errors(lock(entry(sha256="XYZ")), SCHEMA)


def test_schema_rejects_missing_field_and_unknown_field() -> None:
    missing = entry()
    del missing["license_source"]

    assert schema_errors(lock(missing), SCHEMA)
    assert schema_errors(lock(entry(surprise=1)), SCHEMA)


def test_schema_rejects_bad_task_and_bad_nitid_name() -> None:
    assert schema_errors(lock(entry(task="pose")), SCHEMA)
    assert schema_errors(lock(entry(nitid_model="model1s")), SCHEMA)


def test_schema_rejects_empty_entries_and_wrong_version() -> None:
    assert schema_errors({"schema_version": 1, "generated_on": "2026-11-18", "entries": []}, SCHEMA)
    broken = lock()
    broken["schema_version"] = 2
    assert schema_errors(broken, SCHEMA)


def test_rules_are_not_run_when_structure_is_invalid() -> None:
    problems = check_lock(lock(entry(task="pose", license="GPL-3.0-only")), SCHEMA)

    assert problems
    assert all(p.startswith("schema:") for p in problems)


def test_main_returns_zero_for_the_committed_lock(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--lock", str(LOCK_PATH), "--schema", str(SCHEMA_PATH)])

    assert code == 0
    assert "models.lock OK" in capsys.readouterr().out


def test_main_returns_one_and_prints_problems(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = copy.deepcopy(lock(entry(license="CC-BY-NC-4.0", hostable=False)))
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(bad), encoding="utf-8")

    code = main(["--lock", str(path), "--schema", str(SCHEMA_PATH)])

    assert code == 1
    assert "allowlist" in capsys.readouterr().err


def test_main_returns_two_when_the_file_is_missing_or_not_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--lock", str(tmp_path / "nope.json"), "--schema", str(SCHEMA_PATH)]) == 2
    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json", encoding="utf-8")

    assert main(["--lock", str(garbage), "--schema", str(SCHEMA_PATH)]) == 2
    assert "cannot read input" in capsys.readouterr().err
