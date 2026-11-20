"""Unit tests for ``tools/publish_models.py``: selection, the licence gate, catalogue and checksums.

No network and no torch: weights, wrapping and export are injected fakes that write tiny files.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tools import publish_models as pm

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "studio" / "models.lock.json"
FIXTURE = ROOT / "studio" / "src" / "browser" / "__fixtures__" / "hosted-models.json"


def entry(
    model: str = "nitid1s",
    task: str = "detect",
    *,
    variant: str = "coco",
    hostable: bool = True,
    sha: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    ident = (
        f"dfine-{model[-1]}-{variant}" if task == "detect" else f"dfine-seg-{model[-1]}-{variant}"
    )
    return {
        "id": ident,
        "task": task,
        "nitid_model": model,
        "weights_variant": variant,
        "format": "pytorch-checkpoint",
        "upstream_repo": "https://github.com/Peterande/D-FINE",
        "revision": "dfinev1.0",
        "revision_kind": "release-tag",
        "url": f"https://example.test/dfinev1.0/{ident}.pth",
        "size_bytes": 3,
        "sha256": sha or pm.hashlib.sha256(ident.encode()).hexdigest(),
        "sha256_status": "verified",
        "license": "Apache-2.0",
        "license_source": "test",
        "training_data": ["COCO train2017"],
        "open_caveats": [],
        "notice_required": True,
        "notice_files": ["LICENSE"],
        "hostable": hostable,
        "hostable_reason": "test",
        **extra,
    }


def make_lock(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "generated_on": "2026-11-20", "entries": list(entries)}


@pytest.fixture
def lock() -> dict[str, Any]:
    return make_lock(
        entry("nitid1x"),
        entry("nitid1s", "segment"),
        entry("nitid1s"),
        entry("nitid1m"),
        entry("nitid1n"),
        entry("nitid1s", variant="obj2coco", hostable=False),
    )


# ----------------------------------------------------------------------------- selection


def ids(entries: list[dict[str, Any]]) -> list[str]:
    return [e["id"] for e in entries]


def test_default_selection_is_small_sizes_hostable_only_in_catalogue_order(lock):
    chosen = pm.select_entries(lock)
    assert ids(chosen) == ["dfine-n-coco", "dfine-s-coco", "dfine-seg-s-coco"]


def test_size_allow_list_is_an_option(lock):
    assert ids(pm.select_entries(lock, sizes=["m", "x"])) == ["dfine-m-coco", "dfine-x-coco"]


def test_unknown_size_or_task_is_an_error(lock):
    with pytest.raises(pm.PublishError, match="size"):
        pm.select_entries(lock, sizes=["xl"])
    with pytest.raises(pm.PublishError, match="task"):
        pm.select_entries(lock, tasks=["pose"])


def test_task_filter(lock):
    assert ids(pm.select_entries(lock, tasks=["segment"])) == ["dfine-seg-s-coco"]


def test_models_filter_by_public_name_ignores_default_sizes_and_skips_non_hostable(lock):
    # nitid1s also has an obj2coco entry (hostable: false); a public name selects only hostable ones
    assert ids(pm.select_entries(lock, models=["nitid1s"])) == ["dfine-s-coco", "dfine-seg-s-coco"]
    assert ids(pm.select_entries(lock, models=["nitid1x"])) == ["dfine-x-coco"]


def test_explicit_sizes_still_apply_with_models(lock):
    with pytest.raises(pm.PublishError, match="empty"):
        pm.select_entries(lock, models=["nitid1x"], sizes=["n"])


def test_models_filter_by_lock_id(lock):
    assert ids(pm.select_entries(lock, models=["dfine-m-coco"])) == ["dfine-m-coco"]


def test_non_hostable_is_refused_even_if_requested_by_id(lock):
    with pytest.raises(pm.PublishError, match="hostable: false"):
        pm.select_entries(lock, models=["dfine-s-obj2coco"])


def test_non_hostable_is_refused_even_if_the_size_filter_would_drop_it(lock):
    with pytest.raises(pm.PublishError, match="hostable: false"):
        pm.select_entries(lock, models=["dfine-s-obj2coco"], sizes=["x"])


def test_public_name_with_only_non_hostable_entries_is_refused():
    only = make_lock(entry("nitid1l", variant="obj2coco", hostable=False))
    with pytest.raises(pm.PublishError, match="no hostable entry"):
        pm.select_entries(only, models=["nitid1l"])


def test_unknown_model_lists_the_known_ids(lock):
    with pytest.raises(pm.PublishError, match="unknown model 'nitid1q'.*dfine-s-coco"):
        pm.select_entries(lock, models=["nitid1q"])


def test_two_hostable_entries_for_one_model_and_task_are_ambiguous():
    clash = make_lock(entry("nitid1s"), entry("nitid1s", variant="other"))
    with pytest.raises(pm.PublishError, match="one file per"):
        pm.select_entries(clash)


def test_empty_selection_is_an_error():
    with pytest.raises(pm.PublishError, match="empty"):
        pm.select_entries(make_lock(entry("nitid1x")))  # default sizes n,s


def test_selection_is_deterministic_whatever_the_lock_order(lock):
    shuffled = copy.deepcopy(lock)
    shuffled["entries"].reverse()
    assert pm.select_entries(shuffled, sizes=list(pm.SIZE_ORDER)) == pm.select_entries(
        lock, sizes=list(pm.SIZE_ORDER)
    )


# ----------------------------------------------------------------------------- licence gate


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"hostable": False}, "hostable: false"),
        ({"sha256_status": "unverified"}, "no verified sha256"),
        ({"sha256": "nothex"}, "malformed sha256"),
        ({"open_caveats": ["x"]}, "open caveats"),
        ({"notice_files": []}, "notice files"),
        ({"task": "pose"}, "unsupported task"),
    ],
)
def test_check_publishable_refuses(override, message):
    with pytest.raises(pm.PublishError, match=message):
        pm.check_publishable({**entry(), **override})


def test_every_real_hostable_lock_entry_passes_the_gate():
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    hostable = [e for e in lock["entries"] if e["hostable"]]
    assert hostable
    for e in hostable:
        pm.check_publishable(e)


def test_real_lock_default_selection_is_coco_n_and_s_and_never_obj2coco():
    lock = pm.load_lock(LOCK_PATH)
    chosen = pm.select_entries(lock)
    assert ids(chosen) == ["dfine-n-coco", "dfine-seg-n-coco", "dfine-s-coco", "dfine-seg-s-coco"]
    assert all(
        "obj2coco" not in e["id"] for e in pm.select_entries(lock, sizes=list(pm.SIZE_ORDER))
    )


def test_real_lock_refuses_every_obj2coco_entry():
    lock = pm.load_lock(LOCK_PATH)
    for e in lock["entries"]:
        if e["weights_variant"] == "obj2coco":
            with pytest.raises(pm.PublishError, match="hostable: false"):
                pm.select_entries(lock, models=[e["id"]])


def test_load_lock_rejects_an_invalid_lock(tmp_path):
    bad = make_lock(entry(hostable=True, sha256_status="unverified"))
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(pm.PublishError, match="invalid"):
        pm.load_lock(path)
    with pytest.raises(pm.PublishError, match="cannot read"):
        pm.load_lock(tmp_path / "missing.json")


# --------------------------------------------------------------------------- notices, names


def test_notice_slug_and_paths():
    assert pm.notice_slug("https://github.com/ArgoHA/D-FINE-seg") == "ArgoHA-D-FINE-seg"
    assert pm.notice_relative_paths(entry()) == ["notices/Peterande-D-FINE/LICENSE"]
    with pytest.raises(pm.PublishError):
        pm.notice_slug("https://example.test/")
    with pytest.raises(pm.PublishError, match="unsafe"):
        pm.notice_relative_paths(entry(notice_files=["../LICENSE"]))


def test_committed_licence_texts_are_the_apache_text_named_in_the_lock():
    # the lock's license_source records "sha256 prefix c71d239df91726fc, 11357 bytes"
    for slug in ("Peterande-D-FINE", "ArgoHA-D-FINE-seg"):
        path = pm.DEFAULT_NOTICES_DIR / slug / "LICENSE"
        assert path.stat().st_size == 11357
        assert pm.sha256_file(path).startswith("c71d239df91726fc")


def test_every_hostable_lock_entry_has_its_notice_source_committed():
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    for e in lock["entries"]:
        if e["hostable"]:
            slug = pm.notice_slug(e["upstream_repo"])
            for name in e["notice_files"]:
                assert (pm.DEFAULT_NOTICES_DIR / slug / name).is_file(), e["id"]


def test_onnx_file_name():
    assert pm.onnx_file_name("nitid1s", "segment") == "nitid1s-segment.onnx"


# ----------------------------------------------------------------- catalogue and checksums


def sample_records() -> list[dict[str, Any]]:
    return [
        {"id": "nitid1n", "task": "detect", "url": "nitid1n-detect.onnx", "sha256": "1" * 64,
         "label": "nitid1n (detect, coco)"},
        {"id": "nitid1s", "task": "detect", "url": "nitid1s-detect.onnx", "sha256": "2" * 64,
         "label": "nitid1s (detect, coco)"},
        {"id": "nitid1s", "task": "segment", "url": "nitid1s-segment.onnx", "sha256": "3" * 64,
         "label": "nitid1s (segment, coco)"},
    ]  # fmt: skip


def test_catalogue_shape_is_exactly_version_1():
    document = pm.build_catalogue(sample_records())
    assert list(document) == ["models"]
    assert [list(m) for m in document["models"]] == [["id", "task", "url", "sha256", "label"]] * 3


def test_catalogue_matches_the_committed_typescript_fixture_byte_for_byte():
    """The vitest contract test parses this very file (studio/src/browser/hosted-bundle.test.ts)."""
    produced = pm.dump_json(pm.build_catalogue(sample_records()))
    assert FIXTURE.read_text(encoding="utf-8") == produced


def test_dump_json_is_deterministic_and_newline_terminated():
    document = pm.build_catalogue(sample_records())
    assert pm.dump_json(document) == pm.dump_json(copy.deepcopy(document))
    assert pm.dump_json(document).endswith("}\n")


def test_sha256sums_are_sorted_and_sha256sum_compatible():
    text = pm.format_sha256sums({"b.onnx": "b" * 64, "a/LICENSE": "a" * 64})
    assert text == f"{'a' * 64}  a/LICENSE\n{'b' * 64}  b.onnx\n"


def test_sha256_file(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    assert (
        pm.sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_render_notice_mentions_modification_source_and_licence_path():
    text = pm.render_notice([entry()], "9.9.9")
    assert "nitid 9.9.9" in text
    assert "nitid1s-detect.onnx" in text
    assert "notices/Peterande-D-FINE/LICENSE" in text
    assert "modified by exporting" in text


# --------------------------------------------------------------------------- build steps


def fake_onnx(path: Path, task: str = "detect", model: str | None = "nitid1s") -> None:
    """A valid, tiny ONNX carrying a real ``nitid_meta``."""
    import onnx
    from onnx import TensorProto, helper

    from dfine.export_meta import build_onnx_meta, embed_onnx_meta

    node = helper.make_node("Identity", ["images"], ["boxes"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 3, 32, 32])],
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    proto.ir_version = 9
    meta = build_onnx_meta(
        task=task, imgsz=32, names={0: "a", 1: "b"}, topk=5, nitid_version="9.9.9", model=model
    )
    embed_onnx_meta(proto, meta)
    onnx.save(proto, str(path))


class Fakes:
    """Counts calls so tests can prove what was (not) downloaded or exported."""

    def __init__(self) -> None:
        self.fetched: list[str] = []
        self.exported: list[str] = []

    def fetch(self, e: Mapping[str, Any], dest: Path) -> None:
        self.fetched.append(str(e["id"]))
        dest.write_bytes(str(e["id"]).encode())

    @staticmethod
    def prepare(e: Mapping[str, Any], raw: Path, dest: Path) -> Path:
        return raw

    def export(self, checkpoint: Path, task: str, dest: Path) -> None:
        self.exported.append(checkpoint.name)
        fake_onnx(dest, task, model=None)


def run_publish(lock, out, fakes, notices: Path, **kwargs: Any):
    return pm.publish(
        lock,
        out,
        notices_dir=notices,
        nitid_version="9.9.9",
        fetch=fakes.fetch,
        prepare=fakes.prepare,
        export=fakes.export,
        **kwargs,
    )


@pytest.fixture
def notices(tmp_path) -> Path:
    base = tmp_path / "notices-src"
    (base / "Peterande-D-FINE").mkdir(parents=True)
    (base / "Peterande-D-FINE" / "LICENSE").write_text("Apache License", encoding="utf-8")
    return base


@pytest.fixture
def small_lock() -> dict[str, Any]:
    return make_lock(
        entry("nitid1n"), entry("nitid1s"), entry("nitid1s", variant="obj2coco", hostable=False)
    )


def test_publish_writes_the_documented_layout(tmp_path, small_lock, notices):
    out = tmp_path / "bundle"
    built = run_publish(small_lock, out, Fakes(), notices)
    assert [m.file for m in built] == ["nitid1n-detect.onnx", "nitid1s-detect.onnx"]
    names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert names == [
        "NOTICE.txt",
        "SHA256SUMS",
        "models.json",
        "nitid1n-detect.onnx",
        "nitid1n-detect.onnx.meta.json",
        "nitid1s-detect.onnx",
        "nitid1s-detect.onnx.meta.json",
        "notices/Peterande-D-FINE/LICENSE",
    ]
    catalogue = json.loads((out / "models.json").read_text(encoding="utf-8"))
    assert [m["url"] for m in catalogue["models"]] == ["nitid1n-detect.onnx", "nitid1s-detect.onnx"]
    for model in catalogue["models"]:
        assert model["sha256"] == pm.sha256_file(out / model["url"])
    sums = (out / "SHA256SUMS").read_text(encoding="utf-8")
    assert "models.json" in sums and "SHA256SUMS" not in sums


def test_publish_never_writes_weights(tmp_path, small_lock, notices):
    out = tmp_path / "bundle"
    run_publish(small_lock, out, Fakes(), notices)
    assert not [p for p in out.rglob("*") if p.suffix in {".pth", ".pt"}]


def test_sidecar_records_source_and_embedded_meta(tmp_path, small_lock, notices):
    out = tmp_path / "bundle"
    run_publish(small_lock, out, Fakes(), notices)
    sidecar = json.loads((out / "nitid1s-detect.onnx.meta.json").read_text(encoding="utf-8"))
    assert sidecar["source"]["lock_id"] == "dfine-s-coco"
    assert sidecar["source"]["sha256"] == small_lock["entries"][1]["sha256"]
    assert sidecar["sha256"] == pm.sha256_file(out / "nitid1s-detect.onnx")
    assert sidecar["nitid_meta"]["task"] == "detect"
    assert sidecar["nitid_version"] == "9.9.9"


def test_second_run_is_idempotent_and_does_not_download_or_export(tmp_path, small_lock, notices):
    out = tmp_path / "bundle"
    first = Fakes()
    run_publish(small_lock, out, first, notices)
    before = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    second = Fakes()
    built = run_publish(small_lock, out, second, notices)
    assert second.fetched == [] and second.exported == []
    assert all(m.reused for m in built)
    assert before == {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file()}


def test_force_rebuilds(tmp_path, small_lock, notices):
    out = tmp_path / "bundle"
    run_publish(small_lock, out, Fakes(), notices)
    again = Fakes()
    run_publish(small_lock, out, again, notices, force=True)
    assert len(again.exported) == 2


def test_a_changed_source_hash_or_nitid_version_rebuilds(tmp_path, small_lock, notices):
    out = tmp_path / "bundle"
    run_publish(small_lock, out, Fakes(), notices)
    changed = copy.deepcopy(small_lock)
    changed["entries"][0]["sha256"] = "f" * 64
    fakes = Fakes()

    def fetch(e: Mapping[str, Any], dest: Path) -> None:
        dest.write_bytes(b"new")

    with pytest.raises(pm.PublishError, match="checksum mismatch"):
        pm.publish(
            changed, out, notices_dir=notices, nitid_version="9.9.9", fetch=fetch,
            prepare=fakes.prepare, export=fakes.export, entries=[changed["entries"][0]],
        )  # fmt: skip
    newer = Fakes()
    run_publish(small_lock, out, newer, notices)  # same version: reused
    assert newer.exported == []
    upgraded = Fakes()
    pm.publish(
        small_lock, out, notices_dir=notices, nitid_version="10.0.0", fetch=upgraded.fetch,
        prepare=upgraded.prepare, export=upgraded.export,
    )  # fmt: skip
    assert len(upgraded.exported) == 2


def test_a_modified_onnx_is_not_reused(tmp_path, small_lock, notices):
    out = tmp_path / "bundle"
    run_publish(small_lock, out, Fakes(), notices)
    (out / "nitid1n-detect.onnx").write_bytes(b"tampered")
    again = Fakes()
    run_publish(small_lock, out, again, notices)
    assert len(again.exported) == 1


def test_default_selection_when_entries_not_given(tmp_path, small_lock, notices):
    built = run_publish(small_lock, tmp_path / "b", Fakes(), notices)
    assert len(built) == 2  # n and s, hostable only


def test_stale_outputs_of_earlier_runs_are_pruned(tmp_path, small_lock, notices):
    out = tmp_path / "bundle"
    run_publish(small_lock, out, Fakes(), notices)
    only_n = [small_lock["entries"][0]]
    run_publish(small_lock, out, Fakes(), notices, entries=only_n)
    assert not (out / "nitid1s-detect.onnx").exists()
    assert not (out / "nitid1s-detect.onnx.meta.json").exists()
    assert "nitid1s" not in (out / "models.json").read_text(encoding="utf-8")
    assert "nitid1s" not in (out / "SHA256SUMS").read_text(encoding="utf-8")


def test_publish_refuses_a_non_hostable_entry_handed_in_directly(tmp_path, small_lock, notices):
    forbidden = small_lock["entries"][2]
    with pytest.raises(pm.PublishError, match="hostable: false"):
        run_publish(small_lock, tmp_path / "b", Fakes(), notices, entries=[forbidden])
    assert not (tmp_path / "b" / "models.json").exists()


def test_missing_licence_source_blocks_publication(tmp_path, small_lock):
    with pytest.raises(pm.PublishError, match="licence|notice"):
        run_publish(small_lock, tmp_path / "b", Fakes(), tmp_path / "no-notices")


def test_checksum_mismatch_stops_before_anything_is_loaded(tmp_path, small_lock, notices):
    offline = tmp_path / "weights"
    offline.mkdir()
    (offline / "dfine-n-coco.pth").write_bytes(b"not the pinned file")
    fakes = Fakes()
    with pytest.raises(pm.PublishError, match="checksum mismatch"):
        run_publish(small_lock, tmp_path / "b", fakes, notices, offline_from=offline)
    assert fakes.exported == []


def test_offline_from_uses_local_files_and_never_downloads(tmp_path, small_lock, notices):
    offline = tmp_path / "weights"
    offline.mkdir()
    for e in small_lock["entries"][:2]:
        data = e["id"].encode()
        (offline / f"{e['id']}.pth").write_bytes(data)
        e["sha256"] = pm.hashlib.sha256(data).hexdigest()
    fakes = Fakes()
    run_publish(small_lock, tmp_path / "b", fakes, notices, offline_from=offline)
    assert fakes.fetched == []
    assert len(fakes.exported) == 2


def test_offline_from_reports_a_missing_file(tmp_path, small_lock, notices):
    offline = tmp_path / "weights"
    offline.mkdir()
    with pytest.raises(pm.PublishError, match="not found"):
        run_publish(small_lock, tmp_path / "b", Fakes(), notices, offline_from=offline)


def test_download_failure_is_a_clear_error(tmp_path, small_lock, notices):
    def boom(e: Mapping[str, Any], dest: Path) -> None:
        raise OSError("network down")

    with pytest.raises(pm.PublishError, match="download failed.*network down"):
        pm.publish(
            small_lock, tmp_path / "b", notices_dir=notices, nitid_version="1", fetch=boom,
            prepare=Fakes.prepare, export=Fakes().export,
        )  # fmt: skip


def test_onnx_whose_task_contradicts_the_lock_is_refused(tmp_path, small_lock, notices):
    class Wrong(Fakes):
        def export(self, checkpoint: Path, task: str, dest: Path) -> None:
            fake_onnx(dest, "segment", model=None)

    with pytest.raises(pm.PublishError, match="says task"):
        run_publish(small_lock, tmp_path / "b", Wrong(), notices)


def test_onnx_without_meta_is_refused(tmp_path, small_lock, notices):
    import onnx
    from onnx import TensorProto, helper

    class NoMeta(Fakes):
        def export(self, checkpoint: Path, task: str, dest: Path) -> None:
            graph = helper.make_graph(
                [helper.make_node("Identity", ["i"], ["o"])],
                "g",
                [helper.make_tensor_value_info("i", TensorProto.FLOAT, [1])],
                [helper.make_tensor_value_info("o", TensorProto.FLOAT, [1])],
            )
            onnx.save(helper.make_model(graph), str(dest))

    with pytest.raises(pm.PublishError, match="no nitid_meta"):
        run_publish(small_lock, tmp_path / "b", NoMeta(), notices)


# ------------------------------------------------------------------------------------- CLI


def test_cli_dry_run_prints_the_selection_and_writes_nothing(tmp_path, capsys):
    code = pm.main(["--out", str(tmp_path / "o"), "--dry-run"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "nitid1n-detect.onnx" in printed and "nitid1s-segment.onnx" in printed
    assert "obj2coco" not in printed and "nitid1m" not in printed
    assert not (tmp_path / "o").exists()


def test_cli_sizes_option(tmp_path, capsys):
    assert pm.main(["--out", str(tmp_path / "o"), "--dry-run", "--sizes", "x"]) == 0
    assert "nitid1x-detect.onnx" in capsys.readouterr().out


def test_cli_refuses_obj2coco_with_exit_code_1(tmp_path, capsys):
    code = pm.main(["--out", str(tmp_path / "o"), "--models", "dfine-s-obj2coco", "--dry-run"])
    assert code == 1
    assert "hostable: false" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_cli_reports_an_unreadable_lock(tmp_path, capsys):
    assert pm.main(["--out", str(tmp_path), "--lock", str(tmp_path / "nope.json")]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_publisher_adds_the_public_model_name_when_the_exporter_could_not_infer_it(
    tmp_path, small_lock, notices
):
    out = tmp_path / "b"
    run_publish(small_lock, out, Fakes(), notices)  # Fakes export without a model key
    sidecar = json.loads((out / "nitid1s-detect.onnx.meta.json").read_text(encoding="utf-8"))
    assert sidecar["nitid_meta"]["model"] == "nitid1s"
    assert pm.read_meta(out / "nitid1s-detect.onnx")["model"] == "nitid1s"


def test_publisher_refuses_an_onnx_that_names_another_model(tmp_path, small_lock, notices):
    class Other(Fakes):
        def export(self, checkpoint: Path, task: str, dest: Path) -> None:
            fake_onnx(dest, task, model="nitid1x")

    with pytest.raises(pm.PublishError, match="lock says"):
        run_publish(small_lock, tmp_path / "b", Other(), notices)


def test_cache_dir_hit_skips_the_download_but_still_checks_the_hash(tmp_path, small_lock, notices):
    cache = tmp_path / "cache"
    cache.mkdir()
    for e in small_lock["entries"][:2]:
        data = e["id"].encode()
        (cache / f"{e['id']}.pth").write_bytes(data)
        e["sha256"] = pm.hashlib.sha256(data).hexdigest()
    fakes = Fakes()
    run_publish(small_lock, tmp_path / "b", fakes, notices, cache_dir=cache)
    assert fakes.fetched == [] and len(fakes.exported) == 2
    (cache / "dfine-n-coco.pth").write_bytes(b"corrupt")
    with pytest.raises(pm.PublishError, match="checksum mismatch"):
        run_publish(small_lock, tmp_path / "c", Fakes(), notices, cache_dir=cache)


def test_a_failed_download_leaves_no_partial_file_in_the_cache(tmp_path, small_lock, notices):
    cache = tmp_path / "cache"
    cache.mkdir()

    def half(e: Mapping[str, Any], dest: Path) -> None:
        dest.write_bytes(b"half")
        raise OSError("cut off")

    with pytest.raises(pm.PublishError, match="download failed"):
        pm.publish(
            small_lock, tmp_path / "b", notices_dir=notices, nitid_version="1", fetch=half,
            prepare=Fakes.prepare, export=Fakes().export, cache_dir=cache,
        )  # fmt: skip
    assert list(cache.iterdir()) == []
