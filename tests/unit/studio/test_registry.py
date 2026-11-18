"""Unit tests for the model catalogue: registry pairs, local scanning and downloads.

Checkpoints are tiny, random-weight files built at run time; nothing is downloaded.
"""

import logging
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

pytest.importorskip("fastapi", reason="needs the 'studio' extra")

import torch  # noqa: E402

from dfine.nn.build import build_model  # noqa: E402
from dfine.nn.configs import make_model_config  # noqa: E402
from dfine.utils import downloads  # noqa: E402
from dfine.utils.checkpoint import save_checkpoint  # noqa: E402
from nitid.studio import registry  # noqa: E402


def make_checkpoint(path: Path, task: str = "detect", size: str = "s") -> Path:
    config = make_model_config(f"dfine_{size}", task=task, num_classes=2, image_size=(64, 64))
    config["DFINETransformer"]["num_layers"] = 1
    config["DFINETransformer"]["num_queries"] = 10
    config["DFINETransformer"]["num_denoising"] = 0
    config["HybridEncoder"]["depth_mult"] = 0.1
    save_checkpoint(path, build_model(config), config, {0: "a", 1: "b"})
    return path


@pytest.fixture(autouse=True)
def clean_cache():
    registry._META_CACHE.clear()
    yield
    registry._META_CACHE.clear()


# ------------------------------------------------------------------ registry pairs


def pairs():
    return [(e.id, e.task) for e in registry.registry_entries()]


def test_detect_pairs_match_downloads_registry():
    # dfine/utils/downloads.py: MODEL_REGISTRY has dfine_{s,m,l,x} only.
    assert sorted(downloads.MODEL_REGISTRY) == ["dfine_l", "dfine_m", "dfine_s", "dfine_x"]
    detect = [i for i, t in pairs() if t == "detect"]
    assert detect == ["nitid1s", "nitid1m", "nitid1l", "nitid1x"]


def test_segment_pairs_match_downloads_registry():
    # dfine/utils/downloads.py: SEGMENT_MODEL_REGISTRY has dfine_{n,s,m,l,x}.
    segment = [i for i, t in pairs() if t == "segment"]
    assert segment == ["nitid1n", "nitid1s", "nitid1m", "nitid1l", "nitid1x"]


def test_semantic_is_not_exposed():
    assert {t for _, t in pairs()} == {"detect", "segment"}


def test_d8_nano_detect_is_hidden_by_default():
    assert registry.EXPOSE_NANO_DETECT is False
    assert ("nitid1n", "detect") not in pairs()
    assert registry.find_registry_entry("nitid1n", "detect") is None
    # but the id itself is still a registry id because of segment
    assert "nitid1n" in registry.registry_ids()


def test_d8_flag_can_be_flipped(monkeypatch):
    monkeypatch.setattr(registry, "EXPOSE_NANO_DETECT", True)
    assert ("nitid1n", "detect") in pairs()
    entry = registry.find_registry_entry("nitid1n", "detect")
    assert entry is not None and entry.size == "n"
    # No official asset exists for it, so it can be listed but is never "downloaded".
    assert registry.weights_path(Path("/models"), entry) is None
    assert registry.is_downloaded(Path("/models"), entry) is False


@pytest.mark.parametrize(
    "alias", ["model1s", "dfine_s", "NITID1S", "nitid1-s", " nitid1s", "s", ""]
)
def test_ids_are_exact_not_aliases(alias):
    assert registry.find_registry_entry(alias, "detect") is None


def test_unknown_task_has_no_entry():
    assert registry.find_registry_entry("nitid1s", "semantic") is None
    assert registry.find_registry_entry("nitid1s", "pose") is None


# ------------------------------------------------------------------ downloaded flag


def test_downloaded_follows_the_cache_file(tmp_path):
    entry = registry.find_registry_entry("nitid1s", "detect")
    asset = downloads.get_model_asset("dfine_s", "default", task="detect")
    assert registry.weights_path(tmp_path, entry) == tmp_path / asset.filename
    assert registry.is_downloaded(tmp_path, entry) is False
    (tmp_path / asset.filename).write_bytes(b"x")
    assert registry.is_downloaded(tmp_path, entry) is True


def test_listing_marks_downloaded_pairs_only(tmp_path):
    seg = downloads.get_model_asset("dfine_m", "default", task="segment")
    (tmp_path / seg.filename).write_bytes(b"x")
    flags = {(m["id"], m["task"]): m["downloaded"] for m in registry.list_models(tmp_path)}
    assert flags[("nitid1m", "segment")] is True
    assert flags[("nitid1m", "detect")] is False


def test_listing_never_downloads(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("listing must not download")

    monkeypatch.setattr(downloads, "download_model", forbidden)
    monkeypatch.setattr(downloads, "urlretrieve", forbidden)
    assert registry.list_models(tmp_path / "missing-dir")


def test_missing_models_dir_lists_registry_only(tmp_path):
    models = registry.list_models(tmp_path / "nope")
    assert {m["source"] for m in models} == {"registry"}
    assert len(models) == 9


# ------------------------------------------------------------------ local scan


def test_local_checkpoints_are_listed_with_metadata(tmp_path):
    make_checkpoint(tmp_path / "my-detector.pth", "detect", "s")
    make_checkpoint(tmp_path / "my-seg.pth", "segment", "m")
    found = {m.id: m for m in registry.scan_local_models(tmp_path)}
    assert set(found) == {"my-detector", "my-seg"}
    assert (found["my-detector"].task, found["my-detector"].size) == ("detect", "s")
    assert (found["my-seg"].task, found["my-seg"].size) == ("segment", "m")
    assert found["my-seg"].path == (tmp_path / "my-seg.pth").resolve()


def test_local_entries_in_listing_have_contract_shape(tmp_path):
    make_checkpoint(tmp_path / "mine.pth")
    local = [m for m in registry.list_models(tmp_path) if m["source"] == "local"]
    assert local == [
        {
            "id": "mine",
            "task": "detect",
            "size": "s",
            "downloaded": True,
            "source": "local",
            "path": str((tmp_path / "mine.pth").resolve()),
        }
    ]


def test_corrupt_and_foreign_files_are_skipped(tmp_path, caplog):
    make_checkpoint(tmp_path / "good.pth")
    (tmp_path / "corrupt.pth").write_bytes(b"this is not a checkpoint")
    (tmp_path / "empty.pth").write_bytes(b"")
    torch.save({"hello": "world"}, tmp_path / "foreign.pth")  # loads fine, but not wrapped
    torch.save({"model": {}, "config": "nope"}, tmp_path / "badconfig.pth")
    (tmp_path / "notes.txt").write_text("hi")
    (tmp_path / "subdir.pth").mkdir()
    with caplog.at_level(logging.WARNING, logger=registry.logger.name):
        found = registry.scan_local_models(tmp_path)
    assert [m.id for m in found] == ["good"]
    assert "corrupt.pth" in caplog.text


def test_unsafe_pickle_is_rejected_and_never_executed(tmp_path):
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (Path.write_text, (marker, "executed"))

    torch.save({"model": Evil(), "config": {}, "names": {}}, tmp_path / "evil.pth")
    assert registry.scan_local_models(tmp_path) == []
    assert not marker.exists()


def test_oversized_checkpoints_are_not_opened(tmp_path, monkeypatch):
    make_checkpoint(tmp_path / "big.pth")
    monkeypatch.setattr(registry, "MAX_LOCAL_CHECKPOINT_BYTES", 10)

    def forbidden(path):
        raise AssertionError("oversized file must not be opened")

    monkeypatch.setattr(registry, "_read_checkpoint_meta", forbidden)
    assert registry.scan_local_models(tmp_path) == []


def test_scan_is_bounded_in_file_count(tmp_path, monkeypatch):
    for i in range(5):
        (tmp_path / f"f{i}.pth").write_bytes(b"x")
    monkeypatch.setattr(registry, "MAX_LOCAL_SCAN_FILES", 2)
    calls = []
    monkeypatch.setattr(registry, "_read_checkpoint_meta", lambda p: calls.append(p) or None)
    registry.scan_local_models(tmp_path)
    assert len(calls) == 2


def test_official_downloads_are_not_listed_twice(tmp_path):
    asset = downloads.get_model_asset("dfine_s", "default", task="detect")
    make_checkpoint(tmp_path / asset.filename)
    assert registry.scan_local_models(tmp_path) == []


def test_local_file_named_like_an_official_id_is_skipped(tmp_path):
    make_checkpoint(tmp_path / "nitid1s.pth")
    assert registry.scan_local_models(tmp_path) == []


def test_semantic_checkpoints_are_not_exposed(tmp_path):
    path = make_checkpoint(tmp_path / "sem.pth")
    ckpt = torch.load(path, weights_only=True)
    ckpt["task"] = ckpt["config"]["task"] = "semantic"
    torch.save(ckpt, path)
    assert registry.scan_local_models(tmp_path) == []


def test_unknown_architecture_gives_null_size(tmp_path):
    path = make_checkpoint(tmp_path / "odd.pth")
    ckpt = torch.load(path, weights_only=True)
    ckpt["config"]["HGNetv2"]["name"] = "B99"
    torch.save(ckpt, path)
    [model] = registry.scan_local_models(tmp_path)
    assert model.size is None


def test_task_metadata_mismatch_is_invalid(tmp_path):
    path = make_checkpoint(tmp_path / "mixed.pth", task="detect")
    ckpt = torch.load(path, weights_only=True)
    ckpt["task"] = "segment"
    torch.save(ckpt, path)
    assert registry.scan_local_models(tmp_path) == []


def test_legacy_checkpoint_without_task_defaults_to_detect(tmp_path):
    path = make_checkpoint(tmp_path / "legacy.pth")
    ckpt = torch.load(path, weights_only=True)
    ckpt.pop("task")
    ckpt["config"].pop("task")
    torch.save(ckpt, path)
    [model] = registry.scan_local_models(tmp_path)
    assert model.task == "detect"


def test_scan_results_are_cached_until_the_file_changes(tmp_path, monkeypatch):
    path = make_checkpoint(tmp_path / "cached.pth")
    real = registry._read_checkpoint_meta
    calls = []
    monkeypatch.setattr(registry, "_read_checkpoint_meta", lambda p: calls.append(p) or real(p))
    registry.scan_local_models(tmp_path)
    registry.scan_local_models(tmp_path)
    assert len(calls) == 1
    make_checkpoint(path, task="segment")  # new mtime/size -> re-read
    [model] = registry.scan_local_models(tmp_path)
    assert len(calls) == 2
    assert model.task == "segment"


def test_invalid_files_are_cached_too(tmp_path, monkeypatch):
    (tmp_path / "bad.pth").write_bytes(b"junk")
    real = registry._read_checkpoint_meta
    calls = []
    monkeypatch.setattr(registry, "_read_checkpoint_meta", lambda p: calls.append(p) or real(p))
    registry.scan_local_models(tmp_path)
    registry.scan_local_models(tmp_path)
    assert len(calls) == 1


def test_local_model_ids_do_not_open_files(tmp_path, monkeypatch):
    (tmp_path / "a.pth").write_bytes(b"junk")
    (tmp_path / "b.txt").write_text("x")
    monkeypatch.setattr(torch, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert registry.local_model_ids(tmp_path) == frozenset({"a"})
    assert registry.local_model_ids(tmp_path / "missing") == frozenset()


# ------------------------------------------------------------------ download


def fake_download_factory(calls):
    def fake(model, *, task="detect", weights="default", output=None, force=False):
        asset = downloads.get_model_asset(model, weights, task=task)
        out = Path(output) / asset.filename
        out.write_bytes(b"wrapped")
        calls.append({"model": model, "task": task, "weights": weights, "output": Path(output)})
        return out

    return fake


def test_download_places_the_file_atomically(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(downloads, "download_model", fake_download_factory(calls))
    entry = registry.find_registry_entry("nitid1l", "segment")
    seen: list[float] = []
    models_dir = tmp_path / "models"  # does not exist yet
    final = registry.download_registry_model(entry, models_dir, seen.append)
    assert final == registry.weights_path(models_dir, entry)
    assert final.read_bytes() == b"wrapped"
    assert seen == [0.0, 1.0]
    assert [p.name for p in models_dir.iterdir()] == [final.name]  # staging dir is gone
    # The downloader only ever got registry-derived arguments.
    assert calls[0]["model"] == "dfine_l"
    assert calls[0]["task"] == "segment"
    assert calls[0]["output"].parent == models_dir


def test_download_is_skipped_when_already_present(tmp_path, monkeypatch):
    entry = registry.find_registry_entry("nitid1s", "detect")
    final = registry.weights_path(tmp_path, entry)
    final.write_bytes(b"old")
    monkeypatch.setattr(downloads, "download_model", lambda *a, **k: pytest.fail("called"))
    seen: list[float] = []
    registry.download_registry_model(entry, tmp_path, seen.append)
    assert final.read_bytes() == b"old"
    assert seen == [0.0, 1.0]


def test_failed_download_cleans_staging_and_leaves_no_checkpoint(tmp_path, monkeypatch):
    def fail(model, *, task="detect", weights="default", output=None, force=False):
        (Path(output) / "half.pth").write_bytes(b"partial")
        raise URLError("offline")

    monkeypatch.setattr(downloads, "download_model", fail)
    entry = registry.find_registry_entry("nitid1m", "detect")
    with pytest.raises(URLError):
        registry.download_registry_model(entry, tmp_path, lambda p: None)
    assert list(tmp_path.iterdir()) == []
    assert registry.is_downloaded(tmp_path, entry) is False


def test_download_without_registered_asset_is_a_value_error(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "EXPOSE_NANO_DETECT", True)
    entry = registry.find_registry_entry("nitid1n", "detect")
    with pytest.raises(ValueError):
        registry.download_registry_model(entry, tmp_path, lambda p: None)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (URLError("dns failure for https://secret.example/x"), "Network error"),
        (HTTPError("https://secret.example/x", 404, "nf", None, None), "Network error"),
        (TimeoutError("t"), "Network error"),
        (PermissionError("/home/jorge/models denied"), "Could not write"),
        (RuntimeError("Checksum mismatch for /tmp/x: expected a, got b"), "Checksum"),
        (ValueError("Unknown model"), "No official weights"),
        (KeyError("/etc/passwd"), "Download failed (KeyError)"),
    ],
)
def test_error_messages_do_not_leak(exc, expected):
    message = registry.describe_download_error(exc)
    assert expected in message
    for secret in ("secret.example", "/home", "/tmp", "/etc"):
        assert secret not in message
