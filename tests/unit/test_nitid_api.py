from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


class _DummyModel:
    def eval(self) -> None:
        return None

    def parameters(self) -> list[Any]:
        return []


def _patch_checkpoint_loading(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str, Any]]:
    from dfine.utils import checkpoint, downloads

    download_calls: list[tuple[str, str, str, Any]] = []

    def fake_download_model(
        model: str,
        *,
        task: str = "detect",
        weights: str = "default",
        output: str | Path | None = None,
        force: bool = False,
    ) -> Path:
        del force
        download_calls.append((model, task, weights, output))
        asset = downloads.get_model_asset(model, weights=weights, task=task)
        out_path = downloads.resolve_output_path(asset, output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"dummy")
        return out_path

    def fake_load_checkpoint(
        path: str | Path, device: str = "cpu"
    ) -> tuple[_DummyModel, dict, dict]:
        del device
        path_text = str(path)
        task = "detect"
        if "dfine_seg_" in path_text:
            task = "segment"
        if "dfine_semantic_" in path_text:
            task = "semantic"
        return _DummyModel(), {"task": task}, {}

    monkeypatch.setattr(downloads, "download_model", fake_download_model)
    monkeypatch.setattr(checkpoint, "load_checkpoint", fake_load_checkpoint)
    return download_calls


@pytest.mark.parametrize(
    ("name", "normalized", "version", "size"),
    [
        ("nitid1n", "nitid1n", 1, "n"),
        ("nitid1s", "nitid1s", 1, "s"),
        ("NITID-1-M", "nitid1m", 1, "m"),
        ("nitid_1_l", "nitid1l", 1, "l"),
        (" nitid1x ", "nitid1x", 1, "x"),
    ],
)
def test_parse_nitid_model_name(name: str, normalized: str, version: int, size: str) -> None:
    from dfine.nitid import parse_nitid_model_name

    spec = parse_nitid_model_name(name)

    assert spec.name == normalized
    assert spec.version == version
    assert spec.size == size


@pytest.mark.parametrize("name", ["dfine_s", "nitid", "nitid1", "nitid1tiny", "nitid2s"])
def test_parse_nitid_model_name_rejects_invalid_names(name: str) -> None:
    from dfine.nitid import parse_nitid_model_name

    with pytest.raises(ValueError):
        parse_nitid_model_name(name)


@pytest.mark.parametrize(
    ("task", "public_name", "expected_backend"),
    [
        ("detect", "nitid1s", "dfine_s"),
        ("segment", "nitid1n", "dfine_n"),
        ("semantic", "nitid1m", "dfine_m"),
    ],
)
def test_nitid_resolves_existing_tasks_to_backend_models(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    public_name: str,
    expected_backend: str,
) -> None:
    from dfine import NITID

    download_calls = _patch_checkpoint_loading(monkeypatch)

    model = NITID(public_name, task=task, device="cpu", verbose=False)

    assert model.nitid_model == public_name
    assert model.size == public_name[-1]
    assert download_calls[0][0] == expected_backend
    assert download_calls[0][1] == task


def test_nitid_import_package_exposes_public_api() -> None:
    from nitid import DFINE, NITID, parse_nitid_model_name

    assert DFINE is not None
    assert NITID is not None
    assert parse_nitid_model_name("nitid1s").size == "s"


def test_nitid_detect_n_rejected_until_detection_weights_exist() -> None:
    from dfine import NITID

    with pytest.raises(ValueError, match="nitid1n.*detect"):
        NITID("nitid1n", task="detect", device="cpu", verbose=False)


@pytest.mark.parametrize("task", ["pose", "obb"])
def test_nitid_rejects_removed_tasks(task: str) -> None:
    from dfine import NITID

    with pytest.raises(ValueError, match="Unsupported task"):
        NITID("nitid1s", task=task, device="cpu", verbose=False)
