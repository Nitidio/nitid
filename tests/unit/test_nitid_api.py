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
        if "detrpose_" in path_text:
            task = "pose"
        if "nitid1" in path_text:
            task = "obb"
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
        ("pose", "nitid1s", "detrpose_s"),
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


def test_nitid_obb_task_builds_random_rio_model() -> None:
    from dfine import NITID
    from dfine.nn.rio import RioOBBModel
    from dfine.tasks import get_task_contract, normalize_task

    assert normalize_task("oriented_detection") == "obb"
    assert get_task_contract("obb").result_fields == ("obb",)

    model = NITID("nitid1s", task="obb", weights=None, device="cpu", verbose=False)

    assert model.task == "obb"
    assert model.nitid_model == "nitid1s"
    assert model.weights is None
    assert isinstance(model._model, RioOBBModel)
    assert len(model.names) == 15


def test_nitid_obb_default_weights_resolve_through_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dfine import NITID

    download_calls = _patch_checkpoint_loading(monkeypatch)

    model = NITID("nitid1s", task="obb", device="cpu", verbose=False)

    assert model.nitid_model == "nitid1s"
    assert model.task == "obb"
    assert model.weights == "dota_1_ss"
    assert download_calls[0][0] == "nitid1s"
    assert download_calls[0][1] == "obb"
    assert download_calls[0][2] == "dota_1_ss"


def test_nitid_obb_weight_aliases_resolve() -> None:
    from dfine.utils.downloads import get_model_asset, list_models, list_weights

    assert "nitid1s" in list_models(task="obb")
    assert list_weights("nitid1m", task="obb") == ["diorr", "dota_1_ms", "dota_1_ss"]

    default = get_model_asset("nitid1s", task="obb")
    dota = get_model_asset("rio_s", weights="dota", task="obb")
    diorr = get_model_asset("rtdetrv2_obb_s", weights="dior-r", task="obb")

    assert default.weights == "dota_1_ss"
    assert dota.weights == "dota_1_ss"
    assert diorr.weights == "diorr"
    assert default.url.endswith("/dota_1_ss/rtdetrv2_obb_hgnetv2_s_dota_1_ss.pth")
    assert default.sha256 is not None
