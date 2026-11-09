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
        # Canonical public names.
        ("nitid1n", "nitid1n", 1, "n"),
        ("nitid1s", "nitid1s", 1, "s"),
        ("NITID-1-M", "nitid1m", 1, "m"),
        ("nitid_1_l", "nitid1l", 1, "l"),
        (" nitid1x ", "nitid1x", 1, "x"),
        # Hyphenated spelling used in prose and docs.
        ("nitid1-s", "nitid1s", 1, "s"),
        # Legacy names (<= 0.9.4) resolve to the canonical name.
        ("model1s", "nitid1s", 1, "s"),
        ("MODEL-1-M", "nitid1m", 1, "m"),
        # Technical alias: architecture name, current model generation.
        ("d-fine-n", "nitid1n", 1, "n"),
        ("D-FINE-S", "nitid1s", 1, "s"),
        ("dfine_m", "nitid1m", 1, "m"),
        ("dfinel", "nitid1l", 1, "l"),
        ("d-fine-x", "nitid1x", 1, "x"),
    ],
)
def test_parse_nitid_model_name(name: str, normalized: str, version: int, size: str) -> None:
    from dfine.nitid import parse_nitid_model_name

    spec = parse_nitid_model_name(name)

    assert spec.name == normalized
    assert spec.version == version
    assert spec.size == size


@pytest.mark.parametrize(
    "name",
    [
        "dfine",
        "d-fine",
        "d-fine-tiny",
        "dfine2m",
        "nitid",
        "nitid1",
        "nitid1tiny",
        "nitid2s",
        "model",
        "model1",
        "model2s",
        "yolo11l",
    ],
)
def test_parse_nitid_model_name_rejects_invalid_names(name: str) -> None:
    from dfine.nitid import parse_nitid_model_name

    with pytest.raises(ValueError):
        parse_nitid_model_name(name)


def test_parse_nitid_model_name_rejects_non_strings() -> None:
    from dfine.nitid import parse_nitid_model_name

    with pytest.raises(TypeError):
        parse_nitid_model_name(Path("nitid1s"))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("nitid1s", True),
        ("NITID-1-M", True),
        ("model1l", True),
        ("d-fine-x", True),
        ("dfine_s", False),  # backend name: passed through unchanged by the CLI
        ("runs/best.pth", False),
        ("yolo11l", False),
        (Path("nitid1s"), False),
    ],
)
def test_is_public_model_name(name: object, expected: bool) -> None:
    from dfine.nitid import is_public_model_name

    assert is_public_model_name(name) is expected  # type: ignore[arg-type]


@pytest.mark.parametrize("size", ["n", "s", "m", "l", "x"])
def test_every_spelling_of_a_size_resolves_to_the_same_backend(size: str) -> None:
    from dfine.nitid import resolve_nitid_backend_model

    spellings = [f"nitid1{size}", f"nitid1-{size}", f"model1{size}", f"d-fine-{size}"]
    # ``detect`` has no official N checkpoint; ``segment`` covers every size.
    backends = {resolve_nitid_backend_model(name, task="segment") for name in spellings}

    assert backends == {f"dfine_{size}"}


@pytest.mark.parametrize(
    ("task", "public_name", "canonical", "expected_backend"),
    [
        ("detect", "nitid1s", "nitid1s", "dfine_s"),
        ("segment", "nitid1n", "nitid1n", "dfine_n"),
        ("semantic", "nitid1m", "nitid1m", "dfine_m"),
        ("detect", "model1l", "nitid1l", "dfine_l"),
        ("detect", "d-fine-x", "nitid1x", "dfine_x"),
    ],
)
def test_nitid_resolves_existing_tasks_to_backend_models(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    public_name: str,
    canonical: str,
    expected_backend: str,
) -> None:
    from dfine import NITID

    download_calls = _patch_checkpoint_loading(monkeypatch)

    model = NITID(public_name, task=task, device="cpu", verbose=False)

    assert model.nitid_model == canonical
    assert model.nitid_version == 1
    assert model.size == canonical[-1]
    assert download_calls[0][0] == expected_backend
    assert download_calls[0][1] == task


def test_nitid_import_package_exposes_public_api() -> None:
    from nitid import DFINE, NITID, parse_nitid_model_name

    assert DFINE is not None
    assert NITID is not None
    assert parse_nitid_model_name("nitid1s").size == "s"


@pytest.mark.parametrize("name", ["nitid1n", "model1n", "d-fine-n"])
def test_nitid_detect_n_rejected_until_detection_weights_exist(name: str) -> None:
    from dfine import NITID

    # The message always uses the canonical name, whatever spelling was given.
    with pytest.raises(ValueError, match="nitid1n.*detect"):
        NITID(name, task="detect", device="cpu", verbose=False)


@pytest.mark.parametrize("size", ["n", "s", "m", "l", "x"])
def test_infer_nitid_model_spec_from_checkpoint_config(size: str) -> None:
    from dfine.nitid import infer_nitid_model_spec
    from dfine.nn.configs import make_model_config

    config = make_model_config(f"dfine_{size}", task="segment")
    # Training-time overrides must not change the inferred size.
    config["DFINETransformer"]["num_layers"] = 1
    config["HybridEncoder"]["depth_mult"] = 0.1

    spec = infer_nitid_model_spec(config)

    assert spec is not None
    assert spec.name == f"nitid1{size}"


@pytest.mark.parametrize(
    "config",
    [
        {"task": "detect"},
        {"HGNetv2": {"name": "B0"}, "HybridEncoder": {"hidden_dim": 192}},
        {"HGNetv2": "B0", "HybridEncoder": {"hidden_dim": 256}},
    ],
)
def test_infer_nitid_model_spec_returns_none_for_unknown_configs(config: dict) -> None:
    from dfine.nitid import infer_nitid_model_spec

    assert infer_nitid_model_spec(config) is None
