"""Unit tests for model download helpers."""

from pathlib import Path

import pytest

from dfine.utils import downloads


def test_model_registry_contains_architectures_and_weight_variants():
    assert downloads.list_models() == ["dfine_l", "dfine_m", "dfine_n", "dfine_s", "dfine_x"]
    assert downloads.list_weights("dfine_s") == ["coco", "obj2coco"]
    assert downloads.list_weights("dfine_n") == ["coco"]
    assert downloads.list_models("segment") == [
        "dfine_l",
        "dfine_m",
        "dfine_n",
        "dfine_s",
        "dfine_x",
    ]
    assert downloads.list_weights("dfine_s", task="segment") == ["coco"]
    assert downloads.list_models("semantic") == [
        "dfine_l",
        "dfine_m",
        "dfine_n",
        "dfine_s",
        "dfine_x",
    ]
    assert downloads.list_weights("dfine_s", task="semantic") == ["coco"]


def test_nano_detect_asset_is_coco_only_and_carries_a_notice():
    # Decision D8: nitid1n detection is registered, with COCO weights only.
    asset = downloads.get_model_asset("dfine_n")  # "default" falls back to coco
    assert asset.weights == "coco"
    assert asset.url.endswith("/dfine_n_coco.pth")
    assert asset.filename == "dfine_n_coco_wrapped.pth"
    assert asset.sha256 == "41973938d2784d38a9836990d805b8392855ebf611aba55f0f7add90e110744c"
    assert asset.notice is not None and "COCO" in asset.notice
    # Same rule as segmentation: the default variant falls back to coco when a size has
    # no obj2coco weights, so the nano model is never refused for lack of one.
    assert downloads.get_model_asset("dfine_n", "obj2coco").weights == "coco"


def test_larger_detect_models_keep_obj2coco_as_default_and_have_no_notice():
    asset = downloads.get_model_asset("dfine_s")
    assert asset.weights == "obj2coco"
    assert asset.notice is None


def test_segment_asset_uses_coco_mask_checkpoint():
    asset = downloads.get_model_asset("dfine_s", task="segment")
    assert asset.task == "segment"
    assert asset.weights == "coco"
    assert asset.url.endswith("/dfine_seg_s_coco.pt")
    assert asset.filename == "dfine_seg_s_coco_wrapped.pth"


def test_segment_nano_aliases_resolve():
    assert downloads.get_model_asset("n", task="segment").model == "dfine_n"
    assert downloads.get_model_asset("d-fine-n", task="segment").model == "dfine_n"


def test_semantic_asset_uses_instance_checkpoint_as_initialization():
    asset = downloads.get_model_asset("dfine_n", task="semantic")
    assert asset.task == "semantic"
    assert asset.weights == "coco"
    assert asset.url.endswith("/dfine_seg_n_coco.pt")
    assert asset.filename == "dfine_semantic_n_coco_init_wrapped.pth"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("dfine_s", "dfine_s"),
        ("s", "dfine_s"),
        ("D-FINE-M", "dfine_m"),
        ("x", "dfine_x"),
    ],
)
def test_get_model_asset_accepts_model_aliases(name, expected):
    asset = downloads.get_model_asset(name)
    assert asset.model == expected
    assert asset.weights == "obj2coco"


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ("default", "obj2coco"),
        ("obj2coco", "obj2coco"),
        ("objects365-coco", "obj2coco"),
        ("coco", "coco"),
        ("coco-only", "coco"),
    ],
)
def test_get_model_asset_resolves_weight_variants(weights, expected):
    assert downloads.get_model_asset("dfine_s", weights=weights).weights == expected


def test_get_model_asset_rejects_unknown_model_or_weights():
    with pytest.raises(ValueError, match="Unknown model"):
        downloads.get_model_asset("dfine_tiny")
    with pytest.raises(ValueError, match="Unknown weights"):
        downloads.get_model_asset("dfine_s", weights="imagenet")
    with pytest.raises(TypeError, match="weights must be a string"):
        downloads.get_model_asset("dfine_s", weights=True)


def test_resolve_output_path_uses_variant_specific_filename():
    asset = downloads.get_model_asset("dfine_s")
    assert downloads.resolve_output_path(asset) == Path("dfine_s_obj2coco_wrapped.pth")
    assert downloads.resolve_output_path(asset, "models") == Path(
        "models/dfine_s_obj2coco_wrapped.pth"
    )
    assert downloads.resolve_output_path(asset, "custom.pth") == Path("custom.pth")


@pytest.mark.parametrize(
    ("weights", "raw_suffix", "wrapped_name"),
    [
        (
            "default",
            "dfine_s_obj2coco.pth",
            "dfine_s_obj2coco_wrapped.pth",
        ),
        (
            "coco",
            "dfine_s_coco.pth",
            "dfine_s_coco_wrapped.pth",
        ),
    ],
)
def test_download_model_converts_selected_checkpoint(
    monkeypatch, tmp_path, weights, raw_suffix, wrapped_name
):
    calls = {}

    def fake_urlretrieve(url, filename):
        calls["url"] = url
        Path(filename).write_bytes(b"raw")
        return filename, None

    def fake_convert(weights, config, names_file, output):
        calls["weights"] = weights
        calls["config"] = config
        calls["names_file"] = names_file
        Path(output).write_bytes(b"wrapped")

    monkeypatch.setattr(downloads, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(downloads, "convert_checkpoint", fake_convert)

    out = downloads.download_model("dfine_s", weights=weights, output=tmp_path)

    assert out == tmp_path / wrapped_name
    assert out.read_bytes() == b"wrapped"
    assert calls["url"].endswith(raw_suffix)
    assert calls["weights"].endswith(raw_suffix)
    assert calls["config"]["task"] == "detect"
    assert calls["config"]["num_classes"] == 80
    assert calls["config"]["HGNetv2"]["name"] == "B0"
    assert Path(calls["names_file"]).name == "coco_names.yml"


def test_bundled_coco_names_match_dataset_config():
    import yaml

    from dfine.data import COCO_NAMES

    bundled = yaml.safe_load(COCO_NAMES.read_text())
    repo_config = Path(__file__).parents[2] / "configs" / "datasets" / "coco.yml"
    dataset = yaml.safe_load(repo_config.read_text())

    assert bundled["nc"] == 80
    assert len(bundled["names"]) == 80
    assert bundled["names"] == dataset["names"]


def test_download_model_skips_existing_variant_output(monkeypatch, tmp_path):
    out = tmp_path / "dfine_s_obj2coco_wrapped.pth"
    out.write_bytes(b"existing")

    def fail_urlretrieve(url, filename):
        raise AssertionError("download should not run")

    monkeypatch.setattr(downloads, "urlretrieve", fail_urlretrieve)

    assert downloads.download_model("dfine_s", output=tmp_path) == out
    assert out.read_bytes() == b"existing"


def test_download_segment_model_embeds_segment_config(monkeypatch, tmp_path):
    calls = {}

    def fake_urlretrieve(url, filename):
        calls["url"] = url
        Path(filename).write_bytes(b"raw")
        return filename, None

    def fake_convert(weights, config, names_file, output):
        calls["config"] = config
        Path(output).write_bytes(b"wrapped")

    monkeypatch.setattr(downloads, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(downloads, "convert_checkpoint", fake_convert)

    output = downloads.download_model("dfine_s", task="segment", output=tmp_path)

    assert output.name == "dfine_seg_s_coco_wrapped.pth"
    assert calls["url"].endswith("/dfine_seg_s_coco.pt")
    assert calls["config"]["task"] == "segment"
    assert "masks" in calls["config"]["DFINECriterion"]["losses"]


def test_download_semantic_model_transfers_instance_fuser(monkeypatch, tmp_path):
    import torch

    from dfine.nn.native_build import build_native_model
    from dfine.utils.checkpoint import load_checkpoint_state

    instance_model = build_native_model("dfine_n", num_classes=80, task="segment")

    def fake_urlretrieve(url, filename):
        del url
        torch.save({"model": instance_model.state_dict()}, filename)
        return filename, None

    monkeypatch.setattr(downloads, "urlretrieve", fake_urlretrieve)
    output = downloads.download_model("dfine_n", task="semantic", output=tmp_path)
    checkpoint = load_checkpoint_state(output)

    assert output.name == "dfine_semantic_n_coco_init_wrapped.pth"
    assert checkpoint["task"] == "semantic"
    assert checkpoint["config"]["task"] == "semantic"
    assert checkpoint["config"]["num_classes"] == 80
    assert all(
        torch.equal(checkpoint["model"][name], value)
        for name, value in instance_model.state_dict().items()
        if name.startswith("decoder.mask_decoder.")
    )


def test_download_nano_detect_prints_the_coco_only_notice(monkeypatch, tmp_path, capsys):
    import hashlib

    payload = b"raw"
    asset = downloads.get_model_asset("dfine_n")
    monkeypatch.setitem(
        downloads.MODEL_REGISTRY["dfine_n"],
        "coco",
        downloads.ModelAsset(
            model=asset.model,
            task=asset.task,
            weights=asset.weights,
            url=asset.url,
            filename=asset.filename,
            sha256=hashlib.sha256(payload).hexdigest(),
            notice=asset.notice,
        ),
    )

    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(payload)
        return filename, None

    def fake_convert(weights, config, names_file, output):
        Path(output).write_bytes(b"wrapped")

    monkeypatch.setattr(downloads, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(downloads, "convert_checkpoint", fake_convert)

    out = downloads.download_model("dfine_n", output=tmp_path)

    assert out == tmp_path / "dfine_n_coco_wrapped.pth"
    assert "Notice: nitid1n detection weights are trained on COCO train2017 only" in (
        capsys.readouterr().out
    )


def test_download_nano_detect_rejects_a_corrupted_file(monkeypatch, tmp_path):
    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(b"corrupted")
        return filename, None

    monkeypatch.setattr(downloads, "urlretrieve", fake_urlretrieve)

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        downloads.download_model("dfine_n", output=tmp_path)


def test_segment_downloads_are_pinned_and_match_models_lock():
    import json

    lock = json.loads(
        (Path(__file__).resolve().parents[2] / "studio" / "models.lock.json").read_text()
    )
    entries = {e["nitid_model"]: e for e in lock["entries"] if e["task"] == "segment"}
    assert set(entries) == {f"nitid1{s}" for s in "nsmlx"}
    for size in "nsmlx":
        asset = downloads.get_model_asset(f"dfine_{size}", task="segment")
        assert "/resolve/main/" not in asset.url
        assert asset.url == entries[f"nitid1{size}"]["url"]
        assert entries[f"nitid1{size}"]["revision"] in asset.url
