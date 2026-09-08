"""Unit tests for model download helpers."""

from pathlib import Path

import pytest

from dfine.utils import downloads


def test_model_registry_contains_architectures_and_weight_variants():
    assert downloads.list_models() == ["dfine_l", "dfine_m", "dfine_s", "dfine_x"]
    assert downloads.list_weights("dfine_s") == ["coco", "obj2coco"]
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


def test_pose_asset_uses_official_detrpose_checkpoint():
    asset = downloads.get_model_asset("detrpose_n", task="pose", weights="coco")
    assert asset.task == "pose"
    assert asset.weights == "coco"
    assert asset.url.endswith("/model_weights/detrpose_hgnetv2_n.pth")
    assert "SebastianJanampa/DETRPose" in asset.url
    assert asset.filename == "detrpose_n_coco_wrapped.pth"


def test_obb_registry_contains_rio_variants_and_weight_aliases():
    assert downloads.list_models(task="obb") == [
        "nitid1l",
        "nitid1m",
        "nitid1n",
        "nitid1s",
        "nitid1x",
    ]
    assert downloads.list_weights("nitid1s", task="obb") == ["diorr", "dota_1_ss"]
    assert downloads.list_weights("nitid1m", task="obb") == [
        "diorr",
        "dota_1_ms",
        "dota_1_ss",
    ]

    default = downloads.get_model_asset("nitid1s", task="obb")
    dota = downloads.get_model_asset("rio_s", weights="dota", task="obb")
    dota_ms = downloads.get_model_asset("rtdetrv2_obb_m", weights="dota-ms", task="obb")
    diorr = downloads.get_model_asset("nitid1s", weights="dior-r", task="obb")

    assert default.weights == "dota_1_ss"
    assert dota.weights == "dota_1_ss"
    assert dota_ms.weights == "dota_1_ms"
    assert diorr.weights == "diorr"
    assert default.filename == "nitid1s_dota_1_ss_wrapped.pth"
    assert default.url.endswith("/dota_1_ss/rtdetrv2_obb_hgnetv2_s_dota_1_ss.pth")
    assert default.sha256 == "ef0c728aaeb4d85950134431617fb576b3ad6a9ac85878088ce97c0075df2026"


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
    assert calls["names_file"].endswith("configs/datasets/coco.yml")


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


def test_download_obb_model_embeds_rio_config_and_dota_names(monkeypatch, tmp_path):
    calls = {}

    def fake_urlretrieve(url, filename):
        calls["url"] = url
        Path(filename).write_bytes(b"raw-rio")
        return filename, None

    def fake_convert(weights, config, names_file, output):
        import yaml

        calls["weights"] = weights
        calls["config"] = config
        calls["names"] = yaml.safe_load(Path(names_file).read_text(encoding="utf-8"))["names"]
        Path(output).write_bytes(b"wrapped-rio")

    monkeypatch.setattr(downloads, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(downloads, "_verify_sha256", lambda path, expected: None)
    monkeypatch.setattr(downloads, "convert_checkpoint", fake_convert)

    output = downloads.download_model("nitid1s", task="obb", output=tmp_path)

    assert output == tmp_path / "nitid1s_dota_1_ss_wrapped.pth"
    assert output.read_bytes() == b"wrapped-rio"
    assert calls["url"].endswith("/dota_1_ss/rtdetrv2_obb_hgnetv2_s_dota_1_ss.pth")
    assert calls["weights"].endswith("rtdetrv2_obb_hgnetv2_s_dota_1_ss.pth")
    assert calls["config"]["task"] == "obb"
    assert calls["config"]["model"] == "RioOBB"
    assert calls["config"]["num_classes"] == 15
    assert calls["config"]["RTDETRTransformerv2OBB"]["hidden_dim"] == 224
    assert calls["names"][0] == "plane"
    assert calls["names"][-1] == "helicopter"


def test_download_obb_model_wrapped_checkpoint_loads(monkeypatch, tmp_path):
    import torch

    from dfine.nn.native_build import build_native_model
    from dfine.utils.checkpoint import load_checkpoint

    rio_model = build_native_model("nitid1n", num_classes=15, task="obb", image_size=(1024, 1024))

    def fake_urlretrieve(url, filename):
        del url
        torch.save({"model": rio_model.state_dict(), "epoch": 7}, filename)
        return filename, None

    monkeypatch.setattr(downloads, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(downloads, "_verify_sha256", lambda path, expected: None)

    output = downloads.download_model("nitid1n", task="obb", output=tmp_path)
    model, cfg, names = load_checkpoint(output)

    assert cfg["task"] == "obb"
    assert cfg["model"] == "RioOBB"
    assert names[0] == "plane"
    assert names[14] == "helicopter"
    assert model.state_dict().keys() == rio_model.state_dict().keys()
