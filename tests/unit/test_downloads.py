"""Unit tests for model download helpers."""
from pathlib import Path

import pytest

from dfine.utils import downloads


def test_model_registry_contains_coco_variants():
    assert downloads.list_models() == ["dfine_l", "dfine_m", "dfine_s", "dfine_x"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("dfine_s", "dfine_s"),
        ("s", "dfine_s"),
        ("D-FINE-M", "dfine_m"),
        ("x", "dfine_x"),
    ],
)
def test_get_model_asset_accepts_aliases(name, expected):
    assert downloads.get_model_asset(name).name == expected


def test_get_model_asset_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        downloads.get_model_asset("dfine_tiny")


def test_resolve_output_path_defaults_to_wrapped_filename():
    asset = downloads.get_model_asset("dfine_s")
    assert downloads.resolve_output_path(asset) == Path("dfine_s_wrapped.pth")


def test_resolve_output_path_accepts_directory():
    asset = downloads.get_model_asset("dfine_s")
    assert downloads.resolve_output_path(asset, "models") == Path("models/dfine_s_wrapped.pth")


def test_resolve_output_path_accepts_file():
    asset = downloads.get_model_asset("dfine_s")
    assert downloads.resolve_output_path(asset, "custom.pth") == Path("custom.pth")


def test_download_model_converts_checkpoint(monkeypatch, tmp_path):
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

    out = downloads.download_model("dfine_s", output=tmp_path)

    assert out == tmp_path / "dfine_s_wrapped.pth"
    assert out.read_bytes() == b"wrapped"
    assert calls["url"].endswith("dfine_s_coco.pth")
    assert calls["weights"].endswith("dfine_s_coco.pth")
    assert calls["config"].endswith("dfine_hgnetv2_s_coco.yml")
    assert calls["names_file"].endswith("configs/datasets/coco.yml")


def test_download_model_skips_existing_output(monkeypatch, tmp_path):
    out = tmp_path / "dfine_s_wrapped.pth"
    out.write_bytes(b"existing")

    def fail_urlretrieve(url, filename):
        raise AssertionError("download should not run")

    monkeypatch.setattr(downloads, "urlretrieve", fail_urlretrieve)

    assert downloads.download_model("dfine_s", output=tmp_path) == out
    assert out.read_bytes() == b"existing"
