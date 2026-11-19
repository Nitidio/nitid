"""Unit tests for the pure parts of the ONNX endpoints (parsing, ETag matching, file names)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nitid.studio import onnx_service
from nitid.studio.onnx_service import InvalidParams, OnnxArtifact, parse_imgsz, parse_task

CONTRACT = Path(__file__).resolve().parents[3] / "studio" / "api-contract" / "openapi.json"


@pytest.mark.parametrize("value", ["detect", "segment"])
def test_parse_task_accepts_studio_tasks(value):
    assert parse_task(value) == value


@pytest.mark.parametrize("value", [None, "", "pose", "semantic", "DETECT", "detect "])
def test_parse_task_rejects_everything_else(value):
    with pytest.raises(InvalidParams):
        parse_task(value)


def test_parse_imgsz_none_means_model_default():
    assert parse_imgsz(None) is None


@pytest.mark.parametrize(
    ("value", "expected"), [("32", 32), ("640", 640), ("4096", 4096), (" 64 ", 64)]
)
def test_parse_imgsz_accepts_multiples_of_32(value, expected):
    assert parse_imgsz(value) == expected


@pytest.mark.parametrize(
    "value", ["", "0", "16", "31", "33", "4128", "8192", "-32", "+64", "6.4e2", "64.0", "٦٤", "abc"]
)
def test_parse_imgsz_rejects_out_of_range_and_non_integers(value):
    with pytest.raises(InvalidParams):
        parse_imgsz(value)


def test_imgsz_limits_match_the_contract():
    schema = next(
        p
        for p in json.loads(CONTRACT.read_text(encoding="utf-8"))["paths"]["/onnx/{model}"]["get"][
            "parameters"
        ]
        if p["name"] == "imgsz"
    )["schema"]
    assert schema["minimum"] == onnx_service.MIN_IMGSZ
    assert schema["multipleOf"] == onnx_service.IMGSZ_MULTIPLE


def test_artifact_etag_is_the_quoted_sha256_and_meta_gains_only_sha256():
    sha = "ab" * 32
    artifact = OnnxArtifact(Path("x.onnx"), sha, 3, {"task": "detect", "imgsz": 64})
    assert artifact.etag == f'"{sha}"'
    assert artifact.meta_with_sha256() == {"task": "detect", "imgsz": 64, "sha256": sha}
    assert "sha256" not in artifact.meta  # the embedded meta is not mutated


def test_file_name_depends_on_every_key_part(tmp_path, monkeypatch):
    checkpoint = tmp_path / "m.pth"
    checkpoint.write_bytes(b"a")
    name = onnx_service.OnnxService._file_name
    base = name("m", "detect", 64, checkpoint)
    assert base == name("m", "detect", 64, checkpoint)
    assert base != name("n", "detect", 64, checkpoint)
    assert base != name("m", "segment", 64, checkpoint)
    assert base != name("m", "detect", 96, checkpoint)
    monkeypatch.setattr("nitid.__version__", "9.9.9")
    assert base != name("m", "detect", 64, checkpoint)
    monkeypatch.undo()
    checkpoint.write_bytes(b"bb")  # size (and mtime) change
    assert base != name("m", "detect", 64, checkpoint)
    assert base.endswith(".onnx") and "/" not in base and ".." not in base


def test_etag_matching_follows_rfc_9110():
    pytest.importorskip("fastapi", reason="needs the 'studio' extra")
    from nitid.studio.routes_onnx import etag_matches

    etag = '"' + "a" * 64 + '"'
    assert etag_matches(etag, etag)
    assert etag_matches("W/" + etag, etag)
    assert etag_matches('"x",  ' + etag, etag)
    assert etag_matches("*", etag)
    assert not etag_matches(None, etag)
    assert not etag_matches("", etag)
    assert not etag_matches('"' + "b" * 64 + '"', etag)
    assert not etag_matches("a" * 64, etag)  # unquoted is not our validator
