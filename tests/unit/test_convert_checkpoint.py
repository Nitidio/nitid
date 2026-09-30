"""Tests for the maintainer checkpoint converter's command-line parsing."""

from __future__ import annotations

import pytest

import nitid.convert_checkpoint as convert_checkpoint
from dfine.nn.configs import make_model_config


@pytest.fixture()
def captured(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(convert_checkpoint, "convert", lambda *args: calls.append(args))
    return calls


def _run(model: str) -> None:
    convert_checkpoint.main(
        [
            "--weights",
            "raw.pth",
            "--model",
            model,
            "--task",
            "detect",
            "--names",
            "names.yml",
            "--output",
            "out.pth",
        ]
    )


@pytest.mark.parametrize(
    ("model", "backend"),
    [
        ("model1l", "dfine_l"),
        ("MODEL1S", "dfine_s"),
        ("model1n", "dfine_n"),
        ("dfine_m", "dfine_m"),
    ],
)
def test_model_accepts_public_and_dfine_names(captured, model, backend):
    _run(model)

    assert len(captured) == 1
    weights, config, names, output = captured[0]
    assert (weights, names, output) == ("raw.pth", "names.yml", "out.pth")
    assert config == make_model_config(backend, task="detect")


@pytest.mark.parametrize("model", ["model2l", "model1q", "nitid1l", "yolo11l"])
def test_model_rejects_unknown_names(captured, capsys, model):
    with pytest.raises(SystemExit) as exc:
        _run(model)

    assert exc.value.code == 2
    assert "--model" in capsys.readouterr().err
    assert captured == []
