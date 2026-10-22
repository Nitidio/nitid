"""Integration tests for loading self-contained checkpoints through the public NITID API."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.mark.parametrize(
    ("fixture", "task", "expected_model"),
    [
        ("tiny_checkpoint", "detect", "model1s"),
        ("tiny_segment_checkpoint", "segment", "model1s"),
        ("tiny_semantic_checkpoint", "semantic", "model1n"),
    ],
)
def test_nitid_loads_checkpoint_path(request, fixture, task, expected_model):
    from nitid import NITID

    checkpoint = request.getfixturevalue(fixture)

    model = NITID(checkpoint, task=task, device="cpu", verbose=False)

    assert model.task == task
    assert model.nitid_model == expected_model
    assert model.nitid_version == 1
    assert model.size == expected_model[-1]
    assert model.predict(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)


def test_nitid_accepts_path_objects(tiny_checkpoint):
    from nitid import NITID

    model = NITID(Path(tiny_checkpoint), device="cpu", verbose=False)

    assert model.nitid_model == "model1s"


def test_nitid_checkpoint_task_must_match(tiny_segment_checkpoint):
    from nitid import NITID

    with pytest.raises(ValueError, match="Checkpoint task is 'segment'"):
        NITID(tiny_segment_checkpoint, task="detect", device="cpu", verbose=False)


def test_nitid_checkpoint_rejects_registry_weights(tiny_checkpoint):
    from nitid import NITID

    with pytest.raises(ValueError, match="cannot be combined"):
        NITID(tiny_checkpoint, weights="obj365", device="cpu", verbose=False)


@pytest.mark.parametrize("path", ["missing/best.pth", "best.pth"])
def test_nitid_missing_checkpoint_reports_file_not_found(path):
    from nitid import NITID

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        NITID(path, device="cpu", verbose=False)


def test_cli_predict_with_checkpoint_path(tiny_checkpoint, tmp_path, capsys):
    from nitid.cli import main

    image = tmp_path / "frame.jpg"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(image)

    main(["nitid", "predict", f"model={tiny_checkpoint}", f"source={image}", "verbose=False"])

    assert "Results" in capsys.readouterr().out


def test_cli_val_with_checkpoint_path(tiny_checkpoint, tiny_dataset, tmp_path, capsys):
    from nitid.cli import main

    main(
        [
            "nitid",
            "val",
            f"model={tiny_checkpoint}",
            f"data={tiny_dataset}",
            "imgsz=64",
            "batch=2",
            "plots=False",
            f"save_dir={tmp_path / 'val'}",
            "verbose=False",
        ]
    )

    assert "mAP" in capsys.readouterr().out


def test_cli_export_with_checkpoint_path(tiny_checkpoint, tmp_path, capsys):
    from nitid.cli import main

    checkpoint = tmp_path / "best.pth"
    shutil.copy(tiny_checkpoint, checkpoint)

    main(["nitid", "export", f"model={checkpoint}", "format=torchscript", "verbose=False"])

    out = capsys.readouterr().out
    assert "Exported to" in out
    assert Path(out.split("Exported to", 1)[1].strip()).exists()
