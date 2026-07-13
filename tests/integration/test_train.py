"""Integration tests for training and validation (Phase 3)."""

import logging
from pathlib import Path

import pytest
import torch


def test_train_runs(tiny_checkpoint, tiny_dataset, tmp_path):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    metrics = model.train(
        data=tiny_dataset,
        epochs=1,
        batch=2,
        project=str(tmp_path),
        name="test",
        verbose=False,
    )
    assert set(metrics) >= {"loss", "fitness", "mAP50", "mAP50-95", "history"}
    assert len(metrics["history"]) == 1
    assert set(metrics["history"][0]) >= {
        "epoch",
        "loss",
        "mAP50",
        "mAP50-95",
        "precision",
        "recall",
    }
    assert (tmp_path / "test" / "epoch1.pth").exists()
    assert (tmp_path / "test" / "last.pth").exists()
    assert (tmp_path / "test" / "best.pth").exists()
    assert (tmp_path / "test" / "results.csv").exists()
    assert (tmp_path / "test" / "results.png").exists()
    assert (tmp_path / "test" / "confusion_matrix.png").exists()
    assert (tmp_path / "test" / "pr_curve.png").exists()
    assert (tmp_path / "test" / "f1_curve.png").exists()


def test_val_runs(tiny_checkpoint, tiny_dataset):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    metrics = model.val(
        data=tiny_dataset, batch=2, project="runs/pytest_val", name="exp", verbose=False
    )
    assert set(metrics) >= {"mAP50", "mAP50-95", "AR1", "AR100", "precision", "recall", "f1"}
    assert (Path("runs/pytest_val") / "exp" / "confusion_matrix.png").exists()
    assert (Path("runs/pytest_val") / "exp" / "pr_curve.png").exists()
    assert (Path("runs/pytest_val") / "exp" / "f1_curve.png").exists()


def test_val_return_format_includes_per_class_and_summary_fields(tiny_checkpoint, tiny_dataset):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    metrics = model.val(data=tiny_dataset, batch=2, verbose=False)

    assert set(metrics) >= {
        "mAP50",
        "mAP50-95",
        "AR1",
        "AR100",
        "precision",
        "recall",
        "f1",
        "fitness",
        "best_conf",
        "images",
        "instances",
        "per_class",
    }
    assert metrics["images"] == 2
    assert metrics["instances"] == 2
    assert isinstance(metrics["best_conf"], float)
    assert isinstance(metrics["per_class"], list)
    assert len(metrics["per_class"]) > 0

    first_row = metrics["per_class"][0]
    assert set(first_row) == {"class_id", "name", "instances", "ap50", "ap50-95"}
    assert isinstance(first_row["class_id"], int)
    assert isinstance(first_row["name"], str)
    assert isinstance(first_row["instances"], int)
    assert isinstance(first_row["ap50"], float)
    assert isinstance(first_row["ap50-95"], float)


def test_val_logs_ultralytics_style_summary(tiny_checkpoint, tiny_dataset, caplog):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    with caplog.at_level(logging.INFO, logger="dfine"):
        _ = model.val(data=tiny_dataset, batch=2, verbose=True)

    messages = [record.message for record in caplog.records]
    assert any(
        "Class" in message and "Images" in message and "mAP50-95" in message for message in messages
    )
    assert any("all" in message for message in messages)
    assert any("Class" in message and "Instances" in message for message in messages)


def test_train_with_ema(tiny_checkpoint, tiny_dataset, tmp_path):
    """EMA training completes, checkpoint loads, and EMA differs from raw model."""
    from dfine import DFINE
    from dfine.utils.checkpoint import load_checkpoint

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    metrics = model.train(
        data=tiny_dataset,
        epochs=1,
        batch=2,
        ema=True,
        ema_decay=0.5,  # aggressive decay — EMA visibly diverges from raw model
        project=str(tmp_path),
        name="ema_test",
        verbose=False,
    )
    assert "loss" in metrics
    assert len(metrics["history"]) == 1

    ckpt_path = tmp_path / "ema_test" / "epoch1.pth"
    assert ckpt_path.exists()

    # Checkpoint must reload cleanly and have finite weights
    saved_model, _, names = load_checkpoint(str(ckpt_path), device="cpu")
    assert len(names) > 0
    float_params = [p for p in saved_model.parameters() if p.is_floating_point()]
    assert all(p.isfinite().all() for p in float_params)

    # With decay=0.5 and ≥1 optimizer step, the saved EMA weights should differ
    # from the raw model weights (last optimizer state) — because EMA blends history.
    # We only check this when training produced a nonzero update.
    raw_final = {n: p for n, p in model._model.named_parameters() if p.is_floating_point()}
    saved_dict = {n: p for n, p in saved_model.named_parameters() if p.is_floating_point()}
    if metrics["loss"] > 0:
        differs = any(
            not torch.allclose(saved_dict[n], raw_final[n]) for n in raw_final if n in saved_dict
        )
        assert differs, "EMA checkpoint should differ from raw model after nonzero training"


def test_train_amp_disabled_on_cpu(tiny_checkpoint, tiny_dataset, tmp_path, caplog):
    """amp=True on a CPU device logs a warning and training still completes."""
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    with caplog.at_level(logging.WARNING, logger="dfine"):
        metrics = model.train(
            data=tiny_dataset,
            epochs=1,
            batch=2,
            amp=True,
            project=str(tmp_path),
            name="amp_cpu",
            verbose=False,
        )
    assert any("amp" in r.message.lower() for r in caplog.records)
    assert "loss" in metrics
    assert len(metrics["history"]) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_with_amp_cuda(tiny_checkpoint, tiny_dataset, tmp_path):
    """AMP training completes on CUDA without errors."""
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cuda", verbose=False)
    metrics = model.train(
        data=tiny_dataset,
        epochs=1,
        batch=2,
        amp=True,
        project=str(tmp_path),
        name="amp_cuda",
        verbose=False,
    )
    assert "loss" in metrics
    assert len(metrics["history"]) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_with_amp_and_ema_cuda(tiny_checkpoint, tiny_dataset, tmp_path):
    """AMP + EMA together complete without errors on CUDA."""
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cuda", verbose=False)
    metrics = model.train(
        data=tiny_dataset,
        epochs=1,
        batch=2,
        amp=True,
        ema=True,
        project=str(tmp_path),
        name="amp_ema_cuda",
        verbose=False,
    )
    assert "loss" in metrics
    assert len(metrics["history"]) == 1
