"""Integration tests for training and validation (Phase 3)."""
import pytest


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
    assert "loss" in metrics
    assert (tmp_path / "test" / "epoch1.pth").exists()


def test_val_runs(tiny_checkpoint, tiny_dataset):
    from dfine import DFINE
    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    metrics = model.val(data=tiny_dataset, batch=2, verbose=False)
    assert set(metrics) >= {"mAP50", "mAP50-95"}
