"""Integration tests for training and validation (Phase 3)."""
import pytest


@pytest.mark.xfail(reason="Phase 3: DFINETrainer._build_dataloader not yet implemented", strict=True)
def test_train_runs(tiny_checkpoint, tmp_path):
    from dfine import DFINE
    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    model.train(data="configs/datasets/example_custom.yml", epochs=1,
                project=str(tmp_path), name="test")


@pytest.mark.xfail(reason="Phase 3: DFINEValidator not yet implemented", strict=True)
def test_val_runs(tiny_checkpoint):
    from dfine import DFINE
    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    model.val(data="configs/datasets/example_custom.yml")
