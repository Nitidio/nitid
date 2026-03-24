"""Unit tests for DFINETrainer helper methods (no D-FINE submodule needed)."""
import torch
import torch.nn as nn
import pytest

from dfine.trainer import DFINETrainer


@pytest.fixture
def tiny_model():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(4, 4)
        def forward(self, x):
            return self.l(x)
    return M()


@pytest.fixture
def trainer(tiny_model):
    return DFINETrainer(model=tiny_model, cfg={}, device="cpu", names={})


def test_build_optimizer_adamw(trainer):
    opt = trainer._build_optimizer("AdamW", lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_build_optimizer_sgd(trainer):
    opt = trainer._build_optimizer("SGD", lr=5e-3)
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-3)


def test_build_optimizer_invalid(trainer):
    with pytest.raises(ValueError, match="Unknown optimizer"):
        trainer._build_optimizer("LAMB", lr=1e-3)


def test_build_scheduler(trainer):
    opt = trainer._build_optimizer("AdamW", lr=1e-3)
    scheduler = trainer._build_scheduler(opt, epochs=10, lrf=0.1)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)


def test_scheduler_end_lr(trainer):
    """After `epochs` steps the LR should reach lr0 * lrf."""
    lr0, lrf, epochs = 1e-2, 0.01, 5
    opt = trainer._build_optimizer("AdamW", lr=lr0)
    scheduler = trainer._build_scheduler(opt, epochs=epochs, lrf=lrf)
    for _ in range(epochs):
        scheduler.step()
    final_lr = opt.param_groups[0]["lr"]
    assert final_lr == pytest.approx(lr0 * lrf, rel=1e-3)
