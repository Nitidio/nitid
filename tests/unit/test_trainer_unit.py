"""Unit tests for DFINETrainer helper methods (no D-FINE submodule needed)."""
import pytest
import torch
import torch.nn as nn

from dfine.trainer import DFINETrainer, ModelEMA


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


# ── ModelEMA ─────────────────────────────────────────────────────────────────

def test_ema_initial_weights_match_model(tiny_model):
    ema = ModelEMA(tiny_model, decay=0.999)
    for p_model, p_ema in zip(tiny_model.parameters(), ema.ema.parameters()):
        assert torch.allclose(p_model, p_ema)


def test_ema_no_grad(tiny_model):
    ema = ModelEMA(tiny_model, decay=0.999)
    for p in ema.ema.parameters():
        assert not p.requires_grad


def test_ema_update_interpolates(tiny_model):
    """One update with decay=0 should copy float weights exactly."""
    ema = ModelEMA(tiny_model, decay=0.0)
    with torch.no_grad():
        for p in tiny_model.parameters():
            p.fill_(99.0)
    ema.update(tiny_model)
    for p_ema in ema.ema.parameters():
        if p_ema.is_floating_point():
            assert torch.allclose(p_ema, torch.full_like(p_ema, 99.0))


def test_ema_update_integer_params():
    """Integer parameters are copied directly, not blended."""
    class ModelWithIntParam(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter(
                "idx", nn.Parameter(torch.tensor([1, 2, 3], dtype=torch.long),
                                    requires_grad=False)
            )
        def forward(self, x):
            return x

    m = ModelWithIntParam()
    ema = ModelEMA(m, decay=0.999)
    with torch.no_grad():
        m.idx.data.fill_(7)
    ema.update(m)
    assert (ema.ema.idx == 7).all()


def test_ema_update_decay(tiny_model):
    """With decay=0.5 the EMA weight should be 0.5*old + 0.5*new."""
    ema = ModelEMA(tiny_model, decay=0.5)
    initial = {n: p.clone() for n, p in ema.ema.named_parameters()}
    with torch.no_grad():
        for p in tiny_model.parameters():
            p.fill_(10.0)
    ema.update(tiny_model)
    for name, p_ema in ema.ema.named_parameters():
        expected = 0.5 * initial[name] + 0.5 * 10.0
        assert torch.allclose(p_ema, expected)


def test_ema_buffers_copied(tiny_model):
    """Buffers (e.g. BatchNorm running stats) should be copied, not blended."""
    class ModelWithBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(4)
        def forward(self, x):
            return self.bn(x)

    m = ModelWithBN()
    ema = ModelEMA(m, decay=0.999)
    # Simulate a forward pass that updates running_mean
    with torch.no_grad():
        m.bn.running_mean.fill_(7.0)
    ema.update(m)
    assert torch.allclose(ema.ema.bn.running_mean, torch.full((4,), 7.0))


def test_build_ema_returns_model_ema(trainer):
    ema = trainer._build_ema(decay=0.9999)
    assert isinstance(ema, ModelEMA)
    assert ema.decay == pytest.approx(0.9999)
