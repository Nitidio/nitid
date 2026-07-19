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
                "idx", nn.Parameter(torch.tensor([1, 2, 3], dtype=torch.long), requires_grad=False)
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


def test_add_callback_registers_and_runs(trainer):
    calls = []

    trainer.epochs = 3

    def on_train_start(trainer_instance):
        calls.append((trainer_instance, trainer_instance.epochs))

    trainer.add_callback("on_train_start", on_train_start)
    trainer._run_callbacks("on_train_start")

    assert calls == [(trainer, 3)]


def test_add_callback_rejects_unknown_event(trainer):
    with pytest.raises(ValueError, match="Unknown callback event"):
        trainer.add_callback("on_batch_end", lambda *_: None)


def test_add_callbacks_supports_mapping_and_objects(trainer):
    calls = []

    trainer.run_name = "demo"

    class Recorder:
        def on_train_end(self, trainer_instance):
            calls.append(("object", trainer_instance, trainer_instance.run_name))

    trainer.add_callbacks(
        {
            "on_train_start": lambda trainer_instance: calls.append(
                ("mapping", trainer_instance, trainer_instance.run_name)
            )
        }
    )
    trainer.add_callbacks(Recorder())

    trainer._run_callbacks("on_train_start")
    trainer._run_callbacks("on_train_end")

    assert calls == [("mapping", trainer, "demo"), ("object", trainer, "demo")]


def test_add_callbacks_deduplicates_registered_callbacks(trainer):
    calls = []
    trainer.run_name = "demo"

    def on_train_start(trainer_instance):
        calls.append((trainer_instance, trainer_instance.run_name))

    trainer.add_callbacks({"on_train_start": on_train_start})
    trainer.add_callbacks({"on_train_start": on_train_start})
    trainer._run_callbacks("on_train_start")

    assert calls == [(trainer, "demo")]


def test_add_wandb_callback_supports_ultralytics_style_boolean(trainer):
    trainer._add_wandb_callback(True)

    assert len(trainer.callbacks["on_train_start"]) == 1
    callback = trainer.callbacks["on_train_start"][0].__self__
    assert callback.project == "nitid"


def test_add_wandb_callback_accepts_options(trainer):
    trainer._add_wandb_callback({"project": "detectors", "mode": "offline"})

    callback = trainer.callbacks["on_train_start"][0].__self__
    assert callback.project == "detectors"
    assert callback.mode == "offline"


def test_add_wandb_callback_rejects_invalid_value(trainer):
    with pytest.raises(TypeError, match="wandb must be a bool or a mapping"):
        trainer._add_wandb_callback("yes")


def test_add_mlflow_callback_supports_boolean_and_options(trainer):
    trainer._add_mlflow_callback(True)
    default_callback = trainer.callbacks["on_train_start"][0].__self__
    assert default_callback.tracking_uri is None

    configured_trainer = DFINETrainer(model=trainer.model, cfg={}, device="cpu", names={})
    configured_trainer._add_mlflow_callback(
        {"tracking_uri": "runs/custom-mlflow", "experiment_name": "detectors"}
    )
    configured_callback = configured_trainer.callbacks["on_train_start"][0].__self__
    assert configured_callback.tracking_uri == "runs/custom-mlflow"
    assert configured_callback.experiment_name == "detectors"


def test_add_mlflow_callback_rejects_invalid_value(trainer):
    with pytest.raises(TypeError, match="mlflow must be a bool or a mapping"):
        trainer._add_mlflow_callback("yes")


def test_handle_train_error_notifies_callbacks_and_preserves_error(trainer):
    error = RuntimeError("training failed")
    observed = []

    def on_train_error(trainer_instance):
        observed.append(trainer_instance.error)

    trainer.add_callback("on_train_error", on_train_error)
    trainer._handle_train_error(error)

    assert trainer.error is error
    assert observed == [error]


def test_tracking_state_is_serialized(trainer):
    optimizer = trainer._build_optimizer("AdamW", lr=1e-3)
    scheduler = trainer._build_scheduler(optimizer, epochs=2, lrf=0.1)
    trainer.tracking_state = {"wandb": {"run_id": "abc123"}}

    state = trainer._serialize_training_state(
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        ema_model=None,
        history=[],
        best_fitness=0.0,
        train_args={},
    )

    assert state["tracking_state"] == {"wandb": {"run_id": "abc123"}}


def test_public_train_runs_error_callbacks_and_reraises_original(monkeypatch, tiny_model):
    from dfine.model import DFINE

    original_error = RuntimeError("bad batch")
    handled = []

    class FailingTrainer:
        def __init__(self, **kwargs):
            pass

        def train(self, **kwargs):
            raise original_error

        def _handle_train_error(self, error):
            handled.append(error)

    monkeypatch.setattr("dfine.trainer.DFINETrainer", FailingTrainer)
    model = object.__new__(DFINE)
    model._model = tiny_model
    model._cfg = {}
    model._device_str = "cpu"
    model._names = {}
    model._callbacks = {}

    with pytest.raises(RuntimeError) as caught:
        model.train(data="dataset.yaml")

    assert caught.value is original_error
    assert handled == [original_error]
