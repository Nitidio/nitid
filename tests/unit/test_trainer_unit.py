"""Unit tests for DFINETrainer helper methods (no D-FINE submodule needed)."""

import itertools

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


def test_build_optimizer_splits_bias_group(trainer):
    opt = trainer._build_optimizer("AdamW", lr=1e-3)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["is_bias_group"] is False
    assert opt.param_groups[1]["is_bias_group"] is True
    assert opt.param_groups[1]["weight_decay"] == pytest.approx(0.0)


def test_build_optimizer_invalid(trainer):
    with pytest.raises(ValueError, match="Unknown optimizer"):
        trainer._build_optimizer("LAMB", lr=1e-3)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Adam", torch.optim.Adam),
        ("RAdam", torch.optim.RAdam),
        ("NAdam", torch.optim.NAdam),
        ("RMSprop", torch.optim.RMSprop),
        ("Auto", torch.optim.AdamW),
    ],
)
def test_build_optimizer_expanded_choices(trainer, name, expected):
    assert isinstance(trainer._build_optimizer(name, lr=1e-3), expected)


def test_time_limit_overrides_epoch_limit(trainer):
    assert list(trainer._epoch_iterator(0, 2, None)) == [0, 1]
    assert list(itertools.islice(trainer._epoch_iterator(0, 2, 1.0), 4)) == [0, 1, 2, 3]


@pytest.mark.parametrize("batch", ["auto", -1, 0, 0.5, True])
def test_batch_requires_positive_integer(trainer, batch):
    with pytest.raises(ValueError, match="positive integer"):
        trainer._validate_batch_size(batch)


def test_freeze_parameter_pattern(trainer):
    frozen = trainer._apply_freeze("l.weight")

    assert frozen == trainer.model.l.weight.numel()
    assert trainer.model.l.weight.requires_grad is False
    assert trainer.model.l.bias.requires_grad is True


def test_resume_restores_all_saved_controls_with_warning(trainer, caplog):
    saved = {
        "patience": 7,
        "save_period": 3,
        "workers": 2,
        "freeze": ["backbone"],
        "classes": [0, 2],
        "accumulate": 4,
        "multi_scale": True,
        "time": 1.5,
    }
    state = {"training_state": {"train_args": saved}}
    current = {
        "patience": 100,
        "save_period": 1,
        "workers": 0,
        "freeze": None,
        "classes": None,
        "accumulate": 1,
        "multi_scale": False,
        "time": None,
    }

    resolved = trainer._resolve_resume_args(state, current)

    assert resolved == saved
    assert "restored saved value" in caplog.text


def test_build_scheduler(trainer):
    opt = trainer._build_optimizer("AdamW", lr=1e-3)
    scheduler = trainer._build_scheduler(opt, epochs=10, lrf=0.1)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)


def test_build_scheduler_cosine(trainer):
    opt = trainer._build_optimizer("AdamW", lr=1e-3)
    scheduler = trainer._build_scheduler(opt, epochs=10, lrf=0.1, cos_lr=True)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_build_scheduler_warmup_cosine(trainer):
    opt = trainer._build_optimizer("AdamW", lr=1e-3)
    scheduler = trainer._build_scheduler(opt, epochs=10, lrf=0.1, cos_lr=True)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_scheduler_end_lr(trainer):
    """After `epochs` steps the LR should reach lr0 * lrf."""
    lr0, lrf, epochs = 1e-2, 0.01, 5
    opt = trainer._build_optimizer("AdamW", lr=lr0)
    scheduler = trainer._build_scheduler(opt, epochs=epochs, lrf=lrf)
    for _ in range(epochs):
        opt.step()
        scheduler.step()
    final_lr = opt.param_groups[0]["lr"]
    assert final_lr == pytest.approx(lr0 * lrf, rel=1e-3)


def test_cosine_scheduler_end_lr(trainer):
    """Cosine schedule should also end at lr0 * lrf."""
    lr0, lrf, epochs = 1e-2, 0.1, 6
    opt = trainer._build_optimizer("AdamW", lr=lr0)
    scheduler = trainer._build_scheduler(opt, epochs=epochs, lrf=lrf, cos_lr=True)
    for _ in range(epochs):
        opt.step()
        scheduler.step()
    final_lr = opt.param_groups[0]["lr"]
    assert final_lr == pytest.approx(lr0 * lrf, rel=1e-3)


def test_warmup_cosine_scheduler_progression(trainer):
    """Manual warmup should ramp up first, then cosine decay should finish at lr0 * lrf."""
    lr0, lrf, epochs, warmup_epochs = 1e-2, 0.1, 8, 3
    opt = trainer._build_optimizer("AdamW", lr=lr0)
    scheduler = trainer._build_scheduler(opt, epochs=epochs - warmup_epochs, lrf=lrf, cos_lr=True)

    warmup_iters = warmup_epochs * 2
    lrs = []
    for i in range(warmup_iters):
        trainer._apply_warmup(
            optimizer=opt,
            warmup_iter=i,
            total_warmup_iters=warmup_iters,
            lr0=lr0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
        )
        lrs.append(opt.param_groups[0]["lr"])

    assert lrs[0] > 0.0
    assert lrs[-1] == pytest.approx(lr0, rel=1e-3)

    for _ in range(epochs - warmup_epochs):
        opt.step()
        scheduler.step()

    assert opt.param_groups[0]["lr"] == pytest.approx(lr0 * lrf, rel=1e-3)


def test_warmup_bias_lr_and_momentum(trainer):
    opt = trainer._build_optimizer("SGD", lr=1e-2)
    trainer._apply_warmup(
        optimizer=opt,
        warmup_iter=0,
        total_warmup_iters=4,
        lr0=1e-2,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
    )

    weight_group = next(group for group in opt.param_groups if not group["is_bias_group"])
    bias_group = next(group for group in opt.param_groups if group["is_bias_group"])
    assert weight_group["lr"] < bias_group["lr"]
    assert 0.8 < weight_group["momentum"] < 0.9

    trainer._apply_warmup(
        optimizer=opt,
        warmup_iter=3,
        total_warmup_iters=4,
        lr0=1e-2,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
    )
    assert weight_group["lr"] == pytest.approx(1e-2, rel=1e-3)
    assert bias_group["lr"] == pytest.approx(1e-2, rel=1e-3)
    assert weight_group["momentum"] == pytest.approx(0.9, rel=1e-3)


def test_compute_warmup_iters_has_small_dataset_floor(trainer):
    assert trainer._compute_warmup_iters(warmup_epochs=3.0, total_batches=8) == 100
    assert trainer._compute_warmup_iters(warmup_epochs=0.0, total_batches=8) == 0
    assert trainer._compute_warmup_iters(warmup_epochs=0.5, total_batches=300) == 150


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
