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


def test_train_callbacks_receive_lifecycle_events(tiny_checkpoint, tiny_dataset, tmp_path):
    from dfine import DFINE

    events = []

    class Recorder:
        def on_train_start(self, trainer, state):
            events.append(("train_start", state["epochs"], len(state["history"])))
            assert state["save_dir"] == tmp_path / "callbacks"

        def on_train_epoch_start(self, trainer, state):
            events.append(("epoch_start", state["epoch"], state["epoch_index"]))

        def on_val_end(self, trainer, state):
            events.append(("val_end", state["epoch"], "mAP50" in state["val_metrics"]))

        def on_train_epoch_end(self, trainer, state):
            events.append(("epoch_end", state["row"]["epoch"], "fitness" in state["row"]))

        def on_train_end(self, trainer, state):
            events.append(("train_end", len(state["metrics"]["history"]), state["metrics"]["loss"]))

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    metrics = model.train(
        data=tiny_dataset,
        epochs=1,
        batch=2,
        project=str(tmp_path),
        name="callbacks",
        verbose=False,
        callbacks=Recorder(),
    )

    assert len(metrics["history"]) == 1
    assert events[0] == ("train_start", 1, 0)
    assert events[1] == ("epoch_start", 1, 0)
    assert events[2] == ("val_end", 1, True)
    assert events[3] == ("epoch_end", 1, True)
    assert events[4][0] == "train_end"
    assert events[4][1] == 1
    assert isinstance(events[4][2], float)


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


def test_train_and_val_run_with_yolo_txt_labels(tiny_checkpoint, tiny_yolo_dataset, tmp_path):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    metrics = model.train(
        data=tiny_yolo_dataset,
        epochs=1,
        batch=2,
        project=str(tmp_path),
        name="yolo_train",
        verbose=False,
    )
    assert "loss" in metrics
    assert (tmp_path / "yolo_train" / "epoch1.pth").exists()

    val_metrics = model.val(data=tiny_yolo_dataset, batch=2, verbose=False)
    assert set(val_metrics) >= {"mAP50", "mAP50-95", "AR1", "AR100"}


def test_train_and_val_run_with_split_first_yolo_layout(
    tiny_checkpoint, tiny_yolo_splitfirst_dataset, tmp_path
):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    metrics = model.train(
        data=tiny_yolo_splitfirst_dataset,
        epochs=1,
        batch=2,
        project=str(tmp_path),
        name="splitfirst_yolo_train",
        verbose=False,
    )
    assert "loss" in metrics
    assert (tmp_path / "splitfirst_yolo_train" / "epoch1.pth").exists()

    val_metrics = model.val(data=tiny_yolo_splitfirst_dataset, batch=2, verbose=False)
    assert set(val_metrics) >= {"mAP50", "mAP50-95", "AR1", "AR100"}


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


def test_train_resume_restores_history_and_continues_epochs(
    tiny_checkpoint, tiny_dataset, tmp_path
):
    from dfine import DFINE

    run_dir = tmp_path / "resume_test"
    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    first_metrics = model.train(
        data=tiny_dataset,
        epochs=1,
        batch=2,
        ema=True,
        project=str(tmp_path),
        name="resume_test",
        verbose=False,
    )

    resumed_model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    resumed_metrics = resumed_model.train(
        data=tiny_dataset,
        epochs=2,
        batch=2,
        ema=True,
        resume=True,
        project=str(tmp_path),
        name="resume_test",
        verbose=False,
    )

    assert len(first_metrics["history"]) == 1
    assert len(resumed_metrics["history"]) == 2
    assert resumed_metrics["history"][0]["epoch"] == 1
    assert resumed_metrics["history"][1]["epoch"] == 2
    assert resumed_metrics["history"][0]["loss"] == first_metrics["history"][0]["loss"]

    with (run_dir / "results.csv").open() as f:
        rows = f.read().strip().splitlines()
    assert len(rows) == 3  # header + 2 epoch rows

    checkpoint = torch.load(run_dir / "last.pth", map_location="cpu", weights_only=False)
    training_state = checkpoint["training_state"]
    assert "optimizer" in training_state
    assert "scheduler" in training_state
    assert "history" in training_state
    assert "raw_model" in training_state
    assert "ema" in training_state
    assert len(training_state["history"]) == 2

    epoch_checkpoint = torch.load(run_dir / "epoch2.pth", map_location="cpu", weights_only=False)
    best_checkpoint = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    assert epoch_checkpoint["training_state"] == {}
    assert best_checkpoint["training_state"] == {}


def test_train_resume_training_state_omits_raw_model_without_ema(
    tiny_checkpoint, tiny_dataset, tmp_path
):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    model.train(
        data=tiny_dataset,
        epochs=1,
        batch=2,
        ema=False,
        project=str(tmp_path),
        name="resume_no_ema",
        verbose=False,
    )

    checkpoint = torch.load(
        tmp_path / "resume_no_ema" / "last.pth", map_location="cpu", weights_only=False
    )
    training_state = checkpoint["training_state"]
    assert "optimizer" in training_state
    assert "raw_model" not in training_state


def test_train_resume_requires_existing_checkpoint(tiny_checkpoint, tiny_dataset, tmp_path):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    with pytest.raises(FileNotFoundError, match="resume=True requested"):
        model.train(
            data=tiny_dataset,
            epochs=2,
            batch=2,
            resume=True,
            project=str(tmp_path),
            name="missing_resume",
            verbose=False,
        )


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
