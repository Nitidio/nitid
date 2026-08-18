"""Integration tests for training and validation (Phase 3)."""

import logging

import numpy as np
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
    assert (tmp_path / "test" / "args.yaml").exists()
    assert (tmp_path / "test" / "environment.yaml").exists()
    assert model.names == {0: "person", 1: "car"}
    assert model._cfg["num_classes"] == 2
    assert model._model.decoder.num_classes == 2


def test_segment_train_and_validation_run_end_to_end(
    tiny_segment_checkpoint, tiny_dataset, tmp_path
):
    from dfine import DFINE

    model = DFINE(
        tiny_segment_checkpoint,
        task="segment",
        device="cpu",
        verbose=False,
    )
    metrics = model.train(
        data=tiny_dataset,
        epochs=1,
        imgsz=64,
        batch=2,
        augment=False,
        plots=False,
        project=str(tmp_path),
        name="segment",
        verbose=False,
    )

    row = metrics["history"][0]
    assert model.task == "segment"
    assert set(row) >= {
        "loss_mask_bce",
        "loss_mask_dice",
        "mask_mAP50",
        "mask_mAP50-95",
    }
    assert (tmp_path / "segment" / "last.pth").exists()

    reloaded = DFINE(
        tmp_path / "segment" / "last.pth",
        task="segment",
        device="cpu",
        verbose=False,
    )
    result = reloaded.predict(np.zeros((64, 64, 3), dtype=np.uint8), conf=0.99)[0]
    assert result.masks is not None


def test_semantic_train_validation_and_checkpoint_reload_run_end_to_end(
    tiny_semantic_checkpoint, tiny_semantic_dataset, tmp_path
):
    from dfine import DFINE

    model = DFINE(
        tiny_semantic_checkpoint,
        task="semantic",
        device="cpu",
        verbose=False,
    )
    metrics = model.train(
        data=tiny_semantic_dataset,
        epochs=1,
        imgsz=64,
        batch=2,
        augment=False,
        plots=False,
        project=str(tmp_path),
        name="semantic",
        verbose=False,
    )

    row = metrics["history"][0]
    assert set(row) >= {
        "loss_ce",
        "loss_dice",
        "loss_aux",
        "mIoU",
        "pixel_accuracy",
        "fitness",
    }
    assert 0.0 <= metrics["mIoU"] <= 1.0
    assert 0.0 <= metrics["pixel_accuracy"] <= 1.0
    checkpoint = tmp_path / "semantic" / "last.pth"
    assert checkpoint.exists()

    reloaded = DFINE(checkpoint, task="semantic", device="cpu", verbose=False)
    validation = reloaded.val(
        data=tiny_semantic_dataset,
        imgsz=64,
        batch=2,
        plots=False,
        save_dir=tmp_path / "semantic_val",
        verbose=False,
    )
    assert set(validation) >= {"mIoU", "pixel_accuracy", "per_class", "pixels"}
    assert validation["images"] == 2

    resumed = DFINE(
        tiny_semantic_checkpoint,
        task="semantic",
        device="cpu",
        verbose=False,
    )
    resumed_metrics = resumed.train(
        data=tiny_semantic_dataset,
        epochs=2,
        imgsz=64,
        batch=2,
        augment=False,
        plots=False,
        project=str(tmp_path),
        name="semantic",
        resume=True,
        verbose=False,
    )
    assert [row["epoch"] for row in resumed_metrics["history"]] == [1, 2]


def test_semantic_validation_supports_non_native_image_size(
    tiny_semantic_checkpoint, tiny_semantic_dataset, tmp_path
):
    from dfine import DFINE

    model = DFINE(
        tiny_semantic_checkpoint,
        task="semantic",
        device="cpu",
        verbose=False,
    )
    metrics = model.val(
        data=tiny_semantic_dataset,
        imgsz=32,
        batch=2,
        plots=False,
        save_dir=tmp_path / "semantic_non_native_val",
        verbose=False,
    )

    assert metrics["images"] == 2
    assert 0.0 <= metrics["mIoU"] <= 1.0
    assert 0.0 <= metrics["pixel_accuracy"] <= 1.0


def test_repeated_train_calls_increment_run_directory(tiny_checkpoint, tiny_dataset, tmp_path):
    from dfine import DFINE

    for _ in range(2):
        DFINE(tiny_checkpoint, device="cpu", verbose=False).train(
            data=tiny_dataset, epochs=1, batch=2, project=str(tmp_path), verbose=False
        )

    assert (tmp_path / "exp" / "results.csv").exists()
    assert (tmp_path / "exp2" / "results.csv").exists()
    assert (tmp_path / "exp" / "args.yaml").exists()
    assert (tmp_path / "exp2" / "environment.yaml").exists()


def test_train_early_stopping_reports_best_epoch(
    tiny_checkpoint, tiny_dataset, tmp_path, monkeypatch
):
    from dfine import DFINE
    from dfine.trainer import DFINETrainer

    def constant_validation(self, **kwargs):
        return {
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "mAP50": 0.5,
            "mAP50-95": 0.5,
            "fitness": 0.5,
        }

    monkeypatch.setattr(DFINETrainer, "_validate_epoch", constant_validation)
    metrics = DFINE(tiny_checkpoint, device="cpu", verbose=False).train(
        data=tiny_dataset,
        epochs=5,
        batch=2,
        patience=1,
        plots=False,
        project=str(tmp_path),
        verbose=False,
    )

    assert len(metrics["history"]) == 2
    assert metrics["best_epoch"] == 1


def test_train_gradient_accumulation_reduces_optimizer_steps(
    tiny_checkpoint, tiny_dataset, tmp_path, monkeypatch
):
    from dfine import DFINE
    from dfine.trainer import ModelEMA

    observed = {"steps": 0, "batches": 0}
    original_update = ModelEMA.update

    def count_update(self, model):
        observed["steps"] += 1
        return original_update(self, model)

    monkeypatch.setattr(ModelEMA, "update", count_update)

    class CountSteps:
        def on_train_start(self, trainer):
            observed["batches"] = len(trainer.dataloader)

    DFINE(tiny_checkpoint, device="cpu", verbose=False).train(
        data=tiny_dataset,
        epochs=1,
        batch=1,
        accumulate=2,
        ema=True,
        val=False,
        plots=False,
        project=str(tmp_path),
        verbose=False,
        callbacks=CountSteps(),
    )

    assert observed["steps"] == (observed["batches"] + 1) // 2


def test_train_callbacks_receive_lifecycle_events(tiny_checkpoint, tiny_dataset, tmp_path):
    from dfine import DFINE

    events = []

    class Recorder:
        def on_train_start(self, trainer):
            events.append(("train_start", trainer.train_args["epochs"], len(trainer.history)))
            assert trainer.save_dir == tmp_path / "callbacks"

        def on_train_epoch_start(self, trainer):
            assert trainer.current_row is None
            assert trainer.current_val_metrics is None
            assert trainer.metrics is None
            events.append(("epoch_start", trainer.current_epoch))

        def on_val_end(self, trainer):
            assert trainer.current_val_metrics is not None
            events.append(
                ("val_end", trainer.current_epoch, "mAP50" in trainer.current_val_metrics)
            )

        def on_train_epoch_end(self, trainer):
            assert trainer.current_row is not None
            events.append(
                ("epoch_end", trainer.current_row["epoch"], "fitness" in trainer.current_row)
            )

        def on_train_end(self, trainer):
            assert trainer.metrics is not None
            events.append(("train_end", len(trainer.metrics["history"]), trainer.metrics["loss"]))

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
    assert events[1] == ("epoch_start", 1)
    assert events[2] == ("val_end", 1, True)
    assert events[3] == ("epoch_end", 1, True)
    assert events[4][0] == "train_end"
    assert events[4][1] == 1
    assert isinstance(events[4][2], float)


def test_train_can_stop_early_from_callback(tiny_checkpoint, tiny_dataset, tmp_path):
    from dfine import DFINE

    class StopAfterFirstEpoch:
        def on_train_epoch_end(self, trainer):
            if trainer.current_epoch == 1:
                trainer.stop = True

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    metrics = model.train(
        data=tiny_dataset,
        epochs=4,
        batch=2,
        project=str(tmp_path),
        name="stop_early",
        verbose=False,
        callbacks=StopAfterFirstEpoch(),
    )

    assert len(metrics["history"]) == 1
    assert metrics["history"][0]["epoch"] == 1


def test_model_add_callback_registers_persistent_ultralytics_style_callback(
    tiny_checkpoint, tiny_dataset, tmp_path
):
    from dfine import DFINE

    calls = []

    def on_train_end(trainer):
        calls.append(trainer.stop)

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    model.add_callback("on_train_end", on_train_end)
    model.add_callback("on_train_end", on_train_end)
    metrics = model.train(
        data=tiny_dataset,
        epochs=1,
        batch=2,
        project=str(tmp_path),
        name="persistent_callbacks",
        verbose=False,
    )

    assert len(metrics["history"]) == 1
    assert calls == [False]


def test_train_args_match_serialized_training_state(tiny_checkpoint, tiny_dataset, tmp_path):
    from dfine import DFINE

    observed_train_args = {}

    class CaptureTrainArgs:
        def on_train_start(self, trainer):
            observed_train_args.update(trainer.train_args)

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    model.train(
        data=tiny_dataset,
        epochs=1,
        batch=2,
        resume=False,
        project=str(tmp_path),
        name="train_args_consistency",
        verbose=False,
        callbacks=CaptureTrainArgs(),
    )

    checkpoint = torch.load(
        tmp_path / "train_args_consistency" / "last.pth",
        map_location="cpu",
        weights_only=False,
    )
    serialized_train_args = checkpoint["training_state"]["train_args"]

    assert observed_train_args == serialized_train_args
    assert serialized_train_args["resume"] is False
    assert serialized_train_args["verbose"] is False
    assert set(serialized_train_args) >= {
        "patience",
        "save",
        "save_period",
        "val",
        "plots",
        "val_period",
        "workers",
        "cache",
        "seed",
        "deterministic",
        "momentum",
        "backbone_lr",
        "weight_decay",
        "clip_grad",
        "freeze",
        "classes",
        "single_cls",
        "fraction",
        "accumulate",
        "multi_scale",
        "augment",
        "fliplr",
        "scale",
        "translate",
        "crop",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "mosaic",
        "mixup",
        "close_mosaic",
        "time",
    }


def test_val_runs(tiny_checkpoint, tiny_dataset, tmp_path):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    metrics = model.val(
        data=tiny_dataset, batch=2, project=str(tmp_path), name="exp", verbose=False
    )
    assert set(metrics) >= {"mAP50", "mAP50-95", "AR1", "AR100", "precision", "recall", "f1"}
    assert (tmp_path / "exp" / "confusion_matrix.png").exists()
    assert (tmp_path / "exp" / "pr_curve.png").exists()
    assert (tmp_path / "exp" / "f1_curve.png").exists()


def test_repeated_val_calls_increment_run_directory(tiny_checkpoint, tiny_dataset, tmp_path):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    model.val(data=tiny_dataset, batch=2, project=str(tmp_path), verbose=False)
    model.val(data=tiny_dataset, batch=2, project=str(tmp_path), verbose=False)

    assert (tmp_path / "exp" / "args.yaml").exists()
    assert (tmp_path / "exp2" / "environment.yaml").exists()
    assert (tmp_path / "exp" / "confusion_matrix.png").exists()
    assert (tmp_path / "exp2" / "confusion_matrix.png").exists()


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
