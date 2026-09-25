"""
DFINETrainer — fine-tuning engine.
Called internally by DFINE.train(). Not part of the public API.
"""

from __future__ import annotations

import copy
import csv
import fnmatch
import itertools
import math
import random
import resource
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import torch
from tqdm.auto import tqdm

from dfine.utils.checkpoint import load_checkpoint_state, save_checkpoint
from dfine.utils.logging import LOGGER

TrainerCallback = Callable[..., object]


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of the model whose parameters are updated as::

        ema_param = decay * ema_param + (1 - decay) * model_param

    after every optimiser step. Buffers (e.g. BatchNorm running stats) are
    copied directly. EMA weights typically yield better validation accuracy
    than the raw model weights at the end of training.

    Args:
        model: The model being trained.
        decay: EMA decay factor. Higher = slower update (0.9999 is typical).
        warmups: Updates over which the effective decay ramps toward ``decay``.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, warmups: int = 1000) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("EMA decay must be between 0 and 1")
        if warmups < 0:
            raise ValueError("EMA warmups must be non-negative")
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.warmups = warmups
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: torch.nn.Module) -> None:
        """Update shadow weights from the current model state."""
        with torch.no_grad():
            self.updates += 1
            decay = self.decay
            if self.warmups:
                decay *= 1.0 - math.exp(-self.updates / self.warmups)
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                if ema_p.is_floating_point():
                    ema_p.data.mul_(decay).add_(model_p.data, alpha=1.0 - decay)
                else:
                    ema_p.data.copy_(model_p.data)
            for ema_buf, model_buf in zip(self.ema.buffers(), model.buffers()):
                ema_buf.copy_(model_buf)


class FlatCosineLRScheduler:
    """Iteration-based warmup, flat, cosine, and no-augmentation LR schedule."""

    step_per_iteration = True

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        lr_gamma: float,
        iter_per_epoch: int,
        total_epochs: int,
        warmup_iter: int,
        flat_epochs: int,
        no_aug_epochs: int,
    ) -> None:
        if lr_gamma < 0:
            raise ValueError("lr_gamma must be >= 0")
        self.base_lrs = [
            float(group.get("initial_lr", group["lr"])) for group in optimizer.param_groups
        ]
        self.min_lrs = [base_lr * lr_gamma for base_lr in self.base_lrs]
        self.total_iter = max(int(iter_per_epoch) * max(int(total_epochs), 1), 1)
        self.warmup_iter = max(int(warmup_iter), 0)
        self.flat_iter = max(int(iter_per_epoch) * max(int(flat_epochs), 0), self.warmup_iter)
        self.no_aug_iter = max(int(iter_per_epoch) * max(int(no_aug_epochs), 0), 0)
        self.last_iter = -1
        self._last_lr = list(self.base_lrs)

    def step(self, current_iter: int | None = None, optimizer: torch.optim.Optimizer | None = None):
        if current_iter is None:
            current_iter = self.last_iter + 1
        if optimizer is None:
            raise ValueError("optimizer is required for FlatCosineLRScheduler.step()")
        self.last_iter = int(current_iter)
        self._last_lr = [
            self._schedule(self.last_iter, base_lr, min_lr)
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]
        for group, lr in zip(optimizer.param_groups, self._last_lr):
            group["lr"] = lr
        return optimizer

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict[str, object]:
        return {
            "base_lrs": self.base_lrs,
            "min_lrs": self.min_lrs,
            "total_iter": self.total_iter,
            "warmup_iter": self.warmup_iter,
            "flat_iter": self.flat_iter,
            "no_aug_iter": self.no_aug_iter,
            "last_iter": self.last_iter,
            "_last_lr": self._last_lr,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        base_lrs = state_dict.get("base_lrs", self.base_lrs)
        if isinstance(base_lrs, list):
            self.base_lrs = [float(value) for value in base_lrs]
        min_lrs = state_dict.get("min_lrs", self.min_lrs)
        if isinstance(min_lrs, list):
            self.min_lrs = [float(value) for value in min_lrs]
        self.total_iter = _as_int(state_dict.get("total_iter", self.total_iter), self.total_iter)
        self.warmup_iter = _as_int(
            state_dict.get("warmup_iter", self.warmup_iter), self.warmup_iter
        )
        self.flat_iter = _as_int(state_dict.get("flat_iter", self.flat_iter), self.flat_iter)
        self.no_aug_iter = _as_int(
            state_dict.get("no_aug_iter", self.no_aug_iter), self.no_aug_iter
        )
        self.last_iter = _as_int(state_dict.get("last_iter", self.last_iter), self.last_iter)
        last_lr = state_dict.get("_last_lr", self._last_lr)
        if isinstance(last_lr, list):
            self._last_lr = [float(value) for value in last_lr]

    def _schedule(self, current_iter: int, init_lr: float, min_lr: float) -> float:
        if self.warmup_iter > 0 and current_iter <= self.warmup_iter:
            return init_lr * (current_iter / float(self.warmup_iter)) ** 2
        if current_iter <= self.flat_iter:
            return init_lr
        if current_iter >= self.total_iter - self.no_aug_iter:
            return min_lr

        cosine_span = max(self.total_iter - self.flat_iter - self.no_aug_iter, 1)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - self.flat_iter) / cosine_span))
        return min_lr + (init_lr - min_lr) * cosine_decay


class DFINETrainer:
    """
    Fine-tuning loop for D-FINE.

    Loads a COCO-format dataset, trains for the requested number of epochs
    with gradient clipping, and saves a wrapped checkpoint after each epoch.
    The criterion and weight dict come from the checkpoint's embedded config
    so loss weighting stays consistent with the original training setup.

    Optional features:
      - AMP: mixed-precision training via ``torch.amp.autocast`` + ``GradScaler``
        (CUDA only; automatically disabled with a warning on CPU).
      - EMA: exponential moving average of weights; the EMA model is saved to
        the checkpoint so it loads directly with ``DFINE(path)``.

    Callback contract:
      - Callbacks receive the trainer instance only: ``callback(trainer)``.
      - Persistent callbacks can be registered on the model via
        ``DFINE.add_callback(...)`` or passed to ``train(callbacks=...)``.
      - Useful callback attributes include ``stop``, ``current_epoch``,
        ``current_val_metrics``, ``current_row``, ``current_fitness``,
        ``metrics``, ``history``, ``save_dir``, ``results_path``,
        ``optimizer``, ``scheduler``, ``criterion``, ``scaler``, ``ema_model``,
        ``dataloader``, ``train_args``, ``tracking_state``, ``error``, and
        ``start_epoch``.
      - These attributes are live objects, not deep-copied snapshots.
        Mutating them affects the active training run.
    """

    CALLBACK_EVENTS = (
        "on_train_start",
        "on_train_epoch_start",
        "on_train_epoch_end",
        "on_val_end",
        "on_train_end",
        "on_train_error",
    )

    def __init__(
        self,
        model,
        cfg: dict,
        device: str,
        names: dict,
        callbacks: object | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.names = names
        self.stop = False
        self.current_epoch = 0
        self.current_val_metrics: dict[str, object] | None = None
        self.current_row: dict[str, float | int] | None = None
        self.current_fitness = 0.0
        self.metrics: dict | None = None
        self.error: BaseException | None = None
        self.history: list[dict[str, float | int]] = []
        self.tracking_state: dict[str, object] = {}
        self.save_dir: Path | None = None
        self.results_path: Path | None = None
        self.dataloader: object | None = None
        self.optimizer: object | None = None
        self.scheduler: object | None = None
        self.criterion: object | None = None
        self.scaler: torch.cuda.amp.GradScaler | None = None
        self.ema_model: ModelEMA | None = None
        self.train_args: dict[str, object] = {}
        self.start_epoch = 0
        self._display_loss_keys = (
            "loss_bbox",
            "loss_giou",
            "loss_vfl",
            "loss_mal",
            "loss_fgl",
            "loss_mask_bce",
            "loss_mask_dice",
            "loss_ce",
            "loss_dice",
            "loss_aux",
        )
        self._base_callbacks = self._empty_callback_registry()
        self.callbacks = self._empty_callback_registry()
        self._register_callbacks(callbacks, self._base_callbacks)
        self._reset_callbacks()

    def train(
        self,
        data: str,
        epochs: int,
        imgsz: int,
        batch: int,
        lr0: float,
        backbone_lr: float | None,
        lrf: float,
        cos_lr: bool,
        warmup_epochs: float,
        warmup_momentum: float,
        warmup_bias_lr: float,
        optimizer: str,
        momentum: float,
        weight_decay: float,
        clip_grad: float,
        resume: bool,
        amp: bool,
        ema: bool,
        ema_decay: float,
        project: str,
        name: str,
        save_dir: str | Path | None,
        exist_ok: bool,
        patience: int,
        save: bool,
        save_period: int,
        val: bool,
        plots: bool,
        val_period: int,
        workers: int,
        cache: bool | str,
        seed: int,
        deterministic: bool,
        freeze: int | list[int | str] | str | None,
        classes: list[int] | None,
        single_cls: bool,
        fraction: float,
        accumulate: int,
        multi_scale: bool,
        augment: bool,
        fliplr: float,
        scale: float,
        translate: float,
        crop: float,
        hsv_h: float,
        hsv_s: float,
        hsv_v: float,
        mosaic: float,
        mixup: float,
        close_mosaic: int,
        time_limit: float | None,
        recipe: str,
        verbose: bool,
        scheduler: str = "auto",
        warmup_iter: int = 0,
        flat_epochs: int = 0,
        no_aug_epochs: int = 0,
        lr_gamma: float | None = None,
        collate_mixup_prob: float = 0.0,
        collate_mixup_epochs: tuple[int, int] = (0, 0),
        callbacks: object | None = None,
        wandb: bool | Mapping[str, Any] = False,
        mlflow: bool | Mapping[str, Any] = False,
    ) -> dict:
        """
        Run the fine-tuning loop.

        Args:
            data:       Path to the data YAML (ultralytics-style).
            recipe:     Training recipe. ``"default"`` preserves the task's normal path.
            epochs:     Number of training epochs.
            imgsz:      Input image size (square).
            batch:      Batch size.
            lr0:        Initial learning rate.
            backbone_lr: Optional lower learning rate for backbone parameters.
            lrf:        Final LR as a fraction of lr0 at the end of training.
            cos_lr:     Use cosine LR decay instead of linear decay.
            warmup_epochs: Number of warmup epochs before the main decay schedule.
            warmup_momentum: Warmup starting momentum/beta1 value.
            warmup_bias_lr: Warmup starting LR for bias parameters.
            optimizer:  Auto, Adam, AdamW, SGD, RAdam, NAdam, or RMSprop.
            resume:     Restore the latest run state from ``<project>/<name>/last.pth``.
            amp:        Enable AMP mixed-precision (CUDA only).
            ema:        Enable EMA weight averaging.
            ema_decay:  EMA decay factor (ignored when ``ema=False``).
            project:    Root output directory.
            name:       Run name; checkpoints saved to ``<project>/<name>/``.
            verbose:    Print per-epoch loss.
            callbacks:  Optional callback mapping or object with lifecycle-hook methods.
            wandb:      Enable WandB with ``True``, or pass WandB callback options.
            mlflow:     Enable MLflow with ``True``, or pass MLflow callback options.

        Returns:
            Metrics dict containing final scalar metrics plus a ``history``
            list with one metrics row per epoch.
        """
        from dfine.utils.runs import resolve_run_dir, write_run_metadata

        save_dir = resolve_run_dir(
            project=project,
            name=name,
            save_dir=save_dir,
            exist_ok=False if not resume else exist_ok,
            resume=resume,
        )
        results_path = save_dir / "results.csv"
        self.stop = False
        self.current_epoch = 0
        self.current_val_metrics = None
        self.current_row = None
        self.current_fitness = 0.0
        self.metrics = None
        self.error = None
        self.history = []
        self.tracking_state = {}
        self.save_dir = save_dir
        self.results_path = results_path
        self._reset_callbacks()
        self.add_callbacks(callbacks)
        self._add_wandb_callback(wandb)
        self._add_mlflow_callback(mlflow)
        resume_state: dict[str, object] | None = None
        if resume:
            resume_state = self._load_resume_state(save_dir)
            resolved = self._resolve_resume_args(
                resume_state,
                {
                    "data": data,
                    "imgsz": imgsz,
                    "batch": batch,
                    "lr0": lr0,
                    "backbone_lr": backbone_lr,
                    "lrf": lrf,
                    "cos_lr": cos_lr,
                    "warmup_epochs": warmup_epochs,
                    "warmup_momentum": warmup_momentum,
                    "warmup_bias_lr": warmup_bias_lr,
                    "optimizer": optimizer,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                    "clip_grad": clip_grad,
                    "amp": amp,
                    "ema": ema,
                    "ema_decay": ema_decay,
                    "patience": patience,
                    "save": save,
                    "save_period": save_period,
                    "val": val,
                    "plots": plots,
                    "val_period": val_period,
                    "workers": workers,
                    "cache": cache,
                    "seed": seed,
                    "deterministic": deterministic,
                    "freeze": freeze,
                    "classes": classes,
                    "single_cls": single_cls,
                    "fraction": fraction,
                    "accumulate": accumulate,
                    "multi_scale": multi_scale,
                    "augment": augment,
                    "fliplr": fliplr,
                    "scale": scale,
                    "translate": translate,
                    "crop": crop,
                    "hsv_h": hsv_h,
                    "hsv_s": hsv_s,
                    "hsv_v": hsv_v,
                    "mosaic": mosaic,
                    "mixup": mixup,
                    "close_mosaic": close_mosaic,
                    "time": time_limit,
                    "recipe": recipe,
                    "scheduler": scheduler,
                    "warmup_iter": warmup_iter,
                    "flat_epochs": flat_epochs,
                    "no_aug_epochs": no_aug_epochs,
                    "lr_gamma": lr_gamma,
                    "collate_mixup_prob": collate_mixup_prob,
                    "collate_mixup_epochs": collate_mixup_epochs,
                },
            )
            data = str(resolved["data"])
            imgsz = _as_int(resolved["imgsz"])
            batch = _as_int(resolved["batch"])
            lr0, lrf = _as_float(resolved["lr0"]), _as_float(resolved["lrf"])
            backbone_lr = (
                None if resolved["backbone_lr"] is None else _as_float(resolved["backbone_lr"])
            )
            cos_lr = bool(resolved["cos_lr"])
            warmup_epochs = _as_float(resolved["warmup_epochs"])
            warmup_momentum = _as_float(resolved["warmup_momentum"])
            warmup_bias_lr = _as_float(resolved["warmup_bias_lr"])
            optimizer, momentum = str(resolved["optimizer"]), _as_float(resolved["momentum"])
            weight_decay, clip_grad = (
                _as_float(resolved["weight_decay"]),
                _as_float(resolved["clip_grad"]),
            )
            amp, ema, ema_decay = (
                bool(resolved["amp"]),
                bool(resolved["ema"]),
                _as_float(resolved["ema_decay"]),
            )
            patience, save = _as_int(resolved["patience"]), bool(resolved["save"])
            save_period, val, plots = (
                _as_int(resolved["save_period"]),
                bool(resolved["val"]),
                bool(resolved["plots"]),
            )
            val_period, workers = _as_int(resolved["val_period"]), _as_int(resolved["workers"])
            cache_value = resolved["cache"]
            cache = cache_value if isinstance(cache_value, (bool, str)) else False
            seed, deterministic = _as_int(resolved["seed"]), bool(resolved["deterministic"])
            freeze = resolved["freeze"]  # type: ignore[assignment]
            classes = resolved["classes"]  # type: ignore[assignment]
            single_cls, fraction = bool(resolved["single_cls"]), _as_float(resolved["fraction"])
            accumulate, multi_scale = _as_int(resolved["accumulate"]), bool(resolved["multi_scale"])
            augment = bool(resolved["augment"])
            fliplr, scale, translate, crop = (
                _as_float(resolved["fliplr"]),
                _as_float(resolved["scale"]),
                _as_float(resolved["translate"]),
                _as_float(resolved["crop"]),
            )
            hsv_h, hsv_s, hsv_v = (
                _as_float(resolved["hsv_h"]),
                _as_float(resolved["hsv_s"]),
                _as_float(resolved["hsv_v"]),
            )
            mosaic, mixup = _as_float(resolved["mosaic"]), _as_float(resolved["mixup"])
            close_mosaic = _as_int(resolved["close_mosaic"])
            time_limit = (
                resolved["time"] if resolved["time"] is None else _as_float(resolved["time"])
            )
            recipe = str(resolved["recipe"])
            scheduler = str(resolved["scheduler"])
            warmup_iter = _as_int(resolved["warmup_iter"])
            flat_epochs = _as_int(resolved["flat_epochs"])
            no_aug_epochs = _as_int(resolved["no_aug_epochs"])
            lr_gamma = (
                resolved["lr_gamma"]
                if resolved["lr_gamma"] is None
                else _as_float(resolved["lr_gamma"])
            )
            collate_mixup_prob = _as_float(resolved["collate_mixup_prob"])
            mixup_epochs_value = resolved["collate_mixup_epochs"]
            if isinstance(mixup_epochs_value, (list, tuple)) and len(mixup_epochs_value) == 2:
                collate_mixup_epochs = (
                    _as_int(mixup_epochs_value[0]),
                    _as_int(mixup_epochs_value[1]),
                )
            else:
                collate_mixup_epochs = (0, 0)

        self._validate_train_options(
            patience=patience,
            save_period=save_period,
            val_period=val_period,
            workers=workers,
            accumulate=accumulate,
            clip_grad=clip_grad,
            time=time_limit,
        )
        from dfine.tasks import normalize_task
        from dfine.utils.augmentations import AugmentationConfig

        task = normalize_task(str(self.cfg.get("task", "detect")))
        augmentation_profile: Literal["legacy", "dfine", "deim"] = "legacy"
        photometric = 0.0
        zoomout = 0.0
        iou_crop = 0.0
        if task == "detect":
            augmentation_profile = "deim" if recipe == "deim" else "dfine"
            photometric = 0.5
            zoomout = 1.0
            iou_crop = 0.8
            if recipe == "deim" and mosaic == 0.0:
                mosaic = 0.5

        augmentation = AugmentationConfig(
            profile=augmentation_profile,
            enabled=augment,
            fliplr=fliplr,
            scale=scale,
            translate=translate,
            crop=crop,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            mosaic=mosaic,
            mixup=mixup,
            close_mosaic=close_mosaic,
            photometric=photometric,
            zoomout=zoomout,
            iou_crop=iou_crop,
        )
        augmentation.validate()
        semantic_config = self.cfg.get("SemanticSegmentation", {})
        ignore_index = (
            int(semantic_config.get("ignore_index", 255))
            if isinstance(semantic_config, Mapping)
            else 255
        )
        self._set_reproducibility(seed, deterministic)
        batch = self._validate_batch_size(batch)
        self._apply_freeze(freeze)
        dataloader = self._build_dataloader(
            data,
            imgsz,
            batch,
            workers=workers,
            cache=cache,
            seed=seed,
            deterministic=deterministic,
            classes=classes,
            single_cls=single_cls,
            fraction=fraction,
            augment=augmentation,
            collate_mixup_prob=collate_mixup_prob,
            collate_mixup_epochs=collate_mixup_epochs,
        )
        opt = self._build_optimizer(
            optimizer,
            lr0,
            momentum=momentum,
            weight_decay=weight_decay,
            backbone_lr=backbone_lr,
        )
        warmup_epoch_count = max(float(warmup_epochs), 0.0)
        decay_epochs = max(epochs - int(warmup_epoch_count), 1)
        scheduler_obj = self._build_scheduler(
            opt,
            epochs=decay_epochs,
            lrf=lrf,
            cos_lr=cos_lr,
            schedule=scheduler,
            iter_per_epoch=len(dataloader),
            warmup_iter=warmup_iter,
            flat_epochs=flat_epochs,
            no_aug_epochs=no_aug_epochs,
            lr_gamma=lr_gamma,
        )
        criterion = self._build_criterion()
        self.dataloader = dataloader
        self.optimizer = opt
        self.scheduler = scheduler_obj
        self.criterion = criterion

        # AMP: only meaningful on CUDA
        is_cuda = self.device.startswith("cuda")
        if amp and not is_cuda:
            LOGGER.warning("amp=True ignored — AMP requires a CUDA device, got %s", self.device)
            amp = False
        scaler = torch.cuda.amp.GradScaler() if amp else None
        device_type = self.device.split(":")[0]  # "cuda" or "cpu"
        self.scaler = scaler

        # EMA
        ema_model = self._build_ema(ema_decay) if ema else None
        self.ema_model = ema_model

        criterion.train()
        history: list[dict[str, float | int]] = []
        best_fitness = float("-inf")
        best_epoch = 0
        epochs_without_improvement = 0
        start_epoch = 0
        self.train_args = self._build_train_args(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            backbone_lr=backbone_lr,
            lrf=lrf,
            cos_lr=cos_lr,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            warmup_bias_lr=warmup_bias_lr,
            optimizer=optimizer,
            momentum=momentum,
            weight_decay=weight_decay,
            clip_grad=clip_grad,
            resume=resume,
            amp=amp,
            ema=ema,
            ema_decay=ema_decay,
            project=project,
            name=name,
            patience=patience,
            save=save,
            save_period=save_period,
            val=val,
            plots=plots,
            val_period=val_period,
            workers=workers,
            cache=cache,
            seed=seed,
            deterministic=deterministic,
            freeze=freeze,
            classes=classes,
            single_cls=single_cls,
            fraction=fraction,
            accumulate=accumulate,
            multi_scale=multi_scale,
            augment=augment,
            fliplr=fliplr,
            scale=scale,
            translate=translate,
            crop=crop,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            mosaic=mosaic,
            mixup=mixup,
            close_mosaic=close_mosaic,
            time=time_limit,
            recipe=recipe,
            scheduler=scheduler,
            warmup_iter=warmup_iter,
            flat_epochs=flat_epochs,
            no_aug_epochs=no_aug_epochs,
            lr_gamma=lr_gamma,
            collate_mixup_prob=collate_mixup_prob,
            collate_mixup_epochs=collate_mixup_epochs,
            verbose=verbose,
        )
        self.train_args["save_dir"] = str(save_dir)
        self.train_args["exist_ok"] = exist_ok
        write_run_metadata(save_dir, {"mode": "train", **self.train_args})

        if resume_state is not None:
            (
                history,
                best_fitness,
                start_epoch,
                restored_best_epoch,
                restored_no_improve,
            ) = self._restore_training_state(
                resume_state=resume_state,
                optimizer=opt,
                scheduler=scheduler_obj,
                scaler=scaler,
                ema_model=ema_model,
                epochs=epochs,
            )
            self._ensure_results_file(results_path, history)
            if restored_best_epoch > 0:
                best_epoch = restored_best_epoch
                epochs_without_improvement = restored_no_improve
            elif history:
                best_row = max(history, key=lambda row: float(row.get("fitness", float("-inf"))))
                best_epoch = int(best_row["epoch"])
                epochs_without_improvement = max(0, start_epoch - best_epoch)
        self.history = history
        self.start_epoch = start_epoch

        self._run_callbacks("on_train_start")
        if self.stop:
            final_metrics = self._finalize_metrics(history, best_epoch=best_epoch)
            self.metrics = final_metrics
            self._run_callbacks("on_train_end")
            return final_metrics

        if time_limit is None and start_epoch >= epochs:
            LOGGER.warning(
                "resume=True found checkpoint at epoch %s, which already meets/exceeds epochs=%s",
                start_epoch,
                epochs,
            )
            final_metrics = self._finalize_metrics(history, best_epoch=best_epoch)
            self.metrics = final_metrics
            self._run_callbacks("on_train_end")
            return final_metrics

        total_batches = len(dataloader)
        warmup_iters = self._compute_warmup_iters(warmup_epoch_count, total_batches)
        training_started = time.perf_counter()
        time_limit_reached = False

        epoch_iterator = self._epoch_iterator(start_epoch, epochs, time_limit)
        for epoch in epoch_iterator:
            self._set_dataset_epoch(
                dataloader,
                epoch,
                mosaic_open=close_mosaic == 0 or epoch < max(epochs - close_mosaic, 0),
            )
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            batch_count = 0
            instances = 0
            loss_sums: dict[str, float] = defaultdict(float)

            if is_cuda:
                torch.cuda.reset_peak_memory_stats()

            self.model.train()
            self.current_epoch = epoch + 1
            self.current_val_metrics = None
            self.current_row = None
            self.current_fitness = 0.0
            self._run_callbacks("on_train_epoch_start")
            if self.stop:
                break
            progress = tqdm(
                dataloader,
                total=len(dataloader),
                desc=f"{epoch + 1}/time" if time_limit is not None else f"{epoch + 1}/{epochs}",
                leave=False,
                unit="batch",
                disable=not verbose,
            )
            opt.zero_grad()

            for batch_idx, (images, targets) in enumerate(progress):
                if (
                    time_limit is not None
                    and time.perf_counter() - training_started >= time_limit * 3600
                ):
                    time_limit_reached = True
                    LOGGER.info("Maximum training duration of %.3f hours reached", time_limit)
                    break
                ni = epoch * total_batches + batch_idx
                if warmup_iters > 0 and ni < warmup_iters:
                    self._apply_warmup(
                        optimizer=opt,
                        warmup_iter=ni,
                        total_warmup_iters=warmup_iters,
                        lr0=lr0,
                        warmup_momentum=warmup_momentum,
                        warmup_bias_lr=warmup_bias_lr,
                    )

                images = images.to(self.device)
                if multi_scale:
                    size = self._random_multi_scale_size(imgsz)
                    images = torch.nn.functional.interpolate(
                        images, size=(size, size), mode="bilinear", align_corners=False
                    )
                    for target in targets:
                        masks = target.get("masks")
                        if isinstance(masks, torch.Tensor) and masks.numel():
                            target["masks"] = torch.nn.functional.interpolate(
                                masks[:, None].float(), size=(size, size), mode="nearest"
                            )[:, 0].to(masks.dtype)
                        semantic_mask = target.get("sem_mask")
                        if isinstance(semantic_mask, torch.Tensor):
                            target["sem_mask"] = torch.nn.functional.interpolate(
                                semantic_mask[None, None].float(),
                                size=(size, size),
                                mode="nearest",
                            )[0, 0].to(semantic_mask.dtype)
                targets = [
                    {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()
                    }
                    for t in targets
                ]

                with torch.amp.autocast(device_type=device_type, enabled=amp):
                    outputs = self.model(images, targets=targets)
                    loss_dict = criterion(outputs, targets)
                    # DFINECriterion has already applied weight_dict to every returned
                    # primary, auxiliary, encoder, pre-decoder, and denoising term.
                    # Upstream D-FINE optimizes their direct sum. Filtering by the base
                    # weight-dict keys would discard all suffixed supervision terms, and
                    # applying the weights here would weight primary losses twice.
                    loss = self._sum_loss_terms(loss_dict)
                    scaled_loss = loss / accumulate

                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                should_step = (batch_idx + 1) % accumulate == 0 or batch_idx + 1 == total_batches
                if should_step:
                    if scaler is not None:
                        scaler.unscale_(opt)
                    if clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)
                    if scaler is not None:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad()
                    if ema_model is not None:
                        ema_model.update(self.model)
                    self._step_iteration_scheduler(scheduler_obj, ni + 1, opt)

                epoch_loss += loss.item()
                batch_count += 1
                if task == "semantic":
                    instances += sum(
                        int((target["sem_mask"] != ignore_index).sum().item()) for target in targets
                    )
                else:
                    instances += sum(int(t["labels"].numel()) for t in targets)
                for key, value in loss_dict.items():
                    loss_sums[key] += float(value.detach().item())

                if verbose:
                    avg_loss = epoch_loss / batch_count
                    postfix = {
                        "loss": f"{avg_loss:.4f}",
                        "instances": instances,
                        "s/it": f"{((time.perf_counter() - epoch_start) / batch_count):.2f}",
                    }
                    for key, value in self._primary_loss_stats(
                        {name: total / batch_count for name, total in loss_sums.items()}
                    ).items():
                        postfix[key] = f"{value:.4f}"
                    progress.set_postfix(postfix)

            if epoch + 1 > warmup_epoch_count and not self._scheduler_steps_per_iteration(
                scheduler_obj
            ):
                scheduler_obj.step()
            epoch_time = time.perf_counter() - epoch_start
            train_loss = epoch_loss / max(batch_count, 1)
            train_stats = {key: total / max(batch_count, 1) for key, total in loss_sums.items()}
            memory_mb = self._current_memory_mb()
            seconds_per_iter = epoch_time / max(batch_count, 1)

            should_validate = val and (
                (epoch + 1) % val_period == 0
                or (time_limit is None and epoch + 1 == epochs)
                or time_limit_reached
            )
            val_metrics: dict[str, object] = {}
            if should_validate:
                eval_model = ema_model.ema if ema_model is not None else self.model
                val_metrics = self._validate_epoch(
                    model=eval_model,
                    data=data,
                    imgsz=imgsz,
                    batch=batch,
                    save_dir=save_dir,
                    verbose=False,
                    plots=plots,
                    classes=classes,
                    single_cls=single_cls,
                    show_progress=verbose,
                )
                self.current_val_metrics = val_metrics
                self._run_callbacks("on_val_end")
            primary_losses = self._primary_loss_stats(train_stats)
            scalar_val = self._compact_val_metrics(val_metrics)

            row: dict[str, float | int] = {
                "epoch": epoch + 1,
                "validated": int(should_validate),
                "time": epoch_time,
                "s_per_it": seconds_per_iter,
                "lr": opt.param_groups[0]["lr"],
                "imgsz": imgsz,
                "instances": instances,
                "memory_mb": memory_mb,
                "loss": train_loss,
                **primary_losses,
                **scalar_val,
            }
            history.append(row)
            self._write_results_row(results_path, row)
            if plots:
                self._plot_results(save_dir, history)

            # Save EMA weights when available — they are what gets loaded by DFINE(path)
            save_model = ema_model.ema if ema_model is not None else self.model
            fitness = float(row.get("fitness", -train_loss)) if should_validate else -train_loss
            improved = fitness > best_fitness
            if improved:
                best_fitness = fitness
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            elif should_validate or not val:
                epochs_without_improvement += 1
            training_state = self._serialize_training_state(
                optimizer=opt,
                scheduler=scheduler_obj,
                scaler=scaler,
                ema_model=ema_model,
                history=history,
                best_fitness=best_fitness,
                train_args=self.train_args,
            )
            training_state["best_epoch"] = best_epoch
            training_state["epochs_without_improvement"] = epochs_without_improvement
            if save:
                save_checkpoint(
                    save_dir / "last.pth",
                    save_model,
                    self.cfg,
                    self.names,
                    epoch=epoch + 1,
                    metrics=row,
                    training_state=training_state,
                )
                if save_period > 0 and (epoch + 1) % save_period == 0:
                    save_checkpoint(
                        save_dir / f"epoch{epoch + 1}.pth",
                        save_model,
                        self.cfg,
                        self.names,
                        epoch=epoch + 1,
                        metrics=row,
                    )
                if improved:
                    save_checkpoint(
                        save_dir / "best.pth",
                        save_model,
                        self.cfg,
                        self.names,
                        epoch=epoch + 1,
                        metrics=row,
                    )

            self.current_row = row
            self.current_fitness = fitness
            self._run_callbacks("on_train_epoch_end")

            if verbose:
                LOGGER.info(self._format_epoch_row(epoch + 1, epochs, row))

            if self.stop:
                break
            if patience > 0 and epochs_without_improvement >= patience:
                LOGGER.info(
                    "Early stopping at epoch %d; best epoch was %d (patience=%d)",
                    epoch + 1,
                    best_epoch,
                    patience,
                )
                self.stop = True
                break
            if time_limit_reached:
                self.stop = True
                break

        final_metrics = self._finalize_metrics(history, best_epoch=best_epoch)
        if history:
            LOGGER.info(
                "Training complete: best epoch %d, best fitness %.4f",
                best_epoch,
                best_fitness,
            )
        self.metrics = final_metrics
        self._run_callbacks("on_train_end")
        return final_metrics

    def add_callback(self, event: str, callback: TrainerCallback) -> None:
        self._add_callback_to_registry(self.callbacks, event, callback)

    def _add_wandb_callback(self, wandb: bool | Mapping[str, Any]) -> None:
        if wandb is False:
            return
        if wandb is True:
            options: dict[str, Any] = {}
        elif isinstance(wandb, Mapping):
            options = dict(wandb)
        else:
            raise TypeError("wandb must be a bool or a mapping of WandB options")

        from dfine.integrations import WandbCallback

        self.add_callbacks(WandbCallback(**options))

    def _add_mlflow_callback(self, mlflow: bool | Mapping[str, Any]) -> None:
        if mlflow is False:
            return
        if mlflow is True:
            options: dict[str, Any] = {}
        elif isinstance(mlflow, Mapping):
            options = dict(mlflow)
        else:
            raise TypeError("mlflow must be a bool or a mapping of MLflow options")

        from dfine.integrations import MLflowCallback

        self.add_callbacks(MLflowCallback(**options))

    def add_callbacks(self, callbacks: object | None) -> None:
        self._register_callbacks(callbacks, self.callbacks)

    def _normalize_callback_group(self, callback_group: object) -> list[TrainerCallback]:
        if callable(callback_group):
            return [callback_group]

        if isinstance(callback_group, Iterable) and not isinstance(callback_group, (str, bytes)):
            callbacks = list(callback_group)
            if not all(callable(callback) for callback in callbacks):
                raise TypeError("Every callback in a callback group must be callable")
            return callbacks

        raise TypeError("Callbacks must be a callable or an iterable of callables")

    def _empty_callback_registry(self) -> dict[str, list[TrainerCallback]]:
        return {event: [] for event in self.CALLBACK_EVENTS}

    def _reset_callbacks(self) -> None:
        self.callbacks = {event: callbacks[:] for event, callbacks in self._base_callbacks.items()}

    def _register_callbacks(
        self,
        callbacks: object | None,
        registry: dict[str, list[TrainerCallback]],
    ) -> None:
        if callbacks is None:
            return

        if isinstance(callbacks, Mapping):
            for event, callback_group in callbacks.items():
                for callback in self._normalize_callback_group(callback_group):
                    self._add_callback_to_registry(registry, str(event), callback)
            return

        for event in self.CALLBACK_EVENTS:
            callback_obj = getattr(callbacks, event, None)
            if callback_obj is None:
                continue
            if not callable(callback_obj):
                raise TypeError(
                    f"Callback attribute '{event}' must be callable, got "
                    f"{type(callback_obj).__name__}"
                )
            self._add_callback_to_registry(registry, event, callback_obj)

    def _add_callback_to_registry(
        self,
        registry: dict[str, list[TrainerCallback]],
        event: str,
        callback: TrainerCallback,
    ) -> None:
        if event not in registry:
            supported = ", ".join(self.CALLBACK_EVENTS)
            raise ValueError(f"Unknown callback event '{event}'. Supported events: {supported}")
        if not callable(callback):
            raise TypeError(
                f"Callback for '{event}' must be callable, got {type(callback).__name__}"
            )
        if any(
            self._callback_identity(existing) == self._callback_identity(callback)
            for existing in registry[event]
        ):
            return
        registry[event].append(callback)

    def _callback_identity(self, callback: TrainerCallback) -> object:
        bound_self = getattr(callback, "__self__", None)
        bound_func = getattr(callback, "__func__", None)
        if bound_self is not None and bound_func is not None:
            return (id(bound_self), id(bound_func))
        return id(callback)

    def _run_callbacks(self, event: str) -> None:
        for callback in self.callbacks[event]:
            callback(self)

    def _handle_train_error(self, error: BaseException) -> None:
        """Notify error callbacks while preserving the original training exception."""
        self.error = error
        try:
            self._run_callbacks("on_train_error")
        except Exception:
            LOGGER.exception("A callback failed while handling a training error")

    def _build_dataloader(
        self,
        data: str,
        imgsz: int,
        batch: int,
        workers: int = 0,
        cache: bool | str = False,
        seed: int = 0,
        deterministic: bool = True,
        classes: list[int] | None = None,
        single_cls: bool = False,
        fraction: float = 1.0,
        augment=None,
        collate_mixup_prob: float = 0.0,
        collate_mixup_epochs: tuple[int, int] = (0, 0),
    ):
        from dfine.tasks import normalize_task
        from dfine.utils.data import build_detection_dataloader, build_semantic_dataloader

        task = normalize_task(str(self.cfg.get("task", "detect")))
        if task == "semantic":
            if classes is not None or single_cls:
                raise ValueError("Semantic training does not support classes or single_cls")
            return build_semantic_dataloader(
                data,
                split="train",
                imgsz=imgsz,
                batch_size=batch,
                workers=workers,
                cache=cache,
                seed=seed,
                deterministic=deterministic,
                fraction=fraction,
                augment=augment,
            )
        return build_detection_dataloader(
            data,
            split="train",
            imgsz=imgsz,
            batch_size=batch,
            workers=workers,
            cache=cache,
            seed=seed,
            deterministic=deterministic,
            classes=classes,
            single_cls=single_cls,
            fraction=fraction,
            augment=augment,
            task=task,
            collate_mixup_prob=collate_mixup_prob,
            collate_mixup_epochs=collate_mixup_epochs,
        )

    @staticmethod
    def _set_dataset_epoch(dataloader, epoch: int, mosaic_open: bool) -> None:
        dataset = dataloader.dataset
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch, mosaic=mosaic_open)
        collate_fn = getattr(dataloader, "collate_fn", None)
        set_collate_epoch = getattr(collate_fn, "set_epoch", None)
        if callable(set_collate_epoch):
            set_collate_epoch(epoch)

    def _build_optimizer(
        self,
        name: str,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        backbone_lr: float | None = None,
    ):
        if lr <= 0:
            raise ValueError("lr must be greater than 0")
        if backbone_lr is not None and backbone_lr <= 0:
            raise ValueError("backbone_lr must be greater than 0")

        # D-FINE fine-tuning uses a lower LR for the pretrained backbone and no
        # weight decay on normalization parameters or biases. Keep bias groups
        # distinct so the optional warmup_bias_lr behavior remains intact.
        grouped: dict[tuple[float, float, bool, str], list[torch.nn.Parameter]] = {}
        for param_name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_backbone = param_name == "backbone" or param_name.startswith("backbone.")
            group_lr = backbone_lr if is_backbone and backbone_lr is not None else lr
            is_bias = param_name.endswith(".bias")
            lowered = param_name.lower()
            is_norm = param.ndim == 1 or "norm" in lowered or ".bn" in lowered
            decay = 0.0 if is_bias or is_norm else weight_decay
            role = "backbone" if is_backbone else "main"
            grouped.setdefault((group_lr, decay, is_bias, role), []).append(param)

        param_groups: list[dict[str, object]] = []
        for (group_lr, decay, is_bias, role), params in grouped.items():
            param_groups.append(
                {
                    "params": params,
                    "lr": group_lr,
                    "initial_lr": group_lr,
                    "is_bias_group": is_bias,
                    "weight_decay": decay,
                    "parameter_role": role,
                }
            )

        optimizer: torch.optim.Optimizer
        normalized = name.lower()
        if normalized == "auto":
            normalized = "adamw"
            LOGGER.info("optimizer=auto selected AdamW (documented deterministic policy)")
        if normalized == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay
            )
        elif normalized == "adam":
            optimizer = torch.optim.Adam(
                param_groups, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay
            )
        elif normalized == "radam":
            optimizer = torch.optim.RAdam(
                param_groups, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay
            )
        elif normalized == "nadam":
            optimizer = torch.optim.NAdam(
                param_groups, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay
            )
        elif normalized == "rmsprop":
            optimizer = torch.optim.RMSprop(
                param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        elif normalized == "sgd":
            optimizer = torch.optim.SGD(
                param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        else:
            raise ValueError(
                "Unknown optimizer: expected Auto, Adam, AdamW, SGD, RAdam, NAdam, or RMSprop"
            )

        for group in optimizer.param_groups:
            if "momentum" in group:
                group["target_momentum"] = float(group["momentum"])
            if "betas" in group:
                beta1, beta2 = group["betas"]
                group["target_beta1"] = float(beta1)
                group["target_beta2"] = float(beta2)
        return optimizer

    def _validate_train_options(
        self,
        *,
        patience: int,
        save_period: int,
        val_period: int,
        workers: int,
        accumulate: int,
        clip_grad: float,
        time: float | None,
    ) -> None:
        if patience < 0:
            raise ValueError("patience must be >= 0")
        if save_period == 0 or save_period < -1:
            raise ValueError("save_period must be -1 or >= 1")
        if val_period < 1:
            raise ValueError("val_period must be >= 1")
        if workers < 0:
            raise ValueError("workers must be >= 0")
        if accumulate < 1:
            raise ValueError("accumulate must be >= 1")
        if clip_grad < 0:
            raise ValueError("clip_grad must be >= 0")
        if time is not None and time <= 0:
            raise ValueError("time must be greater than 0 hours")

    def _set_reproducibility(self, seed: int, deterministic: bool) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:  # pragma: no cover
            pass
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        torch.use_deterministic_algorithms(deterministic, warn_only=True)

    @staticmethod
    def _epoch_iterator(start_epoch: int, epochs: int, time_limit: float | None):
        """Use epochs as the stop limit only when no duration was requested."""
        if time_limit is not None:
            return itertools.count(start_epoch)
        return iter(range(start_epoch, epochs))

    @staticmethod
    def _validate_batch_size(batch: object) -> int:
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError("batch must be a positive integer")
        return batch

    @staticmethod
    def _random_multi_scale_size(imgsz: int, stride: int = 32) -> int:
        """Select a 0.5x–1.5x training size aligned to the model stride."""
        lower = max(1, math.ceil((imgsz * 0.5) / stride))
        upper = max(lower, math.floor((imgsz * 1.5) / stride))
        return random.randint(lower, upper) * stride

    def _apply_freeze(self, freeze: int | list[int | str] | str | None) -> int:
        for parameter in self.model.parameters():
            if parameter.is_floating_point() or parameter.is_complex():
                parameter.requires_grad_(True)
        if freeze in (None, 0, [], ""):
            return 0

        if isinstance(freeze, list):
            items: list[int | str] = freeze
        elif isinstance(freeze, (int, str)):
            items = [freeze]
        else:
            return 0
        children = [name for name, _ in self.model.named_children()]
        patterns: list[str] = []
        for item in items:
            if isinstance(item, int):
                if item < 0:
                    raise ValueError("freeze stage indices must be >= 0")
                if len(items) == 1:
                    patterns.extend(children[:item])
                elif item < len(children):
                    patterns.append(children[item])
            else:
                patterns.append(str(item))

        frozen = 0
        for name, parameter in self.model.named_parameters():
            if any(
                name == pattern
                or name.startswith(f"{pattern}.")
                or pattern in name
                or fnmatch.fnmatch(name, pattern)
                for pattern in patterns
            ):
                parameter.requires_grad_(False)
                frozen += parameter.numel()
        if frozen == 0:
            LOGGER.warning("freeze=%r did not match any model parameters", freeze)
        else:
            LOGGER.info("Froze %d parameters matching %r", frozen, freeze)
        return frozen

    def _build_scheduler(
        self,
        opt,
        epochs: int,
        lrf: float,
        cos_lr: bool = False,
        *,
        schedule: str = "auto",
        iter_per_epoch: int = 1,
        warmup_iter: int = 0,
        flat_epochs: int = 0,
        no_aug_epochs: int = 0,
        lr_gamma: float | None = None,
    ):
        base_scheduler: object
        epochs = max(int(epochs), 1)
        base_lr = float(opt.param_groups[0]["lr"])
        schedule_name = schedule.lower()

        if schedule_name == "flatcosine":
            base_scheduler = FlatCosineLRScheduler(
                opt,
                lr_gamma=lrf if lr_gamma is None else lr_gamma,
                iter_per_epoch=iter_per_epoch,
                total_epochs=epochs,
                warmup_iter=warmup_iter,
                flat_epochs=flat_epochs,
                no_aug_epochs=no_aug_epochs,
            )
        elif cos_lr:
            base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=epochs,
                eta_min=base_lr * lrf,
            )
        else:
            base_scheduler = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1.0, end_factor=lrf, total_iters=epochs
            )
        return base_scheduler

    @staticmethod
    def _scheduler_steps_per_iteration(scheduler: object) -> bool:
        return bool(getattr(scheduler, "step_per_iteration", False))

    def _step_iteration_scheduler(self, scheduler: object, iteration: int, optimizer) -> None:
        if self._scheduler_steps_per_iteration(scheduler):
            step = getattr(scheduler, "step")
            if callable(step):
                step(iteration, optimizer)

    def _build_criterion(self):
        from dfine.nn.criterion import build_criterion

        return build_criterion(self.cfg)

    @staticmethod
    def _sum_loss_terms(loss_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Sum D-FINE criterion outputs, which are already individually weighted."""
        if not loss_dict:
            raise RuntimeError("D-FINE criterion returned no loss terms")
        terms = iter(loss_dict.values())
        total = next(terms)
        for term in terms:
            total = total + term
        return total

    def _build_ema(self, decay: float) -> ModelEMA:
        return ModelEMA(self.model, decay=decay)

    def _load_resume_state(self, save_dir: Path) -> dict[str, object]:
        checkpoint_path = save_dir / "last.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"resume=True requested but no checkpoint was found at '{checkpoint_path}'"
            )
        return load_checkpoint_state(checkpoint_path)

    def _build_train_args(self, **kwargs: object) -> dict[str, object]:
        return dict(kwargs)

    def _resolve_resume_args(
        self, resume_state: dict[str, object], current_args: dict[str, object]
    ) -> dict[str, object]:
        training_state = resume_state.get("training_state")
        if not isinstance(training_state, dict):
            return current_args.copy()
        saved_args = training_state.get("train_args")
        if not isinstance(saved_args, dict):
            return current_args.copy()

        resolved = current_args.copy()
        for key, current_value in current_args.items():
            if key in saved_args:
                saved_value = saved_args[key]
                if saved_value != current_value:
                    LOGGER.warning(
                        "resume=True ignored %s=%r and restored saved value %r",
                        key,
                        current_value,
                        saved_value,
                    )
                resolved[key] = saved_value
        return resolved

    def _restore_training_state(
        self,
        resume_state: dict[str, object],
        optimizer,
        scheduler,
        scaler,
        ema_model: ModelEMA | None,
        epochs: int,
    ) -> tuple[list[dict[str, float | int]], float, int, int, int]:
        training_state = resume_state.get("training_state")
        if not isinstance(training_state, dict):
            self.model.load_state_dict(resume_state["model"])
            self.model.to(self.device)
            return [], float("-inf"), _as_int(resume_state.get("epoch", 0)), 0, 0

        raw_model_state = training_state.get("raw_model")
        if raw_model_state is None:
            raw_model_state = resume_state["model"]
        self.model.load_state_dict(raw_model_state)
        self.model.to(self.device)

        optimizer_state = training_state.get("optimizer")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)

        scheduler_state = training_state.get("scheduler")
        if isinstance(scheduler_state, dict):
            scheduler.load_state_dict(scheduler_state)
            self._update_scheduler_horizons(
                scheduler=scheduler,
                epochs=epochs,
                train_args=training_state.get("train_args"),
            )

        scaler_state = training_state.get("scaler")
        if scaler is not None and isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)

        ema_state = training_state.get("ema")
        if ema_model is not None and isinstance(ema_state, dict):
            ema_weights = ema_state.get("state_dict")
            if ema_weights is not None:
                ema_model.ema.load_state_dict(ema_weights)
            decay = ema_state.get("decay")
            if isinstance(decay, (int, float)):
                ema_model.decay = float(decay)
            warmups = ema_state.get("warmups")
            if isinstance(warmups, int):
                ema_model.warmups = warmups
            updates = ema_state.get("updates")
            if isinstance(updates, int):
                ema_model.updates = updates

        history_raw = training_state.get("history", [])
        history = self._coerce_history(history_raw)
        tracking_state = training_state.get("tracking_state")
        if isinstance(tracking_state, dict):
            self.tracking_state = tracking_state.copy()
        best_fitness = _as_float(training_state.get("best_fitness", float("-inf")), float("-inf"))
        start_epoch = _as_int(resume_state.get("epoch", 0))
        best_epoch = _as_int(training_state.get("best_epoch", 0))
        no_improve = _as_int(training_state.get("epochs_without_improvement", 0))
        return history, best_fitness, start_epoch, best_epoch, no_improve

    def _update_scheduler_horizons(
        self, scheduler, epochs: int, train_args: object | None = None
    ) -> None:
        warmup_epochs = 0.0
        if isinstance(train_args, dict):
            warmup_epochs = _as_float(train_args.get("warmup_epochs", 0.0))
        decay_epochs = max(epochs - int(max(warmup_epochs, 0.0)), 1)

        if isinstance(scheduler, torch.optim.lr_scheduler.LinearLR):
            scheduler.total_iters = decay_epochs
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.T_max = decay_epochs

    def _compute_warmup_iters(self, warmup_epochs: float, total_batches: int) -> int:
        warmup_epochs = max(float(warmup_epochs), 0.0)
        total_batches = max(int(total_batches), 0)
        if warmup_epochs <= 0.0 or total_batches <= 0:
            return 0
        return max(int(round(warmup_epochs * total_batches)), 100)

    def _apply_warmup(
        self,
        optimizer,
        warmup_iter: int,
        total_warmup_iters: int,
        lr0: float,
        warmup_momentum: float,
        warmup_bias_lr: float,
    ) -> None:
        if total_warmup_iters <= 0:
            return

        progress = min((warmup_iter + 1) / total_warmup_iters, 1.0)
        for group in optimizer.param_groups:
            target_lr = float(group.get("initial_lr", lr0))
            start_lr = warmup_bias_lr if bool(group.get("is_bias_group", False)) else 0.0
            group["lr"] = start_lr + (target_lr - start_lr) * progress

            if "momentum" in group and "target_momentum" in group:
                target_momentum = float(group["target_momentum"])
                group["momentum"] = warmup_momentum + (target_momentum - warmup_momentum) * progress
            elif "betas" in group and "target_beta1" in group and "target_beta2" in group:
                target_beta1 = float(group["target_beta1"])
                target_beta2 = float(group["target_beta2"])
                beta1 = warmup_momentum + (target_beta1 - warmup_momentum) * progress
                group["betas"] = (beta1, target_beta2)

    def _serialize_training_state(
        self,
        optimizer,
        scheduler,
        scaler,
        ema_model: ModelEMA | None,
        history: list[dict[str, float | int]],
        best_fitness: float,
        train_args: dict[str, object],
    ) -> dict[str, object]:
        training_state: dict[str, object] = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best_fitness": best_fitness,
            "train_args": train_args,
            "tracking_state": self.tracking_state,
        }
        if scaler is not None:
            training_state["scaler"] = scaler.state_dict()
        if ema_model is not None:
            training_state["raw_model"] = self.model.state_dict()
            training_state["ema"] = {
                "state_dict": ema_model.ema.state_dict(),
                "decay": ema_model.decay,
                "warmups": ema_model.warmups,
                "updates": ema_model.updates,
            }
        return training_state

    def _coerce_history(self, history_raw: object) -> list[dict[str, float | int]]:
        history: list[dict[str, float | int]] = []
        if not isinstance(history_raw, list):
            return history

        for row in history_raw:
            if not isinstance(row, dict):
                continue
            normalized: dict[str, float | int] = {}
            for key, value in row.items():
                if isinstance(value, bool):
                    normalized[str(key)] = int(value)
                elif isinstance(value, int):
                    normalized[str(key)] = value
                elif isinstance(value, float):
                    normalized[str(key)] = value
            if normalized:
                history.append(normalized)
        return history

    def _ensure_results_file(self, path: Path, history: list[dict[str, float | int]]) -> None:
        if path.exists() or not history:
            return

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._results_fieldnames(history[0]))
            writer.writeheader()
            writer.writerows(history)

    def _finalize_metrics(self, history: list[dict[str, float | int]], best_epoch: int = 0) -> dict:
        if history:
            final_row = history[-1]
            metrics = {
                "loss": float(final_row["loss"]),
                "fitness": float(final_row.get("fitness", 0.0)),
                "mAP50": float(final_row.get("mAP50", 0.0)),
                "mAP50-95": float(final_row.get("mAP50-95", 0.0)),
                "best_epoch": best_epoch,
                "history": history,
            }
            if "mask_mAP50" in final_row:
                metrics["mask_mAP50"] = float(final_row["mask_mAP50"])
                metrics["mask_mAP50-95"] = float(final_row["mask_mAP50-95"])
            if "mIoU" in final_row:
                metrics["mIoU"] = float(final_row["mIoU"])
                metrics["pixel_accuracy"] = float(final_row["pixel_accuracy"])
            return metrics

        metrics = {
            "loss": 0.0,
            "fitness": 0.0,
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "best_epoch": best_epoch,
            "history": [],
        }
        if str(self.cfg.get("task", "detect")).lower() == "segment":
            metrics["mask_mAP50"] = 0.0
            metrics["mask_mAP50-95"] = 0.0
        if str(self.cfg.get("task", "detect")).lower() == "semantic":
            metrics["mIoU"] = 0.0
            metrics["pixel_accuracy"] = 0.0
        return metrics

    def _validate_epoch(
        self,
        model,
        data: str,
        imgsz: int,
        batch: int,
        save_dir: Path,
        verbose: bool,
        plots: bool = True,
        classes: list[int] | None = None,
        single_cls: bool = False,
        show_progress: bool = False,
    ) -> dict:
        from dfine.validator import DFINEValidator

        validator = DFINEValidator(model, self.cfg, self.device, self.names)
        return validator.run(
            data=data,
            imgsz=imgsz,
            batch=batch,
            conf=0.001,
            split="val",
            verbose=verbose,
            save_dir=save_dir,
            plots=plots,
            classes=classes,
            single_cls=single_cls,
            show_progress=show_progress,
        )

    def _primary_loss_stats(self, train_stats: dict[str, float]) -> dict[str, float]:
        return {
            key: float(train_stats[key]) for key in self._display_loss_keys if key in train_stats
        }

    def _compact_val_metrics(self, val_metrics: dict[str, object]) -> dict[str, float]:
        if "mIoU" in val_metrics:
            return {
                key: _as_float(val_metrics.get(key, 0.0))
                for key in ("mIoU", "pixel_accuracy", "fitness")
            }
        keys = ["precision", "recall", "mAP50", "mAP50-95", "fitness"]
        if "mask_mAP50" in val_metrics:
            keys.extend(["mask_mAP50", "mask_mAP50-95"])
        return {key: _as_float(val_metrics.get(key, 0.0)) for key in keys}

    def _write_results_row(self, path: Path, row: dict[str, float | int]) -> None:
        exists = path.exists()
        fieldnames = self._results_fieldnames(row)
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _results_fieldnames(self, row: dict[str, float | int]) -> list[str]:
        priority = [
            "epoch",
            "time",
            "s_per_it",
            "lr",
            "imgsz",
            "instances",
            "memory_mb",
            "loss",
            *self._display_loss_keys,
            "precision",
            "recall",
            "f1",
            "mAP50",
            "mAP50-95",
            "mask_mAP50",
            "mask_mAP50-95",
            "mIoU",
            "pixel_accuracy",
            "fitness",
        ]
        ordered = [key for key in priority if key in row]
        extras = sorted(key for key in row if key not in ordered)
        return ordered + extras

    def _plot_results(self, save_dir: Path, history: list[dict[str, float | int]]) -> None:
        plt = __import__("importlib").import_module("matplotlib.pyplot")

        epochs = [int(row["epoch"]) for row in history]
        metrics = {key: [float(row.get(key, 0.0)) for row in history] for key in history[0]}

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

        axes[0, 0].plot(epochs, metrics["loss"], label="total loss", color="tab:red")
        for key in [k for k in self._display_loss_keys if k in metrics]:
            axes[0, 0].plot(epochs, metrics[key], label=key)
        axes[0, 0].set_title("Loss")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        if "fitness" in metrics:
            axes[0, 1].plot(epochs, metrics["fitness"], label="fitness", color="tab:green")
        if "mAP50-95" in metrics:
            axes[0, 1].plot(epochs, metrics["mAP50-95"], label="mAP50-95", color="tab:blue")
        if "mAP50" in metrics:
            axes[0, 1].plot(epochs, metrics["mAP50"], label="mAP50", color="tab:orange")
        if "mask_mAP50-95" in metrics:
            axes[0, 1].plot(
                epochs, metrics["mask_mAP50-95"], label="mask mAP50-95", color="tab:pink"
            )
        if "mask_mAP50" in metrics:
            axes[0, 1].plot(epochs, metrics["mask_mAP50"], label="mask mAP50", color="tab:olive")
        if "mIoU" in metrics:
            axes[0, 1].plot(epochs, metrics["mIoU"], label="mIoU", color="tab:blue")
        if "pixel_accuracy" in metrics:
            axes[0, 1].plot(
                epochs,
                metrics["pixel_accuracy"],
                label="pixel accuracy",
                color="tab:orange",
            )
        axes[0, 1].set_title("Validation")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        if "precision" in metrics:
            axes[1, 0].plot(epochs, metrics["precision"], label="precision", color="tab:purple")
        if "recall" in metrics:
            axes[1, 0].plot(epochs, metrics["recall"], label="recall", color="tab:brown")
        if "f1" in metrics:
            axes[1, 0].plot(epochs, metrics["f1"], label="f1", color="tab:cyan")
        axes[1, 0].set_title("Precision / Recall")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        if "instances" in metrics:
            axes[1, 1].plot(epochs, metrics["instances"], label="instances", color="tab:gray")
        if "memory_mb" in metrics:
            axes[1, 1].plot(epochs, metrics["memory_mb"], label="memory_mb", color="tab:pink")
        axes[1, 1].set_title("Epoch sanity checks")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        for ax in axes[-1, :]:
            ax.set_xlabel("Epoch")

        fig.tight_layout()
        fig.savefig(save_dir / "results.png", dpi=200)
        plt.close(fig)

    def _format_epoch_row(self, epoch: int, epochs: int, row: dict[str, float | int]) -> str:
        validated = bool(row.get("validated", True))
        val_fields = (
            [
                f"P={float(row.get('precision', 0.0)):.3f}",
                f"R={float(row.get('recall', 0.0)):.3f}",
                f"mAP50={float(row.get('mAP50', 0.0)):.3f}",
                f"mAP50-95={float(row.get('mAP50-95', 0.0)):.3f}",
                f"fitness={float(row.get('fitness', 0.0)):.3f}",
            ]
            if validated
            else ["validation=skipped"]
        )
        if validated and "mIoU" in row:
            val_fields = [
                f"mIoU={float(row['mIoU']):.3f}",
                f"pixel_acc={float(row['pixel_accuracy']):.3f}",
                f"fitness={float(row.get('fitness', 0.0)):.3f}",
            ]
        if validated and "mask_mAP50" in row:
            val_fields.extend(
                [
                    f"mask_mAP50={float(row['mask_mAP50']):.3f}",
                    f"mask_mAP50-95={float(row['mask_mAP50-95']):.3f}",
                ]
            )
        parts = [
            f"Epoch {epoch}/{epochs}",
            f"loss={float(row.get('loss', 0.0)):.4f}",
            *val_fields,
            f"instances={int(row.get('instances', 0))}",
            f"imgsz={int(row.get('imgsz', 0))}",
            f"mem={float(row.get('memory_mb', 0.0)):.1f}MB",
            f"s/it={float(row.get('s_per_it', 0.0)):.2f}",
        ]
        for key in [k for k in self._display_loss_keys if k in row]:
            parts.append(f"{key}={float(row[key]):.4f}")
        return "  ".join(parts)

    def _current_memory_mb(self) -> float:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024**2))

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(rss / (1024**2))
        return float(rss / 1024.0)
