"""
DFINETrainer — fine-tuning engine.
Called internally by DFINE.train(). Not part of the public API.
"""

from __future__ import annotations

import copy
import csv
import resource
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

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
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: torch.nn.Module) -> None:
        """Update shadow weights from the current model state."""
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                if ema_p.is_floating_point():
                    ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
                else:
                    ema_p.data.copy_(model_p.data)
            for ema_buf, model_buf in zip(self.ema.buffers(), model.buffers()):
                ema_buf.copy_(model_buf)


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
        ``dataloader``, ``train_args``, and ``start_epoch``.
      - These attributes are live objects, not deep-copied snapshots.
        Mutating them affects the active training run.
    """

    CALLBACK_EVENTS = (
        "on_train_start",
        "on_train_epoch_start",
        "on_train_epoch_end",
        "on_val_end",
        "on_train_end",
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
        self.history: list[dict[str, float | int]] = []
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
        self._display_loss_keys = ("loss_bbox", "loss_giou", "loss_vfl", "loss_fgl")
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
        lrf: float,
        optimizer: str,
        resume: bool,
        amp: bool,
        ema: bool,
        ema_decay: float,
        project: str,
        name: str,
        verbose: bool,
        callbacks: object | None = None,
    ) -> dict:
        """
        Run the fine-tuning loop.

        Args:
            data:       Path to the data YAML (ultralytics-style).
            epochs:     Number of training epochs.
            imgsz:      Input image size (square).
            batch:      Batch size.
            lr0:        Initial learning rate.
            lrf:        Final LR as a fraction of lr0 (linear decay).
            optimizer:  ``"AdamW"`` or ``"SGD"``.
            resume:     Restore and continue from ``<project>/<name>/last.pth``.
            amp:        Enable AMP mixed-precision (CUDA only).
            ema:        Enable EMA weight averaging.
            ema_decay:  EMA decay factor (ignored when ``ema=False``).
            project:    Root output directory.
            name:       Run name; checkpoints saved to ``<project>/<name>/``.
            verbose:    Print per-epoch loss.
            callbacks:  Optional callback mapping or object with lifecycle-hook methods.

        Returns:
            Metrics dict containing final scalar metrics plus a ``history``
            list with one metrics row per epoch.
        """
        save_dir = Path(project) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        results_path = save_dir / "results.csv"
        self.stop = False
        self.current_epoch = 0
        self.current_val_metrics = None
        self.current_row = None
        self.current_fitness = 0.0
        self.metrics = None
        self.history = []
        self.save_dir = save_dir
        self.results_path = results_path
        self._reset_callbacks()
        self.add_callbacks(callbacks)
        resume_state: dict[str, object] | None = None
        if resume:
            resume_state = self._load_resume_state(save_dir)
            data, imgsz, batch, lr0, lrf, optimizer, amp, ema, ema_decay = (
                self._resolve_resume_args(
                    resume_state=resume_state,
                    data=data,
                    imgsz=imgsz,
                    batch=batch,
                    lr0=lr0,
                    lrf=lrf,
                    optimizer=optimizer,
                    amp=amp,
                    ema=ema,
                    ema_decay=ema_decay,
                )
            )

        dataloader = self._build_dataloader(data, imgsz, batch)
        opt = self._build_optimizer(optimizer, lr0)
        scheduler = self._build_scheduler(opt, epochs, lrf)
        criterion = self._build_criterion()
        self.dataloader = dataloader
        self.optimizer = opt
        self.scheduler = scheduler
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
        weight_dict = criterion.weight_dict
        history: list[dict[str, float | int]] = []
        best_fitness = float("-inf")
        start_epoch = 0
        self.train_args = self._build_train_args(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            lrf=lrf,
            optimizer=optimizer,
            resume=resume,
            amp=amp,
            ema=ema,
            ema_decay=ema_decay,
            project=project,
            name=name,
            verbose=verbose,
        )

        if resume_state is not None:
            history, best_fitness, start_epoch = self._restore_training_state(
                resume_state=resume_state,
                optimizer=opt,
                scheduler=scheduler,
                scaler=scaler,
                ema_model=ema_model,
                epochs=epochs,
            )
            self._ensure_results_file(results_path, history)
        self.history = history
        self.start_epoch = start_epoch

        self._run_callbacks("on_train_start")
        if self.stop:
            final_metrics = self._finalize_metrics(history)
            self.metrics = final_metrics
            self._run_callbacks("on_train_end")
            return final_metrics

        if start_epoch >= epochs:
            LOGGER.warning(
                "resume=True found checkpoint at epoch %s, which already meets/exceeds epochs=%s",
                start_epoch,
                epochs,
            )
            final_metrics = self._finalize_metrics(history)
            self.metrics = final_metrics
            self._run_callbacks("on_train_end")
            return final_metrics

        for epoch in range(start_epoch, epochs):
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
                desc=f"{epoch + 1}/{epochs}",
                leave=False,
                unit="batch",
                disable=not verbose,
            )

            for images, targets in progress:
                images = images.to(self.device)
                targets = [
                    {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()
                    }
                    for t in targets
                ]

                opt.zero_grad()

                with torch.amp.autocast(device_type=device_type, enabled=amp):
                    outputs = self.model(images, targets=targets)
                    loss_dict = criterion(outputs, targets)
                    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                    opt.step()

                if ema_model is not None:
                    ema_model.update(self.model)

                epoch_loss += loss.item()
                batch_count += 1
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

            scheduler.step()
            epoch_time = time.perf_counter() - epoch_start
            train_loss = epoch_loss / max(batch_count, 1)
            train_stats = {key: total / max(batch_count, 1) for key, total in loss_sums.items()}
            memory_mb = self._current_memory_mb()
            seconds_per_iter = epoch_time / max(batch_count, 1)

            eval_model = ema_model.ema if ema_model is not None else self.model
            val_metrics = self._validate_epoch(
                model=eval_model,
                data=data,
                imgsz=imgsz,
                batch=batch,
                save_dir=save_dir,
                verbose=False,
            )
            self.current_val_metrics = val_metrics
            self._run_callbacks("on_val_end")
            primary_losses = self._primary_loss_stats(train_stats)
            scalar_val = self._compact_val_metrics(val_metrics)

            row: dict[str, float | int] = {
                "epoch": epoch + 1,
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
            self._plot_results(save_dir, history)

            # Save EMA weights when available — they are what gets loaded by DFINE(path)
            save_model = ema_model.ema if ema_model is not None else self.model
            fitness = float(row.get("fitness", 0.0))
            training_state = self._serialize_training_state(
                optimizer=opt,
                scheduler=scheduler,
                scaler=scaler,
                ema_model=ema_model,
                history=history,
                best_fitness=max(best_fitness, fitness),
                train_args=self.train_args,
            )
            save_checkpoint(
                save_dir / f"epoch{epoch + 1}.pth",
                save_model,
                self.cfg,
                self.names,
                epoch=epoch + 1,
                metrics=row,
            )
            save_checkpoint(
                save_dir / "last.pth",
                save_model,
                self.cfg,
                self.names,
                epoch=epoch + 1,
                metrics=row,
                training_state=training_state,
            )

            if fitness >= best_fitness:
                best_fitness = fitness
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

        final_metrics = self._finalize_metrics(history)
        self.metrics = final_metrics
        self._run_callbacks("on_train_end")
        return final_metrics

    def add_callback(self, event: str, callback: TrainerCallback) -> None:
        self._add_callback_to_registry(self.callbacks, event, callback)

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

    def _build_dataloader(self, data: str, imgsz: int, batch: int):
        from dfine.utils.data import build_coco_dataloader

        return build_coco_dataloader(data, split="train", imgsz=imgsz, batch_size=batch)

    def _build_optimizer(self, name: str, lr: float):
        # All parameters share the same lr; D-FINE's param-group logic
        # (backbone vs encoder/decoder) is reserved for a future iteration.
        if name == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        if name == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        raise ValueError(f"Unknown optimizer: {name}")

    def _build_scheduler(self, opt, epochs: int, lrf: float):
        # Linear decay: lr starts at lr0, ends at lr0*lrf after `epochs` steps.
        return torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0, end_factor=lrf, total_iters=epochs
        )

    def _build_criterion(self):
        from dfine.nn.criterion import build_criterion

        return build_criterion(self.cfg)

    def _build_ema(self, decay: float) -> ModelEMA:
        return ModelEMA(self.model, decay=decay)

    def _load_resume_state(self, save_dir: Path) -> dict[str, object]:
        checkpoint_path = save_dir / "last.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"resume=True requested but no checkpoint was found at '{checkpoint_path}'"
            )
        return load_checkpoint_state(checkpoint_path)

    def _build_train_args(
        self,
        data: str,
        epochs: int,
        imgsz: int,
        batch: int,
        lr0: float,
        lrf: float,
        optimizer: str,
        resume: bool,
        amp: bool,
        ema: bool,
        ema_decay: float,
        project: str,
        name: str,
        verbose: bool,
    ) -> dict[str, object]:
        return {
            "data": data,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "lr0": lr0,
            "lrf": lrf,
            "optimizer": optimizer,
            "resume": resume,
            "amp": amp,
            "ema": ema,
            "ema_decay": ema_decay,
            "project": project,
            "name": name,
            "verbose": verbose,
        }

    def _resolve_resume_args(
        self,
        resume_state: dict[str, object],
        data: str,
        imgsz: int,
        batch: int,
        lr0: float,
        lrf: float,
        optimizer: str,
        amp: bool,
        ema: bool,
        ema_decay: float,
    ) -> tuple[str, int, int, float, float, str, bool, bool, float]:
        training_state = resume_state.get("training_state")
        if not isinstance(training_state, dict):
            return data, imgsz, batch, lr0, lrf, optimizer, amp, ema, ema_decay

        train_args = training_state.get("train_args")
        if not isinstance(train_args, dict):
            return data, imgsz, batch, lr0, lrf, optimizer, amp, ema, ema_decay

        current_args: dict[str, object] = {
            "data": data,
            "imgsz": imgsz,
            "batch": batch,
            "lr0": lr0,
            "lrf": lrf,
            "optimizer": optimizer,
            "amp": amp,
            "ema": ema,
            "ema_decay": ema_decay,
        }
        resolved = current_args.copy()

        for key in current_args:
            if key in train_args and train_args[key] != current_args[key]:
                LOGGER.warning(
                    "resume=True ignored %s=%r and restored saved value %r",
                    key,
                    current_args[key],
                    train_args[key],
                )
                resolved[key] = train_args[key]

        return (
            str(resolved["data"]),
            _as_int(resolved["imgsz"]),
            _as_int(resolved["batch"]),
            _as_float(resolved["lr0"]),
            _as_float(resolved["lrf"]),
            str(resolved["optimizer"]),
            bool(resolved["amp"]),
            bool(resolved["ema"]),
            _as_float(resolved["ema_decay"]),
        )

    def _restore_training_state(
        self,
        resume_state: dict[str, object],
        optimizer,
        scheduler,
        scaler,
        ema_model: ModelEMA | None,
        epochs: int,
    ) -> tuple[list[dict[str, float | int]], float, int]:
        training_state = resume_state.get("training_state")
        if not isinstance(training_state, dict):
            self.model.load_state_dict(resume_state["model"])
            self.model.to(self.device)
            return [], float("-inf"), _as_int(resume_state.get("epoch", 0))

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
            if hasattr(scheduler, "total_iters"):
                scheduler.total_iters = epochs

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

        history_raw = training_state.get("history", [])
        history = self._coerce_history(history_raw)
        best_fitness = _as_float(training_state.get("best_fitness", float("-inf")), float("-inf"))
        start_epoch = _as_int(resume_state.get("epoch", 0))
        return history, best_fitness, start_epoch

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
        }
        if scaler is not None:
            training_state["scaler"] = scaler.state_dict()
        if ema_model is not None:
            training_state["raw_model"] = self.model.state_dict()
            training_state["ema"] = {
                "state_dict": ema_model.ema.state_dict(),
                "decay": ema_model.decay,
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

    def _finalize_metrics(self, history: list[dict[str, float | int]]) -> dict:
        if history:
            final_row = history[-1]
            return {
                "loss": float(final_row["loss"]),
                "fitness": float(final_row.get("fitness", 0.0)),
                "mAP50": float(final_row.get("mAP50", 0.0)),
                "mAP50-95": float(final_row.get("mAP50-95", 0.0)),
                "history": history,
            }

        return {"loss": 0.0, "fitness": 0.0, "mAP50": 0.0, "mAP50-95": 0.0, "history": []}

    def _validate_epoch(
        self,
        model,
        data: str,
        imgsz: int,
        batch: int,
        save_dir: Path,
        verbose: bool,
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
            plots=True,
        )

    def _primary_loss_stats(self, train_stats: dict[str, float]) -> dict[str, float]:
        return {
            key: float(train_stats[key]) for key in self._display_loss_keys if key in train_stats
        }

    def _compact_val_metrics(self, val_metrics: dict[str, object]) -> dict[str, float]:
        return {
            key: _as_float(val_metrics.get(key, 0.0))
            for key in ("precision", "recall", "mAP50", "mAP50-95", "fitness")
        }

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
        parts = [
            f"Epoch {epoch}/{epochs}",
            f"loss={float(row.get('loss', 0.0)):.4f}",
            f"P={float(row.get('precision', 0.0)):.3f}",
            f"R={float(row.get('recall', 0.0)):.3f}",
            f"mAP50={float(row.get('mAP50', 0.0)):.3f}",
            f"mAP50-95={float(row.get('mAP50-95', 0.0)):.3f}",
            f"fitness={float(row.get('fitness', 0.0)):.3f}",
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
