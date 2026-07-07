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
from pathlib import Path

import torch
from tqdm.auto import tqdm

from dfine.utils.checkpoint import save_checkpoint
from dfine.utils.logging import LOGGER


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
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
    """

    def __init__(self, model, cfg: dict, device: str, names: dict) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.names = names
        self._display_loss_keys = ("loss_bbox", "loss_giou", "loss_vfl", "loss_fgl")

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
            resume:     Reserved for future use (checkpoint resume).
            amp:        Enable AMP mixed-precision (CUDA only).
            ema:        Enable EMA weight averaging.
            ema_decay:  EMA decay factor (ignored when ``ema=False``).
            project:    Root output directory.
            name:       Run name; checkpoints saved to ``<project>/<name>/``.
            verbose:    Print per-epoch loss.

        Returns:
            Metrics dict containing final scalar metrics plus a ``history``
            list with one metrics row per epoch.
        """
        save_dir = Path(project) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        results_path = save_dir / "results.csv"

        dataloader = self._build_dataloader(data, imgsz, batch)
        opt = self._build_optimizer(optimizer, lr0)
        scheduler = self._build_scheduler(opt, epochs, lrf)
        criterion = self._build_criterion()

        # AMP: only meaningful on CUDA
        is_cuda = self.device.startswith("cuda")
        if amp and not is_cuda:
            LOGGER.warning("amp=True ignored — AMP requires a CUDA device, got %s", self.device)
            amp = False
        scaler = torch.cuda.amp.GradScaler() if amp else None
        device_type = self.device.split(":")[0]  # "cuda" or "cpu"

        # EMA
        ema_model = self._build_ema(ema_decay) if ema else None

        criterion.train()
        weight_dict = criterion.weight_dict
        history: list[dict[str, float | int]] = []
        best_fitness = float("-inf")

        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            batch_count = 0
            instances = 0
            loss_sums: dict[str, float] = defaultdict(float)

            if is_cuda:
                torch.cuda.reset_peak_memory_stats()

            self.model.train()
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
            )

            fitness = float(row.get("fitness", 0.0))
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

            if verbose:
                LOGGER.info(self._format_epoch_row(epoch + 1, epochs, row))

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
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

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
