"""
DFINETrainer — fine-tuning engine.
Called internally by DFINE.train(). Not part of the public API.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from dfine.utils.checkpoint import save_checkpoint
from dfine.utils.logging import LOGGER


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
            Metrics dict with at least ``{"loss": <final_epoch_loss>}``.
        """
        save_dir = Path(project) / name
        save_dir.mkdir(parents=True, exist_ok=True)

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

        self.model.train()
        criterion.train()
        weight_dict = criterion.weight_dict
        metrics: dict = {}

        for epoch in range(epochs):
            epoch_loss = 0.0

            for images, targets in dataloader:
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

            scheduler.step()
            metrics["loss"] = epoch_loss

            if verbose:
                LOGGER.info(f"Epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}")

            # Save EMA weights when available — they are what gets loaded by DFINE(path)
            save_model = ema_model.ema if ema_model is not None else self.model
            save_checkpoint(
                save_dir / f"epoch{epoch + 1}.pth",
                save_model,
                self.cfg,
                self.names,
                epoch=epoch + 1,
                metrics=metrics,
            )

        return metrics

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
