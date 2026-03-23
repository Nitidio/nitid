"""
DFINETrainer — fine-tuning engine.
Called internally by DFINE.train(). Not part of the public API.
"""
from __future__ import annotations

from pathlib import Path

import torch
from dfine.utils.checkpoint import save_checkpoint
from dfine.utils.logging import LOGGER


class DFINETrainer:
    """
    Fine-tuning loop for D-FINE.

    Loads a COCO-format dataset, trains for the requested number of epochs
    with gradient clipping, and saves a wrapped checkpoint after each epoch.
    The criterion and weight dict come from the checkpoint's embedded config
    so loss weighting stays consistent with the original training setup.
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
        project: str,
        name: str,
        verbose: bool,
    ) -> dict:
        """
        Run the fine-tuning loop.

        Args:
            data:      Path to the data YAML (ultralytics-style).
            epochs:    Number of training epochs.
            imgsz:     Input image size (square).
            batch:     Batch size.
            lr0:       Initial learning rate.
            lrf:       Final LR as a fraction of lr0 (linear decay).
            optimizer: ``"AdamW"`` or ``"SGD"``.
            resume:    Reserved for future use (checkpoint resume).
            project:   Root output directory.
            name:      Run name; checkpoints saved to ``<project>/<name>/``.
            verbose:   Print per-epoch loss.

        Returns:
            Metrics dict with at least ``{"loss": <final_epoch_loss>}``.
        """
        save_dir = Path(project) / name
        save_dir.mkdir(parents=True, exist_ok=True)

        dataloader = self._build_dataloader(data, imgsz, batch)
        opt = self._build_optimizer(optimizer, lr0)
        scheduler = self._build_scheduler(opt, epochs, lrf)
        criterion = self._build_criterion()

        self.model.train()
        criterion.train()
        weight_dict = criterion.weight_dict
        metrics: dict = {}

        for epoch in range(epochs):
            epoch_loss = 0.0

            for images, targets in dataloader:
                images = images.to(self.device)
                targets = [
                    {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in t.items()}
                    for t in targets
                ]

                opt.zero_grad()
                outputs = self.model(images, targets=targets)
                loss_dict = criterion(outputs, targets)
                loss = sum(
                    loss_dict[k] * weight_dict[k]
                    for k in loss_dict if k in weight_dict
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                opt.step()
                epoch_loss += loss.item()

            scheduler.step()
            metrics["loss"] = epoch_loss

            if verbose:
                LOGGER.info(f"Epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}")

            save_checkpoint(
                save_dir / f"epoch{epoch + 1}.pth",
                self.model,
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
