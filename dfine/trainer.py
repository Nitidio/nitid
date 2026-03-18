"""
DFINETrainer — fine-tuning engine.
Called internally by DFINE.train(). Not part of the public API.
"""
from __future__ import annotations
from pathlib import Path
import torch
from dfine.utils.logging import LOGGER


class DFINETrainer:
    def __init__(self, model, cfg: dict, device: str) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device

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
        Core training loop.

        Pattern:
            for epoch in range(epochs):
                for batch in dataloader:
                    optimizer.zero_grad()
                    loss = criterion(model(batch))
                    loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=0.1)
                    optimizer.step()
                    ema.update()
                scheduler.step()
                save_checkpoint(...)
        """
        save_dir = Path(project) / name
        save_dir.mkdir(parents=True, exist_ok=True)

        dataloader = self._build_dataloader(data, imgsz, batch)
        opt = self._build_optimizer(optimizer, lr0)
        scheduler = self._build_scheduler(opt, epochs, lrf)
        criterion = self._build_criterion()

        self.model.train()
        metrics = {}

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_data in dataloader:
                opt.zero_grad()
                loss = criterion(self.model, batch_data)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                opt.step()
                epoch_loss += loss.item()
            scheduler.step()
            if verbose:
                LOGGER.info(f"Epoch {epoch+1}/{epochs}  loss={epoch_loss:.4f}")

        return metrics

    def _build_dataloader(self, data: str, imgsz: int, batch: int):
        # TODO: implement COCO-style dataset loader
        raise NotImplementedError

    def _build_optimizer(self, name: str, lr: float):
        if name == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        if name == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        raise ValueError(f"Unknown optimizer: {name}")

    def _build_scheduler(self, opt, epochs: int, lrf: float):
        # Linear decay from lr0 to lr0*lrf over all epochs
        return torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0, end_factor=lrf, total_iters=epochs
        )

    def _build_criterion(self):
        from dfine.nn.criterion import build_criterion
        return build_criterion(self.cfg)
