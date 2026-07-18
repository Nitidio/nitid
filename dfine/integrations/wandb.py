"""Weights & Biases integration for :class:`dfine.trainer.DFINETrainer`."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


class WandbCallback:
    """Log a nitid training run to Weights & Biases.

    Pass an instance to ``DFINE.train(callbacks=...)``. The WandB package is
    imported only when training starts, so it remains an optional dependency.

    Args:
        project: WandB project name. Defaults to ``"nitid"``.
        name: Optional WandB run name. Defaults to the nitid training run name.
        entity: Optional WandB team or entity.
        tags: Optional run tags.
        notes: Optional run notes.
        mode: WandB mode, such as ``"online"``, ``"offline"``, or ``"disabled"``.
        log_checkpoints: Upload final ``last.pth`` and ``best.pth`` artifacts.
        checkpoint_interval: Additionally upload ``epochN.pth`` every N epochs.
        init_kwargs: Additional keyword arguments passed to ``wandb.init``.
    """

    def __init__(
        self,
        project: str = "nitid",
        name: str | None = None,
        entity: str | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
        notes: str | None = None,
        mode: str | None = None,
        log_checkpoints: bool = True,
        checkpoint_interval: int | None = None,
        **init_kwargs: Any,
    ) -> None:
        self.project = project
        self.name = name
        self.entity = entity
        self.tags = list(tags) if tags is not None else None
        self.notes = notes
        self.mode = mode
        self.log_checkpoints = log_checkpoints
        if checkpoint_interval is not None and checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be a positive integer or None")
        self.checkpoint_interval = checkpoint_interval
        self.init_kwargs = init_kwargs
        self._wandb: Any = None
        self._run: Any = None

    def on_train_start(self, trainer: Any) -> None:
        """Create the WandB run and record resolved training hyperparameters."""
        try:
            self._wandb = importlib.import_module("wandb")
        except ImportError as exc:
            raise ImportError(
                "Weights & Biases logging requires the optional 'wandb' dependency. "
                "Install it with `pip install nitid[wandb]` or `uv sync --extra wandb`."
            ) from exc

        options = dict(self.init_kwargs)
        options.update(
            {
                "project": self.project,
                "name": self.name or trainer.train_args.get("name"),
                "config": dict(trainer.train_args),
            }
        )
        if self.entity is not None:
            options["entity"] = self.entity
        if self.tags is not None:
            options["tags"] = self.tags
        if self.notes is not None:
            options["notes"] = self.notes
        if self.mode is not None:
            options["mode"] = self.mode

        tracking_state = trainer.tracking_state.get("wandb")
        if isinstance(tracking_state, dict):
            run_id = tracking_state.get("run_id")
            if isinstance(run_id, str) and run_id and "id" not in options:
                options["id"] = run_id
                options.setdefault("resume", "allow")

        self._run = self._wandb.init(**options)
        if self._run is None:
            raise RuntimeError("wandb.init() did not return a run")
        run_id = getattr(self._run, "id", None)
        if isinstance(run_id, str) and run_id:
            trainer.tracking_state["wandb"] = {"run_id": run_id}

    def on_train_epoch_end(self, trainer: Any) -> None:
        """Log finalized losses, validation metrics, and checkpoint artifacts."""
        if self._run is None or trainer.current_row is None:
            return

        row = dict(trainer.current_row)
        step = int(row.get("epoch", trainer.current_epoch))
        self._run.log(row, step=step)

        if (
            self.log_checkpoints
            and self.checkpoint_interval is not None
            and step % self.checkpoint_interval == 0
            and trainer.save_dir is not None
        ):
            path = Path(trainer.save_dir) / f"epoch{step}.pth"
            self._log_checkpoints([path], step, [f"epoch-{step}"])

    def on_train_end(self, trainer: Any) -> None:
        """Write final summary values and close the WandB run."""
        if self._run is None:
            return

        if trainer.metrics is not None:
            for key, value in trainer.metrics.items():
                if key != "history" and isinstance(value, (int, float)):
                    self._run.summary[key] = value
        if self.log_checkpoints and trainer.save_dir is not None:
            save_dir = Path(trainer.save_dir)
            self._log_checkpoints(
                [save_dir / "last.pth", save_dir / "best.pth"],
                trainer.current_epoch,
                ["latest", "best"],
            )
        self._run.finish()
        self._run = None

    def on_train_error(self, trainer: Any) -> None:
        """Mark an interrupted WandB run failed and release its resources."""
        if self._run is None:
            return
        self._run.finish(exit_code=1)
        self._run = None

    def _log_checkpoints(self, paths: list[Path], epoch: int, aliases: list[str]) -> None:
        existing = [path for path in paths if path.is_file()]
        if not existing:
            return

        run_name = getattr(self._run, "name", None) or self.name or "nitid"
        artifact = self._wandb.Artifact(
            name=f"{run_name}-checkpoints",
            type="model",
            metadata={"epoch": epoch},
        )
        for path in existing:
            artifact.add_file(str(path), name=path.name)
        self._run.log_artifact(artifact, aliases=aliases)
