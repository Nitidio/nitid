"""MLflow integration matching Ultralytics' training-logger behavior."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from dfine.utils.logging import LOGGER

_PREFIX = "MLflow: "
_ARTIFACT_SUFFIXES = {".png", ".jpg", ".csv", ".pth", ".yaml", ".yml"}
_TRUE_VALUES = {"1", "true", "yes", "on", "y", "t"}


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _local_store(uri: str) -> Path | None:
    """Return the directory of a tracking URI given as a plain local path.

    MLflow 3 refuses its file-store backend by default, so plain paths are
    backed by a SQLite database inside that directory instead. URIs with an
    explicit scheme (``http://``, ``sqlite://``, ``file:``, ``databricks``) are
    passed to MLflow unchanged.
    """
    if "://" in uri or uri.startswith("file:") or uri == "databricks":
        return None
    return Path(uri).expanduser().resolve()


def _sanitize_metrics(metrics: dict[str, object]) -> dict[str, float]:
    """Remove parentheses from metric names and retain numeric values."""
    sanitized: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            sanitized[key.replace("(", "").replace(")", "")] = float(value)
        elif isinstance(value, (int, float)):
            sanitized[key.replace("(", "").replace(")", "")] = float(value)
    return sanitized


class MLflowCallback:
    """Log nitid training to MLflow using Ultralytics-compatible conventions.

    Environment variables take precedence over constructor values:
    ``MLFLOW_TRACKING_URI``, ``MLFLOW_EXPERIMENT_NAME``, ``MLFLOW_RUN``, and
    ``MLFLOW_KEEP_RUN_ACTIVE``.

    Initialization and logging errors only disable MLflow for the current run;
    they never interrupt model training, matching Ultralytics' behavior.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
        keep_run_active: bool | None = None,
        autolog: bool = True,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.keep_run_active = keep_run_active
        self.autolog = autolog
        self._mlflow: Any = None
        self._active = False
        self._started_run = False
        self._resumed_run = False

    def on_train_start(self, trainer: Any) -> None:
        """Configure MLflow, start or reuse a run, and log training parameters."""
        self._active = False
        self._started_run = False
        self._resumed_run = False
        try:
            self._mlflow = importlib.import_module("mlflow")
        except ImportError:
            LOGGER.warning(
                "%snot installed; install `nitid[mlflow]` or run `uv sync --extra mlflow`",
                _PREFIX,
            )
            return

        uri = os.environ.get("MLFLOW_TRACKING_URI") or self.tracking_uri or "runs/mlflow"
        store = _local_store(uri)
        experiment = (
            os.environ.get("MLFLOW_EXPERIMENT_NAME")
            or self.experiment_name
            or str(trainer.train_args.get("project") or "/Shared/Ultralytics")
        )
        run_name = (
            os.environ.get("MLFLOW_RUN")
            or self.run_name
            or str(trainer.train_args.get("name") or "exp")
        )

        try:
            if store is not None:
                store.mkdir(parents=True, exist_ok=True)
                uri = f"sqlite:///{(store / 'mlflow.db').as_posix()}"
            self._mlflow.set_tracking_uri(uri)
            if store is not None and self._mlflow.get_experiment_by_name(experiment) is None:
                self._mlflow.create_experiment(
                    experiment, artifact_location=(store / "artifacts").as_uri()
                )
            self._mlflow.set_experiment(experiment)
            if self.autolog:
                self._mlflow.autolog()
            active_run = self._mlflow.active_run()
            if active_run is None:
                active_run = self._start_or_resume_run(trainer, run_name)
                self._started_run = True
            run_id = active_run.info.run_id
            trainer.tracking_state["mlflow"] = {"run_id": run_id}
            if not self._resumed_run:
                self._mlflow.log_params(dict(trainer.train_args))
            self._active = True
            LOGGER.info("%slogging run_id(%s) to %s", _PREFIX, run_id, uri)
            if store is not None:
                LOGGER.info(
                    "%sview at http://127.0.0.1:5000 with `mlflow server --backend-store-uri %s`",
                    _PREFIX,
                    uri,
                )
        except Exception as error:
            LOGGER.warning("%sFailed to initialize: %s", _PREFIX, error)
            LOGGER.warning("%sNot tracking this run", _PREFIX)
            self._end_started_run(force=True)

    def on_train_epoch_end(self, trainer: Any) -> None:
        """Log the finalized training and validation metrics for one epoch."""
        if not self._active or trainer.current_row is None:
            return
        metrics = _sanitize_metrics(dict(trainer.current_row))
        self._log_metrics(metrics, step=max(0, trainer.current_epoch - 1))

    def on_train_end(self, trainer: Any) -> None:
        """Log run artifacts and close a run opened by this callback."""
        if self._active and trainer.save_dir is not None:
            try:
                for path in Path(trainer.save_dir).iterdir():
                    if path.is_file() and path.suffix.lower() in _ARTIFACT_SUFFIXES:
                        self._mlflow.log_artifact(str(path))
                LOGGER.info("%sresults logged to %s", _PREFIX, self._mlflow.get_tracking_uri())
            except Exception as error:
                LOGGER.warning("%sfailed to log artifacts: %s", _PREFIX, error)
        self._end_started_run()

    def on_train_error(self, trainer: Any) -> None:
        """Close a callback-owned run as failed after a training exception."""
        if self._started_run:
            try:
                self._mlflow.end_run(status="FAILED")
            except Exception:
                pass
        self._active = False
        self._started_run = False

    def _start_or_resume_run(self, trainer: Any, run_name: str) -> Any:
        tracking_state = trainer.tracking_state.get("mlflow")
        if isinstance(tracking_state, dict):
            run_id = tracking_state.get("run_id")
            if isinstance(run_id, str) and run_id:
                self._resumed_run = True
                return self._mlflow.start_run(run_id=run_id)
        return self._mlflow.start_run(run_name=run_name)

    def _log_metrics(self, metrics: dict[str, float], step: int) -> None:
        try:
            self._mlflow.log_metrics(metrics=metrics, step=step)
        except Exception as error:
            LOGGER.warning(
                "%smetric logging failed, disabling tracking for this run: %s",
                _PREFIX,
                error,
            )
            self._active = False

    def _keep_active(self) -> bool:
        if "MLFLOW_KEEP_RUN_ACTIVE" in os.environ:
            return _env_bool("MLFLOW_KEEP_RUN_ACTIVE")
        return bool(self.keep_run_active)

    def _end_started_run(self, force: bool = False) -> None:
        if self._started_run:
            if self._keep_active() and not force:
                LOGGER.info("%srun still active; close it with `mlflow.end_run()`", _PREFIX)
            else:
                try:
                    self._mlflow.end_run()
                except Exception:
                    pass
        self._active = False
        self._started_run = False
