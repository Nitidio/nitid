"""Unit tests for the optional MLflow callback."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from dfine.integrations import MLflowCallback


class FakeMLflow:
    def __init__(self):
        self.uri = None
        self.experiment = None
        self.autolog_calls = 0
        self.active = None
        self.start_calls = []
        self.params = None
        self.metrics = []
        self.artifacts = []
        self.end_statuses = []
        self.fail_init = False
        self.fail_metrics = False

    def set_tracking_uri(self, uri):
        self.uri = uri

    def set_experiment(self, experiment):
        self.experiment = experiment

    def autolog(self):
        self.autolog_calls += 1

    def active_run(self):
        return self.active

    def start_run(self, **kwargs):
        self.start_calls.append(kwargs)
        run_id = kwargs.get("run_id", "new-run-id")
        self.active = SimpleNamespace(info=SimpleNamespace(run_id=run_id))
        return self.active

    def log_params(self, params):
        if self.fail_init:
            raise RuntimeError("parameter failure")
        self.params = params

    def log_metrics(self, metrics, step):
        if self.fail_metrics:
            raise RuntimeError("metric failure")
        self.metrics.append((metrics, step))

    def log_artifact(self, path):
        self.artifacts.append(path)

    def get_tracking_uri(self):
        return self.uri

    def end_run(self, status=None):
        self.end_statuses.append(status)
        self.active = None


@pytest.fixture
def trainer(tmp_path):
    for filename in ("last.pth", "best.pth", "results.csv", "results.png", "ignored.txt"):
        (tmp_path / filename).write_text("artifact")
    return SimpleNamespace(
        train_args={"project": "runs/train", "name": "experiment", "epochs": 2},
        tracking_state={},
        current_epoch=1,
        current_row={"epoch": 1, "loss(box)": 2.5, "mAP50": 0.6, "label": "ignored"},
        save_dir=tmp_path,
    )


def test_mlflow_logs_params_metrics_artifacts_and_closes_owned_run(monkeypatch, trainer, tmp_path):
    fake = FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    callback = MLflowCallback(
        tracking_uri=str(tmp_path / "mlruns"),
        experiment_name="detectors",
        run_name="baseline",
    )

    callback.on_train_start(trainer)
    callback.on_train_epoch_end(trainer)
    callback.on_train_end(trainer)

    assert fake.uri == str(tmp_path / "mlruns")
    assert fake.experiment == "detectors"
    assert fake.start_calls == [{"run_name": "baseline"}]
    assert fake.params == trainer.train_args
    assert fake.metrics == [({"epoch": 1.0, "lossbox": 2.5, "mAP50": 0.6}, 0)]
    assert {path.rsplit("/", 1)[-1] for path in fake.artifacts} == {
        "last.pth",
        "best.pth",
        "results.csv",
        "results.png",
    }
    assert fake.end_statuses == [None]
    assert trainer.tracking_state == {"mlflow": {"run_id": "new-run-id"}}


def test_mlflow_environment_variables_take_precedence(monkeypatch, trainer):
    fake = FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracker:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "environment-project")
    monkeypatch.setenv("MLFLOW_RUN", "environment-run")

    MLflowCallback(
        tracking_uri="ignored", experiment_name="ignored", run_name="ignored"
    ).on_train_start(trainer)

    assert fake.uri == "http://tracker:5000"
    assert fake.experiment == "environment-project"
    assert fake.start_calls == [{"run_name": "environment-run"}]


def test_mlflow_reuses_external_active_run_without_closing(monkeypatch, trainer):
    fake = FakeMLflow()
    fake.active = SimpleNamespace(info=SimpleNamespace(run_id="external-run"))
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    callback = MLflowCallback()

    callback.on_train_start(trainer)
    callback.on_train_end(trainer)

    assert fake.start_calls == []
    assert fake.end_statuses == []
    assert trainer.tracking_state == {"mlflow": {"run_id": "external-run"}}


def test_mlflow_resume_reconnects_saved_run(monkeypatch, trainer):
    fake = FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    trainer.tracking_state = {"mlflow": {"run_id": "saved-run"}}

    MLflowCallback().on_train_start(trainer)

    assert fake.start_calls == [{"run_id": "saved-run"}]
    assert fake.params is None


@pytest.mark.parametrize("value", ["1", "true", "YES", "on", "y", "T"])
def test_mlflow_keep_run_active_matches_ultralytics_truthy_values(monkeypatch, trainer, value):
    fake = FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_KEEP_RUN_ACTIVE", value)
    callback = MLflowCallback()

    callback.on_train_start(trainer)
    callback.on_train_end(trainer)

    assert fake.end_statuses == []
    assert fake.active is not None


def test_mlflow_initialization_failure_warns_and_does_not_raise(monkeypatch, trainer, caplog):
    fake = FakeMLflow()
    fake.fail_init = True
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    MLflowCallback().on_train_start(trainer)

    assert "Not tracking this run" in caplog.text
    assert fake.end_statuses == [None]


def test_mlflow_metric_failure_disables_logging_without_raising(monkeypatch, trainer, caplog):
    fake = FakeMLflow()
    fake.fail_metrics = True
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    callback = MLflowCallback()
    callback.on_train_start(trainer)

    callback.on_train_epoch_end(trainer)

    assert "disabling tracking" in caplog.text
    assert callback._active is False


def test_mlflow_training_error_marks_owned_run_failed(monkeypatch, trainer):
    fake = FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    callback = MLflowCallback()
    callback.on_train_start(trainer)

    callback.on_train_error(trainer)

    assert fake.end_statuses == ["FAILED"]
    assert callback._active is False


def test_mlflow_missing_dependency_warns_without_raising(monkeypatch, trainer, caplog):
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)

    def missing_mlflow(name):
        if name == "mlflow":
            raise ImportError("missing")
        raise AssertionError(name)

    monkeypatch.setattr("dfine.integrations.mlflow.importlib.import_module", missing_mlflow)

    MLflowCallback().on_train_start(trainer)

    assert "not installed" in caplog.text


def test_mlflow_real_local_file_backend(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    mlflow.end_run()
    save_dir = tmp_path / "training"
    save_dir.mkdir()
    (save_dir / "last.pth").write_bytes(b"checkpoint")
    local_trainer = SimpleNamespace(
        train_args={"project": "runs/train", "name": "local-smoke", "epochs": 1},
        tracking_state={},
        current_epoch=1,
        current_row={"epoch": 1, "loss": 1.25, "mAP50": 0.5},
        save_dir=save_dir,
    )
    tracking_dir = tmp_path / "mlruns"
    callback = MLflowCallback(
        tracking_uri=str(tracking_dir),
        experiment_name="nitid-test",
        run_name="local-smoke",
        autolog=False,
    )

    callback.on_train_start(local_trainer)
    run_id = local_trainer.tracking_state["mlflow"]["run_id"]
    callback.on_train_epoch_end(local_trainer)
    callback.on_train_end(local_trainer)

    run = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_dir)).get_run(run_id)
    assert run.data.params["epochs"] == "1"
    assert run.data.metrics["loss"] == pytest.approx(1.25)
    artifacts = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_dir)).list_artifacts(run_id)
    assert {artifact.path for artifact in artifacts} == {"last.pth"}
