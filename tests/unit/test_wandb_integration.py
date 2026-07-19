"""Unit tests for the optional Weights & Biases callback."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from dfine.integrations import WandbCallback


class FakeArtifact:
    def __init__(self, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path, name):
        self.files.append((path, name))


class FakeRun:
    def __init__(self):
        self.id = "wandb-run-123"
        self.name = "test-run"
        self.logged = []
        self.artifacts = []
        self.summary = {}
        self.finish_exit_codes = []

    def log(self, values, step):
        self.logged.append((values, step))

    def log_artifact(self, artifact, aliases):
        self.artifacts.append((artifact, aliases))

    def finish(self, exit_code=0):
        self.finish_exit_codes.append(exit_code)


class FakeWandb:
    Artifact = FakeArtifact

    def __init__(self):
        self.run = FakeRun()
        self.init_options = None

    def init(self, **kwargs):
        self.init_options = kwargs
        return self.run


@pytest.fixture
def trainer(tmp_path):
    return SimpleNamespace(
        train_args={"epochs": 2, "batch": 4, "name": "nitid-run"},
        tracking_state={},
        current_epoch=1,
        current_row={"epoch": 1, "loss": 2.5, "loss_bbox": 0.4, "mAP50": 0.6},
        save_dir=tmp_path,
        metrics={"loss": 2.5, "mAP50": 0.6, "history": [{"epoch": 1}]},
    )


def test_wandb_callback_logs_config_metrics_checkpoints_and_summary(monkeypatch, trainer):
    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    for filename in ("epoch1.pth", "last.pth", "best.pth"):
        (trainer.save_dir / filename).write_bytes(b"checkpoint")

    callback = WandbCallback(project="detectors", entity="team", tags=["test"], mode="offline")
    callback.on_train_start(trainer)
    callback.on_train_epoch_end(trainer)
    assert fake_wandb.run.artifacts == []
    callback.on_train_end(trainer)

    assert fake_wandb.init_options == {
        "project": "detectors",
        "name": "nitid-run",
        "config": trainer.train_args,
        "entity": "team",
        "tags": ["test"],
        "mode": "offline",
    }
    assert fake_wandb.run.logged == [(trainer.current_row, 1)]
    artifact, aliases = fake_wandb.run.artifacts[0]
    assert artifact.type == "model"
    assert artifact.metadata == {"epoch": 1}
    assert {name for _, name in artifact.files} == {"last.pth", "best.pth"}
    assert aliases == ["latest", "best"]
    assert trainer.tracking_state == {"wandb": {"run_id": "wandb-run-123"}}
    assert fake_wandb.run.summary == {"loss": 2.5, "mAP50": 0.6}
    assert fake_wandb.run.finish_exit_codes == [0]


def test_wandb_callback_can_skip_checkpoint_artifacts(monkeypatch, trainer):
    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    callback = WandbCallback(project="detectors", log_checkpoints=False)

    callback.on_train_start(trainer)
    callback.on_train_epoch_end(trainer)

    assert fake_wandb.run.artifacts == []


def test_wandb_callback_uploads_periodic_epoch_checkpoint(monkeypatch, trainer):
    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    (trainer.save_dir / "epoch2.pth").write_bytes(b"checkpoint")
    trainer.current_epoch = 2
    trainer.current_row["epoch"] = 2
    callback = WandbCallback(project="detectors", checkpoint_interval=2)

    callback.on_train_start(trainer)
    callback.on_train_epoch_end(trainer)

    artifact, aliases = fake_wandb.run.artifacts[0]
    assert [name for _, name in artifact.files] == ["epoch2.pth"]
    assert aliases == ["epoch-2"]


def test_wandb_callback_reconnects_resumed_run(monkeypatch, trainer):
    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    trainer.tracking_state = {"wandb": {"run_id": "existing-run"}}

    WandbCallback(project="detectors").on_train_start(trainer)

    assert fake_wandb.init_options["id"] == "existing-run"
    assert fake_wandb.init_options["resume"] == "allow"


def test_wandb_callback_finishes_failed_run(monkeypatch, trainer):
    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    callback = WandbCallback(project="detectors")
    callback.on_train_start(trainer)

    callback.on_train_error(trainer)

    assert fake_wandb.run.finish_exit_codes == [1]
    assert callback._run is None


def test_wandb_callback_rejects_invalid_checkpoint_interval():
    with pytest.raises(ValueError, match="checkpoint_interval"):
        WandbCallback(checkpoint_interval=0)


def test_wandb_callback_explains_missing_optional_dependency(monkeypatch, trainer):
    monkeypatch.delitem(sys.modules, "wandb", raising=False)

    def missing_wandb(name):
        if name == "wandb":
            raise ImportError("missing")
        raise AssertionError(name)

    monkeypatch.setattr("dfine.integrations.wandb.importlib.import_module", missing_wandb)

    with pytest.raises(ImportError, match=r"nitid\[wandb\]"):
        WandbCallback(project="detectors").on_train_start(trainer)
