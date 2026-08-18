"""Unit tests for checkpoint serialisation."""

import pytest
import torch

from dfine.utils.checkpoint import CHECKPOINT_FORMAT_VERSION, load_checkpoint, save_checkpoint


def test_save_checkpoint_structure(tmp_path):
    """Verify checkpoint contains all required keys."""
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(2, 2)

        def forward(self, x):
            return self.l(x)

    model = TinyModel()
    path = tmp_path / "test.pth"
    save_checkpoint(path, model, cfg={"model": {"num_classes": 80}}, names={0: "person"}, epoch=5)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert "model" in ckpt
    assert "config" in ckpt
    assert "names" in ckpt
    assert "training_state" in ckpt
    assert ckpt["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert ckpt["task"] == "detect"
    assert ckpt["config"]["task"] == "detect"
    assert ckpt["epoch"] == 5
    assert ckpt["names"][0] == "person"


def test_save_checkpoint_includes_optional_training_state(tmp_path):
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(2, 2)

        def forward(self, x):
            return self.l(x)

    model = TinyModel()
    path = tmp_path / "training_state.pth"
    save_checkpoint(
        path,
        model,
        cfg={"model": {"num_classes": 80}},
        names={0: "person"},
        training_state={"history": [{"epoch": 1, "loss": 1.23}]},
    )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["training_state"]["history"][0]["epoch"] == 1


def test_save_checkpoint_canonicalizes_semantic_task_without_mutating_input(tmp_path):
    import torch.nn as nn

    model = nn.Linear(2, 2)
    config = {"task": "sem_seg", "num_classes": 3}
    path = tmp_path / "semantic.pth"

    save_checkpoint(path, model, cfg=config, names={0: "road"})

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["task"] == "semantic"
    assert checkpoint["config"]["task"] == "semantic"
    assert config["task"] == "sem_seg"


def test_semantic_checkpoint_round_trip_builds_native_architecture(tmp_path):
    from dfine.nn.configs import make_model_config
    from dfine.nn.native_build import build_native_model_from_config

    config = make_model_config("dfine_n", task="semantic", num_classes=3, image_size=(64, 64))
    model = build_native_model_from_config(config)
    path = tmp_path / "semantic_native.pth"
    save_checkpoint(
        path,
        model,
        cfg=config,
        names={0: "road", 1: "vehicle", 2: "person"},
    )

    loaded, loaded_config, names = load_checkpoint(path)

    assert loaded_config["task"] == "semantic"
    assert loaded.decoder.__class__.__name__ == "SemSegDecoder"
    assert names == {0: "road", 1: "vehicle", 2: "person"}
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for name, value in loaded.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])


def test_load_checkpoint_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nonexistent.pth")


def test_load_checkpoint_missing_config_key(tmp_path):
    """A raw (unconverted) checkpoint missing 'config' should raise KeyError."""
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(2, 2)

        def forward(self, x):
            return self.l(x)

    path = tmp_path / "raw.pth"
    torch.save({"model": TinyModel().state_dict()}, str(path))

    with pytest.raises(KeyError, match="config"):
        load_checkpoint(path)


def test_load_checkpoint_rejects_conflicting_task_metadata(tmp_path):
    path = tmp_path / "conflicting.pth"
    torch.save(
        {
            "task": "segment",
            "config": {"task": "detect"},
            "model": {},
        },
        path,
    )

    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(path)


def test_load_checkpoint_rejects_newer_format_version(tmp_path):
    path = tmp_path / "future.pth"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION + 1,
            "task": "detect",
            "config": {"task": "detect"},
            "model": {},
        },
        path,
    )

    with pytest.raises(ValueError, match="newer than supported"):
        load_checkpoint(path)
