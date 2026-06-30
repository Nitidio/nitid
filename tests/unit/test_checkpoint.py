"""Unit tests for checkpoint serialisation."""

import pytest
import torch

from dfine.utils.checkpoint import load_checkpoint, save_checkpoint


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
    save_checkpoint(path, model, cfg={"model": {"num_classes": 80}},
                    names={0: "person"}, epoch=5)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert "model" in ckpt
    assert "config" in ckpt
    assert "names" in ckpt
    assert ckpt["epoch"] == 5
    assert ckpt["names"][0] == "person"


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
        def forward(self, x): return self.l(x)

    path = tmp_path / "raw.pth"
    torch.save({"model": TinyModel().state_dict()}, str(path))

    with pytest.raises(KeyError, match="config"):
        load_checkpoint(path)
