"""Isolates web/api's module-level settings/engine/storage per test session.

web/api/config.py and web/api/database.py build a Settings() singleton and a
SQLAlchemy engine at import time, so the NITID_* env vars that redirect them
to a temporary database/storage tree must be set before web.api.* is first
imported anywhere in the test process.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_tmp_root = Path(tempfile.mkdtemp(prefix="nitid-web-test-"))
os.environ["NITID_DATABASE_URL"] = f"sqlite:///{_tmp_root / 'test.db'}"
os.environ["NITID_MODELS_DIR"] = str(_tmp_root / "models")
os.environ["NITID_UPLOADS_DIR"] = str(_tmp_root / "uploads")
os.environ["NITID_RESULTS_DIR"] = str(_tmp_root / "results")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def web_tmp_root() -> Path:
    return _tmp_root


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from web.api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def auth_headers(client):
    client.post("/auth/register", json={"username": "tester", "password": "test-password"})
    response = client.post("/auth/login", data={"username": "tester", "password": "test-password"})
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="session")
def tiny_web_checkpoint(web_tmp_root):
    from dfine.nn.build import build_model
    from dfine.nn.configs import make_detection_config
    from dfine.utils.checkpoint import save_checkpoint

    cfg = make_detection_config("dfine_s")
    cfg["DFINETransformer"]["num_layers"] = 1
    cfg["DFINETransformer"]["num_queries"] = 10
    cfg["DFINETransformer"]["num_denoising"] = 0
    cfg["HybridEncoder"]["depth_mult"] = 0.1

    model = build_model(cfg)
    model.eval()
    names = {i: f"class{i}" for i in range(cfg.get("num_classes", 80))}

    models_dir = Path(os.environ["NITID_MODELS_DIR"])
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / "tiny_web.pth"
    save_checkpoint(checkpoint_path, model, cfg, names)
    return checkpoint_path.name
