"""Unit tests for the /api/v1/health endpoint and the app factory."""

import pytest

pytest.importorskip("fastapi", reason="needs the 'studio' extra")

from starlette.testclient import TestClient  # noqa: E402

import nitid  # noqa: E402
from nitid.studio.app import create_app  # noqa: E402
from nitid.studio.settings import StudioSettings  # noqa: E402

PORT = 8125


@pytest.fixture
def client(tmp_path):
    settings = StudioSettings(port=PORT, token="test-token-123456", models_dir=tmp_path / "m")
    return TestClient(
        create_app(settings, static_dir=tmp_path), base_url=f"http://127.0.0.1:{PORT}"
    )


def test_health_payload(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "nitid_version": nitid.__version__,
        "api_version": 1,
    }


def test_health_is_get_only(client):
    assert (
        client.post(
            "/api/v1/health", headers={"Authorization": "Bearer test-token-123456"}
        ).status_code
        == 405
    )


def test_app_keeps_settings(tmp_path):
    settings = StudioSettings(port=PORT, token="test-token-123456", models_dir=tmp_path)
    assert create_app(settings, static_dir=tmp_path).state.settings is settings
