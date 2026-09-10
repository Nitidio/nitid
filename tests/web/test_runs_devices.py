"""FastAPI TestClient coverage for web/api/: auth, devices, and run creation
with backend/device selection. First test coverage this app has had."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def _fake_image_bytes(shape=(64, 64, 3), seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, shape, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_register_and_login(client):
    register = client.post(
        "/auth/register", json={"username": "alice", "password": "correct-password"}
    )
    assert register.status_code == 201

    login = client.post("/auth/login", data={"username": "alice", "password": "correct-password"})
    assert login.status_code == 200
    assert "access_token" in login.json()

    bad_login = client.post("/auth/login", data={"username": "alice", "password": "wrong"})
    assert bad_login.status_code == 401


def test_devices_endpoint_is_public_and_reports_real_hardware(client):
    response = client.get("/devices")
    assert response.status_code == 200
    body = response.json()
    assert body["torch"]["cpu"] is True
    assert isinstance(body["openvino"]["available"], bool)
    if body["openvino"]["available"]:
        assert "CPU" in body["openvino"]["devices"]


def test_models_list_requires_auth(client):
    assert client.get("/models").status_code == 401


def test_create_run_on_torch_cpu_completes_with_detections_and_speed(
    client, auth_headers, tiny_web_checkpoint
):
    files = [("files", ("frame.jpg", _fake_image_bytes(), "image/jpeg"))]
    params = {
        "model_name": tiny_web_checkpoint,
        "conf": 0.01,
        "imgsz": 640,
        "backend": "torch",
        "device": "cpu",
    }
    import json

    response = client.post(
        "/runs",
        headers=auth_headers,
        files=files,
        data={"params": json.dumps(params)},
    )
    assert response.status_code == 202
    run_id = response.json()["id"]

    detail = client.get(f"/runs/{run_id}", headers=auth_headers)
    assert detail.status_code == 200
    body = detail.json()
    assert body["status"] == "done"
    assert body["backend"] == "torch"
    assert body["device"] == "cpu"
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["speed"] is not None
    assert "inference" in item["speed"]


def test_create_run_on_openvino_cpu_completes(client, auth_headers, tiny_web_checkpoint):
    import json

    import pytest

    pytest.importorskip("openvino", reason="openvino not installed")

    files = [("files", ("frame.jpg", _fake_image_bytes(seed=1), "image/jpeg"))]
    params = {
        "model_name": tiny_web_checkpoint,
        "conf": 0.01,
        "imgsz": 640,
        "backend": "openvino",
        "device": "CPU",
    }
    response = client.post(
        "/runs",
        headers=auth_headers,
        files=files,
        data={"params": json.dumps(params)},
    )
    run_id = response.json()["id"]

    detail = client.get(f"/runs/{run_id}", headers=auth_headers).json()
    assert detail["status"] == "done", detail.get("error_msg")
    assert detail["backend"] == "openvino"
    assert detail["items"][0]["speed"] is not None
