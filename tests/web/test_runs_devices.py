"""FastAPI TestClient coverage for web/api/: auth, devices, and run creation
with backend/device selection. First test coverage this app has had."""

from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

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
    assert item["cpu_percent"] is None or isinstance(item["cpu_percent"], float)
    assert item["device_memory_kib"] is None  # torch backend never reports device memory


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
    # device="CPU" never has an associated GPU/NPU memory footprint
    assert detail["items"][0]["device_memory_kib"] is None


def test_concurrent_runs_against_the_same_device_all_complete(
    client, auth_headers, tiny_web_checkpoint
):
    """Batch/parallel job submission races several requests against the same
    (model, backend, device) cache key — this is exactly the scenario the
    get_model() lock exists for."""
    params = json.dumps(
        {
            "model_name": tiny_web_checkpoint,
            "conf": 0.01,
            "imgsz": 640,
            "backend": "torch",
            "device": "cpu",
        }
    )

    def submit(seed: int) -> int:
        files = [("files", (f"frame-{seed}.jpg", _fake_image_bytes(seed=seed), "image/jpeg"))]
        response = client.post("/runs", headers=auth_headers, files=files, data={"params": params})
        assert response.status_code == 202
        return response.json()["id"]

    with ThreadPoolExecutor(max_workers=6) as pool:
        run_ids = list(pool.map(submit, range(6)))

    deadline = time.monotonic() + 30
    pending = set(run_ids)
    while pending and time.monotonic() < deadline:
        for run_id in list(pending):
            detail = client.get(f"/runs/{run_id}", headers=auth_headers).json()
            if detail["status"] not in ("pending", "running"):
                assert detail["status"] == "done", detail.get("error_msg")
                pending.discard(run_id)
        if pending:
            time.sleep(0.2)

    assert not pending, f"runs still not done after deadline: {pending}"
