"""Integration tests for /devices, /models, /models/{id}/download and /jobs/{job_id}.

The real app (``create_app``) is driven through Starlette's ``TestClient`` with a
token. Downloads are simulated by replacing the downloader; no network, no
pretrained weights. Every response body is validated against the schemas of
``studio/api-contract/openapi.json``.
"""

import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="needs the 'studio' extra")
jsonschema = pytest.importorskip("jsonschema")
referencing = pytest.importorskip("referencing")

import torch  # noqa: E402
from jsonschema import Draft202012Validator  # noqa: E402
from referencing import Registry, Resource  # noqa: E402
from referencing.jsonschema import DRAFT202012  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from dfine.nn.build import build_model  # noqa: E402
from dfine.nn.configs import make_model_config  # noqa: E402
from dfine.utils.checkpoint import save_checkpoint  # noqa: E402
from nitid.studio import registry  # noqa: E402
from nitid.studio.app import create_app  # noqa: E402
from nitid.studio.settings import StudioSettings  # noqa: E402

CONTRACT = Path(__file__).resolve().parents[3] / "studio" / "api-contract" / "openapi.json"
OPENAPI = json.loads(CONTRACT.read_text(encoding="utf-8"))
URN = "urn:nitid-studio:openapi"
SCHEMAS = Registry().with_resource(URN, Resource.from_contents(OPENAPI, DRAFT202012))

PORT = 8126
TOKEN = "models-token-123456"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def validate(instance, component: str) -> None:
    Draft202012Validator(
        {"$ref": f"{URN}#/components/schemas/{component}"}, registry=SCHEMAS
    ).validate(instance)


def make_checkpoint(path: Path, task: str = "detect") -> None:
    config = make_model_config("dfine_s", task=task, num_classes=2, image_size=(64, 64))
    config["DFINETransformer"]["num_layers"] = 1
    config["DFINETransformer"]["num_queries"] = 10
    config["DFINETransformer"]["num_denoising"] = 0
    config["HybridEncoder"]["depth_mult"] = 0.1
    save_checkpoint(path, build_model(config), config, {0: "a", 1: "b"})


@pytest.fixture(autouse=True)
def clean_cache():
    registry._META_CACHE.clear()


@pytest.fixture
def models_dir(tmp_path):
    return tmp_path / "models"


@pytest.fixture
def client(models_dir, tmp_path):
    settings = StudioSettings(port=PORT, token=TOKEN, models_dir=models_dir)
    with TestClient(
        create_app(settings, static_dir=tmp_path), base_url=f"http://127.0.0.1:{PORT}"
    ) as c:
        yield c


def wait_job(client, job_id, states=("done", "failed"), timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}", headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        validate(body, "Job")
        if body["status"] in states:
            return body
        time.sleep(0.01)
    raise AssertionError("job did not finish")


@pytest.fixture
def fake_download(monkeypatch):
    """Replace the downloader by one that writes a placeholder at the registry path."""
    calls: list = []

    def fake(entry, models_dir, progress):
        calls.append((entry.id, entry.task))
        progress(0.5)
        target = registry.weights_path(models_dir, entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"wrapped")
        return target

    monkeypatch.setattr(registry, "download_registry_model", fake)
    return calls


# ------------------------------------------------------------------ auth and guards


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/v1/devices"),
        ("get", "/api/v1/models"),
        ("post", "/api/v1/models/nitid1s/download?task=detect"),
        ("get", "/api/v1/jobs/j_x"),
    ],
)
def test_routes_require_the_token(client, method, path):
    assert getattr(client, method)(path).status_code == 401
    bad = getattr(client, method)(path, headers={"Authorization": "Bearer wrong-token-000"})
    assert bad.status_code == 401


def test_foreign_origin_cannot_start_downloads(client, fake_download):
    response = client.post(
        "/api/v1/models/nitid1s/download?task=detect",
        headers={**AUTH, "Origin": "http://evil.example"},
    )
    assert response.status_code == 403
    assert fake_download == []


def test_dns_rebinding_host_is_refused(client):
    response = client.get("/api/v1/models", headers={**AUTH, "Host": "evil.example:8126"})
    assert response.status_code == 403


def test_security_headers_are_present(client):
    response = client.get("/api/v1/models", headers=AUTH)
    assert response.headers["x-content-type-options"] == "nosniff"


# ------------------------------------------------------------------ /devices


def test_devices_matches_the_contract(client):
    response = client.get("/api/v1/devices", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    validate(body, "Devices")
    assert body["torch"]["cpu"] is True


def test_devices_with_simulated_hardware(client, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Fake GPU")
    import dfine.nn.openvino_runtime as runtime

    monkeypatch.setattr(runtime, "list_openvino_devices", lambda: ["CPU", "GPU", "NPU"])
    body = client.get("/api/v1/devices", headers=AUTH).json()
    validate(body, "Devices")
    assert body == {
        "torch": {"cpu": True, "cuda": [{"index": 0, "name": "Fake GPU"}]},
        "openvino": {"available": True, "devices": ["CPU", "GPU", "NPU"]},
    }


def test_devices_without_openvino(client, monkeypatch):
    import dfine.nn.openvino_runtime as runtime

    def missing():
        raise ImportError("OpenVINO is not installed.")

    monkeypatch.setattr(runtime, "list_openvino_devices", missing)
    body = client.get("/api/v1/devices", headers=AUTH).json()
    validate(body, "Devices")
    assert body["openvino"] == {"available": False, "devices": []}


# ------------------------------------------------------------------ /models


def test_models_registry_only(client):
    response = client.get("/api/v1/models", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    validate(body, "ModelList")
    pairs = {(m["id"], m["task"]) for m in body}
    assert {m["source"] for m in body} == {"registry"}
    assert len(pairs) == len(body) == 9
    assert all(m["downloaded"] is False for m in body)


def test_nano_detect_is_not_exposed(client):
    body = client.get("/api/v1/models", headers=AUTH).json()
    assert {"id": "nitid1n", "task": "segment"} in [
        {"id": m["id"], "task": m["task"]} for m in body
    ]
    assert not any(m["id"] == "nitid1n" and m["task"] == "detect" for m in body)


def test_models_include_local_checkpoints_and_skip_broken_ones(client, models_dir):
    models_dir.mkdir()
    make_checkpoint(models_dir / "my-detector.pth")
    (models_dir / "broken.pth").write_bytes(b"junk")
    body = client.get("/api/v1/models", headers=AUTH).json()
    validate(body, "ModelList")
    local = [m for m in body if m["source"] == "local"]
    assert len(local) == 1
    assert local[0]["id"] == "my-detector"
    assert local[0]["downloaded"] is True
    assert local[0]["path"].endswith("my-detector.pth")


def test_models_reports_downloaded_registry_weights(client, models_dir):
    models_dir.mkdir()
    entry = registry.find_registry_entry("nitid1x", "detect")
    registry.weights_path(models_dir, entry).write_bytes(b"x")
    body = client.get("/api/v1/models", headers=AUTH).json()
    by_pair = {(m["id"], m["task"]): m for m in body}
    assert by_pair[("nitid1x", "detect")]["downloaded"] is True
    assert by_pair[("nitid1x", "segment")]["downloaded"] is False
    assert len(body) == 9  # the downloaded official file is not listed a second time as local


# ------------------------------------------------------------------ download + jobs


def test_download_flow(client, models_dir, fake_download):
    response = client.post(
        "/api/v1/models/nitid1s/download", params={"task": "detect"}, headers=AUTH
    )
    assert response.status_code == 202
    body = response.json()
    validate(body, "JobCreated")
    job = wait_job(client, body["job_id"], states=("done",))
    assert job == {"status": "done", "progress": 1.0}
    assert fake_download == [("nitid1s", "detect")]
    models = client.get("/api/v1/models", headers=AUTH).json()
    flags = {(m["id"], m["task"]): m["downloaded"] for m in models}
    assert flags[("nitid1s", "detect")] is True
    assert flags[("nitid1s", "segment")] is False


def test_download_failure_is_reported_without_leaking(client, monkeypatch):
    def fail(entry, models_dir, progress):
        raise OSError(f"cannot write {models_dir}/secret-location")

    monkeypatch.setattr(registry, "download_registry_model", fail)
    job_id = client.post("/api/v1/models/nitid1m/download?task=detect", headers=AUTH).json()[
        "job_id"
    ]
    job = wait_job(client, job_id, states=("failed",))
    assert job["status"] == "failed"
    assert job["error"]
    assert "secret-location" not in json.dumps(job)
    assert "/" not in job["error"]  # no path-like text


def test_unknown_id_is_404_with_error_envelope(client):
    response = client.post("/api/v1/models/nitid9z/download?task=detect", headers=AUTH)
    assert response.status_code == 404
    validate(response.json(), "ErrorResponse")
    assert response.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize("model_id", ["model1s", "dfine_s", "NITID1S", "nitid1s.pth", "nitid1-s"])
def test_aliases_are_never_accepted(client, fake_download, model_id):
    response = client.post(f"/api/v1/models/{model_id}/download?task=detect", headers=AUTH)
    assert response.status_code == 404
    validate(response.json(), "ErrorResponse")
    assert fake_download == []


@pytest.mark.parametrize(
    "model_id", ["..%2F..%2Fetc%2Fpasswd", "http:%2F%2Fevil.example%2Fw.pth", "a/b"]
)
def test_urls_and_paths_never_reach_the_downloader(client, fake_download, model_id):
    # An encoded slash is decoded before routing, so the path no longer matches the
    # download route at all (404/405); either way nothing is downloaded.
    response = client.post(f"/api/v1/models/{model_id}/download?task=detect", headers=AUTH)
    assert response.status_code in (404, 405)
    assert fake_download == []


def test_nano_detect_download_is_404(client, fake_download):
    response = client.post("/api/v1/models/nitid1n/download?task=detect", headers=AUTH)
    assert response.status_code == 404
    assert fake_download == []
    assert (
        client.post("/api/v1/models/nitid1n/download?task=segment", headers=AUTH).status_code == 202
    )


def test_missing_task_is_422_with_error_envelope(client, fake_download):
    response = client.post("/api/v1/models/nitid1s/download", headers=AUTH)
    assert response.status_code == 422
    validate(response.json(), "ErrorResponse")
    assert response.json()["error"]["code"] == "invalid_params"
    assert fake_download == []


@pytest.mark.parametrize("task", ["semantic", "pose", "", "DETECT", "detect;rm"])
def test_invalid_task_is_422(client, fake_download, task):
    response = client.post("/api/v1/models/nitid1s/download", params={"task": task}, headers=AUTH)
    assert response.status_code == 422
    validate(response.json(), "ErrorResponse")


def test_download_of_a_local_model_is_409(client, models_dir, fake_download):
    models_dir.mkdir()
    make_checkpoint(models_dir / "mine.pth")
    response = client.post("/api/v1/models/mine/download?task=detect", headers=AUTH)
    assert response.status_code == 409
    validate(response.json(), "ErrorResponse")
    assert response.json()["error"]["code"] == "conflict"
    assert fake_download == []


def test_concurrent_download_of_same_pair_is_409(client, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def slow(entry, models_dir, progress):
        started.set()
        release.wait(10)

    monkeypatch.setattr(registry, "download_registry_model", slow)
    url = "/api/v1/models/nitid1l/download"
    first = client.post(url, params={"task": "detect"}, headers=AUTH)
    assert first.status_code == 202
    assert started.wait(10)
    second = client.post(url, params={"task": "detect"}, headers=AUTH)
    assert second.status_code == 409
    validate(second.json(), "ErrorResponse")
    # same id, other task: allowed
    other = client.post(url, params={"task": "segment"}, headers=AUTH)
    assert other.status_code == 202
    running = client.get(f"/api/v1/jobs/{first.json()['job_id']}", headers=AUTH).json()
    assert running["status"] == "running"
    validate(running, "Job")
    release.set()
    wait_job(client, first.json()["job_id"], states=("done",))
    wait_job(client, other.json()["job_id"], states=("done",))


def test_unknown_job_is_404_with_error_envelope(client):
    response = client.get("/api/v1/jobs/j_missing", headers=AUTH)
    assert response.status_code == 404
    validate(response.json(), "ErrorResponse")


def test_expired_jobs_are_404(models_dir, tmp_path, fake_download):
    from fastapi import APIRouter

    from nitid.studio.jobs import JobStore
    from nitid.studio.routes_models import build_models_router

    now = [0.0]
    store = JobStore(ttl=30.0, clock=lambda: now[0])
    router = APIRouter(prefix="/api/v1/t")  # a second copy of the routes, with a fake clock
    router.include_router(build_models_router(store=store))
    settings = StudioSettings(port=PORT, token=TOKEN, models_dir=models_dir)
    app = create_app(settings, static_dir=tmp_path, extra_routers=[router])
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        url = "/api/v1/t/models/nitid1s/download?task=detect"
        job_id = c.post(url, headers=AUTH).json()["job_id"]
        deadline = time.monotonic() + 10
        while store.get(job_id).status != "done" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert c.get(f"/api/v1/t/jobs/{job_id}", headers=AUTH).status_code == 200
        now[0] += 31.0
        response = c.get(f"/api/v1/t/jobs/{job_id}", headers=AUTH)
        assert response.status_code == 404
        validate(response.json(), "ErrorResponse")
    store.shutdown(wait=True)


def test_each_app_has_its_own_jobs(models_dir, tmp_path, fake_download):
    settings = StudioSettings(port=PORT, token=TOKEN, models_dir=models_dir)
    with (
        TestClient(
            create_app(settings, static_dir=tmp_path), base_url=f"http://127.0.0.1:{PORT}"
        ) as a,
        TestClient(
            create_app(settings, static_dir=tmp_path), base_url=f"http://127.0.0.1:{PORT}"
        ) as b,
    ):
        job_id = a.post("/api/v1/models/nitid1s/download?task=detect", headers=AUTH).json()[
            "job_id"
        ]
        wait_job(a, job_id)
        assert b.get(f"/api/v1/jobs/{job_id}", headers=AUTH).status_code == 404


def test_contract_documents_the_four_paths():
    paths = OPENAPI["paths"]
    assert {"/devices", "/models", "/models/{id}/download", "/jobs/{job_id}"} <= set(paths)
