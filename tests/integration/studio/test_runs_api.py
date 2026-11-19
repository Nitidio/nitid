"""Integration tests for ``/api/v1/runs`` through the real app and tiny runtime checkpoints.

No weights are downloaded: ``*.pth`` checkpoints for a compact detection and a compact
segmentation model are built in ``tmp_path``. Every response is validated against the
operation's schema in ``studio/api-contract/openapi.json``.
"""

import io
import json
import time
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="needs the 'studio' extra")
pytest.importorskip("multipart", reason="needs python-multipart (studio extra)")
jsonschema = pytest.importorskip("jsonschema")
referencing = pytest.importorskip("referencing")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from jsonschema import Draft202012Validator  # noqa: E402
from referencing import Registry, Resource  # noqa: E402
from referencing.jsonschema import DRAFT202012  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from dfine.nn.build import build_model  # noqa: E402
from dfine.nn.configs import make_model_config  # noqa: E402
from dfine.utils.checkpoint import save_checkpoint  # noqa: E402
from nitid.studio import registry, runs  # noqa: E402
from nitid.studio.app import create_app  # noqa: E402
from nitid.studio.settings import StudioSettings  # noqa: E402

CONTRACT = Path(__file__).resolve().parents[3] / "studio" / "api-contract" / "openapi.json"
OPENAPI = json.loads(CONTRACT.read_text(encoding="utf-8"))
URN = "urn:nitid-studio:openapi"
REGISTRY = Registry().with_resource(URN, Resource.from_contents(OPENAPI, DRAFT202012))

PORT = 8127
TOKEN = "runs-token-1234567"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def conforms(response, method: str, template: str) -> None:
    """Validate ``response`` against the contract entry for ``method template``."""
    operation = OPENAPI["paths"][template][method]
    declared = operation["responses"]
    assert str(response.status_code) in declared, (
        f"{method.upper()} {template} returned undocumented status {response.status_code}"
    )
    entry = declared[str(response.status_code)]
    if "$ref" in entry:
        entry = OPENAPI["components"]["responses"][entry["$ref"].rsplit("/", 1)[1]]
    content = entry.get("content", {})
    media = response.headers.get("content-type", "").split(";")[0]
    if media == "application/json":
        schema = content["application/json"]["schema"]
        if "$ref" in schema:  # make the local pointer absolute for the registry
            schema = {"$ref": URN + schema["$ref"]}
        Draft202012Validator(schema, registry=REGISTRY).validate(response.json())
    elif content:
        assert media in content, f"{media} not documented for {method} {template}"


def make_checkpoint(path: Path, task: str) -> None:
    config = make_model_config("dfine_s", task=task, num_classes=2, image_size=(64, 64))
    config["DFINETransformer"]["num_layers"] = 1
    config["DFINETransformer"]["num_queries"] = 10
    config["DFINETransformer"]["num_denoising"] = 0
    config["HybridEncoder"]["depth_mult"] = 0.1
    save_checkpoint(path, build_model(config), config, {0: "a", 1: "b"})


def image_bytes(width=80, height=60, fmt=".png") -> bytes:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(fmt, array)
    assert ok
    return buffer.tobytes()


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("models")
    make_checkpoint(directory / "tiny-det.pth", "detect")
    make_checkpoint(directory / "tiny-seg.pth", "segment")
    return directory


@pytest.fixture
def client(models_dir, tmp_path):
    registry._META_CACHE.clear()
    settings = StudioSettings(port=PORT, token=TOKEN, models_dir=models_dir)
    with TestClient(
        create_app(settings, static_dir=tmp_path), base_url=f"http://127.0.0.1:{PORT}"
    ) as c:
        yield c


def params(**overrides) -> dict:
    base = {
        "model": "tiny-det",
        "task": "detect",
        "backend": "torch",
        "device": "cpu",
        "conf": 0.0,
        "imgsz": 64,
    }
    base.update(overrides)
    return base


def post_run(client, files=None, **overrides):
    files = files if files is not None else [("files", ("a.png", image_bytes(), "image/png"))]
    body = overrides.pop("raw_params", None)
    data = {"params": body if body is not None else json.dumps(params(**overrides))}
    return client.post("/api/v1/runs", headers=AUTH, files=files, data=data)


def wait_run(client, run_id, timeout=120.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/runs/{run_id}", headers=AUTH)
        conforms(response, "get", "/runs/{id}")
        if response.json()["status"] in ("done", "failed"):
            return response.json()
        time.sleep(0.05)
    raise AssertionError("run did not finish")


# ------------------------------------------------------------------------ happy paths


def test_detect_run_end_to_end(client):
    response = post_run(client)
    assert response.status_code == 202
    conforms(response, "post", "/runs")
    run = wait_run(client, response.json()["id"])

    assert run["status"] == "done" and run["error_msg"] is None
    assert (run["model"], run["task"], run["backend"], run["device"]) == (
        "tiny-det",
        "detect",
        "torch",
        "cpu",
    )
    assert run["item_count"] == 1 and len(run["items"]) == 1
    item = run["items"][0]
    assert item["id"] == 0 and item["name"] == "a.png"
    assert (item["width"], item["height"]) == (80, 60)
    assert item["detections"], "conf=0 on a random tiny model must yield detections"
    for det in item["detections"]:
        box = det["box"]
        assert 0 <= box["x1"] <= box["x2"] <= 80 and 0 <= box["y1"] <= box["y2"] <= 60
        assert det["name"] in ("a", "b") and det["has_mask"] is False
        assert "segments" not in det


def test_segment_run_has_masks(client):
    response = post_run(client, model="tiny-seg", task="segment")
    assert response.status_code == 202
    run = wait_run(client, response.json()["id"])
    assert run["status"] == "done", run
    detections = run["items"][0]["detections"]
    assert detections and all(d["has_mask"] for d in detections)
    for det in detections:
        assert set(det["segments"]) == {"x", "y"}
        assert len(det["segments"]["x"]) == len(det["segments"]["y"])


def test_multiple_files_get_zero_based_indices_and_formats(client):
    files = [
        ("files", ("../../etc/one.jpg", image_bytes(fmt=".jpg"), "image/jpeg")),
        ("files", ("two.png", image_bytes(40, 30), "image/png")),
        ("files", ("three.bmp", image_bytes(32, 32, ".bmp"), "application/octet-stream")),
    ]
    run = wait_run(client, post_run(client, files=files).json()["id"])
    assert run["status"] == "done" and run["item_count"] == 3
    assert [i["id"] for i in run["items"]] == [0, 1, 2]
    assert [i["name"] for i in run["items"]] == ["one.jpg", "two.png", "three.bmp"]
    assert (run["items"][1]["width"], run["items"][1]["height"]) == (40, 30)


def test_classes_filter_is_applied(client):
    run = wait_run(client, post_run(client, classes=[1]).json()["id"])
    assert {d["class_id"] for d in run["items"][0]["detections"]} <= {1}


def test_list_item_image_csv_zip_and_delete(client):
    run_id = post_run(client).json()["id"]
    wait_run(client, run_id)

    listing = client.get("/api/v1/runs", headers=AUTH)
    conforms(listing, "get", "/runs")
    assert [r["id"] for r in listing.json()][0] == run_id
    assert "items" not in listing.json()[0]

    item = client.get(f"/api/v1/runs/{run_id}/items/0.json", headers=AUTH)
    conforms(item, "get", "/runs/{id}/items/{n}.json")
    assert item.json()["id"] == 0

    image = client.get(f"/api/v1/runs/{run_id}/items/0/image", headers=AUTH)
    conforms(image, "get", "/runs/{id}/items/{n}/image")
    assert image.headers["content-type"] == "image/png"
    assert cv2.imdecode(np.frombuffer(image.content, np.uint8), 1).shape[:2] == (60, 80)

    csv_response = client.get(f"/api/v1/runs/{run_id}/export.csv", headers=AUTH)
    conforms(csv_response, "get", "/runs/{id}/export.csv")
    lines = csv_response.text.strip().splitlines()
    assert lines[0] == "item,image,x1,y1,x2,y2,confidence,class_id,name"
    assert len(lines) == 1 + len(item.json()["detections"])

    zipped = client.get(f"/api/v1/runs/{run_id}/export.zip", headers=AUTH)
    conforms(zipped, "get", "/runs/{id}/export.zip")
    names = set(zipfile.ZipFile(io.BytesIO(zipped.content)).namelist())
    assert {"run.json", "detections.csv", "items/0.json", "images/0.png"} <= names

    deleted = client.delete(f"/api/v1/runs/{run_id}", headers=AUTH)
    assert deleted.status_code == 204
    conforms(deleted, "delete", "/runs/{id}")
    gone = client.get(f"/api/v1/runs/{run_id}", headers=AUTH)
    assert gone.status_code == 404
    conforms(gone, "get", "/runs/{id}")


# ------------------------------------------------------------------------ errors


def test_unknown_run_item_and_bad_index_are_404(client):
    for path, template in [
        ("/api/v1/runs/r_nope", "/runs/{id}"),
        ("/api/v1/runs/r_nope/items/0.json", "/runs/{id}/items/{n}.json"),
        ("/api/v1/runs/r_nope/items/0/image", "/runs/{id}/items/{n}/image"),
        ("/api/v1/runs/r_nope/export.csv", "/runs/{id}/export.csv"),
        ("/api/v1/runs/r_nope/export.zip", "/runs/{id}/export.zip"),
    ]:
        response = client.get(path, headers=AUTH)
        assert response.status_code == 404
        conforms(response, "get", template)
        assert response.json()["error"]["code"] == "not_found"
    assert client.delete("/api/v1/runs/r_nope", headers=AUTH).status_code == 404

    run_id = post_run(client).json()["id"]
    wait_run(client, run_id)
    for bad in ("1", "-1", "x", "99999999999999999999"):
        response = client.get(f"/api/v1/runs/{run_id}/items/{bad}.json", headers=AUTH)
        assert response.status_code == 404, bad
        conforms(response, "get", "/runs/{id}/items/{n}.json")


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        json.dumps({"task": "detect", "backend": "torch"}),
        json.dumps(params(task="pose")),
        json.dumps(params(backend="onnxruntime")),
        json.dumps(params(conf=1.5)),
        json.dumps(params(conf=True)),
        json.dumps(params(imgsz=100)),
        json.dumps(params(classes=[-1])),
        json.dumps(params(confidence=0.5)),
        json.dumps(params(device="tpu:0")),
        json.dumps(params(backend="openvino")),  # not installed here
        json.dumps(params(model="does-not-exist")),
        json.dumps(params(model="tiny-seg")),  # exists, but for another task
    ],
)
def test_invalid_params_are_422(client, raw):
    if "openvino" in raw:
        pytest.importorskip("fastapi")
    response = post_run(client, raw_params=raw)
    if response.status_code == 202:  # openvino present on this machine
        pytest.skip("OpenVINO is installed")
    assert response.status_code == 422
    conforms(response, "post", "/runs")
    assert response.json()["error"]["code"] == "invalid_params"


def test_missing_form_fields_are_422_with_envelope(client):
    response = client.post("/api/v1/runs", headers=AUTH, data={"params": json.dumps(params())})
    assert response.status_code == 422
    conforms(response, "post", "/runs")
    response = client.post(
        "/api/v1/runs",
        headers=AUTH,
        files=[("files", ("a.png", image_bytes(), "image/png"))],
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_params"


def test_official_model_not_downloaded_is_409(client):
    response = post_run(client, model="nitid1s")
    assert response.status_code == 409
    conforms(response, "post", "/runs")
    assert response.json()["error"]["code"] == "conflict"


def test_unsupported_and_empty_files_are_415(client):
    for payload in (b"%PDF-1.4 hello", b"", b"GIF89a" + b"\x00" * 20):
        response = post_run(client, files=[("files", ("x.bin", payload, "image/png"))])
        assert response.status_code == 415, payload[:6]
        conforms(response, "post", "/runs")
        assert response.json()["error"]["code"] == "unsupported_media_type"


def test_oversized_bytes_and_pixels_are_413(client, monkeypatch):
    monkeypatch.setattr(runs, "MAX_IMAGE_BYTES", 1000)
    monkeypatch.setattr("nitid.studio.routes_runs.MAX_IMAGE_BYTES", 1000)
    response = post_run(client, files=[("files", ("a.png", image_bytes(200, 200), "image/png"))])
    assert response.status_code == 413
    conforms(response, "post", "/runs")
    assert response.json()["error"]["code"] == "payload_too_large"

    monkeypatch.setattr(runs, "MAX_IMAGE_BYTES", 25 * 1024 * 1024)
    monkeypatch.setattr(runs, "MAX_IMAGE_PIXELS", 100)
    response = post_run(client)  # 80 x 60 = 4800 px
    assert response.status_code == 413
    assert "megapixels" in response.json()["error"]["message"]


def test_too_many_files_is_422(client, monkeypatch):
    monkeypatch.setattr(runs, "MAX_FILES_PER_RUN", 1)
    files = [("files", (f"{i}.png", image_bytes(), "image/png")) for i in range(2)]
    assert post_run(client, files=files).status_code == 422


def test_requests_need_the_token(client):
    for method, path in [
        ("get", "/api/v1/runs"),
        ("post", "/api/v1/runs"),
        ("get", "/api/v1/runs/x"),
    ]:
        response = getattr(client, method)(path)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"


def test_unknown_api_path_and_wrong_method_use_the_envelope(client):
    response = client.get("/api/v1/nope", headers=AUTH)
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"
    response = client.put("/api/v1/runs", headers=AUTH)
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


# ------------------------------------------------------------------------ queueing / failures


def test_inference_failure_marks_the_run_failed_without_leaking(models_dir, tmp_path):
    secret = "/very/secret/path/model.pth"

    def broken_loader(resolved, run_params):
        raise RuntimeError(secret)

    service = runs.RunService(loader=broken_loader)
    settings = StudioSettings(port=PORT, token=TOKEN, models_dir=models_dir)
    with TestClient(
        create_app(settings, static_dir=tmp_path, run_service=service),
        base_url=f"http://127.0.0.1:{PORT}",
    ) as c:
        run = wait_run(c, post_run(c).json()["id"])
    assert run["status"] == "failed" and run["items"] == []
    assert secret not in json.dumps(run)
    assert run["error_msg"] == "Could not load model 'tiny-det'."


def test_runs_execute_one_at_a_time_in_order(models_dir, tmp_path):
    import threading

    gate = threading.Event()
    active, peak, order = [], [0], []

    class FakeModel:
        def predict(self, image, **kwargs):
            active.append(1)
            peak[0] = max(peak[0], len(active))
            gate.wait(10)
            order.append(image.shape)
            active.pop()
            raise ValueError("stop")

    service = runs.RunService(loader=lambda r, p: FakeModel())
    settings = StudioSettings(port=PORT, token=TOKEN, models_dir=models_dir)
    with TestClient(
        create_app(settings, static_dir=tmp_path, run_service=service),
        base_url=f"http://127.0.0.1:{PORT}",
    ) as c:
        first = post_run(c).json()["id"]
        second = post_run(c).json()["id"]
        deadline = time.monotonic() + 10
        while c.get(f"/api/v1/runs/{first}", headers=AUTH).json()["status"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert c.get(f"/api/v1/runs/{second}", headers=AUTH).json()["status"] == "pending"
        gate.set()
        wait_run(c, first)
        failed = wait_run(c, second)
    assert peak[0] == 1
    assert failed["status"] == "failed" and failed["error_msg"] == "Inference failed (ValueError)."


def test_model_is_loaded_lazily_and_cached(models_dir, tmp_path):
    loads = []

    class Fake:
        def predict(self, image, **kwargs):
            raise ValueError

    def loader(resolved, run_params):
        loads.append(resolved.id)
        return Fake()

    service = runs.RunService(loader=loader)
    settings = StudioSettings(port=PORT, token=TOKEN, models_dir=models_dir)
    with TestClient(
        create_app(settings, static_dir=tmp_path, run_service=service),
        base_url=f"http://127.0.0.1:{PORT}",
    ) as c:
        assert loads == []  # nothing at start-up
        for _ in range(2):
            wait_run(c, post_run(c).json()["id"])
    assert loads == ["tiny-det"]


# ------------------------------------------------------------------------ contract coverage


def test_every_api_route_is_in_the_contract(client):
    import re

    contract = {
        (method.upper(), re.sub(r"\{[^}]+\}", "{}", "/api/v1" + path))
        for path, item in OPENAPI["paths"].items()
        for method in item
    }
    app_routes = {
        (method.upper(), re.sub(r"\{[^}:]+\}", "{}", path))
        for path, item in client.app.openapi()["paths"].items()
        if path.startswith("/api/v1/")
        for method in item
    }
    app_routes -= {("GET", "/api/v1/openapi.json")}
    assert app_routes <= contract, app_routes - contract
    implemented_runs = {r for r in contract if "/runs" in r[1]}
    assert implemented_runs <= app_routes, implemented_runs - app_routes
