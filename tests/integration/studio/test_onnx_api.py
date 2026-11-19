"""Integration tests for ``GET /api/v1/onnx/{model}`` and ``.../meta``.

The real app is driven through Starlette's ``TestClient``. Real exports use tiny
checkpoints built at runtime (64 px, one decoder layer, random weights), so nothing is
downloaded. Cache, serialisation and error behaviour use a fake exporter that writes a
minimal ONNX carrying a real ``nitid_meta``. Responses are validated against
``studio/api-contract/openapi.json`` and ``onnx-meta.schema.json``.
"""

import asyncio
import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="needs the 'studio' extra")
pytest.importorskip("jsonschema")
referencing = pytest.importorskip("referencing")
onnx = pytest.importorskip("onnx")
httpx = pytest.importorskip("httpx")

from jsonschema import Draft202012Validator  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402
from referencing import Registry, Resource  # noqa: E402
from referencing.jsonschema import DRAFT202012  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from dfine.export_meta import build_onnx_meta, embed_onnx_meta  # noqa: E402
from dfine.nn.build import build_model  # noqa: E402
from dfine.nn.configs import make_model_config  # noqa: E402
from dfine.utils.checkpoint import save_checkpoint  # noqa: E402
from nitid import __version__  # noqa: E402
from nitid.studio import registry  # noqa: E402
from nitid.studio.app import create_app  # noqa: E402
from nitid.studio.onnx_service import OnnxService, export_checkpoint  # noqa: E402
from nitid.studio.settings import StudioSettings  # noqa: E402

CONTRACT_DIR = Path(__file__).resolve().parents[3] / "studio" / "api-contract"
OPENAPI = json.loads((CONTRACT_DIR / "openapi.json").read_text(encoding="utf-8"))
META_SCHEMA = json.loads((CONTRACT_DIR / "onnx-meta.schema.json").read_text(encoding="utf-8"))
URN = "urn:nitid-studio:openapi"
SCHEMAS = Registry().with_resource(URN, Resource.from_contents(OPENAPI, DRAFT202012))
META_VALIDATOR = Draft202012Validator(META_SCHEMA)

PORT = 8127
TOKEN = "onnx-token-1234567"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
SHA256_ETAG = r'^"[0-9a-f]{64}"$'


def validate(instance, component: str) -> None:
    Draft202012Validator(
        {"$ref": f"{URN}#/components/schemas/{component}"}, registry=SCHEMAS
    ).validate(instance)


def validate_meta(meta: dict) -> None:
    """Both documents of the contract must accept the meta, plus the cross-field rule."""
    validate(meta, "OnnxMeta")
    META_VALIDATOR.validate(meta)
    shape = meta["input"]["shape"]
    assert shape == [1, 3, meta["imgsz"], meta["imgsz"]]


def make_checkpoint(path: Path, task: str = "detect") -> None:
    config = make_model_config("dfine_s", task=task, num_classes=2, image_size=(64, 64))
    config["DFINETransformer"]["num_layers"] = 1
    config["DFINETransformer"]["num_queries"] = 10
    config["DFINETransformer"]["num_denoising"] = 0
    config["HybridEncoder"]["depth_mult"] = 0.1
    save_checkpoint(path, build_model(config), config, {0: "cat", 1: "dog"})


def fake_onnx(destination: Path, task: str, imgsz: int, label: str = "") -> None:
    """A minimal valid ONNX with a real, contract-valid ``nitid_meta``."""
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, imgsz, imgsz])
    y = helper.make_tensor_value_info("labels", TensorProto.FLOAT, [1, 3, imgsz, imgsz])
    graph = helper.make_graph([helper.make_node("Identity", ["images"], ["labels"])], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    meta = build_onnx_meta(
        task=task,
        imgsz=imgsz,
        names={0: "cat", 1: "dog"},
        topk=10,
        nitid_version=__version__,
        model="fake" + label,
    )
    embed_onnx_meta(model, meta)
    onnx.save(model, str(destination))


class CountingExporter:
    """Fake exporter that records calls and the maximum number of concurrent exports."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[tuple[str, str, int]] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def __call__(self, checkpoint: Path, task: str, imgsz: int, destination: Path) -> None:
        with self._lock:
            self.calls.append((checkpoint.name, task, imgsz))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            fake_onnx(destination, task, imgsz)
        finally:
            with self._lock:
                self.active -= 1


@pytest.fixture(autouse=True)
def clean_cache():
    registry._META_CACHE.clear()


@pytest.fixture
def models_dir(tmp_path):
    return tmp_path / "models"


def build_client(models_dir, tmp_path, service=None):
    settings = StudioSettings(port=PORT, token=TOKEN, models_dir=models_dir)
    app = create_app(settings, static_dir=tmp_path, onnx_service=service)
    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}")


@pytest.fixture
def fake_service():
    return OnnxService(exporter=CountingExporter())


@pytest.fixture
def client(models_dir, tmp_path, fake_service):
    with build_client(models_dir, tmp_path, fake_service) as c:
        yield c


@pytest.fixture
def local_model(models_dir):
    models_dir.mkdir(parents=True)
    make_checkpoint(models_dir / "mine.pth")
    return "mine"


# ------------------------------------------------------------------ auth and guards


@pytest.mark.parametrize("suffix", ["", "/meta"])
def test_routes_require_the_token(client, local_model, suffix):
    response = client.get(f"/api/v1/onnx/{local_model}{suffix}?task=detect")
    assert response.status_code == 401
    response = client.get(
        f"/api/v1/onnx/{local_model}{suffix}?task=detect",
        headers={"Authorization": "Bearer wrong-token-123456"},
    )
    assert response.status_code == 401


def test_foreign_host_is_refused(client, local_model):
    response = client.get(
        f"/api/v1/onnx/{local_model}?task=detect", headers={**AUTH, "Host": "evil.example:8127"}
    )
    assert response.status_code == 403


# ------------------------------------------------------------------ errors


@pytest.mark.parametrize("suffix", ["", "/meta"])
@pytest.mark.parametrize(
    "query",
    [
        "",
        "task=pose",
        "task=semantic",
        "task=detect&imgsz=abc",
        "task=detect&imgsz=-64",
        "task=detect&imgsz=31",
        "task=detect&imgsz=100",
        "task=detect&imgsz=0",
        "task=detect&imgsz=8192",
        "task=detect&imgsz=64.0",
    ],
)
def test_bad_parameters_are_422_with_the_envelope(client, local_model, suffix, query):
    response = client.get(f"/api/v1/onnx/{local_model}{suffix}?{query}", headers=AUTH)
    assert response.status_code == 422
    body = response.json()
    validate(body, "ErrorResponse")
    assert body["error"]["code"] == "invalid_params"


def test_unknown_model_is_404(client):
    response = client.get("/api/v1/onnx/nope?task=detect", headers=AUTH)
    assert response.status_code == 404
    validate(response.json(), "ErrorResponse")
    assert response.json()["error"]["code"] == "not_found"


def test_registry_model_that_is_not_downloaded_is_404_and_says_how_to_fix_it(client):
    response = client.get("/api/v1/onnx/nitid1s?task=detect", headers=AUTH)
    assert response.status_code == 404
    message = response.json()["error"]["message"]
    assert "not downloaded" in message and "/download" in message


def test_local_model_with_another_task_is_404(client, local_model):
    response = client.get(f"/api/v1/onnx/{local_model}?task=segment", headers=AUTH)
    assert response.status_code == 404


def test_path_traversal_ids_never_reach_the_file_system(client, models_dir, local_model):
    for model in ("..%2Fmine", "%2e%2e", "mine.pth", "a%00b"):
        response = client.get(f"/api/v1/onnx/{model}?task=detect", headers=AUTH)
        assert response.status_code in (404, 422), model


def test_exporter_failure_is_500_without_leaking_details(models_dir, tmp_path, local_model):
    def broken(checkpoint, task, imgsz, destination):
        raise RuntimeError(f"secret path {checkpoint}")

    with build_client(models_dir, tmp_path, OnnxService(exporter=broken)) as client:
        response = client.get(f"/api/v1/onnx/{local_model}?task=detect", headers=AUTH)
    assert response.status_code == 500
    validate(response.json(), "ErrorResponse")
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret" not in response.text and str(models_dir) not in response.text
    staging = [p for p in (models_dir / ".cache" / "onnx").iterdir() if p.name.startswith(".")]
    assert staging == []


# ------------------------------------------------------------------ binary: headers, ETag, 304


def test_binary_headers_and_etag(client, local_model):
    response = client.get(f"/api/v1/onnx/{local_model}?task=detect", headers=AUTH)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert int(response.headers["content-length"]) == len(response.content)
    assert response.headers["etag"] == f'"{hashlib.sha256(response.content).hexdigest()}"'
    import re

    assert re.match(SHA256_ETAG, response.headers["etag"])
    assert response.headers["cache-control"] == "private, no-cache"
    assert response.headers["x-content-type-options"] == "nosniff"
    onnx.load_from_string(response.content)  # a parsable ONNX


def test_if_none_match_gives_304_without_a_body(client, local_model):
    url = f"/api/v1/onnx/{local_model}?task=detect"
    etag = client.get(url, headers=AUTH).headers["etag"]
    for value in (etag, f"W/{etag}", f'"other", {etag}', "*"):
        response = client.get(url, headers={**AUTH, "If-None-Match": value})
        assert response.status_code == 304, value
        assert response.content == b""
        assert response.headers["etag"] == etag
    stale = client.get(url, headers={**AUTH, "If-None-Match": '"' + "0" * 64 + '"'})
    assert stale.status_code == 200 and stale.headers["etag"] == etag


def test_not_modified_still_requires_the_token(client, local_model):
    url = f"/api/v1/onnx/{local_model}?task=detect"
    etag = client.get(url, headers=AUTH).headers["etag"]
    assert client.get(url, headers={"If-None-Match": etag}).status_code == 401


def test_range_requests_are_served_by_the_file_response(client, local_model):
    """Not promised by the contract; documented as best effort."""
    url = f"/api/v1/onnx/{local_model}?task=detect"
    full = client.get(url, headers=AUTH)
    part = client.get(url, headers={**AUTH, "Range": "bytes=0-99"})
    assert part.status_code == 206
    assert part.content == full.content[:100]
    assert part.headers["etag"] == full.headers["etag"]


# ------------------------------------------------------------------ meta


def test_meta_matches_both_schemas_and_the_binary(client, local_model):
    url = f"/api/v1/onnx/{local_model}"
    binary = client.get(f"{url}?task=detect", headers=AUTH)
    response = client.get(f"{url}/meta?task=detect", headers=AUTH)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    meta = response.json()
    validate_meta(meta)
    assert meta["sha256"] == hashlib.sha256(binary.content).hexdigest()
    assert response.headers["etag"] == binary.headers["etag"] == f'"{meta["sha256"]}"'
    embedded = json.loads(
        next(e.value for e in onnx.load_from_string(binary.content).metadata_props)
    )
    assert {k: v for k, v in meta.items() if k != "sha256"} == embedded
    assert "sha256" not in embedded  # never embedded in the file it describes


# ------------------------------------------------------------------ cache and serialisation


def test_export_is_cached_by_id_task_imgsz_and_version(models_dir, tmp_path, local_model):
    exporter = CountingExporter()
    with build_client(models_dir, tmp_path, OnnxService(exporter=exporter)) as client:
        get = lambda q: client.get(f"/api/v1/onnx/mine?{q}", headers=AUTH)  # noqa: E731
        first = get("task=detect")
        assert first.status_code == 200
        assert len(exporter.calls) == 1
        assert get("task=detect").content == first.content
        assert client.get("/api/v1/onnx/mine/meta?task=detect", headers=AUTH).status_code == 200
        assert get("task=detect&imgsz=64").content == first.content  # default is 64 for this ckpt
        assert len(exporter.calls) == 1
        get("task=detect&imgsz=96")
        assert [c[2] for c in exporter.calls] == [64, 96]


def test_default_imgsz_is_the_checkpoint_eval_size(models_dir, tmp_path, local_model):
    exporter = CountingExporter()
    with build_client(models_dir, tmp_path, OnnxService(exporter=exporter)) as client:
        meta = client.get("/api/v1/onnx/mine/meta?task=detect", headers=AUTH).json()
    assert meta["imgsz"] == 64 and exporter.calls[0][2] == 64


def test_replacing_the_checkpoint_invalidates_the_cache(models_dir, tmp_path, local_model):
    exporter = CountingExporter()
    with build_client(models_dir, tmp_path, OnnxService(exporter=exporter)) as client:
        client.get("/api/v1/onnx/mine?task=detect", headers=AUTH)
        make_checkpoint(models_dir / "mine.pth")  # new random weights, new size/mtime
        registry._META_CACHE.clear()
        client.get("/api/v1/onnx/mine?task=detect", headers=AUTH)
    assert len(exporter.calls) == 2


def test_cache_survives_a_restart(models_dir, tmp_path, local_model):
    exporter = CountingExporter()
    for _ in range(2):
        with build_client(models_dir, tmp_path, OnnxService(exporter=exporter)) as client:
            assert client.get("/api/v1/onnx/mine?task=detect", headers=AUTH).status_code == 200
    assert len(exporter.calls) == 1


def test_exports_are_serialised_and_deduplicated(models_dir, tmp_path, local_model):
    exporter = CountingExporter(delay=0.15)
    make_checkpoint(models_dir / "other.pth")
    with build_client(models_dir, tmp_path, OnnxService(exporter=exporter)) as client:
        results: list[int] = []

        def fetch(model: str) -> None:
            r = client.get(f"/api/v1/onnx/{model}?task=detect", headers=AUTH)
            results.append(r.status_code)

        threads = [threading.Thread(target=fetch, args=("mine",)) for _ in range(4)]
        threads += [threading.Thread(target=fetch, args=("other",)) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    assert results == [200] * 6
    assert exporter.max_active == 1  # one export at a time
    assert sorted(c[0] for c in exporter.calls) == ["mine.pth", "other.pth"]  # once per file


def test_a_slow_export_does_not_block_the_event_loop(models_dir, tmp_path, local_model):
    exporter = CountingExporter(delay=1.0)
    settings = StudioSettings(port=PORT, token=TOKEN, models_dir=models_dir)
    app = create_app(settings, static_dir=tmp_path, onnx_service=OnnxService(exporter=exporter))

    async def scenario() -> tuple[float, float]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=f"http://127.0.0.1:{PORT}") as c:
            start = time.monotonic()
            slow = asyncio.create_task(c.get("/api/v1/onnx/mine?task=detect", headers=AUTH))
            await asyncio.sleep(0.2)  # the export is running in a worker thread now
            health_started = time.monotonic()
            health = await c.get("/api/v1/health")
            health_elapsed = time.monotonic() - health_started
            assert health.status_code == 200 and not slow.done()
            assert (await slow).status_code == 200
            return health_elapsed, time.monotonic() - start

    health_elapsed, total = asyncio.run(scenario())
    assert health_elapsed < 0.5, "health was held up by the export"
    assert total >= 1.0


def test_cache_directory_is_not_a_model_and_holds_only_onnx_files(
    models_dir, tmp_path, local_model
):
    exporter = CountingExporter()
    with build_client(models_dir, tmp_path, OnnxService(exporter=exporter)) as client:
        client.get("/api/v1/onnx/mine?task=detect", headers=AUTH)
        listing = client.get("/api/v1/models", headers=AUTH).json()
    assert [m["id"] for m in listing if m["source"] == "local"] == ["mine"]
    files = sorted(p.name for p in (models_dir / ".cache" / "onnx").iterdir())
    assert len(files) == 1 and files[0].endswith(".onnx")


# ------------------------------------------------------------------ real exports


def test_registry_model_exports_with_its_public_name(models_dir, tmp_path):
    models_dir.mkdir(parents=True)
    entry = registry.find_registry_entry("nitid1s", "detect")
    make_checkpoint(registry.weights_path(models_dir, entry))
    with build_client(models_dir, tmp_path, OnnxService()) as client:
        response = client.get("/api/v1/onnx/nitid1s/meta?task=detect", headers=AUTH)
    assert response.status_code == 200
    meta = response.json()
    validate_meta(meta)
    assert meta["model"] == "nitid1s" and meta["imgsz"] == 64
    assert meta["names"] == {"0": "cat", "1": "dog"}


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_real_export_is_static_postprocessed_and_consistent(models_dir, tmp_path, task):
    ort = pytest.importorskip("onnxruntime")
    np = pytest.importorskip("numpy")
    models_dir.mkdir(parents=True)
    make_checkpoint(models_dir / "mine.pth", task=task)
    with build_client(models_dir, tmp_path, OnnxService()) as client:
        binary = client.get(f"/api/v1/onnx/mine?task={task}&imgsz=64", headers=AUTH)
        meta = client.get(f"/api/v1/onnx/mine/meta?task={task}&imgsz=64", headers=AUTH).json()
    assert binary.status_code == 200
    validate_meta(meta)
    assert meta["task"] == task and meta["sha256"] == hashlib.sha256(binary.content).hexdigest()

    proto = onnx.load_from_string(binary.content)
    onnx.checker.check_model(proto, full_check=True)
    assert len(proto.graph.value_info) == 0  # no stale shapes for onnxruntime-web

    session = ort.InferenceSession(binary.content, providers=["CPUExecutionProvider"])
    (inp,) = session.get_inputs()
    assert inp.shape == [1, 3, 64, 64]  # static
    outputs = session.run(None, {"images": np.zeros((1, 3, 64, 64), dtype=np.float32)})
    assert [o.name for o in session.get_outputs()] == [o["name"] for o in meta["outputs"]]
    expected_masks = task == "segment"
    assert (len(outputs) == 4) is expected_masks
    if expected_masks:
        assert outputs[3].shape[-2:] == (64 // meta["outputs"][3]["stride"],) * 2


def test_default_exporter_leaves_no_run_metadata_in_the_cache(models_dir, tmp_path):
    models_dir.mkdir(parents=True)
    make_checkpoint(models_dir / "mine.pth")
    service = OnnxService(exporter=export_checkpoint)
    with build_client(models_dir, tmp_path, service) as client:
        assert client.get("/api/v1/onnx/mine?task=detect", headers=AUTH).status_code == 200
    names = sorted(p.name for p in (models_dir / ".cache" / "onnx").iterdir())
    assert len(names) == 1 and names[0].endswith(".onnx")  # no args.yaml / environment.yaml


# ------------------------------------------------------------------ the contract lists what is served


def test_served_routes_are_the_ones_in_the_contract(models_dir, tmp_path):
    settings = StudioSettings(port=PORT, token=TOKEN, models_dir=models_dir)
    generated = create_app(settings, static_dir=tmp_path).openapi()["paths"]
    served = {
        path.removeprefix("/api/v1"): item
        for path, item in generated.items()
        if path.startswith("/api/v1/onnx")
    }
    assert set(served) == {path for path in OPENAPI["paths"] if path.startswith("/onnx")}
    for path, item in served.items():
        assert list(item) == ["get"]
        names = {p["name"] for p in item["get"]["parameters"]}
        declared = {p["name"] for p in OPENAPI["paths"][path]["get"]["parameters"]}
        assert {"model", "task", "imgsz"} <= names
        assert names <= declared | {"model", "task", "imgsz"}
