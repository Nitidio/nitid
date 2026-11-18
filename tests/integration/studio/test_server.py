"""Integration test: a real uvicorn server on a free port, queried with httpx.

No model weights are downloaded and nothing outside ``tmp_path`` is written.
"""

import threading
import time

import pytest

pytest.importorskip("fastapi", reason="needs the 'studio' extra")
pytest.importorskip("uvicorn", reason="needs the 'studio' extra")
httpx = pytest.importorskip("httpx")

import dataclasses  # noqa: E402

import nitid  # noqa: E402
from nitid.studio import launcher  # noqa: E402
from nitid.studio.app import create_app  # noqa: E402
from nitid.studio.settings import StudioSettings  # noqa: E402

TOKEN = "integration-token-1234"
INDEX = "<!doctype html><title>Studio</title>"


@pytest.fixture
def server_url(tmp_path):
    static_dir = tmp_path / "static"
    (static_dir / "assets").mkdir(parents=True)
    (static_dir / "index.html").write_text(INDEX)
    (static_dir / "assets" / "app-0123abcd.js").write_text("export{}")

    settings = StudioSettings(
        port=0, token=TOKEN, models_dir=tmp_path / "models", open_browser=False
    )
    sock = launcher.bind_socket(settings.host, settings.port)
    settings = dataclasses.replace(settings, port=sock.getsockname()[1])
    server = launcher.make_server(create_app(settings, static_dir=static_dir), log_level="warning")

    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    try:
        yield f"http://127.0.0.1:{settings.port}", settings.port
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        sock.close()
    assert not thread.is_alive()


@pytest.fixture
def http(server_url):
    url, _ = server_url
    with httpx.Client(base_url=url, trust_env=False, timeout=10) as client:
        yield client


def test_port_zero_resolved_to_a_real_port(server_url):
    _, port = server_url
    assert port > 0


def test_health_over_real_http(http):
    response = http.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "nitid_version": nitid.__version__,
        "api_version": "1",
    }
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "server" not in response.headers


def test_spa_index_and_fallback(http):
    for path in ["/", "/some/client/route"]:
        response = http.get(path)
        assert response.status_code == 200
        assert response.text == INDEX
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["cross-origin-embedder-policy"] == "require-corp"


def test_hashed_asset_is_immutable(http):
    response = http.get("/assets/app-0123abcd.js")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_foreign_host_is_rejected(http):
    response = http.get("/api/v1/health", headers={"Host": "evil.example.com"})
    assert response.status_code == 403


def test_rebinding_host_with_other_port_is_rejected(http, server_url):
    _, port = server_url
    response = http.get("/api/v1/health", headers={"Host": f"localhost:{port + 1}"})
    assert response.status_code == 403
    assert http.get("/api/v1/health", headers={"Host": f"localhost:{port}"}).status_code == 200


def test_token_is_required_beyond_health(http):
    assert http.get("/api/v1/openapi.json").status_code == 401
    assert (
        http.get("/api/v1/openapi.json", headers={"Authorization": "Bearer nope"}).status_code
        == 401
    )
    ok = http.get("/api/v1/openapi.json", headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.status_code == 200
    assert "/api/v1/health" in ok.json()["paths"]


def test_cross_origin_post_is_rejected(http):
    response = http.post(
        "/api/v1/health",
        headers={"Origin": "http://evil.example.com", "Authorization": f"Bearer {TOKEN}"},
    )
    assert response.status_code == 403


def test_server_serves_concurrent_requests(http):
    results = []

    def hit():
        results.append(http.get("/api/v1/health").status_code)

    threads = [threading.Thread(target=hit) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == [200] * 8
