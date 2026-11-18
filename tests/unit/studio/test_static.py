"""Unit tests for the SPA static serving and fallback (uses tmp_path, never the real static/)."""

import pytest

pytest.importorskip("fastapi", reason="needs the 'studio' extra")

from starlette.testclient import TestClient  # noqa: E402

from nitid.studio.app import create_app  # noqa: E402
from nitid.studio.settings import StudioSettings  # noqa: E402
from nitid.studio.static import IMMUTABLE_CACHE, NO_CACHE, UI_NOT_BUILT_MESSAGE  # noqa: E402

PORT = 8124
INDEX = "<!doctype html><title>Studio</title><div id=root></div>"


def make_client(tmp_path, static_dir):
    settings = StudioSettings(port=PORT, token="test-token-123456", models_dir=tmp_path / "m")
    return TestClient(
        create_app(settings, static_dir=static_dir), base_url=f"http://127.0.0.1:{PORT}"
    )


@pytest.fixture
def static_dir(tmp_path):
    root = tmp_path / "static"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(INDEX)
    (root / "assets" / "index-AbCd1234.js").write_text("console.log(1)")
    (root / "assets" / "style-9f8e7d6c.css").write_text("body{}")
    (root / "assets" / "ort-wasm-simd.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
    (root / "favicon.svg").write_text("<svg/>")
    (tmp_path / "secret.txt").write_text("outside the static root")
    return root


@pytest.fixture
def client(tmp_path, static_dir):
    return make_client(tmp_path, static_dir)


@pytest.mark.parametrize("path", ["/", "/models", "/jobs/123", "/deep/nested/spa/route"])
def test_root_and_unknown_routes_return_index(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.text == INDEX
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == NO_CACHE


def test_index_html_is_no_cache_when_requested_directly(client):
    response = client.get("/index.html")
    assert response.text == INDEX
    assert response.headers["cache-control"] == NO_CACHE


def test_hashed_assets_are_immutable_for_a_year(client):
    for path, kind in [
        ("/assets/index-AbCd1234.js", "javascript"),
        ("/assets/style-9f8e7d6c.css", "css"),
    ]:
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert response.headers["cache-control"] == IMMUTABLE_CACHE
        assert kind in response.headers["content-type"]


def test_wasm_has_its_own_content_type(client):
    response = client.get("/assets/ort-wasm-simd.wasm")
    assert response.headers["content-type"] == "application/wasm"
    assert response.headers["cache-control"] == IMMUTABLE_CACHE


def test_unhashed_top_level_files_are_not_immutable(client):
    response = client.get("/favicon.svg")
    assert response.status_code == 200
    assert response.headers["cache-control"] == NO_CACHE


@pytest.mark.parametrize("path", ["/assets/missing.js", "/missing.png", "/assets/nope.wasm"])
def test_missing_asset_is_404_not_index(client, path):
    response = client.get(path)
    assert response.status_code == 404
    assert INDEX not in response.text


@pytest.mark.parametrize(
    "path",
    [
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/assets/../../secret.txt",
        "/..%2fsecret.txt",
        "/%00",
    ],
)
def test_path_traversal_never_leaks_outside_the_static_root(client, path):
    response = client.get(path)
    assert "outside the static root" not in response.text


def test_symlink_escaping_static_root_is_not_served(tmp_path, static_dir):
    link = static_dir / "leak.txt"
    try:
        link.symlink_to(tmp_path / "secret.txt")
    except OSError:  # pragma: no cover - platforms without symlink rights
        pytest.skip("symlinks unavailable")
    response = make_client(tmp_path, static_dir).get("/leak.txt")
    assert response.status_code == 404
    assert "outside the static root" not in response.text


def test_unknown_api_paths_never_fall_back_to_the_spa(client):
    response = client.get("/api/unknown")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
    assert client.get("/api").status_code == 404


def test_head_is_supported(client):
    response = client.head("/")
    assert response.status_code == 200
    assert response.content == b""


def test_post_to_spa_route_is_not_allowed(client):
    assert client.post("/models").status_code == 405


# --- UI not built ---------------------------------------------------------------------------


@pytest.fixture
def empty_client(tmp_path):
    return make_client(tmp_path, tmp_path / "does-not-exist")


@pytest.mark.parametrize("path", ["/", "/models"])
def test_missing_index_gives_clear_503(empty_client, path):
    response = empty_client.get(path)
    assert response.status_code == 503
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == UI_NOT_BUILT_MESSAGE
    assert "not built" in response.text
    assert "/api/v1/health" in response.text
    assert response.headers["cache-control"] == "no-store"


def test_missing_static_dir_still_serves_the_api(empty_client):
    assert empty_client.get("/api/v1/health").json()["status"] == "ok"


def test_empty_static_dir_gives_503(tmp_path):
    (tmp_path / "static").mkdir()
    assert make_client(tmp_path, tmp_path / "static").get("/").status_code == 503


def test_default_static_dir_is_inside_the_package():
    from nitid.studio import static

    assert static.STATIC_DIR.name == "static"
    assert static.STATIC_DIR.parent.name == "studio"
