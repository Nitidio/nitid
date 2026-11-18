"""Unit tests for the Studio security middleware, driven through the real app."""

import pytest

pytest.importorskip("fastapi", reason="needs the 'studio' extra")

from fastapi import APIRouter  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from nitid.studio.app import create_app  # noqa: E402
from nitid.studio.security import (  # noqa: E402
    CONTENT_SECURITY_POLICY,
    SECURITY_HEADERS,
    SecurityMiddleware,
    bearer_token_matches,
    parse_host_header,
)
from nitid.studio.settings import API_PREFIX, StudioSettings  # noqa: E402

PORT = 8123
TOKEN = "test-token-123456"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def settings(tmp_path):
    return StudioSettings(
        port=PORT, token=TOKEN, models_dir=tmp_path / "models", open_browser=False
    )


@pytest.fixture
def app(settings, tmp_path):
    router = APIRouter(prefix=API_PREFIX)

    @router.get("/_secret")
    def secret():
        return {"secret": True}

    @router.post("/_mutate")
    @router.put("/_mutate")
    @router.patch("/_mutate")
    @router.delete("/_mutate")
    def mutate():
        return {"mutated": True}

    @router.get("/_boom")
    def boom():
        raise RuntimeError("boom")

    return create_app(settings, static_dir=tmp_path / "static", extra_routers=[router])


@pytest.fixture
def client(app):
    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}", raise_server_exceptions=False)


# --- Host guard -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [f"127.0.0.1:{PORT}", f"localhost:{PORT}", f"LOCALHOST:{PORT}", f"[::1]:{PORT}"],
)
def test_valid_hosts_are_accepted(client, host):
    assert client.get("/api/v1/health", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize(
    "host",
    [
        f"evil.example.com:{PORT}",  # foreign name
        f"localhost.evil.com:{PORT}",
        f"127.0.0.1.evil.com:{PORT}",
        f"127.0.0.2:{PORT}",  # other loopback address is not an allowed name
        f"127.0.0.1:{PORT + 1}",  # rebinding to another local service/port
        "127.0.0.1:80",
        "127.0.0.1",  # no port means 80, not the server port
        f"[::1]:{PORT + 1}",
        f"127.0.0.1:{PORT}.evil.com",
        "127.0.0.1:99999",
        "",
        "::1",
        f"@127.0.0.1:{PORT}",
    ],
)
def test_foreign_or_rebinding_hosts_get_403(client, host):
    response = client.get("/api/v1/health", headers={"Host": host})
    assert response.status_code == 403
    assert response.json() == {"detail": "Host not allowed"}


def test_host_guard_applies_to_static_routes_too(client):
    assert client.get("/", headers={"Host": "evil.example.com"}).status_code == 403


def test_ip_literal_hosts_only_allowed_when_bound_beyond_loopback(tmp_path):
    exposed = StudioSettings(host="0.0.0.0", port=PORT, token=TOKEN, models_dir=tmp_path)
    exposed_client = TestClient(
        create_app(exposed, static_dir=tmp_path), base_url=f"http://192.168.1.5:{PORT}"
    )
    assert exposed_client.get("/api/v1/health").status_code == 200
    # A DNS name is still refused (that is the rebinding vector), and so is the wrong port.
    assert (
        exposed_client.get("/api/v1/health", headers={"Host": f"evil.com:{PORT}"}).status_code
        == 403
    )
    assert (
        exposed_client.get(
            "/api/v1/health", headers={"Host": f"192.168.1.5:{PORT + 1}"}
        ).status_code
        == 403
    )

    local = StudioSettings(port=PORT, token=TOKEN, models_dir=tmp_path)
    local_client = TestClient(
        create_app(local, static_dir=tmp_path), base_url=f"http://192.168.1.5:{PORT}"
    )
    assert local_client.get("/api/v1/health").status_code == 403


def test_parse_host_header_rejects_malformed():
    for value in ["", "[::1", "[::1]x", "a:b", "a:1:2", ":80", "a:-1"]:
        assert parse_host_header(value) is None
    assert parse_host_header("Localhost:8000") == ("localhost", 8000)
    assert parse_host_header("[::1]:8000") == ("[::1]", 8000)
    assert parse_host_header("[::1]") == ("[::1]", None)


def test_duplicate_host_headers_are_rejected():
    import asyncio

    seen = []

    async def inner(scope, receive, send):
        seen.append(scope)

    mw = SecurityMiddleware(inner, token=TOKEN, port=PORT)
    sent = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"host", f"127.0.0.1:{PORT}".encode()), (b"host", b"evil.com")],
    }
    asyncio.run(mw(scope, None, send))
    assert sent[0]["status"] == 403
    assert not seen


# --- Origin guard ---------------------------------------------------------------------------


def test_get_without_origin_is_fine(client):
    assert client.get("/api/v1/_secret", headers=AUTH).status_code == 200


def test_get_with_foreign_origin_is_not_blocked_by_origin_guard(client):
    # Origin is only enforced on state-changing methods (CORS blocks reading the response).
    assert (
        client.get("/api/v1/_secret", headers={**AUTH, "Origin": "http://evil.com"}).status_code
        == 200
    )


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_modifying_method_with_foreign_origin_gets_403(client, method):
    response = getattr(client, method)(
        "/api/v1/_mutate", headers={**AUTH, "Origin": "http://evil.example.com"}
    )
    assert response.status_code == 403
    assert response.json() == {"detail": "Origin not allowed"}


@pytest.mark.parametrize(
    "origin",
    [
        "null",
        f"https://127.0.0.1:{PORT}",  # wrong scheme
        f"http://127.0.0.1:{PORT + 1}",  # wrong port
        f"http://127.0.0.1:{PORT}/path",
        "http://evil.com",
        "",
    ],
)
def test_bad_origin_values_on_post_get_403(client, origin):
    assert client.post("/api/v1/_mutate", headers={**AUTH, "Origin": origin}).status_code == 403


@pytest.mark.parametrize(
    "origin",
    [f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}", f"http://[::1]:{PORT}"],
)
def test_own_origin_on_post_is_accepted(client, origin):
    response = client.post("/api/v1/_mutate", headers={**AUTH, "Origin": origin})
    assert response.status_code == 200
    assert response.json() == {"mutated": True}


def test_post_without_origin_is_accepted_for_non_browser_clients(client):
    assert client.post("/api/v1/_mutate", headers=AUTH).status_code == 200


def test_origin_is_checked_before_token(client):
    # A cross-site request never learns whether its token guess was right.
    response = client.post("/api/v1/_mutate", headers={"Origin": "http://evil.com"})
    assert response.status_code == 403


# --- Token ----------------------------------------------------------------------------------


def test_health_needs_no_token(client):
    assert client.get("/api/v1/health").status_code == 200


def test_missing_token_gets_401(client):
    response = client.get("/api/v1/_secret")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "header",
    [
        "Bearer wrong-token-123456",
        "Bearer",
        "Bearer ",
        f"Basic {TOKEN}",
        TOKEN,
        f"Bearer {TOKEN}x",
        f"Bearer {TOKEN[:-1]}",
    ],
)
def test_wrong_token_gets_401(client, header):
    assert client.get("/api/v1/_secret", headers={"Authorization": header}).status_code == 401


def test_correct_token_is_accepted(client):
    assert client.get("/api/v1/_secret", headers=AUTH).json() == {"secret": True}


def test_bearer_scheme_is_case_insensitive(client):
    headers = {"Authorization": f"bearer {TOKEN}"}
    assert client.get("/api/v1/_secret", headers=headers).status_code == 200


def test_token_in_query_string_is_not_accepted(client):
    assert client.get(f"/api/v1/_secret?token={TOKEN}").status_code == 401


def test_unknown_api_path_requires_token_before_404(client):
    assert client.get("/api/v1/nope").status_code == 401
    assert client.get("/api/v1/nope", headers=AUTH).status_code == 404


def test_openapi_schema_is_behind_the_token_and_docs_are_off(client):
    assert client.get("/api/v1/openapi.json").status_code == 401
    assert client.get("/api/v1/openapi.json", headers=AUTH).status_code == 200
    assert client.get("/docs").status_code == 503  # falls to the SPA fallback, not Swagger UI


def test_bearer_token_matches_uses_bytes_safely():
    assert bearer_token_matches(b"Bearer abc", "abc")
    assert not bearer_token_matches(None, "abc")
    assert not bearer_token_matches(b"Bearer \xff\xfe", "abc")
    assert not bearer_token_matches(b"Bearer", "abc")


# --- Security headers -----------------------------------------------------------------------


def _assert_security_headers(response):
    for name, value in SECURITY_HEADERS.items():
        assert response.headers[name] == value, name


def test_security_headers_values_match_the_plan():
    assert SECURITY_HEADERS["X-Content-Type-Options"] == "nosniff"
    assert SECURITY_HEADERS["Cross-Origin-Opener-Policy"] == "same-origin"
    assert SECURITY_HEADERS["Cross-Origin-Embedder-Policy"] == "require-corp"
    assert SECURITY_HEADERS["Cross-Origin-Resource-Policy"] == "same-origin"
    assert SECURITY_HEADERS["Referrer-Policy"] == "no-referrer"
    assert CONTENT_SECURITY_POLICY == (
        "default-src 'self'; worker-src 'self' blob:; script-src 'self' 'wasm-unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'"
    )


@pytest.mark.parametrize(
    ("method", "path", "kwargs", "status"),
    [
        ("get", "/api/v1/health", {}, 200),  # success
        ("get", "/", {}, 503),  # SPA not built
        ("get", "/api/v1/_secret", {}, 401),  # auth rejection
        ("get", "/api/v1/health", {"headers": {"Host": "evil.com"}}, 403),  # host rejection
        ("post", "/api/v1/_mutate", {"headers": {"Origin": "http://evil.com"}}, 403),
        ("get", "/api/v1/_secret", {"headers": AUTH}, 200),
        ("get", "/api/v1/_boom", {"headers": AUTH}, 500),  # unhandled exception
        ("get", "/api/zzz", {}, 404),
    ],
)
def test_security_headers_on_every_response(client, method, path, kwargs, status):
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == status
    _assert_security_headers(response)


def test_no_cors_headers_are_ever_sent(client):
    response = client.get(
        "/api/v1/health",
        headers={"Origin": "http://evil.com", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" not in response.headers
    preflight = client.options("/api/v1/health", headers={"Origin": "http://evil.com"})
    assert "access-control-allow-origin" not in preflight.headers


def test_no_server_banner_or_duplicate_security_headers(client):
    response = client.get("/api/v1/health")
    raw = [k.lower() for k, _ in response.headers.raw]
    assert raw.count(b"x-content-type-options") == 1
    assert raw.count(b"content-security-policy") == 1
