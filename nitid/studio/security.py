"""Security guards for Nitid Studio, as a dependency-free ASGI middleware.

Threat model: Studio is a local tool. The risks are other web pages in the
user's browser talking to it (CSRF, DNS rebinding) and other machines reaching
it when it is bound beyond loopback. The middleware therefore enforces, in order:

1. ``Host`` must be a loopback name with the server's port (DNS rebinding).
2. ``Origin``, when present on a state-changing method, must be our own origin.
3. ``Authorization: Bearer <token>`` for every ``/api/v1/*`` except ``/health``.

Every response, including the rejections above, carries the security headers.
There is deliberately no CORS support: the UI is served from the same origin.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from nitid.studio.settings import API_PREFIX

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

PUBLIC_API_PATHS = frozenset({f"{API_PREFIX}/health"})
MODIFYING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
LOOPBACK_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "[::1]"})

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; worker-src 'self' blob:; script-src 'self' 'wasm-unsafe-eval'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'"
)

SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
}
_SECURITY_HEADERS_RAW = [(k.lower().encode(), v.encode()) for k, v in SECURITY_HEADERS.items()]
_SECURITY_NAMES_RAW = frozenset(k for k, _ in _SECURITY_HEADERS_RAW)


def parse_host_header(value: str) -> tuple[str, int | None] | None:
    """Split ``host[:port]`` (IPv6 literals in brackets). Return ``None`` if malformed."""
    value = value.strip().lower()
    if not value:
        return None
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return None
        hostname, rest = value[: end + 1], value[end + 1 :]
        if rest == "":
            return hostname, None
        if not rest.startswith(":"):
            return None
        port_text = rest[1:]
    else:
        if value.count(":") > 1:
            return None
        hostname, _, port_text = value.partition(":")
        if not hostname:
            return None
        if ":" not in value:
            return hostname, None
    if not port_text.isascii() or not port_text.isdigit():
        return None
    port = int(port_text)
    if port > 65535:
        return None
    return hostname, port


def _is_ip_literal(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        return False
    return True


def bearer_token_matches(header_value: bytes | None, token: str) -> bool:
    """Constant-time check of an ``Authorization: Bearer`` header value."""
    if header_value is None:
        return False
    scheme, _, provided = header_value.partition(b" ")
    if scheme.lower() != b"bearer":
        return False
    return hmac.compare_digest(provided.strip(), token.encode())


class SecurityMiddleware:
    """Pure ASGI middleware applying the Studio security policy (see module docstring)."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        token: str,
        port: int,
        allow_ip_hosts: bool = False,
    ) -> None:
        self.app = app
        self.token = token
        self.port = port
        # Reachable-from-network mode: IP-literal hosts cannot be DNS-rebinding vectors
        # (no name resolution is involved), so they are accepted when the bind is non-loopback.
        self.allow_ip_hosts = allow_ip_hosts

    def host_allowed(self, host_header: str | None) -> bool:
        if host_header is None:
            return False
        parsed = parse_host_header(host_header)
        if parsed is None:
            return False
        hostname, port = parsed
        if (80 if port is None else port) != self.port:
            return False
        if hostname in LOOPBACK_HOSTNAMES:
            return True
        return self.allow_ip_hosts and _is_ip_literal(hostname)

    def origin_allowed(self, origin: str) -> bool:
        prefix = "http://"
        if not origin.lower().startswith(prefix):
            return False  # includes "null" and foreign schemes
        return self.host_allowed(origin[len(prefix) :])

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope["type"]
        if kind == "lifespan":
            await self.app(scope, receive, send)
            return
        if kind != "http":
            # No WebSocket endpoints exist; refuse rather than skip the guards.
            await send({"type": "websocket.close", "code": 1008})
            return

        async def send_secured(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() not in _SECURITY_NAMES_RAW
                ]
                headers.extend(_SECURITY_HEADERS_RAW)
                message = {**message, "headers": headers}
            await send(message)

        headers_in: dict[bytes, list[bytes]] = {}
        for key, value in scope.get("headers", []):
            headers_in.setdefault(key.lower(), []).append(value)

        hosts = headers_in.get(b"host", [])
        host_header = hosts[0].decode("latin-1") if len(hosts) == 1 else None
        if not self.host_allowed(host_header):
            await _reject(send_secured, 403, "Host not allowed")
            return

        origins = headers_in.get(b"origin", [])
        if scope["method"] in MODIFYING_METHODS and origins:
            if len(origins) != 1 or not self.origin_allowed(origins[0].decode("latin-1")):
                await _reject(send_secured, 403, "Origin not allowed")
                return

        path = scope["path"]
        protected = (path == API_PREFIX or path.startswith(API_PREFIX + "/")) and (
            path not in PUBLIC_API_PATHS
        )
        if protected:
            auth = headers_in.get(b"authorization", [])
            value = auth[0] if len(auth) == 1 else None
            if not bearer_token_matches(value, self.token):
                await _reject(
                    send_secured,
                    401,
                    "Missing or invalid token",
                    extra=[(b"www-authenticate", b"Bearer")],
                )
                return

        await self.app(scope, receive, send_secured)


async def _reject(
    send: Send, status: int, detail: str, extra: list[tuple[bytes, bytes]] | None = None
) -> None:
    body = json.dumps({"detail": detail}).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        (b"cache-control", b"no-store"),
        *(extra or []),
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
