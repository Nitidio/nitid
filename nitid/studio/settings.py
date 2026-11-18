"""Validated, immutable settings for Nitid Studio.

Nothing here imports FastAPI, so the CLI can parse and validate arguments
before checking that the optional ``studio`` extra is installed.
"""

from __future__ import annotations

import ipaddress
import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

API_PREFIX = "/api/v1"
API_VERSION = "1"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MODELS_DIR = Path("models")

# Upload limits (enforced by the endpoints that accept images, added in later work packages).
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000  # 20 megapixels

MIN_TOKEN_LENGTH = 8

_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$")
_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off"})
_ALLOWED_KEYS = frozenset({"host", "port", "models", "open", "token"})


class SettingsError(ValueError):
    """Raised when Studio arguments or settings are invalid."""


def normalize_host(host: str) -> str:
    """Return ``host`` without IPv6 brackets, validating its shape."""
    value = host.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value:
        raise SettingsError("host must not be empty")
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        pass
    if not _HOSTNAME_RE.match(value):
        raise SettingsError(f"invalid host {host!r}")
    return value.lower()


def is_loopback_host(host: str) -> bool:
    """True for ``localhost`` and loopback IP addresses (``127.0.0.0/8``, ``::1``)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class StudioSettings:
    """Immutable, validated Studio configuration.

    ``token=None`` generates a random per-start token, which is only allowed for
    loopback hosts: binding to any other interface requires an explicit token.
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    models_dir: Path = DEFAULT_MODELS_DIR
    token: str | None = field(default=None, repr=False)
    open_browser: bool = True

    def __post_init__(self) -> None:
        host = normalize_host(self.host)
        object.__setattr__(self, "host", host)

        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise SettingsError(f"port must be an integer, got {self.port!r}")
        if not 0 <= self.port <= 65535:
            raise SettingsError(f"port must be between 0 and 65535, got {self.port}")

        models_dir = Path(self.models_dir).expanduser().resolve()
        if models_dir.exists() and not models_dir.is_dir():
            raise SettingsError(f"models path is not a directory: {models_dir}")
        object.__setattr__(self, "models_dir", models_dir)

        token = self.token
        if token is None:
            if not is_loopback_host(host):
                raise SettingsError(
                    f"host={host} exposes Studio beyond this machine and requires an explicit token="
                )
            token = secrets.token_urlsafe(32)
        else:
            if len(token) < MIN_TOKEN_LENGTH:
                raise SettingsError(f"token must have at least {MIN_TOKEN_LENGTH} characters")
            if any(ch.isspace() or not ch.isprintable() for ch in token):
                raise SettingsError("token must not contain whitespace or control characters")
        object.__setattr__(self, "token", token)

        object.__setattr__(self, "open_browser", bool(self.open_browser))

    @property
    def is_loopback(self) -> bool:
        """True when the server only listens on this machine."""
        return is_loopback_host(self.host)

    @property
    def api_token(self) -> str:
        """The bearer token (never ``None`` after validation)."""
        assert self.token is not None
        return self.token


def _parse_bool(key: str, raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise SettingsError(f"{key}= expects true or false, got {raw!r}")


def parse_args(tokens: Sequence[str]) -> StudioSettings:
    """Build settings from ``key=value`` tokens and the ``--no-browser`` flag.

    Values stay raw strings (no numeric coercion), so a token such as
    ``token=00112233`` keeps its leading zeros. Unknown keys are errors.
    """
    values: dict[str, str] = {}
    no_browser = False
    for token in tokens:
        if token == "--no-browser":
            no_browser = True
            continue
        if "=" not in token:
            raise SettingsError(f"unexpected argument {token!r} (use key=value)")
        key, raw = token.split("=", 1)
        key = key.strip()
        if key not in _ALLOWED_KEYS:
            allowed = ", ".join(sorted(_ALLOWED_KEYS))
            raise SettingsError(f"unknown option {key!r} (supported: {allowed})")
        values[key] = raw.strip()

    kwargs: dict[str, object] = {}
    if "host" in values:
        kwargs["host"] = values["host"]
    if "port" in values:
        try:
            kwargs["port"] = int(values["port"])
        except ValueError:
            raise SettingsError(f"port= expects an integer, got {values['port']!r}") from None
    if "models" in values:
        if not values["models"]:
            raise SettingsError("models= must not be empty")
        kwargs["models_dir"] = Path(values["models"])
    if "token" in values:
        if not values["token"]:
            raise SettingsError("token= must not be empty")
        kwargs["token"] = values["token"]
    open_browser = _parse_bool("open", values["open"]) if "open" in values else True
    if no_browser:
        open_browser = False
    kwargs["open_browser"] = open_browser
    return StudioSettings(**kwargs)  # type: ignore[arg-type]
