"""Command entry point: ``nitid studio`` and ``python -m nitid.studio``.

FastAPI and uvicorn are optional (``pip install "nitid[studio]"``) and are only
imported by :func:`run` and :func:`make_server`, after the extra has been checked.
"""

from __future__ import annotations

import dataclasses
import importlib
import ipaddress
import os
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote

from nitid.studio.settings import SettingsError, StudioSettings, parse_args

REQUIRED_MODULES = ("fastapi", "uvicorn")
INSTALL_HINT = 'pip install "nitid[studio]"'

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_MISSING_EXTRA = 3

HELP_FLAGS = ("-h", "--help")


def missing_extra_modules() -> list[str]:
    """Names of required optional modules that cannot be imported."""
    missing = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


def display_host(host: str) -> str:
    """Host to show in the URL: wildcard binds are reached through loopback."""
    if host == "0.0.0.0":
        return "127.0.0.1"
    if host == "::":
        return "[::1]"
    try:
        if ipaddress.ip_address(host).version == 6:
            return f"[{host}]"
    except ValueError:
        pass
    return host


def access_url(settings: StudioSettings) -> str:
    """URL to open Studio. The token travels in the fragment, which is never sent to servers."""
    token = quote(settings.api_token, safe="")
    return f"http://{display_host(settings.host)}:{settings.port}/#t={token}"


def bind_socket(host: str, port: int) -> socket.socket:
    """Bind a TCP socket (``port=0`` picks a free port). Raises ``OSError`` on failure."""
    bind_host = "127.0.0.1" if host == "localhost" else host
    family, _, _, _, sockaddr = socket.getaddrinfo(
        bind_host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )[0]
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if os.name != "nt":  # on Windows SO_REUSEADDR would allow port hijacking
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(sockaddr)
    except OSError:
        sock.close()
        raise
    return sock


def make_server(app: Any, *, log_level: str = "info") -> Any:
    """Create a single-process uvicorn server for ``app`` (no reload, no workers)."""
    import uvicorn

    config = uvicorn.Config(
        app,
        log_level=log_level,
        access_log=False,
        server_header=False,
        proxy_headers=False,
        reload=False,
        workers=1,
    )
    return uvicorn.Server(config)


def _announce_when_ready(
    server: Any, settings: StudioSettings, url: str, *, timeout: float = 30.0
) -> bool:
    """Wait until ``server`` is listening, then print the URL and open the browser.

    Nothing is printed if the server never starts (it failed, or was asked to exit
    first), so a URL that leads nowhere is never shown. Returns whether it announced.
    """
    deadline = time.monotonic() + timeout
    while not server.started and not server.should_exit and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        return False
    print(f"Open: {url}")
    print(f"Models directory: {settings.models_dir}")
    print("Press Ctrl+C to stop.")
    sys.stdout.flush()
    if settings.open_browser:
        webbrowser.open(url)
    return True


def run(settings: StudioSettings) -> int:
    """Bind, serve until interrupted and announce the URL once listening.

    Returns a process exit code: :data:`EXIT_OK` after a normal stop (including
    Ctrl+C, after uvicorn has finished its graceful shutdown) and
    :data:`EXIT_FAILURE` if the address cannot be bound.
    """
    from nitid import __version__
    from nitid.studio.app import create_app

    try:
        sock = bind_socket(settings.host, settings.port)
    except OSError as error:
        print(f"ERROR: cannot listen on {settings.host}:{settings.port}: {error}", file=sys.stderr)
        return EXIT_FAILURE

    settings = dataclasses.replace(settings, port=sock.getsockname()[1])
    server = make_server(create_app(settings))
    url = access_url(settings)

    print(f"Nitid app {__version__}")
    if not settings.is_loopback:
        print(
            "WARNING: Studio is listening beyond this machine. Anyone on the network who has the\n"
            "token can use it, and traffic is plain HTTP, so the token is not encrypted in transit.\n"
            "Only use this on a network you trust.",
            file=sys.stderr,
        )
    sys.stdout.flush()

    # The URL is printed (and the browser opened) by this thread only once uvicorn is
    # accepting connections, never before.
    threading.Thread(target=_announce_when_ready, args=(server, settings, url), daemon=True).start()

    try:
        server.run(sockets=[sock])
    except KeyboardInterrupt:
        # uvicorn finishes its graceful shutdown, then re-raises the captured SIGINT.
        # Left alone, the interpreter would die from the signal (exit status -2 / 130).
        pass
    finally:
        sock.close()
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``key=value`` arguments and run Studio. ``argv`` excludes the program name."""
    tokens = list(sys.argv[1:] if argv is None else argv)

    if any(token in HELP_FLAGS for token in tokens):
        from nitid.cli import COMMAND_HELP

        print(COMMAND_HELP["studio"])
        return EXIT_OK

    try:
        settings = parse_args(tokens)
    except SettingsError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print('Run "nitid studio --help" for usage.', file=sys.stderr)
        return EXIT_USAGE

    missing = missing_extra_modules()
    if missing:
        print(
            f"ERROR: Nitid app needs {', '.join(missing)}, which "
            f"{'is' if len(missing) == 1 else 'are'} not installed.\n"
            f"Install the optional extra with:\n  {INSTALL_HINT}",
            file=sys.stderr,
        )
        return EXIT_MISSING_EXTRA

    return run(settings)
