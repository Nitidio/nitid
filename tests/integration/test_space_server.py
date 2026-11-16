"""Start the demo the way the container does: configuration comes from environment variables."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("gradio", reason="gradio is not a dependency of nitid")

REPO_ROOT = Path(__file__).resolve().parents[2]
APP = REPO_ROOT / "spaces" / "demo" / "app.py"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_app_serves_on_the_host_and_port_given_by_the_environment() -> None:
    port = _free_port()
    env = {**os.environ, "GRADIO_SERVER_NAME": "0.0.0.0", "GRADIO_SERVER_PORT": str(port)}
    process = subprocess.Popen(
        [sys.executable, str(APP)],
        cwd=APP.parent,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        status = None
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and process.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                    status = response.status
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(0.5)
        assert status == 200, f"server did not answer on port {port}"
    finally:
        process.terminate()
        try:
            output, _ = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
        if process.returncode not in (0, -15, None) and status != 200:
            print(output)
