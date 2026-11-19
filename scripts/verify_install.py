"""Exercise detection and segmentation through an installed nitid package."""

from __future__ import annotations

import argparse
import importlib.util
import os
import queue
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from importlib.metadata import version
from importlib.resources import as_file
from pathlib import Path

import torch
import yaml

import dfine
from dfine import DFINE, __version__
from dfine.data import COCO_NAMES
from dfine.nn.build import build_model
from dfine.nn.configs import make_model_config


def _checkpoint(root: Path, task: str) -> Path:
    config = make_model_config(
        "dfine_s",
        task=task,
        num_classes=2,
        image_size=(64, 64),
    )
    config["DFINETransformer"]["num_layers"] = 1
    config["DFINETransformer"]["num_queries"] = 10
    config["DFINETransformer"]["num_denoising"] = 0
    config["HybridEncoder"]["depth_mult"] = 0.1
    model = build_model(config)
    path = root / f"{task}.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "names": {0: "class_0", 1: "class_1"},
            "epoch": -1,
            "metrics": {},
        },
        path,
    )
    return path


def _verify_coco_names() -> None:
    """Pretrained downloads embed these names; a missing resource breaks NITID("nitid1s")."""
    with as_file(COCO_NAMES) as path:
        names = yaml.safe_load(path.read_text()).get("names", {})
    if len(names) != 80:
        raise RuntimeError(f"Bundled COCO names are incomplete: {len(names)} of 80 classes")


STUDIO_START_TIMEOUT_S = 120.0  # a cold start imports torch, which is slow on small CI machines
STUDIO_STOP_TIMEOUT_S = 15.0
STUDIO_HTTP_TIMEOUT_S = 10.0
STUDIO_URL_PATTERN = re.compile(r"^Open:\s+(http://\S+)", re.MULTILINE)


def studio_extra_installed() -> bool:
    """True when the optional ``studio`` extra (FastAPI and uvicorn) can be imported."""
    return all(importlib.util.find_spec(name) is not None for name in ("fastapi", "uvicorn"))


def parse_studio_url(output: str) -> str | None:
    """Base URL (``http://host:port``) from the launcher's ``Open: <url>#t=...`` line."""
    match = STUDIO_URL_PATTERN.search(output)
    if match is None:
        return None
    return match.group(1).split("#", 1)[0].rstrip("/")


def _http_get(url: str) -> tuple[int, str]:
    """GET without proxies; returns ``(status, body)`` and does not raise on HTTP errors."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=STUDIO_HTTP_TIMEOUT_S) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


def _http_get_when_ready(url: str, timeout: float = 30.0) -> tuple[int, str]:
    """Like ``_http_get`` but retries refused connections.

    The launcher prints its URL after binding the port but before uvicorn starts listening.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            return _http_get(url)
        except (urllib.error.URLError, ConnectionError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.2)


def _wait_for_url(
    lines: queue.Queue[str | None], process: subprocess.Popen[str], timeout: float
) -> tuple[str | None, str]:
    """Collect output until the URL line appears, the process ends or ``timeout`` passes."""
    collected = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=0.2)
        except queue.Empty:
            if process.poll() is not None and lines.empty():
                break
            continue
        if line is None:
            break
        collected += line
        url = parse_studio_url(collected)
        if url:
            return url, collected
    return None, collected


def _stop(process: subprocess.Popen[str]) -> int:
    """Ask the server to stop (Ctrl+C); kill it only if it does not exit in time."""
    if process.poll() is None:
        if os.name == "nt":
            process.terminate()
        else:
            process.send_signal(signal.SIGINT)
    try:
        return process.wait(timeout=STUDIO_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise RuntimeError(f"Studio did not stop within {STUDIO_STOP_TIMEOUT_S:.0f} s of Ctrl+C")


def verify_studio(*, require_ui: bool = False) -> bool:
    """Start Studio from the installed package and probe it over HTTP.

    Returns ``False`` (after saying why) when the ``studio`` extra is not installed.
    With ``require_ui`` the packaged single-page app must be served at ``/``; without it, a
    missing UI is accepted but must answer with the documented 503 page.
    """
    if not studio_extra_installed():
        print('Skipping the Studio check: install the extra to run it: pip install "nitid[studio]"')
        return False

    token = secrets.token_urlsafe(24)
    command = [sys.executable, "-m", "nitid.studio", "port=0", "--no-browser", f"token={token}"]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=tempfile.gettempdir(),
    )
    assert process.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def pump(stream: object) -> None:
        for line in stream:  # type: ignore[attr-defined]
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, args=(process.stdout,), daemon=True).start()

    output = ""
    try:
        base_url, output = _wait_for_url(lines, process, STUDIO_START_TIMEOUT_S)
        if base_url is None:
            raise RuntimeError(
                f"Studio did not print its URL within {STUDIO_START_TIMEOUT_S:.0f} s "
                f"(exit code {process.poll()}). Output:\n{output}"
            )

        status, body = _http_get_when_ready(f"{base_url}/api/v1/health")
        if status != 200 or '"status"' not in body or '"ok"' not in body:
            raise RuntimeError(f"GET /api/v1/health returned {status}: {body[:200]}")

        status, body = _http_get(f"{base_url}/")
        if require_ui:
            if status != 200 or 'id="root"' not in body:
                raise RuntimeError(
                    f"GET / did not serve the Studio UI (status {status}); the wheel was built "
                    f'without nitid/studio/static/index.html or it has no id="root": {body[:200]}'
                )
            ui = "UI served"
        elif status == 200 and 'id="root"' in body:
            ui = "UI served"
        elif status == 503:
            ui = "no UI in this installation (503 page, as documented)"
        else:
            raise RuntimeError(f"GET / returned an unexpected response ({status}): {body[:200]}")
    finally:
        exit_code = _stop(process)
        process.stdout.close()

    print(f"Studio verified at {base_url}: health ok, {ui}, stopped (exit code {exit_code})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-ui",
        action="store_true",
        help="fail if the installed Studio does not serve the packaged UI at /",
    )
    args = parser.parse_args()

    package_path = Path(dfine.__file__).resolve()
    if "site-packages" not in package_path.parts:
        raise RuntimeError(f"Expected an installed package, imported {package_path}")
    if version("nitid") != __version__:
        raise RuntimeError("Installed distribution version does not match the release candidate")
    _verify_coco_names()

    with tempfile.TemporaryDirectory(prefix="nitid-wheel-") as directory:
        root = Path(directory)
        detector = DFINE(_checkpoint(root, "detect"), task="detect", device="cpu", verbose=False)
        segmenter = DFINE(_checkpoint(root, "segment"), task="segment", device="cpu", verbose=False)

        if detector.task != "detect" or segmenter.task != "segment":
            raise RuntimeError("Installed task API returned an unexpected task")
        if any(key.startswith("decoder.mask_") for key in detector._model.state_dict()):
            raise RuntimeError("Installed detection model unexpectedly contains a mask head")
        if not any(key.startswith("decoder.mask_") for key in segmenter._model.state_dict()):
            raise RuntimeError("Installed segmentation model is missing its mask head")

    print(f"Installed detection and segmentation APIs verified: {package_path}")
    verify_studio(require_ui=args.require_ui)


if __name__ == "__main__":
    main()
