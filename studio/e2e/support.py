"""Helpers shared by the Studio end-to-end tests: processes, prerequisites, driver calls."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

E2E_DIR = Path(__file__).resolve().parent
REPO_ROOT = E2E_DIR.parents[1]
STUDIO_DIR = REPO_ROOT / "studio"
STATIC_INDEX = REPO_ROOT / "nitid" / "studio" / "static" / "index.html"
EXIT_SKIP = 77  # same convention as lib.mjs
OPEN_LINE = re.compile(r"^Open: (http://\S+)$")


def find_chrome() -> str | None:
    """Path of a Chrome/Chromium binary (``CHROME_PATH`` wins), or ``None``."""
    candidates = [
        os.environ.get("CHROME_PATH"),
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    return next((c for c in candidates if c and Path(c).exists()), None)


def child_env() -> dict[str, str]:
    """Environment for child processes: the worktree on ``PYTHONPATH``, no stray proxies."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    env["NO_PROXY"] = "127.0.0.1,localhost," + env.get("NO_PROXY", "")
    return env


@dataclass
class StudioProcess:
    """A real ``nitid studio``-style process listening on a free port."""

    command: list[str]
    process: subprocess.Popen[str] | None = None
    url: str = ""
    output: list[str] = field(default_factory=list)

    @property
    def origin(self) -> str:
        parts = urlsplit(self.url)
        return f"{parts.scheme}://{parts.netloc}"

    @property
    def port(self) -> int:
        return urlsplit(self.url).port or 0

    @property
    def token(self) -> str:
        return urlsplit(self.url).fragment.removeprefix("t=")

    def start(self, timeout: float = 60.0) -> StudioProcess:
        self.process = subprocess.Popen(  # noqa: S603
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=child_env(),
        )
        found = threading.Event()

        def drain() -> None:
            assert self.process is not None and self.process.stdout is not None
            for line in self.process.stdout:
                self.output.append(line.rstrip("\n"))
                match = OPEN_LINE.match(line.strip())
                if match and not found.is_set():
                    self.url = match.group(1)
                    found.set()

        threading.Thread(target=drain, daemon=True).start()
        deadline = time.monotonic() + timeout
        while not found.is_set():
            if self.process.poll() is not None or time.monotonic() > deadline:
                self.stop()
                raise RuntimeError("Studio did not start:\n" + "\n".join(self.output))
            time.sleep(0.05)
        return self

    def stop(self) -> int | None:
        """Ctrl+C the process like a user would; returns its exit status."""
        proc = self.process
        if proc is None:
            return None
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        return proc.returncode


def real_studio(models_dir: Path, *extra: str) -> StudioProcess:
    """``python -m nitid.studio port=0 models=DIR --no-browser`` (the product's own launcher)."""
    command = [
        sys.executable,
        "-m",
        "nitid.studio",
        "port=0",
        f"models={models_dir}",
        "--no-browser",
    ]
    return StudioProcess([*command, *extra])


def harness_studio(models_dir: Path, static_dir: Path) -> StudioProcess:
    """The real app serving the parity harness instead of the product UI."""
    return StudioProcess(
        [sys.executable, str(E2E_DIR / "serve_harness.py"), str(models_dir), str(static_dir)]
    )


def run_driver(script: str, out: Path, *args: str, timeout: float = 600.0) -> dict:
    """Run a Node driver; return its JSON result. Raises ``Skipped`` for exit code 77."""
    node = shutil.which("node")
    if node is None:
        raise Skipped("node not found")
    command = [node, str(E2E_DIR / script), *args, "--out", str(out)]
    result = subprocess.run(  # noqa: S603
        command, capture_output=True, text=True, timeout=timeout, env=child_env(), check=False
    )
    if result.returncode == EXIT_SKIP:
        raise Skipped(result.stdout.strip() or "driver skipped")
    if not out.exists():
        raise AssertionError(
            f"{script} crashed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
    return {**json.loads(out.read_text(encoding="utf-8")), "log": result.stdout + result.stderr}


class Skipped(Exception):
    """A prerequisite is missing; the test should be skipped."""


def failed_checks(result: dict) -> list[str]:
    return [f"{c['name']}: {c['detail']}" for c in result["checks"] if not c["ok"]]
