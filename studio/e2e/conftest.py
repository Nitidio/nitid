"""Fixtures of the Studio end-to-end tests.

Run explicitly (``pytest studio/e2e``); the default ``testpaths`` do not include this folder.
Every missing prerequisite is a clean skip with the reason.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest
from support import (
    REPO_ROOT,
    STATIC_INDEX,
    STUDIO_DIR,
    StudioProcess,
    child_env,
    find_chrome,
    real_studio,
)

for _module in ("fastapi", "uvicorn", "multipart", "onnx", "onnxruntime", "httpx", "cv2"):
    pytest.importorskip(_module, reason=f"the e2e tests need {_module} (extras: studio, dev)")


@pytest.fixture(scope="session")
def chrome() -> str:
    path = find_chrome()
    if path is None:
        pytest.skip("no Chrome/Chromium found (set CHROME_PATH)")
    return path


@pytest.fixture(scope="session")
def node() -> str:
    path = shutil.which("node")
    if path is None:
        pytest.skip("node not found (Node >= 22.13 is needed for the drivers)")
    return path


@pytest.fixture(scope="session")
def fixtures(tmp_path_factory, chrome, node):
    """Tiny checkpoints (models/) and the test image, built at run time."""
    from make_fixtures import build

    return build(tmp_path_factory.mktemp("e2e-fixtures"))


@pytest.fixture(scope="session")
def built_ui(chrome, node) -> None:
    """The product UI in ``nitid/studio/static`` (built with pnpm if it is missing)."""
    if STATIC_INDEX.exists():
        return
    if shutil.which("pnpm") is None:
        pytest.skip("nitid/studio/static is not built and pnpm is missing")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build_studio.py")],
        capture_output=True,
        text=True,
        env=child_env(),
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"the UI build failed ({result.returncode}): {result.stdout[-400:]}")


@pytest.fixture(scope="module")
def studio(fixtures, built_ui):
    """A real ``nitid studio`` process (the product launcher) serving the built UI."""
    server = real_studio(fixtures["detect"].parent).start()
    yield server
    status = server.stop()
    assert status == 0, f"nitid studio did not exit cleanly after Ctrl+C: {status}\n" + "\n".join(
        server.output[-20:]
    )


@pytest.fixture(scope="session")
def studio_dir():
    assert STUDIO_DIR.is_dir()
    return STUDIO_DIR


__all__ = ["StudioProcess"]
