"""The hosted deployment end to end, with no Python backend: ``studio/scripts/smoke-hosted.mjs``.

Builds the UI with ``--mode hosted``, assembles it with a fixture model bundle through
``scripts/assemble_hosted.py``, serves it with the production ``_headers`` and drives real Chrome.
Missing Chrome, Node, pnpm or playwright-core are skips with a reason.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from support import STUDIO_DIR, child_env


def test_hosted_site_runs_in_chrome_without_a_backend(chrome, node):
    if shutil.which("pnpm") is None:
        pytest.skip("pnpm not found")
    if not (STUDIO_DIR / "node_modules").is_dir():
        pytest.skip(
            "studio/node_modules is missing (run: cd studio && pnpm install --frozen-lockfile)"
        )
    result = subprocess.run(  # noqa: S603
        [node, "scripts/smoke-hosted.mjs"],
        cwd=STUDIO_DIR,
        capture_output=True,
        text=True,
        env={**child_env(), "CHROME_PATH": chrome},
        timeout=600,
        check=False,
    )
    output = result.stdout + result.stderr
    print(output)
    if "playwright-core not found" in output:
        pytest.skip("playwright-core not found (set PLAYWRIGHT_CORE_PATH)")
    failures = [line for line in output.splitlines() if line.startswith("FAIL")]
    assert result.returncode == 0 and not failures, "\n".join(failures) or output[-2000:]
    assert output.count("PASS ") >= 15
    assert "smoke:hosted OK" in output
