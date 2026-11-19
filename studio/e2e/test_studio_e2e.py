"""Studio end to end: the real server process, the built UI and real Chrome.

Drivers: ``ui.mjs`` (user flows) and ``security.mjs`` (headers, guards, CSP).
"""

from __future__ import annotations

import json

import pytest
from support import Skipped, failed_checks, run_driver


def drive(script, tmp_path, studio, fixtures):
    out = tmp_path / f"{script}.json"
    try:
        result = run_driver(script, out, "--url", studio.url, "--image", str(fixtures["image"]))
    except Skipped as why:
        pytest.skip(str(why))
    print(result["log"])
    print("DATA", json.dumps(result["data"], indent=1))
    assert not failed_checks(result), "\n".join(failed_checks(result))
    return result


def test_ui_flows(studio, fixtures, tmp_path):
    result = drive("ui.mjs", tmp_path, studio, fixtures)
    assert len(result["checks"]) >= 30


def test_security_in_real_chrome(studio, fixtures, tmp_path):
    drive("security.mjs", tmp_path, studio, fixtures)
