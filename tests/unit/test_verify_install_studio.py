"""Studio smoke check of scripts/verify_install.py (parsing, skipping, and a real launch)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "verify_install", REPO_ROOT / "scripts/verify_install.py"
)
assert _spec is not None and _spec.loader is not None
verify_install = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_install)

needs_extra = pytest.mark.skipif(
    not verify_install.studio_extra_installed(), reason="the studio extra is not installed"
)


def test_parse_studio_url_strips_the_token_fragment():
    output = "Nitid Studio 1.0\nOpen: http://127.0.0.1:51234/#t=abc%2Bdef\nPress Ctrl+C to stop.\n"
    assert verify_install.parse_studio_url(output) == "http://127.0.0.1:51234"


def test_parse_studio_url_ipv6_and_missing():
    assert verify_install.parse_studio_url("Open: http://[::1]:8000/#t=x\n") == "http://[::1]:8000"
    assert verify_install.parse_studio_url("Nitid Studio 1.0\nModels directory: /m\n") is None


def test_skips_without_the_extra(monkeypatch, capsys):
    monkeypatch.setattr(verify_install, "studio_extra_installed", lambda: False)

    def must_not_start(*_args, **_kwargs):
        raise AssertionError("Studio must not be started without the extra")

    monkeypatch.setattr(verify_install.subprocess, "Popen", must_not_start)
    assert verify_install.verify_studio(require_ui=True) is False
    assert 'pip install "nitid[studio]"' in capsys.readouterr().out


@needs_extra
def test_real_launch_serves_health_and_stops(capsys):
    # A source checkout has no built UI, so "/" must answer the documented 503 page.
    static_index = REPO_ROOT / "nitid/studio/static/index.html"
    if static_index.exists():
        pytest.skip("a built UI is present; covered by the require_ui test")
    assert verify_install.verify_studio(require_ui=False) is True
    assert "health ok" in capsys.readouterr().out


@needs_extra
def test_require_ui_fails_without_a_built_ui():
    if (REPO_ROOT / "nitid/studio/static/index.html").exists():
        pytest.skip("a built UI is present")
    with pytest.raises(RuntimeError, match="did not serve the Studio UI"):
        verify_install.verify_studio(require_ui=True)
