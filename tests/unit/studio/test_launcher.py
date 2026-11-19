"""Unit tests for the Studio launcher and the ``nitid studio`` command wiring."""

import importlib
import socket
import time
import subprocess
import sys
from pathlib import Path

import pytest

from nitid.cli import COMMAND_HELP, COMMANDS, main, parse_args
from nitid.studio import launcher
from nitid.studio.settings import StudioSettings

REPO_ROOT = Path(__file__).resolve().parents[3]


# --- lazy imports ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement", ["import nitid.cli", "import nitid.studio", "import nitid.studio.launcher"]
)
def test_importing_cli_and_studio_does_not_import_fastapi_or_uvicorn(statement):
    code = (
        f"import sys; {statement}; "
        "bad = [m for m in ('fastapi', 'starlette', 'uvicorn') if m in sys.modules]; "
        "assert not bad, bad"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr


# --- missing extra --------------------------------------------------------------------------


def _block_imports(monkeypatch, *blocked):
    real = importlib.import_module

    def fake(name, package=None):
        if name.split(".")[0] in blocked:
            raise ImportError(f"No module named {name!r}")
        return real(name, package)

    monkeypatch.setattr(launcher.importlib, "import_module", fake)


def test_missing_extra_prints_install_hint_and_fails(monkeypatch, capsys):
    _block_imports(monkeypatch, "fastapi", "uvicorn")
    called = []
    monkeypatch.setattr(launcher, "run", lambda settings: called.append(settings) or 0)

    code = launcher.main(["--no-browser"])

    assert code == launcher.EXIT_MISSING_EXTRA != 0
    err = capsys.readouterr().err
    assert 'pip install "nitid[studio]"' in err
    assert "fastapi" in err and "uvicorn" in err
    assert not called


def test_only_one_missing_module_is_reported(monkeypatch, capsys):
    _block_imports(monkeypatch, "uvicorn")
    assert launcher.missing_extra_modules() == ["uvicorn"]
    assert launcher.main(["--no-browser"]) == launcher.EXIT_MISSING_EXTRA
    assert "uvicorn, which is not installed" in capsys.readouterr().err


def test_nitid_studio_command_reports_missing_extra_with_nonzero_exit(monkeypatch, capsys):
    _block_imports(monkeypatch, "fastapi", "uvicorn")
    with pytest.raises(SystemExit) as exc_info:
        main(["nitid", "studio", "port=0", "--no-browser"])
    assert exc_info.value.code == launcher.EXIT_MISSING_EXTRA
    assert 'pip install "nitid[studio]"' in capsys.readouterr().err


def test_present_extra_reports_nothing_missing():
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    assert launcher.missing_extra_modules() == []


# --- argument handling ----------------------------------------------------------------------


def test_invalid_arguments_exit_with_usage_error(capsys):
    assert launcher.main(["port=abc"]) == launcher.EXIT_USAGE
    err = capsys.readouterr().err
    assert "port=" in err and "nitid studio --help" in err


def test_wildcard_host_without_token_is_a_usage_error(capsys):
    assert launcher.main(["host=0.0.0.0"]) == launcher.EXIT_USAGE
    assert "explicit token" in capsys.readouterr().err


def test_help_flag_prints_help_without_starting(capsys, monkeypatch):
    monkeypatch.setattr(launcher, "run", lambda s: pytest.fail("must not start"))
    assert launcher.main(["--help"]) == 0
    assert "nitid studio" in capsys.readouterr().out


def test_main_passes_parsed_settings_to_run(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    seen = []
    monkeypatch.setattr(launcher, "run", lambda settings: seen.append(settings) or 0)

    assert (
        launcher.main(["port=0", f"models={tmp_path}", "--no-browser", "token=abcdefgh1234"]) == 0
    )

    (settings,) = seen
    assert settings.port == 0 and settings.open_browser is False
    assert settings.models_dir == tmp_path.resolve()
    assert settings.api_token == "abcdefgh1234"


# --- CLI wiring -----------------------------------------------------------------------------


def test_studio_is_a_registered_command_with_help():
    assert "studio" in COMMANDS
    help_text = COMMAND_HELP["studio"]
    for needle in [
        "host=HOST",
        "port=INT",
        "models=PATH",
        "open=BOOL",
        "--no-browser",
        "token=TOKEN",
        "nitid[studio]",
    ]:
        assert needle in help_text


def test_general_help_lists_studio(capsys):
    with pytest.raises(SystemExit):
        main(["nitid", "--help"])
    assert "studio" in capsys.readouterr().out


def test_studio_help_exits_zero_without_importing_the_package(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "nitid", None)
    with pytest.raises(SystemExit) as exc_info:
        main(["nitid", "studio", "--help"])
    assert exc_info.value.code == 0
    assert "Usage:\n  nitid studio" in capsys.readouterr().out


def test_cli_parse_args_accepts_studio_tokens():
    command, kwargs = parse_args(["nitid", "studio", "port=0", "open=false", "--no-browser"])
    assert command == "studio"
    assert kwargs["port"] == 0 and kwargs["open"] is False


def test_cli_dispatches_raw_tokens_to_studio_main(monkeypatch):
    received = []
    monkeypatch.setattr(launcher, "main", lambda argv: received.append(list(argv)) or 0)

    main(["nitid", "studio", "port=0", "token=00112233", "--no-browser"])

    assert received == [["port=0", "token=00112233", "--no-browser"]]


def test_cli_propagates_nonzero_studio_exit(monkeypatch):
    monkeypatch.setattr(launcher, "main", lambda argv: 7)
    with pytest.raises(SystemExit) as exc_info:
        main(["nitid", "studio"])
    assert exc_info.value.code == 7


def test_report_flag_is_rejected_for_studio(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["nitid", "studio", "--report"])
    assert exc_info.value.code == 1
    assert "--report is supported only for" in capsys.readouterr().out


# --- URL and socket helpers -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "shown"),
    [
        ("127.0.0.1", "127.0.0.1"),
        ("0.0.0.0", "127.0.0.1"),
        ("::", "[::1]"),
        ("::1", "[::1]"),
        ("localhost", "localhost"),
        ("192.168.1.5", "192.168.1.5"),
    ],
)
def test_display_host(host, shown):
    assert launcher.display_host(host) == shown


def test_access_url_has_token_in_fragment_only(tmp_path):
    settings = StudioSettings(port=8000, token="abc-DEF_123456", models_dir=tmp_path)
    assert launcher.access_url(settings) == "http://127.0.0.1:8000/#t=abc-DEF_123456"


def test_access_url_quotes_unusual_tokens(tmp_path):
    settings = StudioSettings(port=8000, token="a b&c=d#e?f".replace(" ", "_"), models_dir=tmp_path)
    url = launcher.access_url(settings)
    assert url.endswith("#t=a_b%26c%3Dd%23e%3Ff")


def test_bind_socket_port_zero_picks_a_free_port():
    sock = launcher.bind_socket("127.0.0.1", 0)
    try:
        host, port = sock.getsockname()[:2]
        assert host == "127.0.0.1" and port > 0
    finally:
        sock.close()


def test_bind_socket_localhost_binds_ipv4_loopback():
    sock = launcher.bind_socket("localhost", 0)
    try:
        assert sock.family == socket.AF_INET
        assert sock.getsockname()[0] == "127.0.0.1"
    finally:
        sock.close()


def test_bind_socket_reports_port_in_use():
    first = launcher.bind_socket("127.0.0.1", 0)
    first.listen(1)
    try:
        with pytest.raises(OSError):
            launcher.bind_socket("127.0.0.1", first.getsockname()[1])
    finally:
        first.close()


def test_run_reports_port_in_use_without_traceback(capsys, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    first = launcher.bind_socket("127.0.0.1", 0)
    first.listen(1)
    try:
        settings = StudioSettings(
            port=first.getsockname()[1], models_dir=tmp_path, open_browser=False
        )
        assert launcher.run(settings) == launcher.EXIT_FAILURE
        assert "cannot listen" in capsys.readouterr().err
    finally:
        first.close()


# --- announcement after start-up and clean Ctrl+C ----------------------------------------------


class FakeServer:
    """Stands in for ``uvicorn.Server``: ``run`` flips ``started`` the way uvicorn does."""

    def __init__(self, *, start=True, interrupt=False):
        self.started = False
        self.should_exit = False
        self._start = start
        self._interrupt = interrupt
        self.announced_before_start = None
        self.seen_output = None

    def run(self, sockets=None):
        # Give a watcher thread the chance to (wrongly) print before we are "listening".
        time.sleep(0.2)
        self.seen_output = self.capture()
        if self._start:
            self.started = True
            time.sleep(0.3)  # let the announcer thread run
        else:
            self.should_exit = True
        if self._interrupt:
            raise KeyboardInterrupt

    capture = staticmethod(lambda: None)


def _patch_server(monkeypatch, fake, capsys):
    pytest.importorskip("fastapi")
    fake.capture = lambda: capsys.readouterr().out
    monkeypatch.setattr(launcher, "make_server", lambda app, **kw: fake)
    opened = []
    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: opened.append(url))
    return opened


def test_url_is_printed_only_after_the_server_is_listening(monkeypatch, capsys, tmp_path):
    fake = FakeServer()
    opened = _patch_server(monkeypatch, fake, capsys)
    settings = StudioSettings(port=0, token="token-123456", models_dir=tmp_path, open_browser=True)

    assert launcher.run(settings) == launcher.EXIT_OK

    assert "Open:" not in fake.seen_output  # nothing announced while still starting
    after = capsys.readouterr().out
    assert "Open: http://127.0.0.1:" in after and "Press Ctrl+C to stop." in after
    assert len(opened) == 1 and opened[0].startswith("http://127.0.0.1:")


def test_nothing_is_announced_and_no_browser_opens_if_the_server_never_starts(
    monkeypatch, capsys, tmp_path
):
    fake = FakeServer(start=False)
    opened = _patch_server(monkeypatch, fake, capsys)
    settings = StudioSettings(port=0, token="token-123456", models_dir=tmp_path, open_browser=True)

    launcher.run(settings)
    time.sleep(0.2)

    assert "Open:" not in fake.seen_output + capsys.readouterr().out
    assert opened == []


def test_browser_is_not_opened_when_open_is_false(monkeypatch, capsys, tmp_path):
    fake = FakeServer()
    opened = _patch_server(monkeypatch, fake, capsys)
    settings = StudioSettings(port=0, token="token-123456", models_dir=tmp_path, open_browser=False)

    launcher.run(settings)

    assert "Open:" in capsys.readouterr().out
    assert opened == []


def test_keyboard_interrupt_after_graceful_shutdown_is_a_clean_exit(monkeypatch, capsys, tmp_path):
    fake = FakeServer(interrupt=True)
    _patch_server(monkeypatch, fake, capsys)
    settings = StudioSettings(port=0, token="token-123456", models_dir=tmp_path, open_browser=False)

    assert launcher.run(settings) == launcher.EXIT_OK == 0


def test_announce_when_ready_returns_false_on_timeout(tmp_path):
    server = FakeServer()
    settings = StudioSettings(port=0, token="token-123456", models_dir=tmp_path)
    assert launcher._announce_when_ready(server, settings, "http://x", timeout=0.05) is False


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT delivery differs on Windows")
def test_real_process_exits_zero_on_sigint(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    import os
    import signal

    process = subprocess.Popen(
        [sys.executable, "-m", "nitid.studio", "port=0", "--no-browser", f"models={tmp_path}"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        deadline = time.monotonic() + 120
        seen = ""
        while "Press Ctrl+C" not in seen:
            assert time.monotonic() < deadline, f"no start-up line; got {seen!r}"
            line = process.stdout.readline()
            assert line != "" or process.poll() is None, process.stderr.read()
            seen += line
        assert "Open: http://127.0.0.1:" in seen
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
