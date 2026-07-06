"""Unit tests for command-line help output."""

import sys

import pytest

from tools.dfine_cli import COMMAND_HELP, COMMANDS, main


@pytest.mark.parametrize(
    ("command", "expected_text"),
    [
        ("predict", ("source=SOURCE", "conf=FLOAT", "dfine predict")),
        ("download", ("model=NAME", "force=BOOL", "dfine download")),
        ("train", ("data=PATH", "epochs=INT", "dfine train")),
        ("val", ("data=PATH", "split=NAME", "dfine val")),
        ("export", ("format=FORMAT", "opset=INT", "dfine export")),
        ("info", ("detailed=BOOL", "dfine info")),
    ],
)
def test_command_help_exits_successfully_without_loading_model(
    command, expected_text, capsys, monkeypatch
):
    monkeypatch.setitem(sys.modules, "dfine", None)

    with pytest.raises(SystemExit) as exc_info:
        main(["dfine", command, "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for text in expected_text:
        assert text in output


def test_every_command_has_help_text():
    assert set(COMMAND_HELP) == COMMANDS


def test_short_help_flag_is_supported(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["dfine", "predict", "-h"])

    assert exc_info.value.code == 0
    assert "Usage:\n  dfine predict" in capsys.readouterr().out


def test_general_help_exits_successfully(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["dfine", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "nitid D-FINE CLI" in output
    assert "dfine COMMAND --help" in output


def test_unknown_command_exits_with_error(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["dfine", "unknown"])

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "ERROR: unknown command 'unknown'" in output
    assert "Commands:" in output
