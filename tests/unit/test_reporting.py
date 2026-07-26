"""Tests for CLI command and standalone bug reports."""

import sys

import pytest

from dfine import BugReport, bugreport
from dfine.utils import runs
from tools import dfine_cli


def _only_report(tmp_path):
    reports = list(tmp_path.glob("*.log"))
    assert len(reports) == 1
    return reports[0]


def test_report_tees_complete_stdout_and_stderr(tmp_path, capsys, monkeypatch):
    def execute(argv):
        assert "--report" not in argv
        print("stdout message")
        print("stderr message", file=sys.stderr)

    monkeypatch.setattr(dfine_cli, "_execute", execute)
    dfine_cli.main(["dfine", "predict", "--report"], report_dir=tmp_path)

    captured = capsys.readouterr()
    assert "stdout message" in captured.out
    assert "stderr message" in captured.err
    assert "Bug report saved to" in captured.out

    report = _only_report(tmp_path).read_text()
    assert "environment:" in report
    assert "stdout message" in report
    assert "stderr message" in report


def test_report_captures_crash_traceback(tmp_path, capsys, monkeypatch):
    def execute(argv):
        print("before crash")
        raise RuntimeError("deliberate crash")

    monkeypatch.setattr(dfine_cli, "_execute", execute)
    with pytest.raises(SystemExit) as error:
        dfine_cli.main(["dfine", "train", "--report"], report_dir=tmp_path)

    assert error.value.code == 1
    assert "Traceback" in capsys.readouterr().err
    report = _only_report(tmp_path).read_text()
    assert "before crash" in report
    assert "Traceback" in report
    assert "RuntimeError: deliberate crash" in report


def test_report_is_published_when_command_exits_with_validation_error(
    tmp_path, capsys, monkeypatch
):
    def execute(argv):
        print("validation failed", file=sys.stderr)
        raise SystemExit(2)

    monkeypatch.setattr(dfine_cli, "_execute", execute)
    with pytest.raises(SystemExit) as error:
        dfine_cli.main(["dfine", "val", "--report"], report_dir=tmp_path)

    assert error.value.code == 2
    assert "Bug report saved to" in capsys.readouterr().out
    assert "validation failed" in _only_report(tmp_path).read_text()


def test_standalone_bugreport_reuses_environment_snapshot(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(runs, "collect_environment", lambda: {"snapshot_marker": "shared"})

    dfine_cli.main(["dfine", "bugreport"], report_dir=tmp_path)

    assert "Bug report saved to" in capsys.readouterr().out
    report = _only_report(tmp_path).read_text()
    assert "snapshot_marker: shared" in report
    assert "command output:" not in report


def test_python_bugreport_context_captures_output(tmp_path, capsys):
    with bugreport("python experiment", report_dir=tmp_path) as report:
        print("python stdout")
        print("python stderr", file=sys.stderr)

    assert isinstance(report, BugReport)
    assert report.path.is_file()
    assert "python-experiment" in report.path.name
    captured = capsys.readouterr()
    assert "python stdout" in captured.out
    assert "python stderr" in captured.err
    assert "Bug report saved to" in captured.out
    contents = report.path.read_text()
    assert "python stdout" in contents
    assert "python stderr" in contents


def test_python_bugreport_context_logs_and_reraises_exception(tmp_path, capsys):
    original = RuntimeError("python API crash")

    with pytest.raises(RuntimeError) as raised:
        with bugreport("train", report_dir=tmp_path) as report:
            print("before Python crash")
            raise original

    assert raised.value is original
    assert "Bug report saved to" in capsys.readouterr().out
    contents = report.path.read_text()
    assert "before Python crash" in contents
    assert "Traceback" in contents
    assert "RuntimeError: python API crash" in contents
