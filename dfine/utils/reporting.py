"""CLI bug-report generation and stdout/stderr tee capture."""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO, cast

from dfine.utils import runs


@dataclass(frozen=True)
class BugReport:
    """A Python API bug report created by :func:`bugreport`."""

    path: Path


class _Tee(io.TextIOBase):
    """Write text to the original terminal stream and a report file."""

    def __init__(self, terminal: TextIO, report: TextIO) -> None:
        self.terminal = terminal
        self.report = report

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.report.write(text)
        self.report.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.report.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def fileno(self) -> int:
        return self.terminal.fileno()

    def __getattr__(self, name: str):
        return getattr(self.terminal, name)


def _report_path(command: str, report_dir: str | Path) -> Path:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    safe_command = re.sub(r"[^a-zA-Z0-9_.-]+", "-", command).strip("-.") or "python"
    return report_dir / f"dfine-{safe_command}-{timestamp}-{os.getpid()}.log"


def _write_header(stream: TextIO, command: str) -> None:
    stream.write("nitid bug report\n")
    stream.write(f"command: {command}\n\n")
    stream.write("environment:\n")
    stream.write(runs.environment_yaml())
    stream.write("\ncommand output:\n")
    stream.flush()


@contextlib.contextmanager
def capture_command_report(
    command: str, report_dir: str | Path = "runs/bugreports"
) -> Iterator[Path]:
    """Tee stdout and stderr to a report containing the shared environment snapshot."""
    path = _report_path(command, report_dir)
    with path.open("w", encoding="utf-8", buffering=1) as report:
        _write_header(report, command)
        stdout = cast(TextIO, _Tee(sys.stdout, report))
        stderr = cast(TextIO, _Tee(sys.stderr, report))
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield path


@contextlib.contextmanager
def bugreport(
    name: str = "python", report_dir: str | Path = "runs/bugreports"
) -> Iterator[BugReport]:
    """Capture Python API stdout/stderr and append any exception before re-raising it."""
    path = _report_path(name, report_dir)
    result = BugReport(path=path)
    try:
        with path.open("w", encoding="utf-8", buffering=1) as report:
            _write_header(report, name)
            stdout = cast(TextIO, _Tee(sys.stdout, report))
            stderr = cast(TextIO, _Tee(sys.stderr, report))
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    yield result
                except BaseException as error:
                    traceback.print_exception(type(error), error, error.__traceback__, file=report)
                    report.flush()
                    raise
    finally:
        print(f"Bug report saved to {path}")


def write_standalone_report(report_dir: str | Path = "runs/bugreports") -> Path:
    """Write an environment-only report for failures outside a model command."""
    path = _report_path("bugreport", report_dir)
    with path.open("w", encoding="utf-8") as report:
        report.write("nitid standalone bug report\n\n")
        report.write("environment:\n")
        report.write(runs.environment_yaml())
    return path
