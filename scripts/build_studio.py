"""Build the Nitid Studio frontend into ``nitid/studio/static/``.

The compiled single-page app is not committed: it is produced from ``studio/``
with pnpm and shipped inside the wheel (see ``docs/studio-packaging.md``).

Exit codes:
    0  success (or ``--check-only`` and ``static/index.html`` exists)
    1  a pnpm command failed, or the build did not produce ``index.html``
    2  the build environment is unusable (Node too old, pnpm missing, no ``studio/``)
    3  ``--check-only`` and ``static/index.html`` is missing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIO_DIR = REPO_ROOT / "studio"
STATIC_DIR = REPO_ROOT / "nitid" / "studio" / "static"

MIN_NODE = (22, 13)

EXIT_OK = 0
EXIT_BUILD_FAILED = 1
EXIT_ENVIRONMENT = 2
EXIT_MISSING_STATIC = 3

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class BuildError(Exception):
    """A build step failed; ``code`` is the process exit code to use."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def parse_version(text: str) -> tuple[int, ...]:
    """Parse ``v22.13.1`` or ``10.28.0`` into a tuple of ints."""
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match is None:
        raise ValueError(f"cannot parse a version from {text!r}")
    return tuple(int(part) for part in match.groups() if part is not None)


def expected_pnpm_version(studio_dir: Path) -> str | None:
    """The pnpm version pinned by ``packageManager`` in ``studio/package.json``."""
    package = json.loads((studio_dir / "package.json").read_text(encoding="utf-8"))
    manager = str(package.get("packageManager", ""))
    if manager.startswith("pnpm@"):
        return manager.split("@", 1)[1].split("+", 1)[0]
    return None


def _version_of(command: Sequence[str], run: Runner) -> str:
    result = run(list(command), capture_output=True, text=True, check=False)
    return str(result.stdout).strip()


def check_node(
    run: Runner = subprocess.run, which: Callable[[str], str | None] = shutil.which
) -> str:
    """Require Node.js >= 22.13; returns the version string."""
    node = which("node")
    if node is None:
        raise BuildError("Node.js was not found. Install Node.js >= 22.13.", EXIT_ENVIRONMENT)
    raw = _version_of([node, "--version"], run)
    try:
        version = parse_version(raw)
    except ValueError:
        raise BuildError(f"Could not read the Node.js version (got {raw!r}).", EXIT_ENVIRONMENT)
    if version[:2] < MIN_NODE:
        needed = ".".join(map(str, MIN_NODE))
        raise BuildError(f"Node.js >= {needed} is required, found {raw}.", EXIT_ENVIRONMENT)
    return raw


def check_pnpm(
    studio_dir: Path,
    run: Runner = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> tuple[str, str]:
    """Require pnpm; returns ``(executable, version)``. A version other than the pinned one only warns."""
    pnpm = which("pnpm")
    if pnpm is None:
        raise BuildError(
            "pnpm was not found. Run `corepack enable` (it picks the version pinned in "
            "studio/package.json) or install pnpm.",
            EXIT_ENVIRONMENT,
        )
    raw = _version_of([pnpm, "--version"], run)
    expected = expected_pnpm_version(studio_dir)
    if expected and raw != expected:
        print(
            f"WARNING: pnpm {raw} found, but studio/package.json pins pnpm {expected}.",
            file=sys.stderr,
        )
    return pnpm, raw


def build(
    studio_dir: Path = STUDIO_DIR,
    static_dir: Path = STATIC_DIR,
    run: Runner = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    env: Mapping[str, str] | None = None,
) -> None:
    """Install dependencies from the lockfile and build into ``static_dir``."""
    if not (studio_dir / "package.json").is_file():
        raise BuildError(
            f"No frontend found at {studio_dir} (missing package.json).", EXIT_ENVIRONMENT
        )
    node = check_node(run, which)
    pnpm, pnpm_version = check_pnpm(studio_dir, run, which)
    print(f"Node.js {node}, pnpm {pnpm_version}")

    build_env = dict(os.environ if env is None else env)
    build_env["STUDIO_OUT_DIR"] = str(static_dir)

    for step in (["install", "--frozen-lockfile"], ["run", "build"]):
        label = "pnpm " + " ".join(step)
        print(f"==> {label}")
        result = run([pnpm, *step], cwd=studio_dir, env=build_env, check=False)
        if result.returncode != 0:
            raise BuildError(
                f"`{label}` failed with exit code {result.returncode}.", EXIT_BUILD_FAILED
            )

    check_static(static_dir, after_build=True)


def check_static(static_dir: Path = STATIC_DIR, *, after_build: bool = False) -> None:
    """Require ``static/index.html``."""
    index = static_dir / "index.html"
    if index.is_file():
        print(f"OK: {index}")
        return
    if after_build:
        raise BuildError(f"The build finished but {index} was not produced.", EXIT_BUILD_FAILED)
    raise BuildError(
        f"{index} is missing. Build the UI with `python scripts/build_studio.py`.",
        EXIT_MISSING_STATIC,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="do not build; only check that nitid/studio/static/index.html exists",
    )
    args = parser.parse_args(argv)
    try:
        if args.check_only:
            check_static(STATIC_DIR)
        else:
            build(STUDIO_DIR, STATIC_DIR)
    except BuildError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return error.code
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
