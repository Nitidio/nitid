"""``nitid.__version__`` must exist and agree with the package metadata sources."""

from __future__ import annotations

import re
from pathlib import Path


def _pyproject_version() -> str:
    text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    # Only look inside the ``[project]`` table (up to the next table header).
    project = text.split("[project]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"', project, flags=re.MULTILINE)
    assert match is not None, "no version found in the [project] table of pyproject.toml"
    return match.group(1)


def test_nitid_exposes_version() -> None:
    import nitid

    assert isinstance(nitid.__version__, str)
    assert nitid.__version__


def test_nitid_version_matches_dfine_version() -> None:
    import dfine
    import nitid

    assert nitid.__version__ == dfine.__version__


def test_nitid_version_matches_pyproject() -> None:
    import nitid

    assert nitid.__version__ == _pyproject_version()


def test_version_is_part_of_public_all() -> None:
    import nitid

    assert "__version__" in nitid.__all__
    # ``from nitid import *`` must not fail because of the new name.
    namespace: dict[str, object] = {}
    exec("from nitid import *", namespace)  # noqa: S102
    assert namespace["__version__"] == nitid.__version__
