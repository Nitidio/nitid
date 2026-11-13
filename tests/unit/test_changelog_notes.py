from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "changelog_notes", REPO_ROOT / "scripts/changelog_notes.py"
)
assert _spec is not None and _spec.loader is not None
changelog_notes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(changelog_notes)

CHANGELOG = """# Changelog

## [Unreleased]

### Added

- Something not released yet.

## [0.9.4] - 2026-11-06

### Added

- `nitid benchmark`.

### Fixed

- A bug.

## [0.9.3] - 2026-11-05

### Fixed

- An older bug.

## [0.9.2] - 2026-10-09
"""


def test_extracts_only_the_requested_section() -> None:
    notes = changelog_notes.extract_notes(CHANGELOG, "0.9.4")

    assert notes.startswith("### Added")
    assert "`nitid benchmark`" in notes
    assert "A bug." in notes
    assert "older bug" not in notes
    assert "not released yet" not in notes
    assert "## [" not in notes


def test_extracts_the_first_and_the_unreleased_sections() -> None:
    assert "not released yet" in changelog_notes.extract_notes(CHANGELOG, "Unreleased")
    assert "older bug" in changelog_notes.extract_notes(CHANGELOG, "0.9.3")


def test_missing_version_raises_with_a_useful_message() -> None:
    with pytest.raises(ValueError, match=r"no '## \[0\.9\.5\]' section"):
        changelog_notes.extract_notes(CHANGELOG, "0.9.5")


def test_version_prefix_is_not_a_match() -> None:
    # "0.9" must not match "[0.9.4]".
    with pytest.raises(ValueError):
        changelog_notes.extract_notes(CHANGELOG, "0.9")


def test_empty_section_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        changelog_notes.extract_notes(CHANGELOG, "0.9.2")


def test_main_writes_notes_file(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    output = tmp_path / "notes.md"

    code = changelog_notes.main(["0.9.3", "--changelog", str(changelog), "--output", str(output)])

    assert code == 0
    assert output.read_text(encoding="utf-8") == "### Fixed\n\n- An older bug.\n"


def test_main_fails_for_unknown_version(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")

    assert changelog_notes.main(["1.0.0", "--changelog", str(changelog)]) == 1
    assert "1.0.0" in capsys.readouterr().err


def test_repository_changelog_has_a_section_for_the_current_release() -> None:
    import nitid

    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    # The published version must always have notes the release workflow can use.
    assert changelog_notes.extract_notes(text, nitid.__version__)
