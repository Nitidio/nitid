from __future__ import annotations

from pathlib import Path

import pytest

from tools.check_roadmap import check_roadmap, issue_refs, main, split_sections

ROADMAP = """# Title

Intro with [#1](https://github.com/Nitidio/nitid/issues/1) before any section.

## Current status

Done: [#2](https://github.com/Nitidio/nitid/issues/2).

## Near-term priorities

1. First ([#8](https://github.com/Nitidio/nitid/issues/8)).
2. Second ([#7](https://github.com/Nitidio/nitid/issues/7)), again [#7](https://github.com/Nitidio/nitid/issues/7).

## Documentation

- [Quick Start](quickstart.md)
"""

STATES = {2: "closed", 7: "open", 8: "open"}


def test_split_sections_drops_text_before_first_heading() -> None:
    sections = split_sections(ROADMAP)

    assert list(sections) == ["Current status", "Near-term priorities", "Documentation"]
    assert "#1" not in sections["Current status"]


def test_issue_refs_are_sorted_and_unique() -> None:
    body = split_sections(ROADMAP)["Near-term priorities"]

    assert issue_refs(body) == [7, 8]
    assert issue_refs("no links here") == []
    assert issue_refs("https://github.com/Nitidio/nitid/pull/9 and /issues/x") == []


def test_consistent_roadmap_has_no_problems() -> None:
    assert check_roadmap(ROADMAP, STATES.get) == []


def test_missing_issue_is_reported_in_any_section() -> None:
    states = {2: "closed", 7: "open"}  # #8 does not exist

    problems = check_roadmap(ROADMAP, states.get)

    assert problems == ["#8 [Near-term priorities]: the issue does not exist"]


def test_closed_issue_in_near_term_priorities_is_reported() -> None:
    states = {**STATES, 7: "closed"}

    problems = check_roadmap(ROADMAP, states.get)

    assert len(problems) == 1
    assert "#7" in problems[0]
    assert "closed" in problems[0]


def test_closed_issue_outside_near_term_priorities_is_fine() -> None:
    # #2 is closed and linked from "Current status": that is allowed.
    assert check_roadmap(ROADMAP, STATES.get) == []


def test_each_issue_is_looked_up_once_per_run() -> None:
    calls: list[int] = []

    def lookup(number: int) -> str | None:
        calls.append(number)
        return "open"

    check_roadmap(ROADMAP, lookup)

    assert sorted(calls) == [2, 7, 8]


def test_main_returns_nonzero_and_prints_problems(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(ROADMAP, encoding="utf-8")

    code = main(["--roadmap", str(roadmap)], lookup={2: "closed", 7: "closed", 8: "open"}.get)

    assert code == 1
    assert "#7" in capsys.readouterr().err


def test_main_returns_zero_for_consistent_roadmap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(ROADMAP, encoding="utf-8")

    assert main(["--roadmap", str(roadmap)], lookup=STATES.get) == 0
    assert "consistent" in capsys.readouterr().out


def test_repository_roadmap_links_are_well_formed() -> None:
    # Offline sanity check on the real file: it parses and has the sections we check.
    text = (Path(__file__).resolve().parents[2] / "docs" / "roadmap.md").read_text(encoding="utf-8")

    assert "Near-term priorities" in split_sections(text)
    assert issue_refs(text), "the roadmap is expected to link at least one issue"
