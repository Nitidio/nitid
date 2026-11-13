"""Check the issue links in ``docs/roadmap.md`` against GitHub.

Two rules keep the roadmap honest:

* every ``https://github.com/Nitidio/nitid/issues/N`` link must point to an
  issue (or pull request) that exists;
* the "Near-term priorities" section may only link to issues that are still
  open, because anything closed there is already delivered.

Usage::

    GITHUB_TOKEN=... python -m tools.check_roadmap [--roadmap docs/roadmap.md]

The token is optional (it only raises the API rate limit).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

REPO = "Nitidio/nitid"
ISSUE_URL_RE = re.compile(r"https://github\.com/Nitidio/nitid/issues/(\d+)")
OPEN_ONLY_SECTIONS = frozenset({"Near-term priorities"})

# Maps an issue number to "open", "closed", or ``None`` when it does not exist.
StateLookup = Callable[[int], str | None]


def split_sections(text: str) -> dict[str, str]:
    """Split Markdown into ``{"## heading": body}`` chunks (text before the first heading is dropped)."""
    sections: dict[str, str] = {}
    heading: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if heading is not None:
                sections[heading] = "\n".join(lines)
            heading, lines = line[3:].strip(), []
        elif heading is not None:
            lines.append(line)
    if heading is not None:
        sections[heading] = "\n".join(lines)
    return sections


def issue_refs(text: str) -> list[int]:
    """Return the sorted, de-duplicated issue numbers linked in ``text``."""
    return sorted({int(number) for number in ISSUE_URL_RE.findall(text)})


def check_roadmap(text: str, lookup: StateLookup) -> list[str]:
    """Return one message per problem found; an empty list means the roadmap is consistent."""
    cache: dict[int, str | None] = {}
    problems: list[str] = []
    for section, body in split_sections(text).items():
        for number in issue_refs(body):
            if number not in cache:
                cache[number] = lookup(number)
            state = cache[number]
            if state is None:
                problems.append(f"#{number} [{section}]: the issue does not exist")
            elif section in OPEN_ONLY_SECTIONS and state != "open":
                problems.append(
                    f"#{number} [{section}]: the issue is {state}; remove it or move it"
                )
    return problems


def fetch_issue_state(number: int, token: str | None = None) -> str | None:
    """Ask the GitHub API for an issue's state; ``None`` when it does not exist (404)."""
    request = urllib.request.Request(  # noqa: S310 - fixed https URL
        f"https://api.github.com/repos/{REPO}/issues/{number}",
        headers={"Accept": "application/vnd.github+json"},
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            state = json.load(response)["state"]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    return str(state)


def main(argv: Sequence[str] | None = None, lookup: StateLookup | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--roadmap", type=Path, default=Path("docs/roadmap.md"))
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN")
    state_of = lookup or (lambda number: fetch_issue_state(number, token))
    problems = check_roadmap(args.roadmap.read_text(encoding="utf-8"), state_of)
    for problem in problems:
        print(f"{args.roadmap}: {problem}", file=sys.stderr)
    if not problems:
        print(f"{args.roadmap}: all issue links are consistent")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
