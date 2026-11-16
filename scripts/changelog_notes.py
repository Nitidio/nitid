"""Print one version's section of CHANGELOG.md, for use as GitHub Release notes.

The changelog follows Keep a Changelog: each release is a ``## [X.Y.Z] - DATE``
heading followed by ``###`` subsections. This script returns the text under the
requested version's heading (without the heading itself) and fails when that
section is missing or empty, so a tag cannot be released without notes.

    python scripts/changelog_notes.py 0.9.4 [--changelog CHANGELOG.md] [--output notes.md]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HEADING_RE = re.compile(r"^## \[(?P<version>[^\]]+)\][^\n]*$", re.MULTILINE)


def extract_notes(text: str, version: str) -> str:
    """Return the body of ``## [version]`` in ``text``.

    Raises:
        ValueError: the version has no section, or its section is empty.
    """
    headings = list(_HEADING_RE.finditer(text))
    for index, heading in enumerate(headings):
        if heading.group("version") != version:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = text[heading.end() : end].strip()
        if not body:
            raise ValueError(f"CHANGELOG.md has a section for {version} but it is empty")
        return body
    raise ValueError(
        f"CHANGELOG.md has no '## [{version}]' section; rename '[Unreleased]' and add the date "
        "before tagging"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract a version's CHANGELOG section.")
    parser.add_argument("version", help="version without the leading 'v', for example 0.9.4")
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--output", type=Path, help="write to this file instead of stdout")
    args = parser.parse_args(argv)

    try:
        notes = extract_notes(args.changelog.read_text(encoding="utf-8"), args.version)
    except ValueError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1

    if args.output is None:
        print(notes)
    else:
        args.output.write_text(notes + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
