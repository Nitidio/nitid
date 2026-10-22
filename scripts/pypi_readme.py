"""Rewrite README.md's repository-relative links and images as absolute URLs.

PyPI renders the README as the project description but cannot resolve paths
inside the repository, so images such as the logo and the demo would show as
broken and links such as ``docs/quickstart.md`` would 404. The README in the
repository keeps relative paths, which GitHub renders even while the
repository is private; the release workflow runs this script before building.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPOSITORY = "Nitidio/nitid"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

# Markdown targets: ](path) and HTML attributes: src="path", srcset="path", href="path".
_MARKDOWN_TARGET = re.compile(r"(\]\()([^)\s]+)(\))")
_HTML_ATTRIBUTE = re.compile(r'(\b(?:src|srcset|href)=")([^"]+)(")')


def _is_relative(target: str) -> bool:
    return not re.match(r"^(?:[a-z][a-z0-9+.-]*:|#|/)", target, re.IGNORECASE)


def absolute_url(target: str, ref: str) -> str:
    """Return the absolute URL for a repository-relative ``target`` at ``ref``."""
    if not _is_relative(target):
        return target
    path, _, anchor = target.partition("#")
    if Path(path).suffix.lower() in IMAGE_SUFFIXES:
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{ref}/{path}"
    else:
        url = f"https://github.com/{REPOSITORY}/blob/{ref}/{path}"
    return f"{url}#{anchor}" if anchor else url


def rewrite(text: str, ref: str) -> str:
    """Rewrite every repository-relative link and image in ``text``."""

    def replace(match: re.Match[str]) -> str:
        return f"{match.group(1)}{absolute_url(match.group(2), ref)}{match.group(3)}"

    return _HTML_ATTRIBUTE.sub(replace, _MARKDOWN_TARGET.sub(replace, text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", required=True, help="Git tag or branch the URLs point at")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args()
    args.readme.write_text(rewrite(args.readme.read_text(encoding="utf-8"), args.ref), "utf-8")


if __name__ == "__main__":
    main()
