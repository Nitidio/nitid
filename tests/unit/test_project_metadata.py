"""Guards the public metadata that search engines and PyPI show for Nitid."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAGLINE = "Nitid Object detection that's actually open source"


def _url(name: str) -> str:
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(rf'^{name} = "([^"]+)"', text, flags=re.MULTILINE)
    assert match, f"{name} missing from [project.urls]"
    return match.group(1)


def test_homepage_is_the_canonical_site():
    assert _url("Homepage") == "https://nitid.io"


def test_documentation_points_to_the_docs_site():
    assert _url("Documentation") == "https://nitidio.github.io/nitid/"


def test_tagline_is_consistent_across_readme_docs_and_mkdocs():
    for name in ("README.md", "docs/index.md", "mkdocs.yml"):
        assert TAGLINE in (ROOT / name).read_text(), name
