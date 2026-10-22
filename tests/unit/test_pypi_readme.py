from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("pypi_readme", REPO_ROOT / "scripts/pypi_readme.py")
assert _spec is not None and _spec.loader is not None
pypi_readme = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pypi_readme)

RAW = "https://raw.githubusercontent.com/Nitidio/nitid/v0.1.0"
BLOB = "https://github.com/Nitidio/nitid/blob/v0.1.0"


def test_images_point_at_raw_files() -> None:
    text = '![demo](docs/assets/demo.jpg)\n<img src="docs/assets/logo.png" alt="nitid">'

    out = pypi_readme.rewrite(text, "v0.1.0")

    assert f"![demo]({RAW}/docs/assets/demo.jpg)" in out
    assert f'src="{RAW}/docs/assets/logo.png"' in out


def test_links_point_at_blob_pages_and_keep_anchors() -> None:
    text = '[Guide](docs/quickstart.md#coming-from-yolo) [License](LICENSE) <a href="LICENSE">'

    out = pypi_readme.rewrite(text, "v0.1.0")

    assert f"[Guide]({BLOB}/docs/quickstart.md#coming-from-yolo)" in out
    assert f"[License]({BLOB}/LICENSE)" in out
    assert f'href="{BLOB}/LICENSE"' in out


def test_absolute_urls_and_anchors_are_unchanged() -> None:
    text = (
        "[Docs](https://nitidio.github.io/nitid/) [Install](#installation) "
        '<a href="https://pypi.org/project/nitid/"> [Mail](mailto:a@b.c)'
    )

    assert pypi_readme.rewrite(text, "v0.1.0") == text


def test_repository_readme_has_no_relative_targets_left() -> None:
    out = pypi_readme.rewrite((REPO_ROOT / "README.md").read_text(encoding="utf-8"), "main")

    targets = re.findall(r'\]\(([^)\s]+)\)|(?:src|srcset|href)="([^"]+)"', out)
    relative = [t for pair in targets for t in pair if t and not t.startswith(("http", "#"))]
    assert relative == []
