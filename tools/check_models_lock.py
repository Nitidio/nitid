"""Validate ``studio/models.lock.json``, the licence-audited list of pretrained weights.

Two layers of checks:

1. *Structure*: the file must satisfy ``studio/models.lock.schema.json`` (JSON Schema 2020-12).
2. *Business rules* that a schema cannot express well:

   * ids are unique;
   * the licence of every entry is in :data:`ALLOWED_LICENSES` (permissive licences only);
   * the revision is pinned: floating names such as ``main`` or ``latest`` are rejected, and so
     are URLs that contain one as a path or query segment;
   * ``notice_required: true`` needs at least one entry in ``notice_files``;
   * ``hostable: true`` is only allowed when the sha256 is verified, ``notice_required`` is
     covered by ``notice_files`` and no ``open_caveats`` remain.

``hostable`` is the result of this repository's gate, not a legal opinion. See
``docs/licensing/studio-weights.md``.

Usage::

    python -m tools.check_models_lock [--lock studio/models.lock.json] [--schema studio/models.lock.schema.json]

``jsonschema`` is required (``pip install jsonschema``); it is not a dependency of the nitid wheel.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ALLOWED_LICENSES = frozenset({"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC"})
FLOATING_REVISIONS = frozenset({"main", "master", "latest", "head", "develop", "dev", "trunk"})
_FLOATING_URL_SEGMENT = re.compile(
    r"(?:/|=)(?:refs/heads/)?(?:%s)(?:/|$|&|\?)" % "|".join(sorted(FLOATING_REVISIONS)),
    re.IGNORECASE,
)

DEFAULT_LOCK = Path("studio/models.lock.json")
DEFAULT_SCHEMA = Path("studio/models.lock.schema.json")


def schema_errors(lock: Any, schema: dict[str, Any]) -> list[str]:
    """Return one message per violation of ``schema``, sorted by location."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - exercised only without jsonschema
        raise SystemExit(
            "check_models_lock needs the 'jsonschema' package: pip install jsonschema"
        ) from exc

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(lock), key=lambda e: [str(p) for p in e.absolute_path])
    return [
        f"schema: {'/'.join(str(p) for p in error.absolute_path) or '<root>'}: {error.message}"
        for error in errors
    ]


def rule_errors(lock: dict[str, Any]) -> list[str]:
    """Return one message per business-rule violation. Assumes the structure is valid."""
    problems: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(lock.get("entries", [])):
        name = str(entry.get("id", f"#{index}"))
        if name in seen:
            problems.append(f"{name}: duplicate id")
        seen.add(name)

        licence = entry.get("license")
        if licence not in ALLOWED_LICENSES:
            allowed = ", ".join(sorted(ALLOWED_LICENSES))
            problems.append(f"{name}: licence {licence!r} is not in the allowlist ({allowed})")

        revision = str(entry.get("revision", ""))
        if revision.strip().lower() in FLOATING_REVISIONS:
            problems.append(f"{name}: revision {revision!r} is not pinned")
        if _FLOATING_URL_SEGMENT.search(str(entry.get("url", ""))):
            problems.append(f"{name}: url points at a floating revision")

        if entry.get("notice_required") and not entry.get("notice_files"):
            problems.append(f"{name}: notice_required is true but notice_files is empty")

        if entry.get("hostable") is True:
            if entry.get("sha256_status") != "verified" or not entry.get("sha256"):
                problems.append(f"{name}: hostable requires a verified sha256")
            if entry.get("notice_required") and not entry.get("notice_files"):
                problems.append(f"{name}: hostable requires notice_required to be covered")
            if entry.get("open_caveats"):
                problems.append(f"{name}: hostable is not allowed while open_caveats remain")
    return problems


def check_lock(lock: Any, schema: dict[str, Any]) -> list[str]:
    """Return all problems; an empty list means the lock is valid."""
    problems = schema_errors(lock, schema)
    if problems:
        return problems  # business rules assume the structure is right
    return rule_errors(lock)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args(argv)

    try:
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        schema = json.loads(args.schema.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"check_models_lock: cannot read input: {exc}", file=sys.stderr)
        return 2

    problems = check_lock(lock, schema)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    entries = lock["entries"]
    hostable = sum(1 for entry in entries if entry["hostable"])
    print(f"models.lock OK: {len(entries)} entries, {hostable} hostable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
