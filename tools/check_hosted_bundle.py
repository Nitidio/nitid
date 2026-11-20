"""Validate a hosted model bundle written by ``tools.publish_models``.

Re-reads the folder from scratch (it trusts nothing the publisher said) and checks::

    python -m tools.check_hosted_bundle dist-models [--lock studio/models.lock.json]
                                                    [--max-total-bytes N]

* ``models.json`` has the shape ``studio/src/browser/catalog.ts`` accepts (format version 1),
  and additionally every entry has a ``sha256`` and a relative ``url`` of an existing file;
* the ``sha256`` of every entry equals the SHA-256 of its file;
* every ONNX carries a ``nitid_meta`` that satisfies ``studio/api-contract/onnx-meta.schema.json``
  and agrees with the catalogue (task, model, static ``[1, 3, S, S]`` input);
* every sidecar names a lock entry that is ``hostable: true`` with a verified sha256, and its
  source hash equals the lock's;
* the licence files each lock entry requires are present and ``NOTICE.txt`` exists;
* no checkpoint or other non-ONNX weight format is in the bundle;
* ``SHA256SUMS`` lists exactly the files of the bundle and every hash matches;
* a size report (per file and total) is printed; ``--max-total-bytes`` turns it into a limit.

Exit status 0 when there are no problems, 1 otherwise, 2 when the folder cannot be read. Needs
``onnx`` and ``jsonschema`` (dev dependencies); CI runs it on the bundle it builds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from tools.publish_models import (
    CATALOGUE_NAME,
    CHECKSUMS_NAME,
    DEFAULT_LOCK,
    NOTICE_NAME,
    SIDECAR_FORMAT_VERSION,
    SIDECAR_SUFFIX,
    PublishError,
    notice_relative_paths,
    read_meta,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
META_SCHEMA_PATH = ROOT / "studio" / "api-contract" / "onnx-meta.schema.json"

FORBIDDEN_SUFFIXES = (".pth", ".pt", ".ckpt", ".safetensors", ".torchscript", ".engine", ".bin")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUMS_LINE_RE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")


@dataclass
class BundleReport:
    """Outcome of :func:`check_bundle`."""

    problems: list[str] = field(default_factory=list)
    sizes: dict[str, int] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(self.sizes.values())

    @property
    def ok(self) -> bool:
        return not self.problems


def catalogue_errors(document: Any) -> list[str]:
    """Problems with a ``models.json`` document, mirroring ``parseHostedCatalog`` in catalog.ts.

    The extra rules (sha256 required, non-empty list, relative url) are bundle policy, not
    something catalog.ts needs.
    """
    models = document.get("models") if isinstance(document, dict) else None
    if not isinstance(models, list):
        return ['the catalogue must be {"models": [...]}']
    problems: list[str] = []
    if not models:
        problems.append("the catalogue lists no models")
    seen: set[str] = set()
    for index, raw in enumerate(models):
        where = f"models[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{where} is not an object")
            continue
        model_id, task, url = raw.get("id"), raw.get("task"), raw.get("url")
        if not isinstance(model_id, str) or not model_id:
            problems.append(f"{where}.id must be a non-empty string")
        if task not in ("detect", "segment"):
            problems.append(f'{where}.task must be "detect" or "segment"')
        if not isinstance(url, str) or not url:
            problems.append(f"{where}.url must be a non-empty string")
        elif _url_problem(url):
            problems.append(f"{where}.url {url!r}: {_url_problem(url)}")
        sha = raw.get("sha256")
        if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
            problems.append(f"{where}.sha256 must be 64 lowercase hex characters (required here)")
        if "label" in raw and not isinstance(raw["label"], str):
            problems.append(f"{where}.label must be a string")
        key = f"{task}:{model_id}"
        if key in seen:
            problems.append(f"{where}: duplicate model {key}")
        seen.add(key)
    return problems


def _url_problem(url: str) -> str | None:
    if "://" in url or url.startswith(("/", "\\", "//")) or "\\" in url:
        return "must be a relative file name (same origin as the UI)"
    parts = PurePosixPath(url).parts
    if len(parts) != 1 or parts[0] in (".", ".."):
        return "must be a plain file name in the bundle root"
    if not url.endswith(".onnx"):
        return "must end in .onnx"
    return None


def meta_errors(meta: Mapping[str, Any]) -> list[str]:
    """Schema plus cross-field problems of a ``nitid_meta`` document."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("check_hosted_bundle needs 'jsonschema' (a dev dependency)") from exc

    schema = json.loads(META_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(dict(meta)),
        key=lambda e: [str(p) for p in e.absolute_path],
    )
    problems = [
        f"nitid_meta {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in errors
    ]
    if problems:
        return problems
    shape = meta["input"]["shape"]
    if shape[2] != shape[3] or shape[2] != meta["imgsz"]:
        problems.append(f"nitid_meta input shape {shape} does not match imgsz={meta['imgsz']}")
    return problems


def _read_json(path: Path, problems: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        problems.append(f"{path.name}: cannot read as JSON ({exc})")
        return None


def parse_sha256sums(text: str) -> tuple[dict[str, str], list[str]]:
    """``(path -> hash, problems)`` of a ``SHA256SUMS`` file."""
    hashes: dict[str, str] = {}
    problems: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = _SUMS_LINE_RE.match(line)
        if match is None:
            problems.append(f"{CHECKSUMS_NAME}:{number}: malformed line")
            continue
        if match.group(2) in hashes:
            problems.append(f"{CHECKSUMS_NAME}:{number}: {match.group(2)} listed twice")
        hashes[match.group(2)] = match.group(1)
    return hashes, problems


def _check_model(
    bundle: Path,
    raw: Mapping[str, Any],
    entries_by_id: Mapping[str, Mapping[str, Any]],
    problems: list[str],
    referenced: list[Mapping[str, Any]],
) -> None:
    url, sha = raw.get("url"), raw.get("sha256")
    if not isinstance(url, str) or _url_problem(url):
        return  # already reported by catalogue_errors
    onnx_path = bundle / url
    name = f"{raw.get('id')}/{raw.get('task')}"
    if not onnx_path.is_file():
        problems.append(f"{name}: {url} is referenced by the catalogue but missing")
        return
    if isinstance(sha, str) and sha256_file(onnx_path) != sha:
        problems.append(f"{name}: sha256 of {url} does not match the catalogue")

    meta: dict[str, Any] | None = None
    try:
        meta = read_meta(onnx_path)
    except PublishError as exc:
        problems.append(f"{name}: {exc}")
    else:
        problems.extend(f"{name}: {p}" for p in meta_errors(meta))
        if meta.get("task") != raw.get("task"):
            problems.append(f"{name}: nitid_meta task {meta.get('task')!r} differs from catalogue")
        if "model" in meta and meta["model"] != raw.get("id"):
            problems.append(f"{name}: nitid_meta model {meta['model']!r} differs from catalogue id")

    sidecar_path = bundle / (url + SIDECAR_SUFFIX)
    if not sidecar_path.is_file():
        problems.append(f"{name}: sidecar {sidecar_path.name} is missing")
        return
    sidecar = _read_json(sidecar_path, problems)
    if not isinstance(sidecar, dict):
        return
    if sidecar.get("format_version") != SIDECAR_FORMAT_VERSION:
        problems.append(f"{name}: sidecar format_version is not {SIDECAR_FORMAT_VERSION}")
    for key in ("id", "task", "sha256"):
        if sidecar.get(key) != raw.get(key):
            problems.append(f"{name}: sidecar {key} differs from the catalogue")
    if sidecar.get("file") != url:
        problems.append(f"{name}: sidecar file differs from the catalogue url")
    if sidecar.get("size_bytes") != onnx_path.stat().st_size:
        problems.append(f"{name}: sidecar size_bytes does not match the file")
    source = sidecar.get("source")
    lock_id = source.get("lock_id") if isinstance(source, dict) else None
    entry = entries_by_id.get(str(lock_id))
    if entry is None:
        problems.append(f"{name}: sidecar names lock entry {lock_id!r}, which does not exist")
        return
    if entry.get("hostable") is not True:
        problems.append(f"{name}: lock entry {lock_id} is hostable: false and must not be here")
    if entry.get("sha256_status") != "verified":
        problems.append(f"{name}: lock entry {lock_id} has no verified sha256")
    if isinstance(source, dict) and source.get("sha256") != entry.get("sha256"):
        problems.append(f"{name}: source sha256 differs from the lock entry {lock_id}")
    if entry.get("nitid_model") != raw.get("id") or entry.get("task") != raw.get("task"):
        problems.append(
            f"{name}: lock entry {lock_id} is {entry.get('nitid_model')} {entry.get('task')}"
        )
    if meta is not None and isinstance(sidecar.get("nitid_meta"), dict):
        if sidecar["nitid_meta"] != meta:
            problems.append(f"{name}: sidecar nitid_meta differs from the one embedded in the ONNX")
    referenced.append(entry)


def check_bundle(
    bundle: Path,
    lock: Mapping[str, Any],
    *,
    max_total_bytes: int | None = None,
) -> BundleReport:
    """Verify ``bundle`` against the rules in the module docstring."""
    report = BundleReport()
    problems = report.problems
    if not bundle.is_dir():
        problems.append(f"{bundle} is not a directory")
        return report

    files = {p.relative_to(bundle).as_posix(): p for p in sorted(bundle.rglob("*")) if p.is_file()}
    report.sizes = {name: path.stat().st_size for name, path in files.items()}

    for name in files:
        if name.lower().endswith(FORBIDDEN_SUFFIXES):
            problems.append(f"{name}: weights in this format must never be hosted")

    entries_by_id = {str(e["id"]): e for e in lock.get("entries", [])}
    referenced: list[Mapping[str, Any]] = []

    if CATALOGUE_NAME not in files:
        problems.append(f"{CATALOGUE_NAME} is missing")
    else:
        document = _read_json(files[CATALOGUE_NAME], problems)
        if document is not None:
            shape_problems = catalogue_errors(document)
            problems.extend(f"{CATALOGUE_NAME}: {p}" for p in shape_problems)
            if not shape_problems:
                for raw in document["models"]:
                    _check_model(bundle, raw, entries_by_id, problems, referenced)
                listed = {m["url"] for m in document["models"]}
                for name in files:
                    if name.endswith(".onnx") and name not in listed:
                        problems.append(f"{name}: ONNX file is not in the catalogue")

    for entry in referenced:
        try:
            required = notice_relative_paths(entry)
        except PublishError as exc:
            problems.append(str(exc))
            continue
        for relative in required:
            if relative not in files or files[relative].stat().st_size == 0:
                problems.append(f"{entry['id']}: licence file {relative} is missing or empty")
    if NOTICE_NAME not in files:
        problems.append(f"{NOTICE_NAME} is missing")

    if CHECKSUMS_NAME not in files:
        problems.append(f"{CHECKSUMS_NAME} is missing")
    else:
        listed_hashes, sums_problems = parse_sha256sums(
            files[CHECKSUMS_NAME].read_text(encoding="utf-8")
        )
        problems.extend(sums_problems)
        for name, path in files.items():
            if name == CHECKSUMS_NAME:
                continue
            if name not in listed_hashes:
                problems.append(f"{CHECKSUMS_NAME}: {name} is not listed")
            elif sha256_file(path) != listed_hashes[name]:
                problems.append(f"{CHECKSUMS_NAME}: hash of {name} does not match")
        for name in listed_hashes:
            if name not in files:
                problems.append(f"{CHECKSUMS_NAME}: lists {name}, which does not exist")

    if max_total_bytes is not None and report.total_bytes > max_total_bytes:
        problems.append(
            f"total size {report.total_bytes} bytes exceeds the limit {max_total_bytes}"
        )
    return report


def format_size_report(report: BundleReport) -> str:
    """Per-file sizes, largest first, and the total."""
    width = max((len(n) for n in report.sizes), default=0)
    lines = [
        f"{name:<{width}}  {size:>12,} bytes ({size / 1_048_576:.2f} MiB)"
        for name, size in sorted(report.sizes.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    total = report.total_bytes
    lines.append(f"{'total':<{width}}  {total:>12,} bytes ({total / 1_048_576:.2f} MiB)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--max-total-bytes", type=int)
    args = parser.parse_args(argv)
    try:
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"check_hosted_bundle: cannot read the lock: {exc}", file=sys.stderr)
        return 2
    report = check_bundle(args.bundle, lock, max_total_bytes=args.max_total_bytes)
    print(format_size_report(report))
    if report.problems:
        print("\n".join(f"PROBLEM: {p}" for p in report.problems), file=sys.stderr)
        return 1
    print("hosted bundle OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
