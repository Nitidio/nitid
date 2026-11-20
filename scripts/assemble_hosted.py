"""Assemble the static directory of the hosted Nitid app (``app.nitid.io``).

One directory, two shops behind one counter: the UI (``studio/dist-hosted``, built with
``pnpm run build:hosted``) and the model bundle (``models.json``, ``*.onnx``, ``SHA256SUMS``,
licence files; produced by ``tools/publish_models.py``) are merged into a single folder that
Cloudflare serves from one origin::

    out/
      index.html, assets/, ort/      <- the UI
      models/models.json, *.onnx ... <- the bundle
      _headers                       <- Cloudflare response headers

The script is the last gate before a deploy, so it is strict: a bundle without ``models.json``,
a referenced file that is missing or whose SHA-256 differs, a training checkpoint (``.pth``/``.pt``
...), a file Cloudflare would refuse (over 25 MiB) or a symlink stops the run with exit status 1 and
leaves nothing behind. Standard library only.

Usage::

    python scripts/assemble_hosted.py --bundle DIR [--ui studio/dist-hosted] \\
        [--headers studio/hosting/cloudflare/_headers] [--out studio/dist-hosted-site] [--clean]

Exit status: 0 assembled, 1 validation failed, 2 bad usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UI = REPO_ROOT / "studio" / "dist-hosted"
DEFAULT_HEADERS = REPO_ROOT / "studio" / "hosting" / "cloudflare" / "_headers"
DEFAULT_OUT = REPO_ROOT / "studio" / "dist-hosted-site"

CATALOG_FILE = "models.json"
SUMS_FILE = "SHA256SUMS"
MODELS_SUBDIR = "models"

#: Cloudflare Workers static assets: 25 MiB per file, 20,000 files on the Free plan
#: (https://developers.cloudflare.com/workers/platform/limits/, checked 2026-11-20).
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_FILES = 20_000

#: Training checkpoints and other formats that must never be published as a browser model.
#: ``.pth``/``.pt``/``.ckpt``/``.pkl``/``.pickle`` are pickle-based (code execution on load).
FORBIDDEN_SUFFIXES = frozenset(
    {".pth", ".pt", ".ckpt", ".safetensors", ".pkl", ".pickle", ".engine", ".torchscript"}
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUMS_LINE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


class AssembleError(Exception):
    """The inputs are not publishable; ``problems`` lists every reason found."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


@dataclass
class Report:
    """What an assembly produced."""

    out: Path
    models: list[str] = field(default_factory=list)
    files: int = 0
    total_bytes: int = 0
    warnings: list[str] = field(default_factory=list)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> list[Path]:
    """Every file below ``root`` (sorted), symlinks included so they can be refused."""
    return sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink())


def _resolve_inside(root: Path, relative: str) -> Path | None:
    """``root / relative`` when it stays inside ``root`` and is a plain relative path, else ``None``."""
    if not relative or "\\" in relative or relative.startswith("/") or _SCHEME.match(relative):
        return None
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        return None
    return target


def check_tree(root: Path, label: str, problems: list[str]) -> list[Path]:
    """Refuse symlinks, forbidden formats and files above the Cloudflare limits."""
    files = _files(root)
    if len(files) > MAX_FILES:
        problems.append(f"{label}: {len(files)} files; Cloudflare allows {MAX_FILES} per version")
    for path in files:
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            problems.append(f"{label}: {rel} is a symlink (links are not published)")
        elif path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(
                f"{label}: {rel} is a {path.suffix} file; only ONNX models are published"
            )
        elif path.stat().st_size > MAX_FILE_BYTES:
            mib = path.stat().st_size / (1024 * 1024)
            problems.append(f"{label}: {rel} is {mib:.1f} MiB; Cloudflare allows 25 MiB per file")
    return files


def validate_bundle(bundle: Path) -> tuple[list[str], list[str], list[str]]:
    """Check a model bundle. Returns ``(problems, warnings, model ids)``."""
    problems: list[str] = []
    warnings: list[str] = []
    ids: list[str] = []
    if not bundle.is_dir():
        return [f"bundle {bundle} is not a directory"], warnings, ids
    files = check_tree(bundle, "bundle", problems)

    catalog_path = bundle / CATALOG_FILE
    if not catalog_path.is_file():
        problems.append(f"bundle has no {CATALOG_FILE}")
        return problems, warnings, ids
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        problems.append(f"{CATALOG_FILE} is not valid JSON: {error}")
        return problems, warnings, ids
    entries = catalog.get("models") if isinstance(catalog, dict) else None
    if not isinstance(entries, list) or not entries:
        problems.append(f'{CATALOG_FILE} must be {{"models": [...]}} with at least one model')
        return problems, warnings, ids

    referenced: set[Path] = set()
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        where = f"{CATALOG_FILE} models[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{where} is not an object")
            continue
        model_id, task, url, digest = (entry.get(k) for k in ("id", "task", "url", "sha256"))
        if not isinstance(model_id, str) or not model_id:
            problems.append(f"{where}.id must be a non-empty string")
            continue
        ids.append(model_id)
        if task not in ("detect", "segment"):
            problems.append(f'{where}.task must be "detect" or "segment"')
        if (model_id, str(task)) in seen:
            problems.append(f"{where}: duplicate model {task}:{model_id}")
        seen.add((model_id, str(task)))
        if not isinstance(url, str) or not url:
            problems.append(f"{where}.url must be a non-empty string")
            continue
        target = _resolve_inside(bundle, url)
        if target is None:
            problems.append(f"{where}.url {url!r} must be a relative path inside the bundle")
            continue
        if not target.is_file():
            problems.append(f"{where}.url {url!r}: file does not exist")
            continue
        referenced.add(target)
        if digest is None:
            warnings.append(f"{where} has no sha256: the browser cannot verify or cache it by hash")
        elif not isinstance(digest, str) or not _SHA256.match(digest):
            problems.append(f"{where}.sha256 must be 64 lowercase hex characters")
        elif sha256_of(target) != digest:
            problems.append(f"{where}: sha256 of {url} does not match {CATALOG_FILE}")

    sums_path = bundle / SUMS_FILE
    if not sums_path.is_file():
        warnings.append(f"bundle has no {SUMS_FILE}")
    else:
        _check_sums(bundle, sums_path, problems)

    for path in files:
        if path.suffix.lower() == ".onnx" and path.resolve() not in referenced:
            warnings.append(
                f"{path.relative_to(bundle).as_posix()} is not listed in {CATALOG_FILE}"
            )
    return problems, warnings, ids


def _check_sums(bundle: Path, sums_path: Path, problems: list[str]) -> None:
    lines = [ln for ln in sums_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        problems.append(f"{SUMS_FILE} is empty")
    for line in lines:
        match = _SUMS_LINE.match(line)
        if match is None:
            problems.append(f"{SUMS_FILE}: cannot parse {line!r}")
            continue
        digest, name = match.group(1).lower(), match.group(2)
        target = _resolve_inside(bundle, name)
        if target is None or not target.is_file():
            problems.append(f"{SUMS_FILE}: {name!r} is not a file of the bundle")
        elif sha256_of(target) != digest:
            problems.append(f"{SUMS_FILE}: sha256 of {name} does not match")


def validate_ui(ui: Path, headers: Path) -> list[str]:
    problems: list[str] = []
    if not (ui / "index.html").is_file():
        problems.append(f"UI {ui} has no index.html (run: cd studio && pnpm run build:hosted)")
    if not list((ui / "ort").glob("*.wasm")):
        problems.append(f"UI {ui} has no ort/*.wasm (the in-browser engine would not start)")
    if (ui / MODELS_SUBDIR).exists():
        problems.append(
            f"UI {ui} already has a {MODELS_SUBDIR}/ folder; it is reserved for the bundle"
        )
    if not headers.is_file():
        problems.append(f"headers file {headers} does not exist")
    if ui.is_dir():
        check_tree(ui, "ui", problems)
    return problems


def _unsafe_output(out: Path, *inputs: Path) -> str | None:
    """Why ``out`` must not be (re)created, or ``None``."""
    out = out.resolve()
    for other in inputs:
        other = other.resolve()
        if out == other or out.is_relative_to(other) or other.is_relative_to(out):
            return f"output {out} overlaps the input {other}"
    return None


def assemble(
    ui: Path, bundle: Path, out: Path, headers: Path = DEFAULT_HEADERS, *, clean: bool = False
) -> Report:
    """Validate everything, then write ``out``. Raises :class:`AssembleError`; ``out`` is untouched then."""
    problems = validate_ui(ui, headers)
    bundle_problems, warnings, ids = validate_bundle(bundle)
    problems += bundle_problems
    overlap = _unsafe_output(out, ui, bundle)
    if overlap:
        problems.append(overlap)
    if out.exists() and any(out.iterdir()) and not clean:
        problems.append(f"output {out} is not empty (use --clean to replace it)")
    if problems:
        raise AssembleError(problems)

    # Build in a sibling temporary folder, verify it, then swap: a failure never leaves a half site.
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        shutil.copytree(ui, staging, dirs_exist_ok=True)
        shutil.copytree(bundle, staging / MODELS_SUBDIR)
        shutil.copyfile(headers, staging / "_headers")
        final_problems, _, _ = validate_bundle(staging / MODELS_SUBDIR)
        if final_problems:
            raise AssembleError([f"after copying: {p}" for p in final_problems])
        if out.exists():
            shutil.rmtree(out)
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    files = _files(out)
    return Report(
        out=out,
        models=ids,
        files=len(files),
        total_bytes=sum(p.stat().st_size for p in files),
        warnings=warnings,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument(
        "--bundle", type=Path, required=True, help="model bundle folder (models.json, *.onnx, ...)"
    )
    parser.add_argument(
        "--ui", type=Path, default=DEFAULT_UI, help="built UI (default: studio/dist-hosted)"
    )
    parser.add_argument(
        "--headers", type=Path, default=DEFAULT_HEADERS, help="Cloudflare _headers file"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="output folder (default: studio/dist-hosted-site)",
    )
    parser.add_argument(
        "--clean", action="store_true", help="replace the output folder if it is not empty"
    )
    args = parser.parse_args(argv)
    try:
        report = assemble(args.ui, args.bundle, args.out, args.headers, clean=args.clean)
    except AssembleError as error:
        print("assemble_hosted: refusing to assemble:", file=sys.stderr)
        for problem in error.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    for warning in report.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    size = report.total_bytes / (1024 * 1024)
    print(
        f"assembled {report.out}: {report.files} files, {size:.1f} MiB, models: {', '.join(report.models)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
