"""Assemble the static directory of the hosted Nitid app (``app.nitid.io``).

The UI (``studio/dist-hosted``, built with ``pnpm run build:hosted``) is written, together with the
Cloudflare response headers rendered from ``studio/hosting/cloudflare/_headers.template``, into one folder
that Cloudflare serves. Where the ONNX models live decides the mode (exactly one is required):

**Cross-origin models** (the production mode)::

    python scripts/assemble_hosted.py --models-base-url https://models.example.org/ \\
        [--validate-bundle DIR] [--bundle-check full|basic]

  The model bundle (``models.json``, ``*.onnx``, notices, ``SHA256SUMS``; produced by
  ``tools/publish_models.py``) is hosted by the owner on a server of his own. NOTHING of the bundle is
  copied into the site (Cloudflare static assets accept 25 MiB per file at most, and ``nitid1s`` is
  larger). The only effect on the site is the CSP: ``connect-src`` gets exactly that origin. The UI must
  have been built with the same ``STUDIO_MODELS_BASE_URL``; this is checked. ``--validate-bundle DIR``
  additionally validates a local copy of the bundle (nothing is copied, no size limit applies to it).
  ``--models-origin`` is the same with the bare origin instead of the base URL.

**Same-origin models** (small bundles only)::

    python scripts/assemble_hosted.py --bundle DIR

  The bundle is copied to ``models/`` in the site, so every file of it must stay under the 25 MiB limit
  of Cloudflare static assets; the UI must be built with ``STUDIO_MODELS_BASE_URL=/models/``.

In both modes the script is the last gate before a deploy, so it is strict: a bundle without
``models.json``, a referenced file that is missing or whose SHA-256 differs, a training checkpoint
(``.pth``/``.pt`` ...), a symlink, an origin that is not a plain ``https://host[:port]``, a wildcard, a UI
built for another models URL, or (for files that end up in the site) a file Cloudflare would refuse, stop
the run with exit status 1 and leave nothing behind. Standard library only (``--bundle-check full``
imports ``tools.check_hosted_bundle``, which needs ``onnx`` and ``jsonschema``).

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
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UI = REPO_ROOT / "studio" / "dist-hosted"
DEFAULT_HEADERS = REPO_ROOT / "studio" / "hosting" / "cloudflare" / "_headers.template"
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
    models_origin: str | None = None
    mode: str = "same-origin"


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


def check_tree(
    root: Path, label: str, problems: list[str], *, site_limits: bool = True
) -> list[Path]:
    """Refuse symlinks and forbidden formats; with ``site_limits`` also the Cloudflare file limits.

    The limits (count and 25 MiB per file) only concern files that end up in the site.
    """
    files = _files(root)
    if site_limits and len(files) > MAX_FILES:
        problems.append(f"{label}: {len(files)} files; Cloudflare allows {MAX_FILES} per version")
    for path in files:
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            problems.append(f"{label}: {rel} is a symlink (links are not published)")
        elif path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(
                f"{label}: {rel} is a {path.suffix} file; only ONNX models are published"
            )
        elif site_limits and path.stat().st_size > MAX_FILE_BYTES:
            mib = path.stat().st_size / (1024 * 1024)
            problems.append(f"{label}: {rel} is {mib:.1f} MiB; Cloudflare allows 25 MiB per file")
    return files


def validate_bundle(
    bundle: Path, *, site_limits: bool = True
) -> tuple[list[str], list[str], list[str]]:
    """Check a model bundle. Returns ``(problems, warnings, model ids)``.

    ``site_limits=False`` skips the Cloudflare per-file limits: used for a bundle that is hosted
    elsewhere and never copied into the site.
    """
    problems: list[str] = []
    warnings: list[str] = []
    ids: list[str] = []
    if not bundle.is_dir():
        return [f"bundle {bundle} is not a directory"], warnings, ids
    files = check_tree(bundle, "bundle", problems, site_limits=site_limits)

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


# --------------------------------------------------------------------------- the models origin

_HOST_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_HOST = re.compile(rf"^{_HOST_LABEL}(?:\.{_HOST_LABEL})*$")
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1"})
PLACEHOLDER = "{{MODELS_CONNECT_SRC}}"


class OriginError(ValueError):
    """The models origin is not something that may be written into the CSP."""


def _origin_of(value: str, *, allow_path: bool, allow_http_loopback: bool) -> str:
    """Strict ``scheme://host[:port]`` of ``value``; the single place that decides what the CSP gets."""
    if not isinstance(value, str) or not value.strip():
        raise OriginError("the models origin is empty")
    raw = value.strip()
    if "*" in raw:
        raise OriginError(f"{raw!r}: wildcards are not allowed in the models origin")
    if re.search(r"[\s;,'\"\\]", raw):
        raise OriginError(f"{raw!r}: whitespace and the characters ; , ' \" \\ are not allowed")
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError as error:
        raise OriginError(f"{raw!r}: not a valid URL ({error})") from error
    host = (parts.hostname or "").lower()
    if parts.scheme == "http":
        if not (allow_http_loopback and host in LOOPBACK_HOSTS):
            raise OriginError(
                f"{raw!r}: the models origin must use https"
                + ("" if not allow_http_loopback else " (http is accepted for localhost only)")
            )
    elif parts.scheme != "https":
        raise OriginError(f"{raw!r}: the models origin must be an https:// URL")
    if not host or not _HOST.match(host) or len(host) > 253:
        raise OriginError(f"{raw!r}: not a valid host name")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise OriginError(f"{raw!r}: credentials are not allowed")
    if parts.query or parts.fragment or "?" in raw or "#" in raw:
        raise OriginError(f"{raw!r}: no query string or fragment is allowed")
    if not allow_path and parts.path not in ("", "/"):
        raise OriginError(f"{raw!r}: an origin has no path (use --models-base-url for a base URL)")
    if port is not None and not 1 <= port <= 65535:
        raise OriginError(f"{raw!r}: invalid port")
    default = 443 if parts.scheme == "https" else 80
    suffix = "" if port is None or port == default else f":{port}"
    return f"{parts.scheme}://{host}{suffix}"


def parse_models_origin(value: str, *, allow_http_loopback: bool = False) -> str:
    """Validate a bare origin (``https://models.example.org[:port]``, no path) and normalise it."""
    return _origin_of(value, allow_path=False, allow_http_loopback=allow_http_loopback)


def origin_of_base_url(value: str, *, allow_http_loopback: bool = False) -> str:
    """The origin of a base URL (``https://host/some/path/``), validated like :func:`parse_models_origin`."""
    return _origin_of(value, allow_path=True, allow_http_loopback=allow_http_loopback)


def render_headers(
    template: str, models_origin: str | None, *, allow_http_loopback: bool = False
) -> str:
    """Fill the placeholder of the headers template.

    ``None`` -> ``connect-src 'self'`` (models on the same origin); an origin -> that origin, and only
    that, is appended to ``connect-src``. The origin must already be validated (it is checked again).
    """
    lines = template.splitlines(keepends=True)
    hits = [
        i for i, ln in enumerate(lines) if PLACEHOLDER in ln and not ln.lstrip().startswith("#")
    ]
    if len(hits) != 1 or lines[hits[0]].count(PLACEHOLDER) != 1:
        raise AssembleError(
            [
                f"headers template must use {PLACEHOLDER} exactly once, in the CSP line (found {len(hits)})"
            ]
        )
    if "connect-src 'self'" + PLACEHOLDER not in lines[hits[0]]:
        raise AssembleError([f"{PLACEHOLDER} must directly follow connect-src 'self'"])
    extra = ""
    if models_origin is not None:
        origin = _origin_of(
            models_origin, allow_path=False, allow_http_loopback=allow_http_loopback
        )
        if origin != models_origin:
            raise AssembleError([f"models origin {models_origin!r} is not normalised ({origin!r})"])
        extra = f" {origin}"
    lines[hits[0]] = lines[hits[0]].replace(PLACEHOLDER, extra)
    rendered = "".join(lines)
    leftover = [
        ln for ln in rendered.splitlines() if "{{" in ln and not ln.lstrip().startswith("#")
    ]
    if leftover:
        raise AssembleError([f"unresolved placeholder in the headers: {leftover[0].strip()[:80]}"])
    return rendered


def check_ui_matches_base(ui: Path, base: str) -> list[str]:
    """The UI must have been built for ``base``: its models URL is baked into the JavaScript.

    A UI built for another URL would request the models from an origin the CSP does not allow (or from
    the wrong server), and the page would only show "catalogue could not be reached".
    """
    needle = base.rstrip("/")
    assets = sorted((ui / "assets").glob("*.js")) if (ui / "assets").is_dir() else []
    for path in assets:
        if needle in path.read_text(encoding="utf-8", errors="replace"):
            return []
    return [
        f"UI {ui} was not built for {base}: that URL is not in assets/*.js. "
        f"Rebuild it with: cd studio && STUDIO_MODELS_BASE_URL={base} pnpm run build:hosted"
    ]


def full_bundle_problems(bundle: Path, lock: Path | None = None) -> list[str]:
    """The strict check of ``tools/check_hosted_bundle.py`` (needs ``onnx`` and ``jsonschema``).

    ``lock`` is the models lock the bundle was published from (default: ``studio/models.lock.json``).
    """
    added = str(REPO_ROOT) not in sys.path
    if added:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from tools import check_hosted_bundle, publish_models

        lock_path = lock or publish_models.DEFAULT_LOCK
        document = json.loads(lock_path.read_text(encoding="utf-8"))
        return list(check_hosted_bundle.check_bundle(bundle, document).problems)
    except ImportError as error:
        return [
            f"--bundle-check full needs the dev dependencies ({error}); run `uv sync --extra dev`, "
            "or use --bundle-check basic"
        ]
    except Exception as error:  # noqa: BLE001 - a corrupt file must be a reported problem, not a traceback
        return [
            f"tools/check_hosted_bundle.py could not read the bundle: {type(error).__name__}: {error}"
        ]
    finally:
        if added:
            sys.path.remove(str(REPO_ROOT))


# --------------------------------------------------------------------------- assembling


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
        problems.append(f"headers template {headers} does not exist")
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
    ui: Path,
    out: Path,
    headers_template: Path = DEFAULT_HEADERS,
    *,
    bundle: Path | None = None,
    models_base_url: str | None = None,
    models_origin: str | None = None,
    validate_bundle_dir: Path | None = None,
    bundle_check: str = "full",
    bundle_lock: Path | None = None,
    allow_http_loopback: bool = False,
    clean: bool = False,
) -> Report:
    """Validate everything, then write ``out``. Raises :class:`AssembleError`; ``out`` is untouched then.

    Exactly one of ``bundle`` (same-origin models, copied into the site), ``models_base_url`` or
    ``models_origin`` (models elsewhere, nothing copied) must be given.
    """
    given = [
        name
        for name, v in (
            ("bundle", bundle),
            ("models_base_url", models_base_url),
            ("models_origin", models_origin),
        )
        if v is not None
    ]
    if len(given) != 1:
        raise AssembleError(
            [
                "give exactly one of --bundle (same origin), --models-base-url or --models-origin (elsewhere)"
            ]
        )
    if bundle is not None and validate_bundle_dir is not None:
        raise AssembleError(
            ["--validate-bundle is for models hosted elsewhere; --bundle is already validated"]
        )
    if bundle_check not in ("full", "basic"):
        raise AssembleError([f"bundle_check must be 'full' or 'basic', not {bundle_check!r}"])

    problems = validate_ui(ui, headers_template)
    csp_origin: str | None = None
    ids: list[str] = []
    warnings: list[str] = []
    try:
        if models_base_url is not None:
            csp_origin = origin_of_base_url(
                models_base_url, allow_http_loopback=allow_http_loopback
            )
        elif models_origin is not None:
            csp_origin = parse_models_origin(models_origin, allow_http_loopback=allow_http_loopback)
    except OriginError as error:
        problems.append(str(error))
    if csp_origin is not None and (ui / "assets").is_dir():
        problems += check_ui_matches_base(ui, models_base_url or csp_origin)

    inputs = [ui]
    if bundle is not None:
        bundle_problems, warnings, ids = validate_bundle(bundle)
        problems += bundle_problems
        inputs.append(bundle)
    elif validate_bundle_dir is not None:
        bundle_problems, warnings, ids = validate_bundle(validate_bundle_dir, site_limits=False)
        problems += [f"--validate-bundle: {p}" for p in bundle_problems]
        if not bundle_problems and bundle_check == "full":
            problems += [
                f"--validate-bundle: {p}"
                for p in full_bundle_problems(validate_bundle_dir, bundle_lock)
            ]
        inputs.append(validate_bundle_dir)
    overlap = _unsafe_output(out, *inputs)
    if overlap:
        problems.append(overlap)
    if out.exists() and any(out.iterdir()) and not clean:
        problems.append(f"output {out} is not empty (use --clean to replace it)")
    rendered = ""
    if not problems:
        try:
            rendered = render_headers(
                headers_template.read_text(encoding="utf-8"),
                csp_origin,
                allow_http_loopback=allow_http_loopback,
            )
        except AssembleError as error:
            problems += error.problems
    if problems:
        raise AssembleError(problems)

    # Build in a sibling temporary folder, verify it, then swap: a failure never leaves a half site.
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        shutil.copytree(ui, staging, dirs_exist_ok=True)
        if bundle is not None:
            shutil.copytree(bundle, staging / MODELS_SUBDIR)
        (staging / "_headers").write_text(rendered, encoding="utf-8")
        if bundle is not None:
            final_problems, _, _ = validate_bundle(staging / MODELS_SUBDIR)
            final_problems += [p for p in (_site_problems(staging),) if p]
            if final_problems:
                raise AssembleError([f"after copying: {p}" for p in final_problems])
        else:
            site_problems: list[str] = []
            check_tree(staging, "site", site_problems)
            if (staging / MODELS_SUBDIR).exists():
                site_problems.append(
                    f"site has a {MODELS_SUBDIR}/ folder; no bundle was supposed to be copied"
                )
            if site_problems:
                raise AssembleError([f"after copying: {p}" for p in site_problems])
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
        models_origin=csp_origin,
        mode="same-origin" if bundle is not None else "cross-origin",
    )


def _site_problems(site: Path) -> str | None:
    """Files of the whole site (UI and copied bundle) against the Cloudflare limits."""
    problems: list[str] = []
    check_tree(site, "site", problems)
    return "; ".join(problems) if problems else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else None,
        epilog="Give --bundle (models in the site) or --models-base-url / --models-origin (models elsewhere).",
    )
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument(
        "--bundle",
        type=Path,
        help="same-origin mode: model bundle folder copied to models/ (every file under 25 MiB)",
    )
    where.add_argument(
        "--models-base-url",
        help="cross-origin mode: absolute https base URL the models are served from, e.g. https://models.example.org/",
    )
    where.add_argument(
        "--models-origin",
        help="cross-origin mode: the bare https origin (no path) to allow in connect-src",
    )
    parser.add_argument(
        "--validate-bundle",
        type=Path,
        metavar="DIR",
        help="cross-origin mode: also validate this local copy of the bundle (not copied, no size limit)",
    )
    parser.add_argument(
        "--bundle-check",
        choices=("full", "basic"),
        default="full",
        help="full: also tools/check_hosted_bundle.py (needs onnx, jsonschema); basic: standard library only",
    )
    parser.add_argument(
        "--bundle-lock",
        type=Path,
        help="models lock for --bundle-check full (default: studio/models.lock.json)",
    )
    parser.add_argument(
        "--allow-http-loopback",
        action="store_true",
        help="accept http://localhost and http://127.0.0.1 as models origin (tests only; never in production)",
    )
    parser.add_argument(
        "--ui", type=Path, default=DEFAULT_UI, help="built UI (default: studio/dist-hosted)"
    )
    parser.add_argument(
        "--headers-template",
        "--headers",
        dest="headers_template",
        type=Path,
        default=DEFAULT_HEADERS,
        help="Cloudflare _headers template (default: studio/hosting/cloudflare/_headers.template)",
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
        report = assemble(
            args.ui,
            args.out,
            args.headers_template,
            bundle=args.bundle,
            models_base_url=args.models_base_url,
            models_origin=args.models_origin,
            validate_bundle_dir=args.validate_bundle,
            bundle_check=args.bundle_check,
            bundle_lock=args.bundle_lock,
            allow_http_loopback=args.allow_http_loopback,
            clean=args.clean,
        )
    except AssembleError as error:
        print("assemble_hosted: refusing to assemble:", file=sys.stderr)
        for problem in error.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    for warning in report.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    size = report.total_bytes / (1024 * 1024)
    where_models = (
        f"models: {', '.join(report.models) or 'none'} (in the site)"
        if report.mode == "same-origin"
        else f"models hosted at {report.models_origin} (CSP connect-src), "
        + (f"validated: {', '.join(report.models)}" if report.models else "bundle not validated")
    )
    print(f"assembled {report.out}: {report.files} files, {size:.1f} MiB, {where_models}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
