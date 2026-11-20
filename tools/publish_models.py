"""Build the static model bundle that the hosted Nitid Studio serves (WP-6.2).

The hosted site (``vite build --mode hosted``) has no backend: the browser engine downloads
ONNX files from a folder next to the UI. This tool produces that folder from the audited list
``studio/models.lock.json``::

    python -m tools.publish_models --out dist-models                  # n and s, detect + segment
    python -m tools.publish_models --out dist-models --sizes n,s,m    # a wider catalogue
    python -m tools.publish_models --out dist-models --models nitid1s --tasks detect
    python -m tools.publish_models --out dist-models --offline-from ~/weights   # no network
    python -m tools.publish_models --out dist-models --dry-run        # only print the selection

Bundle layout (everything is a plain static file; ``url`` values are relative names, so the
bundle is served from the same origin as the UI)::

    models.json                       catalogue read by studio/src/browser/catalog.ts (version 1)
    nitid1s-detect.onnx               one post-processed, static-shape ONNX per (model, task)
    nitid1s-detect.onnx.meta.json     sidecar: sha256, size, source checkpoint, nitid_meta copy
    notices/<owner>-<repo>/LICENSE    the upstream licence files named by ``notice_files``
    NOTICE.txt                        attribution and the "modified by exporting to ONNX" notice
    SHA256SUMS                        ``sha256sum -c`` compatible, every file except itself

Rules enforced here (the licence gate is mechanical, see ``docs/licensing/studio-weights.md``):

* only ``hostable: true`` lock entries are ever published. Asking for a ``hostable: false``
  entry by id (for example an ``obj2coco`` one) is an error, never a silent skip;
* an entry needs a *verified* sha256: the downloaded weights are hashed before they are loaded;
* the licence files named by ``notice_files`` must exist (``--notices-dir``) and are copied;
* no checkpoint (``.pth``/``.pt``) is ever written to the bundle.

Selection: with no ``--models`` every hostable entry is a candidate and the size allow-list
``--sizes`` (default ``n,s``: small sizes until real sizes are measured) applies. With
``--models`` (lock ids such as ``dfine-s-coco`` or public names such as ``nitid1s``) the default
size allow-list is not applied, only an explicit ``--sizes`` is.

The run is idempotent: when ``<file>.meta.json`` says the ONNX was built from the same source
sha256 with the same nitid version and the file still has the recorded hash, it is reused
without downloading or exporting again (``--force`` rebuilds). Output order is deterministic.
``torch`` and ``onnx`` are imported only when a model is actually built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlretrieve

from tools.check_models_lock import rule_errors, schema_errors

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "studio" / "models.lock.json"
DEFAULT_SCHEMA = ROOT / "studio" / "models.lock.schema.json"
DEFAULT_NOTICES_DIR = ROOT / "studio" / "notices"

#: Version of the sidecar document written next to every ONNX.
SIDECAR_FORMAT_VERSION = 1
#: Size letters in catalogue order; also the vocabulary of ``--sizes``.
SIZE_ORDER = ("n", "s", "m", "l", "x")
#: Small sizes only until the real ONNX sizes are measured (decision of the owner).
DEFAULT_SIZES = ("n", "s")
TASK_ORDER = ("detect", "segment")

CATALOGUE_NAME = "models.json"
CHECKSUMS_NAME = "SHA256SUMS"
NOTICE_NAME = "NOTICE.txt"
NOTICES_DIR_NAME = "notices"
SIDECAR_SUFFIX = ".meta.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class PublishError(Exception):
    """A request or an input that must not be published; the message is for the operator."""


# --------------------------------------------------------------------------- small helpers


def sha256_file(path: Path) -> str:
    """Lowercase hex SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry_size(entry: Mapping[str, Any]) -> str:
    """Size letter of an entry (``nitid1s`` -> ``s``)."""
    return str(entry["nitid_model"])[-1]


def onnx_file_name(model_id: str, task: str) -> str:
    """Relative name of the ONNX of ``(model_id, task)`` inside the bundle."""
    return f"{model_id}-{task}.onnx"


def notice_slug(upstream_repo: str) -> str:
    """Folder name for the notices of an upstream repository (``Owner-Repo``)."""
    parts = [p for p in urlparse(upstream_repo).path.split("/") if p]
    slug = "-".join(parts)
    if not slug or not _SAFE_NAME_RE.fullmatch(slug):
        raise PublishError(f"cannot derive a safe notices folder from {upstream_repo!r}")
    return slug


def notice_relative_paths(entry: Mapping[str, Any]) -> list[str]:
    """Bundle-relative paths of the licence files an entry requires."""
    slug = notice_slug(str(entry["upstream_repo"]))
    paths = []
    for name in entry.get("notice_files", []):
        if not isinstance(name, str) or not _SAFE_NAME_RE.fullmatch(name):
            raise PublishError(f"{entry['id']}: unsafe notice file name {name!r}")
        paths.append(f"{NOTICES_DIR_NAME}/{slug}/{name}")
    return paths


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [part.strip() for part in value.split(",") if part.strip()]
    return items or None


# ------------------------------------------------------------------------------ the lock


def load_lock(path: Path, schema_path: Path | None = DEFAULT_SCHEMA) -> dict[str, Any]:
    """Read the lock and run the same checks as ``tools.check_models_lock``.

    The JSON Schema layer needs ``jsonschema`` (a dev dependency); when it is not installed only
    the business rules run, which is enough for the hostable gate.
    """
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishError(f"cannot read the lock {path}: {exc}") from exc
    if not isinstance(lock, dict) or not isinstance(lock.get("entries"), list):
        raise PublishError(f"{path}: not a models lock (no 'entries' list)")

    problems: list[str] = []
    if schema_path is not None:
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            pass
        else:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            problems = schema_errors(lock, schema)
    if not problems:
        problems = rule_errors(lock)
    if problems:
        raise PublishError("the lock is invalid:\n  " + "\n  ".join(problems))
    return lock


def check_publishable(entry: Mapping[str, Any]) -> None:
    """The licence gate for one entry. Raises :class:`PublishError` on any miss."""
    name = entry.get("id", "?")
    if entry.get("hostable") is not True:
        reason = entry.get("hostable_reason", "no reason recorded")
        raise PublishError(f"{name} is hostable: false and is never published ({reason})")
    sha = entry.get("sha256")
    if entry.get("sha256_status") != "verified" or not isinstance(sha, str):
        raise PublishError(f"{name} has no verified sha256 and cannot be published")
    if not _SHA256_RE.fullmatch(sha):
        raise PublishError(f"{name} has a malformed sha256")
    if entry.get("open_caveats"):
        raise PublishError(f"{name} still has open caveats and cannot be published")
    if entry.get("notice_required") and not entry.get("notice_files"):
        raise PublishError(f"{name} needs notice files but lists none")
    if entry.get("task") not in TASK_ORDER:
        raise PublishError(f"{name} has an unsupported task {entry.get('task')!r}")
    if entry_size(entry) not in SIZE_ORDER:
        raise PublishError(f"{name} has an unsupported model name {entry.get('nitid_model')!r}")


def _sort_key(entry: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        SIZE_ORDER.index(entry_size(entry)),
        TASK_ORDER.index(str(entry["task"])),
        str(entry["id"]),
    )


def select_entries(
    lock: Mapping[str, Any],
    *,
    models: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    sizes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Choose the lock entries to publish, in catalogue order.

    ``models`` items are lock ids (exact entry) or public names such as ``nitid1s`` (every
    *hostable* entry of that name). A lock id of a ``hostable: false`` entry is refused, and so
    is a public name that has no hostable entry at all. ``sizes`` defaults to
    :data:`DEFAULT_SIZES` only when ``models`` is not given.
    """
    entries: list[dict[str, Any]] = list(lock["entries"])
    by_id = {str(e["id"]): e for e in entries}

    chosen: list[dict[str, Any]]
    if models:
        chosen = []
        for token in models:
            if token in by_id:
                entry = by_id[token]
                check_publishable(entry)  # refuses hostable: false, however it was requested
                chosen.append(entry)
                continue
            named = [e for e in entries if str(e["nitid_model"]).lower() == token.lower()]
            if not named:
                known = ", ".join(sorted(by_id))
                raise PublishError(f"unknown model {token!r}. Lock ids: {known}")
            hostable = [e for e in named if e.get("hostable") is True]
            if not hostable:
                raise PublishError(
                    f"{token!r} has no hostable entry in the lock "
                    f"({', '.join(str(e['id']) for e in named)} are hostable: false)"
                )
            for entry in hostable:
                check_publishable(entry)
                chosen.append(entry)
    else:
        chosen = [e for e in entries if e.get("hostable") is True]
        for entry in chosen:
            check_publishable(entry)

    if tasks:
        unknown = sorted(set(tasks) - set(TASK_ORDER))
        if unknown:
            raise PublishError(f"unknown task(s) {unknown}; choose from {list(TASK_ORDER)}")
        chosen = [e for e in chosen if e["task"] in tasks]

    effective_sizes = sizes if sizes else (None if models else DEFAULT_SIZES)
    if effective_sizes is not None:
        unknown = sorted(set(effective_sizes) - set(SIZE_ORDER))
        if unknown:
            raise PublishError(f"unknown size(s) {unknown}; choose from {list(SIZE_ORDER)}")
        chosen = [e for e in chosen if entry_size(e) in effective_sizes]

    unique: dict[str, dict[str, Any]] = {}
    for entry in chosen:
        unique[str(entry["id"])] = entry
    result = sorted(unique.values(), key=_sort_key)

    seen: dict[tuple[str, str], str] = {}
    for entry in result:
        key = (str(entry["nitid_model"]), str(entry["task"]))
        if key in seen:
            raise PublishError(
                f"{seen[key]} and {entry['id']} are both {key[0]} {key[1]}: "
                "a catalogue holds one file per (model, task); select one with --models"
            )
        seen[key] = str(entry["id"])
    if not result:
        raise PublishError("the selection is empty: nothing to publish")
    return result


# -------------------------------------------------------------------------- catalogue text


def catalogue_label(entry: Mapping[str, Any]) -> str:
    """Human label shown in the model picker."""
    return f"{entry['nitid_model']} ({entry['task']}, {entry['weights_variant']})"


def build_catalogue(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The ``models.json`` document (format version 1) for built records.

    Each record needs ``id``, ``task``, ``url``, ``sha256`` and ``label``; order is kept.
    """
    return {
        "models": [
            {
                "id": r["id"],
                "task": r["task"],
                "url": r["url"],
                "sha256": r["sha256"],
                "label": r["label"],
            }
            for r in records
        ]
    }


def dump_json(document: Any) -> str:
    """Deterministic, human-readable JSON with a trailing newline."""
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def format_sha256sums(hashes: Mapping[str, str]) -> str:
    """``sha256sum -c`` compatible text, sorted by relative path."""
    return "".join(f"{hashes[path]}  {path}\n" for path in sorted(hashes))


def render_notice(entries: Sequence[Mapping[str, Any]], nitid_version: str) -> str:
    """Attribution text. Apache-2.0 asks derived files to carry a change notice."""
    lines = [
        "Nitid Studio hosted models: notices",
        "",
        "Every .onnx file in this folder is a derivative work: the upstream PyTorch checkpoint",
        f"was modified by exporting it to ONNX with nitid {nitid_version} (static input shape,",
        "post-processing included, graph simplified, class names and preprocessing embedded in",
        "the 'nitid_meta' metadata). The weights themselves are unchanged.",
        "",
        "The licence of each upstream project is in notices/<owner>-<repo>/ and applies to the",
        "files derived from it, listed below.",
        "",
    ]
    for entry in entries:
        file_name = onnx_file_name(str(entry["nitid_model"]), str(entry["task"]))
        lines += [
            f"{file_name}",
            f"  source:   {entry['url']}",
            f"  upstream: {entry['upstream_repo']} (revision {entry['revision']})",
            f"  licence:  {entry['license']} (see {', '.join(notice_relative_paths(entry))})",
            f"  trained:  {', '.join(entry.get('training_data', [])) or 'not recorded'}",
            "",
        ]
    return "\n".join(lines)


# ------------------------------------------------------------------------ build steps


Fetch = Callable[[Mapping[str, Any], Path], None]
Prepare = Callable[[Mapping[str, Any], Path, Path], Path]
Export = Callable[[Path, str, Path], None]


def default_fetch(entry: Mapping[str, Any], destination: Path) -> None:
    """Download the pinned release asset (same mechanism as ``dfine.utils.downloads``)."""
    urlretrieve(str(entry["url"]), destination)


def wrap_checkpoint(entry: Mapping[str, Any], raw: Path, destination: Path) -> Path:
    """Return a self-contained nitid checkpoint for ``raw`` (already verified against the lock).

    A file that already holds ``config``/``model``/``names`` is used as it is. A raw upstream
    checkpoint is wrapped like ``dfine.utils.downloads.download_model`` does.
    """
    import torch

    loaded = torch.load(raw, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict) and {"config", "model", "names"} <= set(loaded):
        return raw
    del loaded

    from importlib.resources import as_file

    from dfine.data import COCO_NAMES
    from dfine.nitid import resolve_nitid_backend_model
    from dfine.nn.configs import make_model_config
    from nitid.convert_checkpoint import convert

    task = str(entry["task"])
    backend = resolve_nitid_backend_model(str(entry["nitid_model"]), task=task)
    with as_file(COCO_NAMES) as names:
        convert(
            weights=str(raw),
            config=make_model_config(backend, task=task),
            names_file=str(names),
            output=str(destination),
        )
    return destination


def default_export(checkpoint: Path, task: str, destination: Path) -> None:
    """Export with the exact settings Studio's local server uses at the checkpoint's default size."""
    from nitid.studio.onnx_service import default_imgsz, export_checkpoint

    export_checkpoint(checkpoint, task, default_imgsz(checkpoint), destination)


def _nitid_version() -> str:
    from dfine import __version__

    return str(__version__)


def ensure_meta_model(path: Path, model_id: str) -> None:
    """Make sure ``nitid_meta.model`` is the public name from the lock.

    The exporter infers the name from the architecture; a checkpoint it cannot recognise gets no
    ``model`` key, which the meta schema requires. The lock is the authority here: add the name
    when it is missing and refuse a file that already says something else.
    """
    meta = read_meta(path)
    if meta.get("model") == model_id:
        return
    if "model" in meta:
        raise PublishError(
            f"{path.name}: nitid_meta.model is {meta['model']!r} but the lock says {model_id!r}"
        )
    import onnx

    from dfine.export_meta import embed_onnx_meta

    proto = onnx.load(str(path))
    embed_onnx_meta(proto, {**meta, "model": model_id})
    onnx.save(proto, str(path))


def read_meta(path: Path) -> dict[str, Any]:
    """The ``nitid_meta`` embedded in an ONNX file; raises :class:`PublishError` if unusable."""
    from dfine.export_meta import read_onnx_meta

    try:
        meta = read_onnx_meta(path)
    except ValueError as exc:
        raise PublishError(f"{path.name}: unreadable nitid_meta ({exc})") from exc
    if meta is None:
        raise PublishError(f"{path.name}: the ONNX carries no nitid_meta")
    return meta


@dataclass(frozen=True)
class BuiltModel:
    """One published model, as recorded in the catalogue and the sidecar."""

    entry: Mapping[str, Any]
    file: str
    sha256: str
    size_bytes: int
    reused: bool

    def record(self) -> dict[str, Any]:
        return {
            "id": str(self.entry["nitid_model"]),
            "task": str(self.entry["task"]),
            "url": self.file,
            "sha256": self.sha256,
            "label": catalogue_label(self.entry),
        }


def _sidecar_document(
    entry: Mapping[str, Any],
    file: str,
    sha: str,
    size: int,
    meta: Mapping[str, Any],
    nitid_version: str,
) -> dict[str, Any]:
    return {
        "format_version": SIDECAR_FORMAT_VERSION,
        "id": str(entry["nitid_model"]),
        "task": str(entry["task"]),
        "file": file,
        "sha256": sha,
        "size_bytes": size,
        "nitid_version": nitid_version,
        "source": {
            "lock_id": str(entry["id"]),
            "url": str(entry["url"]),
            "sha256": str(entry["sha256"]),
            "revision": str(entry["revision"]),
            "upstream_repo": str(entry["upstream_repo"]),
            "license": str(entry["license"]),
            "weights_variant": str(entry["weights_variant"]),
        },
        "nitid_meta": dict(meta),
    }


def _reusable(
    out: Path, entry: Mapping[str, Any], file: str, nitid_version: str
) -> tuple[str, int] | None:
    """``(sha256, size)`` of an existing, still valid build of ``entry``; else ``None``."""
    onnx_path = out / file
    sidecar_path = out / (file + SIDECAR_SUFFIX)
    if not onnx_path.is_file() or not sidecar_path.is_file():
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        source = sidecar["source"]
        same = (
            sidecar["format_version"] == SIDECAR_FORMAT_VERSION
            and source["sha256"] == entry["sha256"]
            and source["lock_id"] == entry["id"]
            and sidecar["nitid_version"] == nitid_version
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not same:
        return None
    sha = sha256_file(onnx_path)
    if sha != sidecar.get("sha256"):
        return None
    return sha, onnx_path.stat().st_size


def obtain_weights(
    entry: Mapping[str, Any],
    workdir: Path,
    *,
    offline_from: Path | None,
    fetch: Fetch,
) -> Path:
    """Get the weight file of ``entry`` and prove it is the pinned one before anything loads it."""
    asset_name = Path(urlparse(str(entry["url"])).path).name
    if offline_from is not None:
        raw = offline_from / asset_name
        if not raw.is_file():
            raise PublishError(f"{entry['id']}: {asset_name} not found in {offline_from}")
    else:
        raw = workdir / asset_name
        try:
            fetch(entry, raw)
        except Exception as exc:
            raise PublishError(f"{entry['id']}: download failed ({exc})") from exc
    actual = sha256_file(raw)
    if actual != entry["sha256"]:
        raise PublishError(
            f"{entry['id']}: checksum mismatch for {asset_name}: "
            f"lock says {entry['sha256']}, file is {actual}"
        )
    return raw


def build_one(
    entry: Mapping[str, Any],
    out: Path,
    *,
    nitid_version: str,
    offline_from: Path | None,
    cache_dir: Path | None,
    force: bool,
    fetch: Fetch,
    prepare: Prepare,
    export: Export,
) -> BuiltModel:
    """Produce ``<id>-<task>.onnx`` and its sidecar in ``out`` (or reuse an identical build)."""
    check_publishable(entry)
    file = onnx_file_name(str(entry["nitid_model"]), str(entry["task"]))

    if not force:
        reusable = _reusable(out, entry, file, nitid_version)
        if reusable is not None:
            return BuiltModel(entry, file, reusable[0], reusable[1], reused=True)

    with tempfile.TemporaryDirectory(prefix="nitid-publish-") as scratch_name:
        scratch = Path(scratch_name)
        raw = obtain_weights(
            entry,
            cache_dir if cache_dir is not None else scratch,
            offline_from=offline_from,
            fetch=fetch,
        )
        checkpoint = prepare(entry, raw, scratch / "wrapped.pth")
        staging = scratch / "export"
        staging.mkdir()
        produced = staging / "model.onnx"
        export(checkpoint, str(entry["task"]), produced)
        if not produced.is_file():
            raise PublishError(f"{entry['id']}: the exporter produced no file")
        ensure_meta_model(produced, str(entry["nitid_model"]))
        meta = read_meta(produced)
        if meta.get("task") != entry["task"]:
            raise PublishError(
                f"{entry['id']}: the ONNX says task {meta.get('task')!r}, lock says "
                f"{entry['task']!r}"
            )
        sha = sha256_file(produced)
        size = produced.stat().st_size
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced, out / file)

    sidecar = _sidecar_document(entry, file, sha, size, meta, nitid_version)
    (out / (file + SIDECAR_SUFFIX)).write_text(dump_json(sidecar), encoding="utf-8")
    return BuiltModel(entry, file, sha, size, reused=False)


def copy_notices(entries: Sequence[Mapping[str, Any]], out: Path, notices_dir: Path) -> list[str]:
    """Copy the licence files the entries require. Missing or empty sources are errors."""
    copied: list[str] = []
    for entry in entries:
        slug = notice_slug(str(entry["upstream_repo"]))
        for name, relative in zip(entry["notice_files"], notice_relative_paths(entry), strict=True):
            source = notices_dir / slug / name
            if not source.is_file() or source.stat().st_size == 0:
                raise PublishError(
                    f"{entry['id']}: required notice file {source} is missing or empty; "
                    "no licence file, no publication"
                )
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if relative not in copied:
                copied.append(relative)
    return sorted(copied)


def _prune_stale(out: Path, keep: set[str]) -> list[str]:
    """Remove ONNX files (and sidecars) of earlier runs that are no longer selected."""
    removed = []
    for path in sorted(out.glob("*.onnx")) + sorted(out.glob("*.onnx" + SIDECAR_SUFFIX)):
        base = path.name.removesuffix(SIDECAR_SUFFIX)
        if base not in keep:
            path.unlink()
            removed.append(path.name)
    return removed


def publish(
    lock: Mapping[str, Any],
    out: Path,
    *,
    entries: Sequence[Mapping[str, Any]] | None = None,
    offline_from: Path | None = None,
    cache_dir: Path | None = None,
    notices_dir: Path = DEFAULT_NOTICES_DIR,
    force: bool = False,
    nitid_version: str | None = None,
    fetch: Fetch = default_fetch,
    prepare: Prepare = wrap_checkpoint,
    export: Export = default_export,
    log: Callable[[str], None] = lambda message: None,
) -> list[BuiltModel]:
    """Build the whole bundle in ``out``; returns the built models in catalogue order."""
    selected = list(entries) if entries is not None else select_entries(lock)
    for entry in selected:
        check_publishable(entry)
    version = nitid_version or _nitid_version()
    out.mkdir(parents=True, exist_ok=True)

    built: list[BuiltModel] = []
    for entry in selected:
        log(f"{entry['id']}: {entry['nitid_model']} {entry['task']}")
        model = build_one(
            entry,
            out,
            nitid_version=version,
            offline_from=offline_from,
            cache_dir=cache_dir,
            force=force,
            fetch=fetch,
            prepare=prepare,
            export=export,
        )
        log(f"  {'reused' if model.reused else 'built'} {model.file} ({model.size_bytes} bytes)")
        built.append(model)

    notice_paths = copy_notices(selected, out, notices_dir)
    _prune_stale(out, {m.file for m in built})
    stale_notices = (
        [p for p in sorted((out / NOTICES_DIR_NAME).rglob("*")) if p.is_file()]
        if (out / NOTICES_DIR_NAME).is_dir()
        else []
    )
    for path in stale_notices:
        if path.relative_to(out).as_posix() not in notice_paths:
            path.unlink()

    (out / CATALOGUE_NAME).write_text(
        dump_json(build_catalogue(m.record() for m in built)), encoding="utf-8"
    )
    (out / NOTICE_NAME).write_text(render_notice(selected, version), encoding="utf-8")

    hashes = {
        path.relative_to(out).as_posix(): sha256_file(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != CHECKSUMS_NAME
    }
    (out / CHECKSUMS_NAME).write_text(format_sha256sums(hashes), encoding="utf-8")
    return built


# ------------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.publish_models",
        description=(__doc__ or "").split("\n\n")[0],
    )
    parser.add_argument("--out", type=Path, required=True, help="bundle folder (created)")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--models",
        help="comma separated lock ids (dfine-s-coco) or public names (nitid1s); "
        "hostable: false entries are refused",
    )
    parser.add_argument("--tasks", help="comma separated: detect,segment (default: both)")
    parser.add_argument(
        "--sizes",
        help="size allow-list, comma separated letters from n,s,m,l,x "
        f"(default {','.join(DEFAULT_SIZES)} unless --models is given)",
    )
    parser.add_argument(
        "--offline-from",
        type=Path,
        help="folder with already downloaded weights, named like the release assets "
        "(dfine_s_coco.pth); nothing is downloaded. They are still checked against the lock",
    )
    parser.add_argument("--cache-dir", type=Path, help="keep downloaded weights here")
    parser.add_argument("--notices-dir", type=Path, default=DEFAULT_NOTICES_DIR)
    parser.add_argument("--force", action="store_true", help="rebuild even if up to date")
    parser.add_argument("--dry-run", action="store_true", help="print the selection and stop")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        lock = load_lock(args.lock, args.schema)
        entries = select_entries(
            lock,
            models=_split_csv(args.models),
            tasks=_split_csv(args.tasks),
            sizes=_split_csv(args.sizes),
        )
        if args.dry_run:
            for entry in entries:
                print(
                    f"{entry['id']}: {onnx_file_name(entry['nitid_model'], entry['task'])} "
                    f"({entry['license']}, {entry['size_bytes']} bytes of weights)"
                )
            return 0
        built = publish(
            lock,
            args.out,
            entries=entries,
            offline_from=args.offline_from,
            cache_dir=args.cache_dir,
            notices_dir=args.notices_dir,
            force=args.force,
            log=print,
        )
    except PublishError as exc:
        print(f"publish_models: {exc}", file=sys.stderr)
        return 1
    total = sum(m.size_bytes for m in built)
    print(f"published {len(built)} model(s), {total} bytes of ONNX, to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
