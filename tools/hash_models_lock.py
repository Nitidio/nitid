"""Download every weight listed in ``studio/models.lock.json`` and compute its SHA-256.

This is the helper behind the "SHA-256 not verified" gap recorded in the lock: the audit
environment could not download assets larger than 50 MB. Run it on a machine with normal
internet access and send the report back (or commit it).

Only the Python standard library is used, so it runs with any Python >= 3.10, no install::

    python tools/hash_models_lock.py                       # every entry, report to models-lock-hashes.json
    python tools/hash_models_lock.py --only dfine-m-coco   # a subset (comma separated ids)
    python tools/hash_models_lock.py --missing-only        # skip entries that already have a verified sha256
    python tools/hash_models_lock.py --huggingface         # also read ArgoSA/D-FINE-seg from the Hub API
    python tools/hash_models_lock.py --huggingface-only    # ONLY the Hub API query, no downloads
    python tools/hash_models_lock.py --update-lock         # also write the hashes into the lock file

What it does, and does not do:

* Each file is streamed to a hash, never kept on disk (``--keep DIR`` keeps a copy).
* The report is rewritten after every file, so an interruption loses nothing.
* A hash that disagrees with one already in the lock, or a size that disagrees with
  ``size_bytes``, is reported as a MISMATCH and the exit code is 1. Nothing is ever overwritten
  silently: ``--update-lock`` refuses to replace an existing, different sha256.
* ``--update-lock`` sets ``sha256`` and ``sha256_status: "verified"`` only. It does **not** touch
  ``hostable``, ``hostable_reason``, ``open_caveats`` or ``sha256_note``: whether an entry may be
  hosted is a licence decision (see ``docs/licensing/studio-weights.md``), reviewed by a person.
* ``--huggingface`` asks the Hub API (``/api/models/<repo>?blobs=true``) for the commit and the
  LFS SHA-256 of each file. It does not download weights. Hub licence and file names are reported
  as found; read the model card yourself before trusting a licence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

DEFAULT_LOCK = Path("studio/models.lock.json")
DEFAULT_REPORT = Path("models-lock-hashes.json")
HF_REPO = "ArgoSA/D-FINE-seg"
HF_API = "https://huggingface.co/api/models/{repo}?blobs=true"
CHUNK = 1024 * 1024
USER_AGENT = "nitid-hash-models-lock/1"


class FetchError(RuntimeError):
    """A download failed after all retries."""


def sha256_stream(chunks: Iterator[bytes]) -> tuple[str, int]:
    """Hash an iterator of byte chunks. Returns ``(hexdigest, total_bytes)``."""
    digest = hashlib.sha256()
    total = 0
    for chunk in chunks:
        digest.update(chunk)
        total += len(chunk)
    return digest.hexdigest(), total


def _open(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - URL comes from the audited lock


def iter_url(
    url: str,
    *,
    timeout: float = 60.0,
    opener: Callable[[str, float], Any] = _open,
    on_progress: Callable[[int, int | None], None] | None = None,
    keep: Path | None = None,
) -> Iterator[bytes]:
    """Yield the body of ``url`` in chunks (redirects are followed by urllib)."""
    sink = keep.open("wb") if keep is not None else None
    try:
        with opener(url, timeout) as response:
            length_header = response.headers.get("Content-Length")
            expected = int(length_header) if length_header and length_header.isdigit() else None
            done = 0
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                done += len(chunk)
                if sink is not None:
                    sink.write(chunk)
                if on_progress is not None:
                    on_progress(done, expected)
                yield chunk
            if expected is not None and done != expected:
                raise FetchError(f"truncated download: got {done} of {expected} bytes")
    finally:
        if sink is not None:
            sink.close()


def hash_url(
    url: str,
    *,
    retries: int = 3,
    timeout: float = 60.0,
    opener: Callable[[str, float], Any] = _open,
    on_progress: Callable[[int, int | None], None] | None = None,
    keep: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, int]:
    """Download ``url`` and return ``(sha256, size_bytes)``, retrying transient failures."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return sha256_stream(
                iter_url(url, timeout=timeout, opener=opener, on_progress=on_progress, keep=keep)
            )
        except (urllib.error.URLError, OSError, FetchError) as error:
            last = error
            if isinstance(error, urllib.error.HTTPError) and 400 <= error.code < 500:
                break  # a 404 will not fix itself
            if attempt < retries:
                sleep(2.0 * attempt)
    raise FetchError(f"{url}: {last}")


def select_entries(
    entries: Sequence[dict[str, Any]], *, only: set[str] | None, missing_only: bool
) -> list[dict[str, Any]]:
    """Filter lock entries by id and, optionally, by "no verified sha256 yet"."""
    selected = []
    for entry in entries:
        if only is not None and entry["id"] not in only:
            continue
        if missing_only and entry.get("sha256") and entry.get("sha256_status") == "verified":
            continue
        selected.append(entry)
    unknown = (only or set()) - {e["id"] for e in entries}
    if unknown:
        raise SystemExit(f"error: unknown id(s) in --only: {', '.join(sorted(unknown))}")
    return selected


def compare_with_lock(entry: dict[str, Any], sha256: str, size: int) -> list[str]:
    """Return human-readable disagreements between a download and the lock entry."""
    problems = []
    locked_sha = entry.get("sha256")
    if locked_sha and locked_sha != sha256:
        problems.append(f"sha256 differs from the lock ({locked_sha} in lock, {sha256} downloaded)")
    locked_size = entry.get("size_bytes")
    if locked_size and locked_size != size:
        problems.append(f"size differs from the lock ({locked_size} in lock, {size} downloaded)")
    return problems


def fetch_huggingface(
    repo: str = HF_REPO,
    *,
    opener: Callable[[str, float], Any] = _open,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Read commit, licence tag and per-file size and LFS SHA-256 from the Hub API (no weights)."""
    with opener(HF_API.format(repo=repo), timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    card = data.get("cardData") or {}
    files = []
    for sibling in data.get("siblings", []):
        lfs = sibling.get("lfs") or {}
        files.append(
            {
                "file": sibling.get("rfilename"),
                "size_bytes": sibling.get("size") or lfs.get("size"),
                "sha256": lfs.get("sha256"),  # None for small files stored without LFS
            }
        )
    return {
        "repo": repo,
        "revision": data.get("sha"),
        "license_tag": card.get("license")
        or next(
            (t.split(":", 1)[1] for t in data.get("tags", []) if t.startswith("license:")), None
        ),
        "files": files,
    }


def update_lock(lock: dict[str, Any], results: dict[str, dict[str, Any]]) -> list[str]:
    """Write verified hashes into ``lock`` in place. Returns the ids that were changed.

    Only ``sha256`` and ``sha256_status`` change. An existing, different sha256 is never replaced.
    """
    changed = []
    for entry in lock["entries"]:
        result = results.get(entry["id"])
        if not result or result.get("status") != "ok":
            continue
        locked = entry.get("sha256")
        if locked and locked != result["sha256"]:
            continue
        if locked == result["sha256"] and entry.get("sha256_status") == "verified":
            continue
        entry["sha256"] = result["sha256"]
        entry["sha256_status"] = "verified"
        if not entry.get("size_bytes"):
            entry["size_bytes"] = result["size_bytes"]
        changed.append(entry["id"])
    return changed


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _progress_printer(label: str) -> Callable[[int, int | None], None]:
    state = {"last": 0.0}

    def show(done: int, total: int | None) -> None:
        now = time.monotonic()
        if now - state["last"] < 0.5 and not (total and done >= total):
            return
        state["last"] = now
        pct = f" {100 * done / total:5.1f}%" if total else ""
        sys.stderr.write(f"\r  {label}: {done / 1e6:8.1f} MB{pct}")
        sys.stderr.flush()

    return show


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the weights in models.lock.json and compute their SHA-256."
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--only", help="comma separated entry ids")
    parser.add_argument(
        "--missing-only", action="store_true", help="skip entries with a verified sha256"
    )
    parser.add_argument(
        "--huggingface", action="store_true", help=f"also read {HF_REPO} from the Hub API"
    )
    parser.add_argument(
        "--huggingface-only",
        action="store_true",
        help=f"only query {HF_REPO} on the Hub API; download nothing",
    )
    parser.add_argument(
        "--update-lock", action="store_true", help="write hashes into the lock file"
    )
    parser.add_argument("--keep", type=Path, help="directory where downloaded files are kept")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    opener: Callable[[str, float], Any] = _open,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    args = build_parser().parse_args(argv)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None
    entries = select_entries(lock["entries"], only=only, missing_only=args.missing_only)
    if args.huggingface_only:
        entries = []
        args.huggingface = True
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "tool": "tools/hash_models_lock.py",
        "lock": str(args.lock),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": {},
    }
    exit_code = 0
    print(f"{len(entries)} file(s) to hash", file=sys.stderr)
    for index, entry in enumerate(entries, 1):
        label = f"[{index}/{len(entries)}] {entry['id']}"
        print(label, file=sys.stderr)
        keep = args.keep / entry["url"].rsplit("/", 1)[-1] if args.keep else None
        try:
            sha, size = hash_url(
                entry["url"],
                retries=args.retries,
                timeout=args.timeout,
                opener=opener,
                on_progress=_progress_printer(entry["id"]),
                keep=keep,
                sleep=sleep,
            )
            sys.stderr.write("\n")
            problems = compare_with_lock(entry, sha, size)
            result = {
                "status": "mismatch" if problems else "ok",
                "url": entry["url"],
                "size_bytes": size,
                "sha256": sha,
                "problems": problems,
            }
            if problems:
                exit_code = 1
        except FetchError as error:
            sys.stderr.write("\n")
            result = {"status": "error", "url": entry["url"], "error": str(error)}
            exit_code = 1
        report["results"][entry["id"]] = result
        write_json(args.report, report)  # after every file: an interruption loses nothing

    if args.huggingface:
        print(f"Hugging Face: {HF_REPO}", file=sys.stderr)
        try:
            report["huggingface"] = fetch_huggingface(opener=opener, timeout=args.timeout)
        except (urllib.error.URLError, OSError, ValueError) as error:
            report["huggingface"] = {"repo": HF_REPO, "status": "error", "error": str(error)}
            exit_code = 1
        write_json(args.report, report)

    if args.update_lock:
        changed = update_lock(lock, report["results"])
        if changed:
            write_json(args.lock, lock)
        print(
            f"lock updated for {len(changed)} entr(y/ies): {', '.join(changed) or '-'}",
            file=sys.stderr,
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nreport written to {args.report}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
