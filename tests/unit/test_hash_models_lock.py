from __future__ import annotations

import hashlib
import http.server
import io
import json
import threading
import urllib.error
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tools import hash_models_lock as h

PAYLOAD_A = b"A" * (2 * h.CHUNK + 123)  # more than one chunk, so streaming is exercised
PAYLOAD_B = b"nitid-weights-b"
SHA_A = hashlib.sha256(PAYLOAD_A).hexdigest()
SHA_B = hashlib.sha256(PAYLOAD_B).hexdigest()


class _Handler(http.server.BaseHTTPRequestHandler):
    files = {"/a.pth": PAYLOAD_A, "/b.pth": PAYLOAD_B}

    def log_message(self, *args: Any) -> None:  # keep test output quiet
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/redirect.pth":
            self.send_response(302)
            self.send_header("Location", "/b.pth")
            self.end_headers()
            return
        if self.path == "/truncated.pth":
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"x" * 10)  # fewer bytes than announced
            return
        body = self.files.get(self.path)
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _lock(server: str, **overrides: Any) -> dict[str, Any]:
    entries = [
        {
            "id": "a",
            "url": f"{server}/a.pth",
            "size_bytes": len(PAYLOAD_A),
            "sha256": None,
            "sha256_status": "unverified",
            "hostable": False,
            "sha256_note": "keep me",
        },
        {
            "id": "b",
            "url": f"{server}/b.pth",
            "size_bytes": None,
            "sha256": SHA_B,
            "sha256_status": "verified",
            "hostable": True,
        },
    ]
    for entry in entries:
        entry.update(overrides.get(entry["id"], {}))
    return {"schema_version": 1, "entries": entries}


def _run(tmp_path: Path, lock: dict[str, Any], *extra: str) -> tuple[int, dict[str, Any], Path]:
    lock_path = tmp_path / "models.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    report = tmp_path / "report.json"
    code = h.main(
        ["--lock", str(lock_path), "--report", str(report), "--retries", "1", *extra],
        sleep=lambda _: None,
    )
    return code, json.loads(report.read_text(encoding="utf-8")), lock_path


def test_sha256_stream_matches_hashlib():
    digest, total = h.sha256_stream(iter([b"ab", b"", b"c"]))
    assert digest == hashlib.sha256(b"abc").hexdigest()
    assert total == 3


def test_hash_url_streams_a_multi_chunk_file(server: str):
    assert h.hash_url(f"{server}/a.pth") == (SHA_A, len(PAYLOAD_A))


def test_hash_url_follows_redirects(server: str):
    assert h.hash_url(f"{server}/redirect.pth") == (SHA_B, len(PAYLOAD_B))


def test_hash_url_reports_progress(server: str):
    seen: list[tuple[int, int | None]] = []
    h.hash_url(f"{server}/a.pth", on_progress=lambda done, total: seen.append((done, total)))
    assert seen[-1] == (len(PAYLOAD_A), len(PAYLOAD_A))
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


def test_hash_url_keeps_a_copy(server: str, tmp_path: Path):
    keep = tmp_path / "a.pth"
    h.hash_url(f"{server}/a.pth", keep=keep)
    assert keep.read_bytes() == PAYLOAD_A


def test_hash_url_does_not_retry_a_404(server: str):
    calls: list[float] = []
    with pytest.raises(h.FetchError, match="404"):
        h.hash_url(f"{server}/missing.pth", retries=3, sleep=calls.append)
    assert calls == []  # a 404 will not fix itself


def test_hash_url_detects_a_truncated_download_and_retries(server: str):
    calls: list[float] = []
    with pytest.raises(h.FetchError, match="truncated|IncompleteRead"):
        h.hash_url(f"{server}/truncated.pth", retries=3, sleep=calls.append)
    assert len(calls) == 2  # retried, with growing back-off
    assert calls == [2.0, 4.0]


def test_select_entries_filters_and_rejects_unknown_ids():
    entries = [
        {"id": "x", "sha256": None},
        {"id": "y", "sha256": "f" * 64, "sha256_status": "verified"},
    ]
    assert [e["id"] for e in h.select_entries(entries, only=None, missing_only=False)] == ["x", "y"]
    assert [e["id"] for e in h.select_entries(entries, only=None, missing_only=True)] == ["x"]
    assert [e["id"] for e in h.select_entries(entries, only={"y"}, missing_only=False)] == ["y"]
    with pytest.raises(SystemExit, match="unknown id"):
        h.select_entries(entries, only={"nope"}, missing_only=False)


def test_compare_with_lock():
    entry = {"sha256": "a" * 64, "size_bytes": 5}
    assert h.compare_with_lock(entry, "a" * 64, 5) == []
    problems = h.compare_with_lock(entry, "b" * 64, 6)
    assert len(problems) == 2
    assert h.compare_with_lock({"sha256": None, "size_bytes": None}, "b" * 64, 6) == []


def test_main_writes_a_report_and_leaves_the_lock_alone(server: str, tmp_path: Path):
    code, report, lock_path = _run(tmp_path, _lock(server))
    assert code == 0
    assert report["results"]["a"]["sha256"] == SHA_A
    assert report["results"]["a"]["status"] == "ok"
    assert report["results"]["b"]["status"] == "ok"  # matches the lock's sha256
    assert json.loads(lock_path.read_text())["entries"][0]["sha256"] is None


def test_main_reports_a_mismatch_and_exits_1(server: str, tmp_path: Path):
    code, report, _ = _run(tmp_path, _lock(server, b={"sha256": "0" * 64}))
    assert code == 1
    assert report["results"]["b"]["status"] == "mismatch"
    assert "differs from the lock" in report["results"]["b"]["problems"][0]


def test_main_reports_a_failed_download_and_keeps_going(server: str, tmp_path: Path):
    lock = _lock(server, a={"url": f"{server}/missing.pth"})
    code, report, _ = _run(tmp_path, lock)
    assert code == 1
    assert report["results"]["a"]["status"] == "error"
    assert report["results"]["b"]["status"] == "ok"


def test_update_lock_writes_only_sha_and_status(server: str, tmp_path: Path):
    code, _, lock_path = _run(tmp_path, _lock(server), "--update-lock")
    assert code == 0
    entry = json.loads(lock_path.read_text())["entries"][0]
    assert entry["sha256"] == SHA_A
    assert entry["sha256_status"] == "verified"
    assert entry["hostable"] is False  # a licence decision: never touched
    assert entry["sha256_note"] == "keep me"


def test_update_lock_never_replaces_a_different_hash(server: str, tmp_path: Path):
    code, _, lock_path = _run(tmp_path, _lock(server, b={"sha256": "0" * 64}), "--update-lock")
    assert code == 1
    assert json.loads(lock_path.read_text())["entries"][1]["sha256"] == "0" * 64


def test_only_and_missing_only(server: str, tmp_path: Path):
    _, report, _ = _run(tmp_path, _lock(server), "--missing-only")
    assert list(report["results"]) == ["a"]
    _, report, _ = _run(tmp_path, _lock(server), "--only", "b")
    assert list(report["results"]) == ["b"]


class _FakeResponse(io.BytesIO):
    headers: dict[str, str] = {}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def test_fetch_huggingface_parses_commit_license_and_lfs_hashes():
    payload = {
        "sha": "c0ffee" * 6 + "abcd",
        "cardData": {"license": "apache-2.0"},
        "siblings": [
            {
                "rfilename": "dfine_seg_s_coco.pt",
                "size": 123,
                "lfs": {"sha256": "f" * 64, "size": 123},
            },
            {"rfilename": "README.md", "size": 10},
        ],
    }
    seen: list[str] = []

    def opener(url: str, timeout: float) -> _FakeResponse:
        seen.append(url)
        return _FakeResponse(json.dumps(payload).encode())

    info = h.fetch_huggingface(opener=opener)
    assert seen == ["https://huggingface.co/api/models/ArgoSA/D-FINE-seg?blobs=true"]
    assert info["revision"] == payload["sha"]
    assert info["license_tag"] == "apache-2.0"
    assert info["files"][0] == {
        "file": "dfine_seg_s_coco.pt",
        "size_bytes": 123,
        "sha256": "f" * 64,
    }
    assert info["files"][1]["sha256"] is None


def test_fetch_huggingface_falls_back_to_the_license_tag():
    payload = {"sha": "a" * 40, "tags": ["license:apache-2.0", "x"], "siblings": []}
    info = h.fetch_huggingface(
        opener=lambda url, timeout: _FakeResponse(json.dumps(payload).encode())
    )
    assert info["license_tag"] == "apache-2.0"


def test_main_records_a_huggingface_failure(server: str, tmp_path: Path):
    def opener(url: str, timeout: float) -> Any:
        if "huggingface.co" in url:
            raise urllib.error.URLError("blocked")
        return h._open(url, timeout)

    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(_lock(server)), encoding="utf-8")
    report_path = tmp_path / "r.json"
    code = h.main(
        ["--lock", str(lock_path), "--report", str(report_path), "--huggingface", "--retries", "1"],
        opener=opener,
        sleep=lambda _: None,
    )
    report = json.loads(report_path.read_text())
    assert code == 1
    assert report["huggingface"]["status"] == "error"
    assert report["results"]["a"]["status"] == "ok"  # the downloads were not lost


def test_huggingface_only_downloads_nothing(server: str, tmp_path: Path):
    payload = {"sha": "a" * 40, "cardData": {"license": "apache-2.0"}, "siblings": []}
    urls: list[str] = []

    def opener(url: str, timeout: float) -> _FakeResponse:
        urls.append(url)
        return _FakeResponse(json.dumps(payload).encode())

    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(_lock(server)), encoding="utf-8")
    report_path = tmp_path / "r.json"
    code = h.main(
        ["--lock", str(lock_path), "--report", str(report_path), "--huggingface-only"],
        opener=opener,
        sleep=lambda _: None,
    )
    report = json.loads(report_path.read_text())
    assert code == 0
    assert report["results"] == {}
    assert report["huggingface"]["revision"] == "a" * 40
    assert urls == ["https://huggingface.co/api/models/ArgoSA/D-FINE-seg?blobs=true"]


def test_the_real_lock_is_readable_by_the_script():
    root = Path(__file__).resolve().parents[2]
    lock = json.loads((root / "studio" / "models.lock.json").read_text(encoding="utf-8"))
    assert all(e["url"].startswith("https://") for e in lock["entries"])
    pending = h.select_entries(lock["entries"], only=None, missing_only=True)
    # "pending" are exactly the entries without a verified sha256 (none today; new ones will show up)
    assert all(not (e.get("sha256") and e.get("sha256_status") == "verified") for e in pending)
    assert len(pending) == sum(
        1 for e in lock["entries"] if not (e.get("sha256") and e.get("sha256_status") == "verified")
    )
