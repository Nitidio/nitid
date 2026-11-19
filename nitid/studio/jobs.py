"""In-memory store for short background jobs (checkpoint downloads).

Design constraints, all deliberate:

* **Local single-user tool**: jobs live in process memory and vanish on restart.
* **Thread-safe**: one lock guards the table; work runs in a small
  ``ThreadPoolExecutor`` (:data:`DEFAULT_MAX_WORKERS`).
* **One active job per key**: submitting a key that is still ``pending`` or
  ``running`` raises :class:`JobConflict` (the API maps it to 409).
* **Expiry**: finished jobs are forgotten :data:`JOB_TTL_SECONDS` after they
  finish. Active jobs never expire.
* **No leaks**: the error text a client sees comes from ``describe_error`` (a
  caller-supplied sanitiser), never from ``str(exception)``. The full exception is
  logged server-side.

The clock is injectable so TTL behaviour is testable without sleeping.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections.abc import Callable, Hashable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

#: How long a *finished* job stays queryable.
JOB_TTL_SECONDS = 3600.0
#: Concurrent downloads.
DEFAULT_MAX_WORKERS = 2

JobStatus = Literal["pending", "running", "done", "failed"]
ProgressCallback = Callable[[float], None]
JobFunction = Callable[[ProgressCallback], object]

_GENERIC_ERROR = "The job failed."


class JobConflict(Exception):
    """A job with the same key is already pending or running."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"job {job_id} is still active")
        self.job_id = job_id


@dataclass(frozen=True)
class JobSnapshot:
    """Immutable view of a job, safe to hand out of the lock."""

    id: str
    status: JobStatus
    progress: float
    error: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Shape of the ``Job`` schema in the API contract (``error`` only on failure)."""
        payload: dict[str, object] = {"status": self.status, "progress": self.progress}
        if self.status == "failed":
            payload["error"] = self.error or _GENERIC_ERROR
        return payload


@dataclass
class _Record:
    id: str
    key: Hashable
    status: JobStatus = "pending"
    progress: float = 0.0
    error: str | None = None
    finished_at: float | None = None

    def snapshot(self) -> JobSnapshot:
        return JobSnapshot(self.id, self.status, self.progress, self.error)


class JobStore:
    """Thread-safe job table plus the executor that runs the work."""

    def __init__(
        self,
        *,
        ttl: float = JOB_TTL_SECONDS,
        max_workers: int = DEFAULT_MAX_WORKERS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl < 0:
            raise ValueError("ttl must be >= 0")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._ttl = ttl
        self._max_workers = max_workers
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: dict[str, _Record] = {}
        self._executor: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------ public API

    def submit(
        self,
        key: Hashable,
        function: JobFunction,
        *,
        describe_error: Callable[[BaseException], str] | None = None,
    ) -> str:
        """Start ``function(progress)`` in the background and return its job id.

        ``function`` reports progress by calling ``progress(x)`` with ``x`` in
        ``[0, 1]`` (values are clamped). Raises :class:`JobConflict` if a job with
        the same ``key`` is still active.
        """
        with self._lock:
            self._expire_locked()
            for record in self._jobs.values():
                if record.key == key and record.status in ("pending", "running"):
                    raise JobConflict(record.id)
            job_id = f"j_{secrets.token_hex(6)}"
            record = _Record(id=job_id, key=key)
            self._jobs[job_id] = record
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._max_workers, thread_name_prefix="nitid-studio-job"
                )
            future: Future[None] = self._executor.submit(
                self._run, record, function, describe_error
            )
            del future  # outcome is captured in the record; _run never raises
        return job_id

    def get(self, job_id: str) -> JobSnapshot | None:
        """Current state of ``job_id``, or ``None`` if unknown or expired."""
        with self._lock:
            self._expire_locked()
            record = self._jobs.get(job_id)
            return record.snapshot() if record is not None else None

    def shutdown(self, *, wait: bool = False) -> None:
        """Stop accepting work. Used by tests; the server relies on process exit."""
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)

    # ------------------------------------------------------------------ internals

    def _expire_locked(self) -> None:
        now = self._clock()
        stale = [
            job_id
            for job_id, record in self._jobs.items()
            if record.finished_at is not None and now - record.finished_at >= self._ttl
        ]
        for job_id in stale:
            del self._jobs[job_id]

    def _set_progress(self, record: _Record, value: float) -> None:
        clamped = min(1.0, max(0.0, float(value)))
        with self._lock:
            if record.status == "running":
                record.progress = clamped

    def _run(
        self,
        record: _Record,
        function: JobFunction,
        describe_error: Callable[[BaseException], str] | None,
    ) -> None:
        with self._lock:
            record.status = "running"
        try:
            function(lambda value: self._set_progress(record, value))
        except Exception as exc:  # a job must never kill its worker thread
            logger.exception("Job %s failed", record.id)
            try:
                message = describe_error(exc) if describe_error is not None else _GENERIC_ERROR
            except Exception:
                message = _GENERIC_ERROR
            with self._lock:
                record.status = "failed"
                record.error = message
                record.finished_at = self._clock()
            return
        with self._lock:
            record.status = "done"
            record.progress = 1.0
            record.finished_at = self._clock()
