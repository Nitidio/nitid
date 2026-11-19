"""Unit tests for the in-memory job store (threads synchronised with events, no sleeping)."""

import threading

import pytest

pytest.importorskip("fastapi", reason="needs the 'studio' extra")

from nitid.studio import jobs  # noqa: E402
from nitid.studio.jobs import JobConflict, JobStore  # noqa: E402


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def wait_until(store, job_id, states, timeout=10.0):
    event = threading.Event()
    deadline = threading.Timer(timeout, event.set)
    deadline.start()
    try:
        while not event.is_set():
            snap = store.get(job_id)
            if snap is not None and snap.status in states:
                return snap
            event.wait(0.005)
    finally:
        deadline.cancel()
    raise AssertionError(f"job {job_id} never reached {states}")


@pytest.fixture
def store():
    s = JobStore(max_workers=2)
    yield s
    s.shutdown(wait=True)


def test_job_runs_to_done(store):
    job_id = store.submit(("a", "detect"), lambda progress: None)
    snap = wait_until(store, job_id, {"done"})
    assert snap.progress == 1.0
    assert snap.to_payload() == {"status": "done", "progress": 1.0}


def test_job_ids_are_opaque_and_unique(store):
    ids = {store.submit(("k", i), lambda progress: None) for i in range(5)}
    assert len(ids) == 5
    assert all(i.startswith("j_") for i in ids)


def test_progress_is_reported_and_clamped(store):
    seen = threading.Event()
    release = threading.Event()

    def work(progress):
        progress(0.4)
        seen.set()
        release.wait(10)
        progress(7)  # clamped to 1.0 (but status is still running)

    job_id = store.submit("k", work)
    assert seen.wait(10)
    snap = store.get(job_id)
    assert snap.status == "running"
    assert snap.progress == pytest.approx(0.4)
    release.set()
    assert wait_until(store, job_id, {"done"}).progress == 1.0


def test_negative_progress_is_clamped(store):
    seen = threading.Event()
    release = threading.Event()

    def work(progress):
        progress(-3)
        seen.set()
        release.wait(10)

    job_id = store.submit("k", work)
    assert seen.wait(10)
    assert store.get(job_id).progress == 0.0
    release.set()
    wait_until(store, job_id, {"done"})


def test_pending_state_when_workers_are_busy():
    store = JobStore(max_workers=1)
    release = threading.Event()
    started = threading.Event()

    def blocker(progress):
        started.set()
        release.wait(10)

    first = store.submit("a", blocker)
    assert started.wait(10)
    second = store.submit("b", lambda progress: None)
    assert store.get(second).status == "pending"
    release.set()
    wait_until(store, first, {"done"})
    wait_until(store, second, {"done"})
    store.shutdown(wait=True)


def test_failure_hides_the_exception_text(store):
    def boom(progress):
        raise RuntimeError("/secret/internal/path.pth exploded")

    job_id = store.submit("k", boom)
    snap = wait_until(store, job_id, {"failed"})
    assert snap.error is not None
    assert "secret" not in snap.error
    assert snap.to_payload() == {"status": "failed", "progress": 0.0, "error": snap.error}


def test_failure_uses_describe_error(store):
    def boom(progress):
        progress(0.5)
        raise OSError("disk /x full")

    job_id = store.submit("k", boom, describe_error=lambda exc: f"safe:{type(exc).__name__}")
    snap = wait_until(store, job_id, {"failed"})
    assert snap.error == "safe:OSError"
    assert snap.progress == pytest.approx(0.5)  # progress is kept on failure


def test_describe_error_that_raises_falls_back_to_generic(store):
    def bad_describe(exc):
        raise ValueError("oops")

    job_id = store.submit(
        "k", lambda progress: (_ for _ in ()).throw(RuntimeError("x")), describe_error=bad_describe
    )
    assert wait_until(store, job_id, {"failed"}).error == "The job failed."


def test_conflict_while_active_then_allowed_after(store):
    release = threading.Event()
    started = threading.Event()

    def blocker(progress):
        started.set()
        release.wait(10)

    first = store.submit(("nitid1s", "detect"), blocker)
    assert started.wait(10)
    with pytest.raises(JobConflict) as info:
        store.submit(("nitid1s", "detect"), lambda progress: None)
    assert info.value.job_id == first
    # A different key is fine.
    other = store.submit(("nitid1s", "segment"), lambda progress: None)
    wait_until(store, other, {"done"})
    release.set()
    wait_until(store, first, {"done"})
    # After it finished the same key can run again.
    again = store.submit(("nitid1s", "detect"), lambda progress: None)
    assert again != first
    wait_until(store, again, {"done"})


def test_failed_job_does_not_block_a_retry(store):
    def boom(progress):
        raise RuntimeError("x")

    first = store.submit("k", boom)
    wait_until(store, first, {"failed"})
    second = store.submit("k", lambda progress: None)
    wait_until(store, second, {"done"})


def test_unknown_job_is_none(store):
    assert store.get("j_nope") is None


def test_finished_jobs_expire_after_ttl():
    clock = FakeClock()
    store = JobStore(ttl=60.0, clock=clock)
    job_id = store.submit("k", lambda progress: None)
    wait_until(store, job_id, {"done"})
    clock.now += 59.0
    assert store.get(job_id) is not None
    clock.now += 1.0
    assert store.get(job_id) is None
    store.shutdown(wait=True)


def test_active_jobs_never_expire():
    clock = FakeClock()
    store = JobStore(ttl=1.0, clock=clock)
    release = threading.Event()
    started = threading.Event()

    def blocker(progress):
        started.set()
        release.wait(10)

    job_id = store.submit("k", blocker)
    assert started.wait(10)
    clock.now += 10_000
    assert store.get(job_id).status == "running"
    release.set()
    wait_until(store, job_id, {"done"})
    store.shutdown(wait=True)


def test_expired_failed_jobs_free_memory():
    clock = FakeClock()
    store = JobStore(ttl=5.0, clock=clock)
    job_id = store.submit("k", lambda progress: 1 / 0)
    wait_until(store, job_id, {"failed"})
    clock.now += 5
    assert store.get(job_id) is None
    store.shutdown(wait=True)


def test_default_ttl_is_a_named_constant():
    assert jobs.JOB_TTL_SECONDS > 0
    assert JobStore()._ttl == jobs.JOB_TTL_SECONDS


@pytest.mark.parametrize("kwargs", [{"ttl": -1}, {"max_workers": 0}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        JobStore(**kwargs)


def test_concurrent_submissions_of_one_key_allow_exactly_one():
    store = JobStore(max_workers=2)
    release = threading.Event()
    results: list[str] = []
    conflicts: list[JobConflict] = []
    barrier = threading.Barrier(8)

    def attempt():
        barrier.wait(10)
        try:
            results.append(store.submit("same", lambda progress: release.wait(10)))
        except JobConflict as exc:
            conflicts.append(exc)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    release.set()
    assert len(results) == 1
    assert len(conflicts) == 7
    wait_until(store, results[0], {"done"})
    store.shutdown(wait=True)
