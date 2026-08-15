"""`sim.workers` lends process pools out and takes them back. Pin the bookkeeping.

The failure these exist for is not a wrong answer, it is a run that dies four
minutes in on `OSError: [Errno 24] Too many open files`. The pool cache used to
live in a `threading.local()`, and every thread that reached it was created and
destroyed per batch by the round-hook and model-update thread pools -- so each
round built `concurrency` new pools, orphaned the previous round's, and never shut
any of them down. Nothing about that is visible in an episode's output, so it has
to be asserted on the registry directly.
"""

from __future__ import annotations

import threading

import pytest

from sim import workers

JOBS = 2


def _square(value: int) -> int:
    """Module level so a spawned worker can unpickle it by name."""
    return value * value


@pytest.fixture(autouse=True)
def _clean_registry():
    """Every test starts and ends with no pools, so counts mean what they say."""
    workers.shutdown_worker_pools()
    try:
        yield
    finally:
        workers.shutdown_worker_pools()


def test_a_pool_outlives_the_thread_that_used_it():
    """The regression. A per-thread cache misses here and spawns a second pool."""

    seen: list[object] = []

    def borrow() -> None:
        with workers.worker_pool(JOBS) as pool:
            assert list(pool.map(_square, [1, 2, 3])) == [1, 4, 9]
            seen.append(pool)

    first = threading.Thread(target=borrow)
    first.start()
    first.join()

    second = threading.Thread(target=borrow)
    second.start()
    second.join()

    assert seen[0] is seen[1]
    assert workers.live_pool_count() == 1


def test_live_pools_track_concurrency_not_rounds():
    """Ten rounds of two concurrent callers is two pools, not twenty."""

    concurrency = 2
    for _ in range(10):
        start = threading.Barrier(concurrency)
        release = threading.Event()

        def borrow() -> None:
            with workers.worker_pool(JOBS) as pool:
                # Hold both pools at once, so the round genuinely needs two.
                start.wait(timeout=60)
                assert list(pool.map(_square, [4])) == [16]
                release.wait(timeout=60)

        threads = [threading.Thread(target=borrow) for _ in range(concurrency)]
        for thread in threads:
            thread.start()
        release.set()
        for thread in threads:
            thread.join()

        assert workers.live_pool_count() == concurrency


def test_widths_do_not_share_a_pool():
    with workers.worker_pool(JOBS) as narrow:
        pass
    with workers.worker_pool(JOBS + 1) as wide:
        pass
    assert narrow is not wide
    assert workers.live_pool_count() == 2

    # The pin state of the caller is not part of the key any more: pools are
    # spawned pinned from inside `worker_pool`, so a pinned caller reuses what an
    # unpinned one left behind instead of getting a second copy.
    with workers.single_threaded_workers():
        with workers.worker_pool(JOBS) as again:
            assert again is narrow
    assert workers.live_pool_count() == 2


def test_a_pool_is_spawned_pinned_regardless_of_the_caller():
    """Unpinned caller, still-pinned workers -- what the warm-up map buys."""

    assert not workers.pin_active()
    with workers.worker_pool(JOBS) as pool:
        threads = list(pool.map(_omp_threads, range(JOBS)))
    assert threads == ["1"] * JOBS


def _omp_threads(_index: int) -> str | None:
    import os

    return os.environ.get("OMP_NUM_THREADS")


def test_discarded_pools_do_not_come_back():
    with workers.worker_pool(JOBS) as broken:
        workers.discard_pool(broken)
    assert workers.live_pool_count() == 0

    with workers.worker_pool(JOBS) as fresh:
        assert fresh is not broken
        assert list(fresh.map(_square, [5])) == [25]


def test_shutdown_empties_both_registries_and_repeats_cleanly():
    with workers.worker_pool(JOBS):
        pass
    assert workers.live_pool_count() == 1

    workers.shutdown_worker_pools()
    assert workers.live_pool_count() == 0
    workers.shutdown_worker_pools()
    assert workers.live_pool_count() == 0


def test_the_pin_survives_a_failed_spawn():
    """A pool that cannot be built must not leave the BLAS pin held."""

    with pytest.raises(ValueError):
        with workers.worker_pool(0):
            pass
    assert not workers.pin_active()
    assert workers.live_pool_count() == 0
