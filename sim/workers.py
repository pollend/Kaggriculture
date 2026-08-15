"""Process pools for episode play, shared by rollout collection and the arena.

Two costs live here, and both used to be paid once per *batch* of episodes:

1. Spawning workers. Eight `spawn` children each start a fresh interpreter and
   import numpy and `kaggle_environments` before playing anything. A round hook
   measured ~25s against ~13s of ideal simulation, and most of the gap was this.
2. BLAS oversubscription. Parallelism here is across episodes, not inside one,
   so every worker's BLAS opening a machine-sized thread pool means `jobs`
   workers asking for `jobs * cores` threads.

Pools are borrowed from a process-global free-list, one per concurrent caller. A
population run drives `training_concurrency` learners from one process, one per
thread, so a single shared pool would collapse `training_concurrency * jobs`
workers down to `jobs`; keying on the *thread* instead leaks a pool per round,
because those threads are recreated per batch. See `_IDLE` below.
"""

from __future__ import annotations

import atexit
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

#: Thread-count variables every BLAS backend numpy might be linked against reads.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: Reference count and saved values for the pin below. Module state rather than
#: closure state because the thing being guarded -- `os.environ` -- is itself
#: process-global, so the guard has to be too.
_PIN_LOCK = threading.Lock()
_pin_depth = 0
_pin_saved: dict[str, str | None] = {}


def pin_active() -> bool:
    """Whether any thread currently holds the BLAS pin."""
    with _PIN_LOCK:
        return _pin_depth > 0


@contextmanager
def single_threaded_workers():
    """Pin spawned episode workers to one BLAS thread each.

    Left alone, every worker's BLAS opens a thread pool sized to the whole
    machine, so `jobs` workers ask for `jobs * cores` threads: measured at ~350%
    CPU per worker, eight workers wanting ~29 cores of a 32-core box, and a
    16-episode batch taking 17 minutes against an ideal of well under one.

    Set in the parent rather than in an initializer because `spawn` children
    inherit the environment at launch, and the variables have to be set before
    numpy is imported in the child -- which happens while the task is being
    unpickled, ahead of any initializer running.

    Reference counted, because `os.environ` is process-global and this is now
    entered from several threads at once. One learner per thread is how a
    population run keeps a 32-core box busy, and the naive save/restore is a race
    on exactly the variable it exists to control: the second thread in saves the
    "1"s the first one wrote, and when *it* exits it restores them as though they
    were the originals -- so the pin either leaks forever or, far worse, is torn
    down while another thread is still spawning workers. That worker inherits an
    unpinned environment, and the docstring above measures what that costs: ~350%
    CPU per worker and a 16-episode batch taking 17 minutes.

    The innermost enter pins and the outermost exit restores, so a single-threaded
    caller sees byte-for-byte the previous behaviour.
    """

    global _pin_depth
    with _PIN_LOCK:
        if _pin_depth == 0:
            _pin_saved.clear()
            _pin_saved.update({name: os.environ.get(name) for name in _THREAD_VARS})
            os.environ.update({name: "1" for name in _THREAD_VARS})
        _pin_depth += 1
    try:
        yield
    finally:
        with _PIN_LOCK:
            _pin_depth -= 1
            if _pin_depth == 0:
                for name, value in _pin_saved.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
                _pin_saved.clear()


#: Pools available for checkout, keyed by width, and the ones currently lent out.
#:
#: Process-global rather than thread-local, and that is the whole point. The
#: threads that reach here do not live long enough to own a cache: both
#: `run_round_hooks` and `_update_session_models` open a `ThreadPoolExecutor` per
#: batch and close it again, so a thread-keyed cache misses on every round, builds
#: `concurrency` fresh pools, and orphans the previous round's -- unreachable
#: through the dead thread's local but pinned by the registry, so never shut down.
#: Measured: cohorts of `concurrency * jobs` workers arriving one per round and
#: none ever exiting, 193 live children and 543 open descriptors five minutes in,
#: round wall clock drifting 43s to 79s as they oversubscribed the box, and the run
#: dying on `OSError: [Errno 24] Too many open files` while opening a connection.
#:
#: Borrowed and returned, so a pool outlives the thread that used it and the live
#: count settles at the high-water mark of *simultaneous* callers rather than
#: growing with elapsed time. Isolation is unchanged: a caller still gets a pool
#: no one else is mapping on.
_IDLE: dict[int, list[ProcessPoolExecutor]] = {}
_BUSY: set[ProcessPoolExecutor] = set()
_REGISTRY_LOCK = threading.Lock()


def _noop(_index: int) -> None:
    """Warm-up task. Module level because a worker has to unpickle it by name."""


def _spawn_pool(jobs: int) -> ProcessPoolExecutor:
    """Build a pool and spawn its workers, both under the BLAS pin.

    Pinning here rather than trusting the caller is what lets the free-list key on
    width alone. A worker inherits the BLAS variables at launch and keeps them for
    life, so a pool spawned by an unpinned caller is oversubscribed for good --
    which is why the cache used to key on `pin_active()` too, and why it therefore
    held two pools per width. `play_series` calls in unpinned and `collect_batch`
    calls in pinned, so that second copy was not hypothetical.

    Workers are spawned lazily by `submit`, so the pin has to still be held when
    the first task goes in: the warm-up map is what forces the spawn while the
    variables are set, instead of leaving it to whatever the first real caller
    happens to be. `_pin_depth` is refcounted, so entering under a caller that
    already holds the pin is a no-op rather than a nested save/restore.
    """

    with single_threaded_workers():
        pool = ProcessPoolExecutor(
            max_workers=jobs, mp_context=multiprocessing.get_context("spawn")
        )
        try:
            list(pool.map(_noop, range(jobs)))
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return pool


@contextmanager
def worker_pool(jobs: int):
    """Lend out a `jobs`-wide pool, building one if none is free.

    The pool goes back on the free-list when the block exits, so the next round --
    on whatever thread it happens to run -- reuses these workers instead of
    spawning a fresh interpreter per batch. That reuse is the point: eight `spawn`
    children each import numpy and `kaggle_environments` before playing anything,
    ~25s of a round hook against ~13s of ideal simulation.

    A pool that broke mid-map must not come back here. Call `discard_pool` inside
    the block and it is dropped instead of returned.
    """

    with _REGISTRY_LOCK:
        free = _IDLE.get(jobs)
        pool = free.pop() if free else None
        if pool is not None:
            _BUSY.add(pool)
    if pool is None:
        pool = _spawn_pool(jobs)
        with _REGISTRY_LOCK:
            _BUSY.add(pool)
    try:
        yield pool
    finally:
        with _REGISTRY_LOCK:
            if pool in _BUSY:
                _BUSY.discard(pool)
                _IDLE.setdefault(jobs, []).append(pool)


def discard_pool(pool: ProcessPoolExecutor) -> None:
    """Drop a pool that broke mid-map, so the next checkout builds a fresh one."""
    with _REGISTRY_LOCK:
        _BUSY.discard(pool)
        for width, free in _IDLE.items():
            _IDLE[width] = [cached for cached in free if cached is not pool]
    pool.shutdown(wait=False, cancel_futures=True)


def shutdown_worker_pools() -> None:
    """Tear every live pool down. Registered at exit; safe to call early.

    `wait=True` because this is where the children are actually reaped. Returning
    before they exit is what leaves the `resource_tracker` complaining about leaked
    semaphores at interpreter shutdown.
    """
    with _REGISTRY_LOCK:
        pools = [pool for free in _IDLE.values() for pool in free]
        pools.extend(_BUSY)
        _IDLE.clear()
        _BUSY.clear()
    for pool in pools:
        pool.shutdown(wait=True, cancel_futures=True)


def live_pool_count() -> int:
    """How many pools exist right now, idle plus lent out. For tests."""
    with _REGISTRY_LOCK:
        return sum(len(free) for free in _IDLE.values()) + len(_BUSY)


atexit.register(shutdown_worker_pools)
