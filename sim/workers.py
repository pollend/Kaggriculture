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
import faulthandler
import multiprocessing
import os
import signal
import sys
import threading
import time
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


#: Ceiling on one `map` over a checked-out pool, in seconds.
#:
#: Sized off the measurement in `_update_session_models`: 222.6 seconds per update
#: of which 96% is simulation, and an update is two series back to back. Half an
#: hour is roughly eight times the worst observed series, so nothing healthy trips
#: it -- it exists because the alternative is what four runs did on 2026-08-15,
#: which is sit at "model updates started" forever with 32 idle workers, no log
#: line, and no traceback. A stall has to become a failure to be a bug you can see.
MAP_TIMEOUT_SECONDS = 1800.0

#: How often a worker checks whether it still has the parent it was spawned for.
_ORPHAN_POLL_SECONDS = 5.0


def _noop(_index: int) -> None:
    """Warm-up task. Module level because a worker has to unpickle it by name."""


def _worker_guard(parent_pid: int) -> None:
    """Worker initializer: exit if the parent process goes away.

    Nothing in a `spawn` child notices its parent dying, so a parent killed while
    a pool is checked out leaves `jobs` workers reparented to init, holding their
    memory and their share of the cores forever. Three runs left 99 such orphans
    behind on 2026-08-15 -- 21GB and 272% of the box -- because `atexit` does not
    run on SIGTERM and there was no other backstop.

    `PR_SET_PDEATHSIG` is the obvious mechanism and the wrong one here: it fires
    when the *creating thread* exits, and the free-list above exists precisely so a
    pool outlives the thread that spawned it. Round-hook threads are recreated per
    batch, so pdeathsig would shoot the workers of a perfectly live pool the moment
    its original borrower was recycled. Polling `getppid` is keyed on the process,
    which is the thing that actually has to still be there.

    `os._exit` rather than `sys.exit`: this is a daemon thread in a child whose
    parent is already gone, and there is no one left to flush to.
    """

    def watch() -> None:
        while True:
            if os.getppid() != parent_pid:
                os._exit(1)
            time.sleep(_ORPHAN_POLL_SECONDS)

    threading.Thread(target=watch, daemon=True, name="orphan-guard").start()


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
            max_workers=jobs,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_guard,
            initargs=(os.getpid(),),
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


def _reap(pool: ProcessPoolExecutor, grace: float) -> None:
    """Shut a pool down and make sure its workers are actually gone.

    `shutdown(wait=False, cancel_futures=True)` cancels what is still *queued*; it
    has no say over a worker already inside a task, and `wait=True` just blocks on
    one. Either way `concurrent.futures.process` installs its own `atexit` hook
    that joins every live worker at interpreter shutdown, so a single wedged worker
    means the process cannot exit at all and the `resource_tracker` prints its
    leaked-semaphore warning on the way down. Reproduced directly: a pool discarded
    after a stalled `map` left the interpreter unkillable and leaked 5 semaphores --
    the same signature the hung runs of 2026-08-15 left behind.

    So: ask nicely, wait `grace` seconds in total, then SIGKILL whatever is left.
    A pool reaches here only on the way out, so being blunt costs nothing.

    `_processes` is private and there is no public way to reach the workers of an
    executor being torn down, hence the defensive read.
    """

    workers = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    deadline = time.monotonic() + grace
    for worker in workers:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            try:
                worker.join(remaining)
            except (OSError, ValueError, AssertionError):
                pass
        if worker.exitcode is None:
            try:
                worker.kill()
            except (OSError, ValueError, AttributeError):
                pass


def discard_pool(pool: ProcessPoolExecutor) -> None:
    """Drop a pool that broke or stalled, so the next checkout builds a fresh one."""
    with _REGISTRY_LOCK:
        _BUSY.discard(pool)
        for width, free in _IDLE.items():
            _IDLE[width] = [cached for cached in free if cached is not pool]
    # No grace: this pool is already known bad, and the caller is usually about to
    # spawn its replacement.
    _reap(pool, grace=0.0)


#: How long a pool gets to drain on the way out before its workers are killed.
_SHUTDOWN_GRACE_SECONDS = 10.0


def shutdown_worker_pools() -> None:
    """Tear every live pool down. Registered at exit; safe to call early.

    This is where the children are actually reaped -- returning before they exit is
    what leaves the `resource_tracker` complaining about leaked semaphores. But it
    is also on the SIGTERM path, and an unbounded `wait=True` there means a run
    that wedged mid-arena cannot be stopped by the signal that is supposed to stop
    it: the handler blocks on the very worker that is stuck. Bounded instead, so
    the teardown always finishes. See `_reap`.
    """
    with _REGISTRY_LOCK:
        pools = [pool for free in _IDLE.values() for pool in free]
        pools.extend(_BUSY)
        _IDLE.clear()
        _BUSY.clear()
    for pool in pools:
        _reap(pool, grace=_SHUTDOWN_GRACE_SECONDS)


def live_pool_count() -> int:
    """How many pools exist right now, idle plus lent out. For tests."""
    with _REGISTRY_LOCK:
        return sum(len(free) for free in _IDLE.values()) + len(_BUSY)


atexit.register(shutdown_worker_pools)


def install_signal_handlers() -> None:
    """Make SIGTERM tear pools down, and SIGUSR1 dump every thread's stack.

    `atexit` covers a clean return and it covers `KeyboardInterrupt`, because that
    unwinds through the interpreter. It does not cover SIGTERM, whose default
    disposition kills the process outright -- so `kill` on a long run, which is how
    one is actually stopped, is exactly the case the `atexit` hook above misses.
    That is where the 99 orphans came from.

    SIGUSR1 is the other half. A run that wedges gives no traceback because nothing
    raised: on 2026-08-15 four runs stopped at "model updates started" with all 32
    workers idle and the main thread parked on the `ThreadPoolExecutor` join, and
    the process had to be killed to be cleared, which destroyed the only evidence.
    `kill -USR1 <pid>` now writes all thread stacks to stderr -- the nohup log --
    and leaves the process running, so the next one can be diagnosed instead of
    just cleaned up.

    Idempotent, and a no-op on platforms without the signals.
    """

    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):
        # `chain=False` deliberately. SIGUSR1's default disposition is to terminate,
        # so chaining dumps the stacks and then kills the very process you were
        # trying to inspect -- which is the whole thing this exists to avoid.
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

    def terminate(signum: int, _frame) -> None:
        print(
            f"\nreceived signal {signum}; shutting {live_pool_count()} worker pool(s) down",
            file=sys.stderr,
            flush=True,
        )
        shutdown_worker_pools()
        # Re-raise with the handler cleared, so the exit status is the signal's
        # rather than a plain 0 -- a supervisor has to be able to tell a run that
        # was stopped from one that finished.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for name in ("SIGTERM", "SIGHUP"):
        handled = getattr(signal, name, None)
        if handled is not None:
            signal.signal(handled, terminate)
