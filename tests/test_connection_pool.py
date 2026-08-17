"""The connection pool behind the dashboard's reads.

Pooling only pays off if a broken connection is never handed to the next caller,
so these tests break connections on purpose: terminate one from another session,
raise inside a lease, and overflow the size limit.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from kaggriculture_lab.dashboard.errors import DashboardUnavailable
from kaggriculture_lab.dashboard.repository import DashboardRepository
from kaggriculture_lab.database import ConnectionPool, DatabaseTarget, connect_postgres
from tests.postgres_support import postgres_database


@pytest.fixture
def database(tmp_path: Path) -> DatabaseTarget:
    return postgres_database(tmp_path, "pool")


def _backend_pid(connection) -> int:
    return int(connection.execute("SELECT pg_backend_pid()").fetchone()[0])


def test_a_released_connection_is_reused(database: DatabaseTarget):
    pool = ConnectionPool(database, size=4)
    try:
        with pool.lease() as connection:
            first = _backend_pid(connection)
        assert pool.idle_count() == 1
        with pool.lease() as connection:
            assert _backend_pid(connection) == first
    finally:
        pool.close()
    assert pool.idle_count() == 0


def test_a_lease_that_raises_discards_its_connection(database: DatabaseTarget):
    """Whatever went wrong may have left session state the next caller would inherit."""
    pool = ConnectionPool(database, size=4)
    try:
        with pytest.raises(RuntimeError):
            with pool.lease() as connection:
                _backend_pid(connection)
                raise RuntimeError("boom")
        assert pool.idle_count() == 0
    finally:
        pool.close()


def test_a_terminated_connection_is_discarded_rather_than_reissued(database: DatabaseTarget):
    pool = ConnectionPool(database, size=4)
    killer = connect_postgres(database, initialize=False)
    try:
        with pool.lease() as connection:
            pid = _backend_pid(connection)
        assert pool.idle_count() == 1

        killer.execute("SELECT pg_terminate_backend(%s)", [pid])

        # The pool notices the corpse, drops it, and the next lease still works.
        with pool.lease() as connection:
            assert _backend_pid(connection) != pid
            assert connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        killer.close()
        pool.close()


def test_the_pool_does_not_grow_past_its_size(database: DatabaseTarget):
    pool = ConnectionPool(database, size=2)
    try:
        connections = [pool.acquire() for _ in range(4)]
        for connection in connections:
            pool.release(connection)
        assert pool.idle_count() == 2
        # The two that did not fit were closed, not leaked.
        assert sum(1 for connection in connections if connection.closed) == 2
    finally:
        pool.close()


def test_concurrent_requests_share_a_bounded_set_of_backends(database: DatabaseTarget):
    pool = ConnectionPool(database, size=3)
    seen: set[int] = set()
    lock = threading.Lock()
    start = threading.Barrier(4)

    def worker() -> None:
        start.wait()
        for _ in range(6):
            with pool.lease() as connection:
                with lock:
                    seen.add(_backend_pid(connection))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        pool.close()
    # Four threads may hold four connections at once, but repeated leases must
    # reuse rather than open a fresh backend every time.
    assert len(seen) <= 8


def test_pool_size_must_be_positive(database: DatabaseTarget):
    with pytest.raises(ValueError):
        ConnectionPool(database, size=0)


def test_repository_reports_a_dead_connection_as_unavailable(database: DatabaseTarget):
    """The handler's 503 branch depends on this, and pooling created a new way in."""
    repository = DashboardRepository(database, pool_size=2)
    killer = connect_postgres(database, initialize=False)
    try:
        with repository.connect() as connection:
            pid = _backend_pid(connection)
        killer.execute("SELECT pg_terminate_backend(%s)", [pid])

        # Kill it again while leased: the health check cannot catch this one, so
        # the failure has to be translated at query time.
        with pytest.raises(DashboardUnavailable):
            with repository.connect() as connection:
                killer.execute("SELECT pg_terminate_backend(%s)", [_backend_pid(connection)])
                connection.execute("SELECT 1")
    finally:
        killer.close()
        repository.close()


def test_repository_lets_ordinary_query_errors_through(database: DatabaseTarget):
    """A typo in SQL is not the database being unavailable."""
    repository = DashboardRepository(database, pool_size=2)
    try:
        with pytest.raises(Exception) as caught:
            with repository.connect() as connection:
                connection.execute("SELECT * FROM no_such_table")
        assert not isinstance(caught.value, DashboardUnavailable)
    finally:
        repository.close()


def test_a_stream_connection_is_not_taken_from_the_pool(database: DatabaseTarget):
    repository = DashboardRepository(database, pool_size=2)
    try:
        with repository.connect_dedicated() as connection:
            assert connection.execute("SELECT 1").fetchone()[0] == 1
            assert repository.pool.idle_count() == 0
        assert repository.pool.idle_count() == 0
    finally:
        repository.close()
