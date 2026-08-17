"""Connection handling for the dashboard's reads."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from kaggriculture_lab.dashboard.errors import DashboardUnavailable
from kaggriculture_lab.database import ConnectionPool, DatabaseTarget, connect_postgres

DEFAULT_POOL_SIZE = 8


class DashboardRepository:
    """Narrow access to the shared league database.

    Requests lease a pooled connection; event streams take a dedicated one for
    their lifetime. Streams are deliberately kept out of the pool: a tab left open
    overnight holds its connection for the whole time, and if that came from the
    pool a handful of forgotten tabs would starve ordinary page loads.
    """

    def __init__(self, database: str | DatabaseTarget, *, pool_size: int = DEFAULT_POOL_SIZE):
        self.target = (
            database if isinstance(database, DatabaseTarget) else DatabaseTarget(database)
        )
        self.pool = ConnectionPool(self.target, size=pool_size)

    @contextmanager
    def connect(self) -> Iterator[Any]:
        """Lease a pooled connection for the duration of one request.

        Both the connect and the queries run inside the `DashboardUnavailable`
        translation. Before pooling, only a failed connect could raise
        `OperationalError` and the handler's error ladder relied on that; a pooled
        connection can also die between lease and use, and that has to land on the
        same 503 rather than a 500.
        """
        try:
            connection = self.pool.acquire()
        except Exception as error:
            raise DashboardUnavailable("The database is temporarily unavailable.") from error
        try:
            yield connection
        except Exception as error:
            self.pool.release(connection, discard=True)
            if _is_connection_failure(error):
                raise DashboardUnavailable("The database is temporarily unavailable.") from error
            raise
        else:
            self.pool.release(connection)

    @contextmanager
    def connect_dedicated(self) -> Iterator[Any]:
        """Open a connection outside the pool, for a long-lived reader."""
        try:
            connection = connect_postgres(self.target, initialize=False)
        except Exception as error:
            raise DashboardUnavailable("The database is temporarily unavailable.") from error
        try:
            yield connection
        finally:
            connection.close()

    def validate(self) -> bool:
        """Validate that PostgreSQL's public schema contains the lab tables."""
        try:
            with self.connect() as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = %s",
                        ["public"],
                    ).fetchall()
                }
        except Exception as error:
            raise ValueError("not a Kaggriculture PostgreSQL database") from error
        if not {"runs", "episodes", "training_sessions"} <= tables:
            raise ValueError("not a Kaggriculture PostgreSQL database")
        return True

    def close(self) -> None:
        self.pool.close()


def _is_connection_failure(error: BaseException) -> bool:
    """Whether this is the database going away rather than a bad query.

    Matched on psycopg's class rather than on message text, and imported lazily so
    the module keeps working when psycopg is not installed.
    """
    try:
        import psycopg
    except ImportError:  # pragma: no cover - psycopg is a hard runtime dependency
        return False
    return isinstance(error, (psycopg.OperationalError, psycopg.InterfaceError))
