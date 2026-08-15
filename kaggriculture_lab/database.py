"""PostgreSQL targets and connections for the shared league database."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ENV_FILE = Path(".kaggriculture/postgres.env")


def _env_file_values(path: Path = DEFAULT_ENV_FILE) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True, slots=True)
class DatabaseTarget:
    url: str

    def __post_init__(self) -> None:
        if not self.url.startswith(("postgresql://", "postgres://")):
            raise ValueError("live database URL must use postgresql://")

    def redacted(self) -> str:
        from urllib.parse import urlsplit, urlunsplit

        parsed = urlsplit(self.url)
        host = parsed.hostname or ""
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        user = parsed.username
        netloc = f"{user}:***@{host}" if user else host
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def database_target(*, url: str | None = None) -> DatabaseTarget:
    """Resolve a project-local PostgreSQL target without logging credentials."""

    if url is None:
        values = _env_file_values()
        url = os.environ.get("KAGGRICULTURE_DATABASE_URL") or values.get(
            "KAGGRICULTURE_DATABASE_URL"
        )
    if not url:
        raise RuntimeError(
            "PostgreSQL is not configured; run ./lab-stack.sh up or set "
            "KAGGRICULTURE_DATABASE_URL"
        )
    return DatabaseTarget(url)


class PostgresConnection:
    """Connection facade matching the subset used throughout the lab."""

    def __init__(self, target: DatabaseTarget, *, initialize: bool = True):
        import psycopg
        from psycopg.types.json import set_json_loads

        self.target = target
        self._connection = psycopg.connect(target.url, autocommit=True)
        set_json_loads(
            lambda value: value.decode("utf-8")
            if isinstance(value, (bytes, bytearray, memoryview))
            else str(value),
            context=self._connection,
        )
        self._connection.execute("SET search_path TO public")
        if initialize:
            self._initialize_schema()

    def _initialize_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        schema = schema_path.read_text(encoding="utf-8")
        self._connection.execute(schema)

    def execute(self, query: Any, parameters: Sequence[Any] | None = None):
        return self._connection.execute(query, parameters)

    def executemany(self, query: str, parameters: Iterable[Sequence[Any]]):
        with self._connection.cursor() as cursor:
            cursor.executemany(query, parameters)

    @contextmanager
    def advisory_lock(self, namespace: str, identifier: str):
        """Serialize one logical resource without blocking unrelated resources."""

        key = f"{namespace}:{identifier}"
        self.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", [key])
        try:
            yield
        finally:
            self.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", [key])

    @property
    def closed(self) -> bool:
        return bool(self._connection.closed)

    def alive(self) -> bool:
        """Whether this connection can still answer a query.

        A pooled connection can be terminated out from under the pool -- by a
        `pg_terminate_backend`, a server restart, or an idle timeout -- and the
        failure only surfaces on the next statement. The pool checks this before
        handing one out so the caller never sees somebody else's dead socket.
        """
        if self.closed:
            return False
        try:
            self._connection.execute("SELECT 1")
        except Exception:
            return False
        return True

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> PostgresConnection:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def connect_postgres(
    target: DatabaseTarget, *, initialize: bool = True
) -> PostgresConnection:
    return PostgresConnection(target, initialize=initialize)


class ConnectionPool:
    """A small LIFO pool of read connections.

    Connecting to PostgreSQL is a TCP handshake, authentication and a
    `SET search_path` -- tens of milliseconds -- and the dashboard's event streams
    turn that into a per-second cost per open browser tab. Reusing connections
    removes it.

    LIFO rather than round-robin on purpose: reusing the most recently returned
    connection keeps the working set small, so a pool that grew during a burst
    lets its extra connections go idle and be discarded rather than keeping every
    one of them warm.

    The rules that matter:

    * A connection returned after an exception is **discarded**, never reused.
      Whatever went wrong may have left the session in a state the next caller
      would inherit.
    * A connection is health-checked before being handed out, because it may have
      been killed while idle.
    * Over the size limit, a returned connection is closed rather than queued.

    Connections are autocommit, so a leased connection holds no transaction and a
    long-lived lease causes no bloat.
    """

    def __init__(self, target: DatabaseTarget, *, size: int = 8):
        if size < 1:
            raise ValueError("pool size must be at least 1")
        import threading

        self.target = target
        self.size = size
        self._idle: list[PostgresConnection] = []
        self._lock = threading.Lock()

    def acquire(self) -> PostgresConnection:
        while True:
            with self._lock:
                connection = self._idle.pop() if self._idle else None
            if connection is None:
                return connect_postgres(self.target, initialize=False)
            if connection.alive():
                return connection
            connection.close()

    def release(self, connection: PostgresConnection, *, discard: bool = False) -> None:
        if discard or connection.closed:
            connection.close()
            return
        with self._lock:
            if len(self._idle) < self.size:
                self._idle.append(connection)
                return
        connection.close()

    @contextmanager
    def lease(self):
        connection = self.acquire()
        try:
            yield connection
        except BaseException:
            self.release(connection, discard=True)
            raise
        else:
            self.release(connection)

    def close(self) -> None:
        with self._lock:
            idle, self._idle = self._idle, []
        for connection in idle:
            connection.close()

    def idle_count(self) -> int:
        with self._lock:
            return len(self._idle)
