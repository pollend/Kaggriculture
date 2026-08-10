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


def database_target(
    *,
    url: str | None = None,
    role: str = "writer",
) -> DatabaseTarget:
    """Resolve a project-local PostgreSQL target without logging credentials."""

    values = _env_file_values()
    if url is None:
        env_key = (
            "KAGGRICULTURE_READER_DATABASE_URL"
            if role == "reader"
            else "KAGGRICULTURE_DATABASE_URL"
        )
        url = os.environ.get(env_key) or values.get(env_key)
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
