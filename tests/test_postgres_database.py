from __future__ import annotations

import threading
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

from kaggriculture_lab.database import DatabaseTarget, connect_postgres, database_target
from kaggriculture_lab.storage import ExperimentStore
from tests.postgres_support import postgres_database


def _reader_target(database: DatabaseTarget) -> DatabaseTarget:
    reader = database_target(role="reader")
    source = urlsplit(reader.url)
    destination = urlsplit(database.url)
    return DatabaseTarget(
        urlunsplit((source.scheme, source.netloc, destination.path, source.query, source.fragment))
    )


def test_store_rejects_file_database_paths():
    with pytest.raises(ValueError, match="postgresql"):
        ExperimentStore("artifacts/experiments.duckdb")


def test_schema_uses_json_and_bytea(tmp_path: Path):
    target = postgres_database(tmp_path, "schema-types")
    with connect_postgres(target, initialize=False) as connection:
        columns = dict(
            connection.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND ((table_name = 'runs' AND column_name = 'config_json')
                    OR (table_name = 'episode_payloads' AND column_name = 'replay_gzip'))
                """,
            ).fetchall()
        )
    assert columns == {"config_json": "json", "replay_gzip": "bytea"}


def test_reader_role_cannot_mutate_database(tmp_path: Path):
    target = _reader_target(postgres_database(tmp_path, "reader-role"))
    with connect_postgres(target, initialize=False) as connection:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            connection.execute("DELETE FROM runs")


def test_two_session_writers_progress_concurrently(tmp_path: Path):
    target = postgres_database(tmp_path, "concurrent-writers")
    writers = [(target, "alpha"), (target, "beta")]
    barrier = threading.Barrier(2)
    completed: list[str] = []
    errors: list[BaseException] = []

    def write(target: DatabaseTarget, name: str) -> None:
        try:
            with connect_postgres(target, initialize=False) as connection:
                connection.execute("BEGIN")
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, name, status, seed, config_json, versions_json
                    ) VALUES (%s, %s, 'RUNNING', 1, %s, %s)
                    """,
                    [f"concurrency-smoke-{name}", name, "{}", "{}"],
                )
                barrier.wait(timeout=5)
                assert connection.execute(
                    "SELECT name FROM runs WHERE run_id = %s", [f"concurrency-smoke-{name}"]
                ).fetchone() == (name,)
                completed.append(name)
                connection.execute("ROLLBACK")
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=write, args=item) for item in writers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert sorted(completed) == ["alpha", "beta"]
