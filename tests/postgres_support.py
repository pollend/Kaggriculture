"""Isolated PostgreSQL databases for tests that persist lab state."""

from __future__ import annotations

import atexit
import hashlib
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql

from kaggriculture_lab.database import DatabaseTarget, connect_postgres, database_target
from kaggriculture_lab.postgres_admin import DEFAULT_ADMIN_URL

_databases: set[str] = set()


def _database_url(url: str, name: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{name}", parsed.query, parsed.fragment))


def postgres_database(tmp_path: Path, label: str) -> DatabaseTarget:
    """Create one stable, empty database for a pytest temporary directory."""

    digest = hashlib.sha256(f"{tmp_path}:{label}".encode()).hexdigest()[:20]
    database_name = f"test_{digest}"
    base = database_target()
    target = DatabaseTarget(_database_url(base.url, database_name))
    if database_name in _databases:
        return target
    admin_url = os.environ.get("KAGGRICULTURE_ADMIN_DATABASE_URL", DEFAULT_ADMIN_URL)
    try:
        with psycopg.connect(admin_url, autocommit=True) as connection:
            connection.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                [database_name],
            )
            connection.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(database_name))
            )
            connection.execute(
                sql.SQL("CREATE DATABASE {} OWNER kag_writer").format(
                    sql.Identifier(database_name)
                )
            )
        with connect_postgres(target):
            pass
    except Exception as error:
        raise RuntimeError("PostgreSQL tests require ./lab-stack.sh up") from error
    _databases.add(database_name)
    return target


def seed_reference_rows(
    connection: object,
    *,
    run_id: str | None = None,
    session_id: str | None = None,
    generation_id: str | None = None,
    family_id: str | None = None,
    revision_ids: tuple[str, ...] = (),
    match_ids: tuple[str, ...] = (),
) -> None:
    """Insert the parent rows a fixture's children reference.

    Tests hand-build narrow slices of league state -- a pairing without its
    generation, a match without its run. The schema's foreign keys reject that,
    so fixtures declare their parents here rather than each restating the same
    inserts."""

    if run_id is not None:
        connection.execute(
            """
            INSERT INTO runs VALUES (%s, %s, current_timestamp, 'RUNNING', 0, '{}', '{}', NULL)
            ON CONFLICT (run_id) DO NOTHING
            """,
            [run_id, run_id],
        )
    if session_id is not None:
        connection.execute(
            """
            INSERT INTO training_sessions (
                session_id, league, name, status, seed, config_json
            ) VALUES (%s, 'test', %s, 'ACTIVE', 0, '{}')
            ON CONFLICT (session_id) DO NOTHING
            """,
            [session_id, session_id],
        )
    if generation_id is not None:
        if session_id is None:
            raise ValueError("generation_id requires session_id")
        connection.execute(
            """
            INSERT INTO generations (generation_id, session_id, generation_index, status)
            VALUES (%s, %s, 0, 'OPEN')
            ON CONFLICT (generation_id) DO NOTHING
            """,
            [generation_id, session_id],
        )
    if family_id is not None:
        connection.execute(
            """
            INSERT INTO agent_families VALUES (
                %s, %s, 'agent.toml', 'agent:agent', NULL, NULL, 1, current_timestamp
            )
            ON CONFLICT (family_id) DO NOTHING
            """,
            [family_id, family_id],
        )
    for revision_id in revision_ids:
        if family_id is None:
            raise ValueError("revision_ids requires family_id")
        connection.execute(
            """
            INSERT INTO agent_revisions (
                revision_id, family_id, strategy_id, entry_point, agent_spec,
                code_fingerprint, parameter_schema_version, params_json, source
            ) VALUES (%s, %s, NULL, 'agent:agent', %s, 'fingerprint', 1, '{}', 'test')
            ON CONFLICT (revision_id) DO NOTHING
            """,
            [revision_id, family_id, revision_id],
        )
    for match_id in match_ids:
        if run_id is None:
            raise ValueError("match_ids requires run_id")
        connection.execute(
            """
            INSERT INTO episodes
            VALUES (%s, %s, 'test', 0, 'seat0', 'seat1', 'seat0', 'seat1', 0, 0.0, false, NULL)
            ON CONFLICT (match_id) DO NOTHING
            """,
            [match_id, run_id],
        )


@atexit.register
def _drop_test_databases() -> None:
    if not _databases:
        return
    admin_url = os.environ.get("KAGGRICULTURE_ADMIN_DATABASE_URL", DEFAULT_ADMIN_URL)
    try:
        with psycopg.connect(admin_url, autocommit=True) as connection:
            for database_name in _databases:
                connection.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = %s AND pid <> pg_backend_pid()",
                    [database_name],
                )
                connection.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {}").format(
                        sql.Identifier(database_name)
                    )
                )
    except Exception:
        pass
