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
        test_admin_url = _database_url(admin_url, database_name)
        with psycopg.connect(test_admin_url, autocommit=True) as connection:
            connection.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO kag_reader").format(
                    sql.Identifier(database_name)
                )
            )
            connection.execute("GRANT USAGE ON SCHEMA public TO kag_reader")
            connection.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO kag_reader")
    except Exception as error:
        raise RuntimeError("PostgreSQL tests require ./lab-stack.sh up") from error
    _databases.add(database_name)
    return target


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
