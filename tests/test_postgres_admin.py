from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql

from kaggriculture_lab.database import DatabaseTarget, connect_postgres, database_target
from kaggriculture_lab.postgres_admin import migrate_catalogs
from tests.postgres_support import postgres_database


def _database_url(url: str, database: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{database}", parsed.query, parsed.fragment))


def _seed_legacy_schema(admin_url: str, schema: str, session_id: str, name: str) -> None:
    schema_text = Path("kaggriculture_lab/schema.sql").read_text(encoding="utf-8")
    legacy_schema = schema_text.replace("    league VARCHAR NOT NULL,\n", "")
    with psycopg.connect(admin_url, autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE SCHEMA {} AUTHORIZATION kag_writer").format(sql.Identifier(schema))
        )
        connection.execute(
            sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema))
        )
        connection.execute(legacy_schema)
        connection.execute(
            sql.SQL(
                "INSERT INTO {}.training_sessions "
                "(session_id, name, status, seed, config_json) "
                "VALUES (%s, %s, 'RUNNING', 17, %s)"
            ).format(sql.Identifier(schema)),
            [session_id, name, "{}"],
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS public.kaggriculture_catalogs (
                catalog_name VARCHAR PRIMARY KEY
            )
            """
        )
        connection.execute(
            "INSERT INTO public.kaggriculture_catalogs VALUES (%s)", [schema]
        )


def _migration_target(tmp_path: Path, label: str) -> tuple[DatabaseTarget, str]:
    target = postgres_database(tmp_path, label)
    database = urlsplit(target.url).path.lstrip("/")
    admin = database_target(role="writer").url.replace(
        "kag_writer:kag_writer_local", "kag_admin:kag_admin_local"
    )
    return target, _database_url(admin, database)


def test_migration_merges_sessions_and_removes_legacy_schemas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target, admin_url = _migration_target(tmp_path, "migration-success")
    _seed_legacy_schema(admin_url, "fixed_dqn", "fixed-session", "Fixed training")
    _seed_legacy_schema(admin_url, "planner", "planner-session", "Planner training")
    monkeypatch.setenv("KAGGRICULTURE_ADMIN_DATABASE_URL", admin_url)

    backup = migrate_catalogs(tmp_path / "backups")

    assert backup is not None and backup.is_file()
    with connect_postgres(target, initialize=False) as connection:
        rows = connection.execute(
            "SELECT session_id, league, config_json ->> 'league' "
            "FROM training_sessions ORDER BY session_id"
        ).fetchall()
        assert rows == [
            ("fixed-session", "fixed_dqn", "fixed_dqn"),
            ("planner-session", "planner", "planner"),
        ]
        assert connection.execute(
            "SELECT to_regnamespace('fixed_dqn'), to_regnamespace('planner'), "
            "to_regclass('public.kaggriculture_catalogs')"
        ).fetchone() == (None, None, None)


def test_migration_rejects_global_session_id_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target, admin_url = _migration_target(tmp_path, "migration-collision")
    _seed_legacy_schema(admin_url, "fixed_dqn", "duplicate", "Fixed training")
    _seed_legacy_schema(admin_url, "planner", "duplicate", "Different training")
    monkeypatch.setenv("KAGGRICULTURE_ADMIN_DATABASE_URL", admin_url)

    with pytest.raises(RuntimeError, match="conflicting training_sessions"):
        migrate_catalogs(tmp_path / "backups")

    with connect_postgres(target, initialize=False) as connection:
        assert connection.execute("SELECT count(*) FROM training_sessions").fetchone() == (0,)
        assert connection.execute(
            "SELECT to_regnamespace('fixed_dqn'), to_regnamespace('planner')"
        ).fetchone() == ("fixed_dqn", "planner")
