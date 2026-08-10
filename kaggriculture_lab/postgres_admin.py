"""Bootstrap and migrate the local PostgreSQL lab database."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

from kaggriculture_lab.database import DatabaseTarget, connect_postgres

DEFAULT_ADMIN_URL = "postgresql://kag_admin:kag_admin_local@127.0.0.1:54329/kaggriculture"
DEFAULT_WRITER_URL = "postgresql://kag_writer:kag_writer_local@127.0.0.1:54329/kaggriculture"
DEFAULT_READER_URL = "postgresql://kag_reader:kag_reader_local@127.0.0.1:54329/kaggriculture"
BACKUP_DIRECTORY = Path("artifacts/postgres-backups")
MIGRATION_LOCK = "kaggriculture:single-schema-migration"


def _url(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


def _schema_sql() -> str:
    return Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")


def _schema_tables() -> tuple[str, ...]:
    return tuple(
        re.findall(r"CREATE TABLE IF NOT EXISTS\s+([a-z][a-z0-9_]*)", _schema_sql())
    )


def _ensure_roles(connection: object) -> None:
    from psycopg import sql

    roles = (("kag_writer", "kag_writer_local"), ("kag_reader", "kag_reader_local"))
    for role, password in roles:
        exists = connection.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", [role]
        ).fetchone()
        command = "ALTER ROLE {} LOGIN PASSWORD {}" if exists else "CREATE ROLE {} LOGIN PASSWORD {}"
        connection.execute(
            sql.SQL(command).format(sql.Identifier(role), sql.Literal(password))
        )
    connection.execute("GRANT CONNECT ON DATABASE kaggriculture TO kag_writer, kag_reader")
    connection.execute("GRANT USAGE, CREATE ON SCHEMA public TO kag_writer")
    connection.execute("GRANT USAGE ON SCHEMA public TO kag_reader")


def _grant_application_tables(connection: object) -> None:
    from psycopg import sql

    for table in _schema_tables():
        connection.execute(
            sql.SQL("ALTER TABLE public.{} OWNER TO kag_writer").format(sql.Identifier(table))
        )
    connection.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO kag_reader")
    connection.execute(
        "ALTER DEFAULT PRIVILEGES FOR ROLE kag_writer IN SCHEMA public "
        "GRANT SELECT ON TABLES TO kag_reader"
    )


def _legacy_schemas(connection: object) -> list[str]:
    registry_exists = connection.execute(
        "SELECT to_regclass('public.kaggriculture_catalogs')"
    ).fetchone()[0]
    if registry_exists:
        return [
            str(row[0])
            for row in connection.execute(
                """
                SELECT catalog_name FROM public.kaggriculture_catalogs
                WHERE to_regclass(quote_ident(catalog_name) || '.training_sessions') IS NOT NULL
                ORDER BY catalog_name
                """
            ).fetchall()
        ]
    return [
        str(row[0])
        for row in connection.execute(
            """
            SELECT table_schema FROM information_schema.tables
            WHERE table_name = 'training_sessions'
              AND table_schema NOT IN ('public', 'pg_catalog', 'information_schema')
            ORDER BY table_schema
            """
        ).fetchall()
    ]


def _backup_database(admin_url: str, directory: Path = BACKUP_DIRECTORY) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = directory / f"before-single-schema-{stamp}.sql"
    pg_dump = shutil.which("pg_dump")
    if pg_dump:
        parsed = urlsplit(admin_url)
        environment = dict(os.environ)
        if parsed.password:
            environment["PGPASSWORD"] = unquote(parsed.password)
        command = [
            pg_dump,
            "--host", parsed.hostname or "127.0.0.1",
            "--port", str(parsed.port or 5432),
            "--username", unquote(parsed.username or "kag_admin"),
            "--dbname", parsed.path.lstrip("/") or "kaggriculture",
            "--no-password",
        ]
    elif shutil.which("docker"):
        environment = None
        parsed = urlsplit(admin_url)
        command = [
            "docker", "exec", "kaggriculture-postgres", "pg_dump",
            "--username", unquote(parsed.username or "kag_admin"),
            "--dbname", parsed.path.lstrip("/") or "kaggriculture",
        ]
    else:
        raise RuntimeError("migration requires pg_dump or the lab Docker container")
    with destination.open("wb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.PIPE, env=environment)
    if result.returncode:
        destination.unlink(missing_ok=True)
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"PostgreSQL backup failed: {message}")
    destination.chmod(0o600)
    return destination


def _unique_keys(connection: object, schema: str, table: str) -> list[tuple[str, ...]]:
    rows = connection.execute(
        """
        SELECT array_agg(a.attname ORDER BY key_position)
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY keys(attnum, key_position)
        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = keys.attnum
        WHERE n.nspname = %s AND c.relname = %s AND i.indisunique
        GROUP BY i.indexrelid
        """,
        [schema, table],
    ).fetchall()
    return [tuple(str(column) for column in row[0]) for row in rows]


def _preflight_table(connection: object, source: str, table: str) -> None:
    from psycopg import sql

    session_table = table == "training_sessions"
    target_json = sql.SQL("to_jsonb(target_row)")
    for key in _unique_keys(connection, "public", table):
        join = sql.SQL(" AND ").join(
            sql.SQL("target_row.{} IS NOT DISTINCT FROM source_row.{}").format(
                sql.Identifier(column), sql.Identifier(column)
            )
            for column in key
        )
        difference = (
            sql.SQL("true")
            if session_table
            else sql.SQL("{} <> to_jsonb(source_row)").format(target_json)
        )
        conflict = connection.execute(
            sql.SQL(
                "SELECT 1 FROM public.{} target_row JOIN {}.{} source_row ON {} "
                "WHERE {} LIMIT 1"
            ).format(
                sql.Identifier(table),
                sql.Identifier(source),
                sql.Identifier(table),
                join,
                difference,
            )
        ).fetchone()
        if conflict:
            columns = ", ".join(key)
            raise RuntimeError(
                f"conflicting {table} row while merging {source}; unique key: {columns}"
            )


def _copy_table(connection: object, source: str, table: str) -> int:
    from psycopg import sql

    source_columns = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position
            """,
            [source, table],
        ).fetchall()
    ]
    if not source_columns:
        return 0
    target_columns = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position
            """,
            [table],
        ).fetchall()
    ]
    select_values: list[object] = []
    for column in target_columns:
        if table == "training_sessions" and column == "league":
            select_values.append(sql.Literal(source))
        elif table == "training_sessions" and column == "config_json":
            select_values.append(
                sql.SQL("{}::jsonb || jsonb_build_object('league', {})").format(
                    sql.Identifier(column), sql.Literal(source)
                )
            )
        elif column in source_columns:
            select_values.append(sql.Identifier(column))
        else:
            raise RuntimeError(f"legacy {source}.{table} is missing required column {column}")
    cursor = connection.execute(
        sql.SQL("INSERT INTO public.{} ({}) SELECT {} FROM {}.{} ON CONFLICT DO NOTHING").format(
            sql.Identifier(table),
            sql.SQL(", ").join(map(sql.Identifier, target_columns)),
            sql.SQL(", ").join(select_values),
            sql.Identifier(source),
            sql.Identifier(table),
        )
    )
    return max(0, cursor.rowcount)


REFERENCE_CHECKS = (
    ("generations", "session_id", "training_sessions", "session_id"),
    ("generation_members", "generation_id", "generations", "generation_id"),
    ("league_pairings", "generation_id", "generations", "generation_id"),
    ("pairing_episodes", "pairing_id", "league_pairings", "pairing_id"),
    ("pairing_episodes", "match_id", "episodes", "match_id"),
    ("model_round_hooks", "session_id", "training_sessions", "session_id"),
    ("model_round_hooks", "generation_id", "generations", "generation_id"),
    ("model_round_hooks", "model_id", "training_models", "model_id"),
    ("training_batches", "model_id", "training_models", "model_id"),
    ("model_updates", "model_id", "training_models", "model_id"),
)


def _validate_references(connection: object) -> None:
    from psycopg import sql

    for child, child_key, parent, parent_key in REFERENCE_CHECKS:
        missing = connection.execute(
            sql.SQL(
                "SELECT 1 FROM public.{} child LEFT JOIN public.{} parent "
                "ON parent.{} = child.{} WHERE child.{} IS NOT NULL "
                "AND parent.{} IS NULL LIMIT 1"
            ).format(
                sql.Identifier(child), sql.Identifier(parent),
                sql.Identifier(parent_key), sql.Identifier(child_key),
                sql.Identifier(child_key), sql.Identifier(parent_key),
            )
        ).fetchone()
        if missing:
            raise RuntimeError(f"migration produced an orphan reference: {child}.{child_key}")


def migrate_catalogs(backup_directory: Path = BACKUP_DIRECTORY) -> Path | None:
    """Atomically merge registered catalog schemas into ``public``."""

    import psycopg
    from psycopg import sql

    admin_url = _url("KAGGRICULTURE_ADMIN_DATABASE_URL", DEFAULT_ADMIN_URL)
    with psycopg.connect(admin_url, autocommit=True) as connection:
        sources = _legacy_schemas(connection)
        if not sources:
            return None
        active_locks = int(
            connection.execute(
                "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND granted"
                " AND database = (SELECT oid FROM pg_database WHERE datname = current_database())"
            ).fetchone()[0]
        )
        if active_locks:
            raise RuntimeError("stop all league trainers before migrating; advisory locks are active")
        public_exists = connection.execute(
            "SELECT to_regclass('public.training_sessions')"
        ).fetchone()[0]
        if public_exists and connection.execute(
            "SELECT 1 FROM public.training_sessions LIMIT 1"
        ).fetchone():
            raise RuntimeError("public already contains league data; refusing to merge twice")

    backup = _backup_database(admin_url, backup_directory)
    with psycopg.connect(admin_url, autocommit=True) as connection:
        connection.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", [MIGRATION_LOCK])
        try:
            with connection.transaction():
                _ensure_roles(connection)
                connection.execute("SET LOCAL search_path TO public")
                connection.execute(_schema_sql())
                for source in sources:
                    for table in _schema_tables():
                        if connection.execute(
                            "SELECT to_regclass(%s)", [f"{source}.{table}"]
                        ).fetchone()[0] is not None:
                            connection.execute(
                                sql.SQL("LOCK TABLE {}.{} IN ACCESS EXCLUSIVE MODE").format(
                                    sql.Identifier(source), sql.Identifier(table)
                                )
                            )
                inserted: dict[str, int] = {table: 0 for table in _schema_tables()}
                for source in sources:
                    for table in _schema_tables():
                        if connection.execute(
                            "SELECT to_regclass(%s)", [f"{source}.{table}"]
                        ).fetchone()[0] is None:
                            continue
                        _preflight_table(connection, source, table)
                        inserted[table] += _copy_table(connection, source, table)
                for table, expected in inserted.items():
                    actual = int(
                        connection.execute(
                            sql.SQL("SELECT count(*) FROM public.{}").format(sql.Identifier(table))
                        ).fetchone()[0]
                    )
                    if actual != expected:
                        raise RuntimeError(
                            f"row-count validation failed for {table}: expected {expected}, found {actual}"
                        )
                _validate_references(connection)
                for source in sources:
                    connection.execute(
                        sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(source))
                    )
                connection.execute("DROP TABLE IF EXISTS public.kaggriculture_catalogs")
                _grant_application_tables(connection)
        finally:
            connection.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", [MIGRATION_LOCK])
    return backup


def bootstrap() -> None:
    import psycopg

    admin_url = _url("KAGGRICULTURE_ADMIN_DATABASE_URL", DEFAULT_ADMIN_URL)
    writer_url = _url("KAGGRICULTURE_DATABASE_URL", DEFAULT_WRITER_URL)
    with psycopg.connect(admin_url, autocommit=True) as connection:
        if _legacy_schemas(connection):
            raise RuntimeError(
                "legacy catalog schemas contain data; run `python -m "
                "kaggriculture_lab.postgres_admin migrate-catalogs` first"
            )
        _ensure_roles(connection)
    with connect_postgres(DatabaseTarget(writer_url)):
        pass
    with psycopg.connect(admin_url, autocommit=True) as connection:
        _grant_application_tables(connection)


def write_env(path: Path = Path(".kaggriculture/postgres.env")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"KAGGRICULTURE_ADMIN_DATABASE_URL={DEFAULT_ADMIN_URL}\n"
        f"KAGGRICULTURE_DATABASE_URL={DEFAULT_WRITER_URL}\n"
        f"KAGGRICULTURE_READER_DATABASE_URL={DEFAULT_READER_URL}\n"
    )
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bootstrap", "write-env", "migrate-catalogs"))
    parser.add_argument("--env-file", type=Path, default=Path(".kaggriculture/postgres.env"))
    parser.add_argument("--backup-directory", type=Path, default=BACKUP_DIRECTORY)
    args = parser.parse_args(argv)
    if args.command == "write-env":
        print(write_env(args.env_file))
    elif args.command == "migrate-catalogs":
        backup = migrate_catalogs(args.backup_directory)
        print(f"Merged legacy catalogs; backup: {backup}" if backup else "No legacy catalogs found")
    else:
        bootstrap()
        print("PostgreSQL public schema initialized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
