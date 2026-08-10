"""Bootstrap the local PostgreSQL lab database."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from kaggriculture_lab.database import DatabaseTarget, connect_postgres

DEFAULT_ADMIN_URL = "postgresql://kag_admin:kag_admin_local@127.0.0.1:54329/kaggriculture"
DEFAULT_WRITER_URL = "postgresql://kag_writer:kag_writer_local@127.0.0.1:54329/kaggriculture"


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

    role, password = "kag_writer", "kag_writer_local"
    exists = connection.execute(
        "SELECT 1 FROM pg_roles WHERE rolname = %s", [role]
    ).fetchone()
    command = "ALTER ROLE {} LOGIN PASSWORD {}" if exists else "CREATE ROLE {} LOGIN PASSWORD {}"
    connection.execute(
        sql.SQL(command).format(sql.Identifier(role), sql.Literal(password))
    )
    connection.execute("GRANT CONNECT ON DATABASE kaggriculture TO kag_writer")
    connection.execute("GRANT USAGE, CREATE ON SCHEMA public TO kag_writer")


def _grant_application_tables(connection: object) -> None:
    from psycopg import sql

    for table in _schema_tables():
        connection.execute(
            sql.SQL("ALTER TABLE public.{} OWNER TO kag_writer").format(sql.Identifier(table))
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


def bootstrap() -> None:
    import psycopg

    admin_url = _url("KAGGRICULTURE_ADMIN_DATABASE_URL", DEFAULT_ADMIN_URL)
    writer_url = _url("KAGGRICULTURE_DATABASE_URL", DEFAULT_WRITER_URL)
    with psycopg.connect(admin_url, autocommit=True) as connection:
        legacy = _legacy_schemas(connection)
        if legacy:
            # The one-shot merge into `public` is gone. Refusing here keeps a
            # bootstrap from quietly stranding whatever those schemas still hold.
            raise RuntimeError(
                "legacy catalog schemas still contain league data: "
                f"{', '.join(legacy)}; export or drop them before bootstrapping"
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
    )
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bootstrap", "write-env"))
    parser.add_argument("--env-file", type=Path, default=Path(".kaggriculture/postgres.env"))
    args = parser.parse_args(argv)
    if args.command == "write-env":
        print(write_env(args.env_file))
    else:
        bootstrap()
        print("PostgreSQL public schema initialized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
