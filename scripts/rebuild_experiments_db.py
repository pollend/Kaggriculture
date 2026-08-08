"""Recreate the shared experiment catalog from kaggriculture_lab/schema.sql."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from kaggriculture_lab.storage import SCHEMA_VERSION, ExperimentStore


def rebuild(database: Path, *, force: bool = False) -> tuple[Path, Path | None]:
    database = database.resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists() and not force:
        raise FileExistsError(f"{database} exists; pass --force to back it up and replace it")

    temporary = database.with_name(f".{database.name}.rebuild-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary database already exists: {temporary}")
    backup: Path | None = None
    try:
        with ExperimentStore(temporary) as store:
            version = store.connection.execute(
                "SELECT max(version) FROM schema_migrations"
            ).fetchone()[0]
            required = {
                "training_models",
                "training_batches",
                "model_updates",
                "model_training_checkpoints",
                "model_round_hooks",
            }
            present = {
                str(row[0])
                for row in store.connection.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                ).fetchall()
            }
            if version != SCHEMA_VERSION or not required <= present:
                raise RuntimeError("new database failed schema validation")
        if database.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup = database.with_name(f"{database.name}.{stamp}.bak")
            os.replace(database, backup)
        os.replace(temporary, database)
        return database, backup
    except Exception:
        if temporary.exists():
            temporary.unlink()
        if backup is not None and backup.exists() and not database.exists():
            os.replace(backup, database)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("artifacts/experiments.duckdb"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    database, backup = rebuild(args.database, force=args.force)
    print(database)
    if backup:
        print(f"backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
