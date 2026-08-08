"""Parser-backed read-only SQL execution for extension hooks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import duckdb


def fetchall_select(
    connection: Any,
    sql: str,
    parameters: Sequence[Any] = (),
    *,
    reader_name: str,
) -> list[tuple[Any, ...]]:
    """Execute exactly one SELECT statement.

    Prefix checks are not a security boundary: ``SELECT 1; DELETE ...`` starts
    with SELECT. DuckDB's own parser supplies the authoritative statement count
    and type, including SELECT queries beginning with WITH.
    """

    statements = connection.extract_statements(sql)
    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
        raise ValueError(f"{reader_name} only permits SELECT/WITH queries")
    return connection.execute(sql, list(parameters)).fetchall()
