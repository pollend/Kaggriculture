"""Bounded read-only PostgreSQL execution for extension hooks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_MUTATION = re.compile(
    r"\b(?:insert|update|delete|merge|copy|call|create|alter|drop|truncate|grant|revoke|vacuum|refresh)\b",
    re.IGNORECASE,
)


def _unquoted_sql(sql: str) -> str:
    """Blank strings/comments so statement and keyword checks cannot be spoofed."""

    return re.sub(
        r"(?s)(?:--[^\n]*(?:\n|$)|/\*.*?\*/|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")",
        " ",
        sql,
    )


def _validate_select(_connection: Any, sql: str, *, reader_name: str) -> None:
    visible = _unquoted_sql(sql).strip()
    if ";" in visible or not re.match(r"(?is)^(?:select|with)\b", visible) or _MUTATION.search(visible):
        raise ValueError(f"{reader_name} only permits SELECT/WITH queries")


def fetchall_select(
    connection: Any,
    sql: str,
    parameters: Sequence[Any] = (),
    *,
    reader_name: str,
) -> list[tuple[Any, ...]]:
    """Execute exactly one SELECT statement.

    The query is validated before execution, and dashboard connections use a
    PostgreSQL role that has SELECT privileges only.
    """

    _validate_select(connection, sql, reader_name=reader_name)
    return connection.execute(sql, list(parameters)).fetchall()


def fetchmany_select(
    connection: Any,
    sql: str,
    parameters: Sequence[Any] = (),
    *,
    limit: int,
    reader_name: str,
) -> tuple[list[str], list[tuple[Any, ...]], bool]:
    """Execute one SELECT and return at most ``limit`` rows plus truncation state."""

    if limit < 1:
        raise ValueError("limit must be positive")
    _validate_select(connection, sql, reader_name=reader_name)
    cursor = connection.execute(sql, list(parameters))
    columns = [str(item[0]) for item in cursor.description]
    rows = cursor.fetchmany(limit + 1)
    return columns, rows[:limit], len(rows) > limit
