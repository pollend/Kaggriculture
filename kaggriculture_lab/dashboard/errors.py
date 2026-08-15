"""Errors the dashboard translates into HTTP status codes."""

from __future__ import annotations


class DashboardUnavailable(RuntimeError):
    """The database cannot currently be opened."""


class DashboardNotFound(LookupError):
    """A requested dashboard resource does not exist."""


class DashboardBadRequest(ValueError):
    """A dashboard route or query parameter is invalid."""
