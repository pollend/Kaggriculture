"""Lab-wide defaults shared by the constants-driven training runners."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from kaggriculture_lab.database import DatabaseTarget, database_target

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
# The block length a full league run schedules. Short-horizon runners that train a
# single model (train_dqn, train_rf_market) set their own instead.
DEFAULT_GENERATIONS = 48


@cache
def default_database() -> DatabaseTarget:
    """The project-local PostgreSQL target, resolved once per process.

    Resolution is deferred rather than evaluated at import so that importing this
    module on a machine without the lab stack running does not raise.
    """

    return database_target()
