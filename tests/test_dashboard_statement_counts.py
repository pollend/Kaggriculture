"""Ceilings on how many statements each page issues.

This is the guard that keeps an N+1 from creeping back in. The numbers are
deliberately exact rather than generous: a page that starts issuing one more query
should fail here and be looked at, even if the extra query is cheap, because the
cost of these pages is dominated by how many round trips they make rather than by
any one of them.

Raising a number is fine when the page genuinely gained a section. Raising it
because "the test broke" is the thing this file exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaggriculture_lab.dashboard import routes
from kaggriculture_lab.database import DatabaseTarget, connect_postgres
from kaggriculture_lab.storage import ExperimentStore
from tests.postgres_support import postgres_database, seed_reference_rows

# One page render, one statement each unless the page needs more; these are the
# counts as of the split, with a short note on where each budget goes.
EXPECTED = {
    "/": 3,  # sessions, rollups, leaders
    "/sessions/s": 6,  # session, generations, trend, evaluations, models, head-to-head
    "/generations/g": 14,  # header, work, leaderboard, pairings + episodes, filters,
    #                        hooks, and the six performance panels
    "/episodes/m": 4,  # episode, seats, evaluations, replay probe
    "/models/model": 7,  # model, checkpoint, updates, checkpoints, evaluations,
    #                      batch sources, batches, hooks (batches share one call)
    "/sessions/s/failures": 5,  # session, hooks, updates, episodes, pairings
}


class CountingConnection:
    def __init__(self, connection):
        self._connection = connection
        self.statements: list[str] = []

    def execute(self, query, parameters=None):
        self.statements.append(str(query))
        return self._connection.execute(query, parameters)

    def __getattr__(self, name):
        return getattr(self._connection, name)


def _seed(database: DatabaseTarget) -> None:
    with ExperimentStore(database) as store:
        connection = store.connection
        connection.execute(
            """
            INSERT INTO training_sessions (session_id, league, name, status, seed, config_json)
            VALUES ('s', 'counts', 'Counts', 'RUNNING', 1, '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO generations (generation_id, session_id, generation_index, status, created_at)
            VALUES ('g', 's', 0, 'COMPLETE', now())
            """
        )
        seed_reference_rows(connection, run_id="s", family_id="family")
        connection.execute(
            "INSERT INTO strategies VALUES ('strategy-0', '{}', 'test', NULL), "
            "('strategy-1', '{}', 'test', NULL)"
        )
        connection.execute(
            """
            INSERT INTO agent_revisions (
                revision_id, family_id, strategy_id, entry_point, agent_spec,
                code_fingerprint, parameter_schema_version, params_json, source
            ) VALUES
                ('r0', 'family', 'strategy-0', 'agent:agent', 'a', 'fp', 1, '{}', 'test'),
                ('r1', 'family', 'strategy-1', 'agent:agent', 'b', 'fp', 1, '{}', 'test')
            """
        )
        connection.execute(
            "INSERT INTO generation_members VALUES ('g', 'r0', 'population', 1500, 200, 0.06), "
            "('g', 'r1', 'population', 1500, 200, 0.06)"
        )
        connection.execute(
            "INSERT INTO rating_snapshots VALUES ('g', 'r0', 1500, 100, 0.06, 1, false), "
            "('g', 'r1', 1400, 100, 0.06, 1, false)"
        )
        connection.execute(
            "INSERT INTO league_pairings VALUES ('p', 'g', 0, 'population', 1, 'r0', 'r1', 'COMPLETE')"
        )
        connection.execute(
            """
            INSERT INTO episodes (
                match_id, run_id, stage, seed, seat0_id, seat1_id, seat0_spec,
                seat1_spec, steps, wallclock, failed
            ) VALUES ('m', 's', 'league:g', 1, 'r0', 'r1', 'a', 'b', 720, 1.0, false)
            """
        )
        connection.execute("INSERT INTO pairing_episodes VALUES ('p', 0, 'm')")
        connection.execute("INSERT INTO pairing_results VALUES ('p', 200, 1, false, 1, '{}')")
        connection.execute(
            """
            INSERT INTO episode_players (
                match_id, run_id, stage, seed, seat, strategy_id, opponent_id,
                status, failed, reward, final_money, opponent_money,
                money_difference, win, productive_action_percentage,
                movement_percentage, idle_percentage, final_shed_items,
                final_carried_items, metrics_json
            ) VALUES
                ('m', 's', 'league:g', 1, 0, 'r0', 'r1', 'DONE', false,
                 3200, 3200, 3000, 200, 1, 0.6, 0.3, 0.1, 0, 0, '{}'),
                ('m', 's', 'league:g', 1, 1, 'r1', 'r0', 'DONE', false,
                 3000, 3000, 3200, -200, 0, 0.5, 0.3, 0.2, 0, 0, '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO training_models (
                model_id, family_id, trainer_path, min_new_batches, config_json,
                last_attempted_batch_seq
            ) VALUES ('model', 'family', 'a.train:trainer', 2, '{}', 0)
            """
        )
        connection.execute(
            """
            INSERT INTO training_batches (
                model_id, batch_seq, batch_id, source_kind, source_id,
                sample_count, metadata_json
            ) VALUES ('model', 1, 'b1', 'league_generation', 'g', 2, '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO model_round_hooks (
                hook_id, model_id, session_id, generation_id, scope,
                round_index, status, batches_recorded
            ) VALUES ('h', 'model', 's', 'g', 'round', 0, 'COMPLETE', 1)
            """
        )


@pytest.fixture
def database(tmp_path: Path) -> DatabaseTarget:
    target = postgres_database(tmp_path, "counts")
    _seed(target)
    return target


@pytest.fixture
def connection(database: DatabaseTarget):
    live = connect_postgres(database, initialize=False)
    try:
        yield CountingConnection(live)
    finally:
        live.close()


@pytest.mark.parametrize("path,expected", sorted(EXPECTED.items()))
def test_a_page_stays_within_its_statement_budget(connection, path, expected):
    document = routes.resolve(path, {})(connection)
    assert "<main>" in document
    assert len(connection.statements) == expected, (
        f"{path} issued {len(connection.statements)} statements, expected {expected}:\n"
        + "\n".join(
            f"  {index}. {' '.join(statement.split())[:110]}"
            for index, statement in enumerate(connection.statements, 1)
        )
    )


def test_a_larger_page_does_not_issue_more_statements(connection):
    """The count is per page, not per row: this is the N+1 guard."""
    baseline = len(connection.statements)
    routes.resolve("/generations/g", {})(connection)
    first = len(connection.statements) - baseline

    with connect_postgres(connection.target, initialize=False) as writer:
        for index in range(1, 12):
            writer.execute(
                "INSERT INTO league_pairings VALUES (%s, 'g', %s, 'population', %s, 'r0', 'r1', 'COMPLETE')",
                [f"p{index}", index, index],
            )

    before = len(connection.statements)
    routes.resolve("/generations/g", {})(connection)
    assert len(connection.statements) - before == first
