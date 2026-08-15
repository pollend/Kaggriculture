"""Semantics of the rollups shared by the dashboard and the MCP tools.

Each test here pins down one definition the dashboard and `progress_mcp` used to
disagree about. They are written against deliberately awkward rows -- a round only
half of whose models finished, an episode that crashed after writing one seat --
because those are exactly the rows the two copies answered differently for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaggriculture_lab import league_queries as lq
from kaggriculture_lab.database import DatabaseTarget, connect_postgres
from kaggriculture_lab.storage import ExperimentStore
from tests.postgres_support import postgres_database, seed_reference_rows


def _seed(database: DatabaseTarget) -> None:
    with ExperimentStore(database) as store:
        connection = store.connection
        connection.execute(
            """
            INSERT INTO training_sessions (session_id, league, name, status, seed, config_json)
            VALUES ('s', 'edge', 'Edge cases', 'RUNNING', 1, '{"rounds_per_generation": 2}')
            """
        )
        connection.execute(
            """
            INSERT INTO generations (generation_id, session_id, generation_index, status, created_at)
            VALUES ('g', 's', 0, 'RUNNING', now())
            """
        )
        seed_reference_rows(connection, run_id="s", family_id="family")
        connection.execute(
            "INSERT INTO strategies VALUES ('strategy-0', '{}', 'test', NULL), ('strategy-1', '{}', 'test', NULL)"
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
            """
            INSERT INTO rating_snapshots VALUES
                ('g', 'r0', 1500, 100, 0.06, 1, false),
                ('g', 'r1', 1400, 100, 0.06, 1, false)
            """
        )
        # One settled pairing and one FAILED pairing. `league_pairings.status` is
        # only ever written COMPLETE or FAILED, never ERROR.
        connection.execute(
            """
            INSERT INTO league_pairings VALUES
                ('good', 'g', 0, 'population', 1, 'r0', 'r1', 'COMPLETE'),
                ('bad', 'g', 1, 'population', 2, 'r0', 'r1', 'FAILED')
            """
        )
        # `ok` completed; `crashed` is a failed episode that still wrote a seat row
        # whose own `failed` flag is clear -- the case the two copies split on.
        connection.execute(
            """
            INSERT INTO episodes (
                match_id, run_id, stage, seed, seat0_id, seat1_id, seat0_spec,
                seat1_spec, steps, wallclock, failed
            ) VALUES
                ('ok', 's', 'league:g', 1, 'r0', 'r1', 'a', 'b', 720, 1.0, false),
                ('crashed', 's', 'league:g', 2, 'r0', 'r1', 'a', 'b', 10, 0.5, true)
            """
        )
        connection.execute(
            "INSERT INTO pairing_episodes VALUES ('good', 0, 'ok'), ('bad', 0, 'crashed')"
        )
        connection.execute("INSERT INTO pairing_results VALUES ('good', 200, 1, false, 1, '{}')")
        connection.execute(
            """
            INSERT INTO episode_players (
                match_id, run_id, stage, seed, seat, strategy_id, opponent_id,
                status, failed, reward, final_money, opponent_money,
                money_difference, win, productive_action_percentage,
                movement_percentage, idle_percentage, final_shed_items,
                final_carried_items, metrics_json
            ) VALUES
                ('ok', 's', 'league:g', 1, 0, 'r0', 'r1', 'DONE', false,
                 3200, 3200, 3000, 200, 1, 0.6, 0.3, 0.1, 0, 0, '{}'),
                ('ok', 's', 'league:g', 1, 1, 'r1', 'r0', 'DONE', false,
                 3000, 3000, 3200, -200, 0, 0.5, 0.3, 0.2, 0, 0, '{}'),
                ('crashed', 's', 'league:g', 2, 0, 'r0', 'r1', 'DONE', false,
                 99999, 99999, 0, 99999, 1, 0.9, 0.1, 0.0, 0, 0, '{}')
            """
        )
        # Round 0: two models, one errored -- so the round is NOT complete.
        # Round 1: both models complete.
        connection.execute(
            """
            INSERT INTO training_models (
                model_id, family_id, trainer_path, min_new_batches, config_json,
                last_attempted_batch_seq
            ) VALUES ('m0', 'family', 'a.train:trainer', 1, '{}', 0),
                     ('m1', 'family', 'b.train:trainer', 1, '{}', 0)
            """
        )
        connection.execute(
            """
            INSERT INTO model_round_hooks (
                hook_id, model_id, session_id, generation_id, scope,
                round_index, status, batches_recorded
            ) VALUES
                ('h0', 'm0', 's', 'g', 'round', 0, 'COMPLETE', 1),
                ('h1', 'm1', 's', 'g', 'round', 0, 'ERROR', 0),
                ('h2', 'm0', 's', 'g', 'round', 1, 'COMPLETE', 1),
                ('h3', 'm1', 's', 'g', 'round', 1, 'COMPLETE', 1)
            """
        )
        # An evaluation that raised but still carries the re-evaluation flag.
        connection.execute(
            """
            INSERT INTO episode_evaluations (
                evaluation_id, match_id, revision_id, seat, evaluator_path,
                evaluator_fingerprint, status, result_json, error_text
            ) VALUES
                ('e-ok', 'ok', 'r0', 0, 'agent:evaluate', 'fp', 'COMPLETE',
                 '{"should_reevaluate": true}', NULL),
                ('e-bad', 'ok', 'r1', 1, 'agent:evaluate', 'fp', 'ERROR',
                 '{"should_reevaluate": true}', 'boom')
            """
        )


@pytest.fixture
def edge_database(tmp_path: Path) -> DatabaseTarget:
    database = postgres_database(tmp_path, "edges")
    _seed(database)
    return database


@pytest.fixture
def connection(edge_database: DatabaseTarget):
    live = connect_postgres(edge_database, initialize=False)
    try:
        yield live
    finally:
        live.close()


def test_a_round_needs_every_model_to_finish(connection):
    """Round 0 has an errored hook, so only round 1 counts."""
    assert lq.completed_round_count(connection, generation_id="g") == 1
    assert lq.completed_round_count(connection, session_id="s") == 1
    assert int(lq.generation_work(connection, "g")["completed_rounds"]) == 1
    generations = lq.session_generations(connection, "s")
    assert int(generations[0]["completed_rounds"]) == 1
    assert int(generations[0]["hook_errors"]) == 1
    rollup = lq.session_rollups(connection, ["s"])[0]
    assert int(rollup["completed_rounds"]) == 1


def test_completed_round_count_rejects_ambiguous_scope(connection):
    with pytest.raises(ValueError):
        lq.completed_round_count(connection)
    with pytest.raises(ValueError):
        lq.completed_round_count(connection, session_id="s", generation_id="g")


def test_pairing_failure_is_failed_not_error(connection):
    work = lq.generation_work(connection, "g")
    assert int(work["pairings"]) == 2
    assert int(work["completed_pairings"]) == 1
    assert int(work["failed_pairings"]) == 1
    assert "pairing_errors" not in work
    rollup = lq.session_rollups(connection, ["s"])[0]
    assert int(rollup["failed_pairings"]) == 1


def test_a_crashed_episode_is_excluded_whole(connection):
    """`crashed` wrote a seat row with `failed` clear; it must not reach the money stats."""
    work = lq.generation_work(connection, "g")
    assert int(work["episodes"]) == 2
    assert int(work["failed_episodes"]) == 1

    seats = {int(row["seat"]): row for row in lq.seat_balance(connection, "g")}
    assert int(seats[0]["episodes"]) == 1
    assert float(seats[0]["mean_final_money"]) == 3200

    agents = {row["revision_id"]: row for row in lq.agent_episode_stats(connection, "g")}
    assert int(agents["r0"]["episodes"]) == 1
    assert float(agents["r0"]["mean_money"]) == 3200

    leaders = {row["revision_id"]: row for row in lq.generation_leaderboard(connection, "g")}
    assert int(leaders["r0"]["episodes"]) == 1
    assert float(leaders["r0"]["mean_final_money"]) == 3200


def test_reevaluation_requests_ignore_errored_evaluations(connection):
    rollup = lq.evaluation_rollup(connection, generation_id="g")
    assert int(rollup["evaluations"]) == 2
    assert int(rollup["errors"]) == 1
    assert int(rollup["reevaluation_requests"]) == 1

    session_rollup = lq.evaluation_rollup(connection, session_id="s")
    assert int(session_rollup["reevaluation_requests"]) == 1

    by_evaluator = {row["status"]: row for row in lq.evaluation_rollup_by_evaluator(connection, "g")}
    assert int(by_evaluator["COMPLETE"]["reevaluation_requests"]) == 1
    assert int(by_evaluator["ERROR"]["reevaluation_requests"]) == 0


def test_rating_trend_uses_canonical_column_names(connection):
    trend = lq.rating_trend(connection, "s")
    assert [row["generation_index"] for row in trend] == [0]
    assert float(trend[0]["champion_conservative"]) == 1300
    assert float(trend[0]["median_conservative"]) == 1250


def test_outcomes_read_the_seat_cancelled_margin(connection):
    outcomes = lq.generation_outcomes(connection, "g")
    assert int(outcomes["settled_pairings"]) == 1
    assert float(outcomes["mean_paired_margin"]) == 200
    assert float(outcomes["min_paired_margin"]) == 200
    assert float(outcomes["max_paired_margin"]) == 200
    assert int(outcomes["drawn"]) == 0


def _settle(connection, pairing_id: str, generation_id: str, margin: float) -> None:
    """One more settled pairing, so a test can shape the margin distribution."""
    connection.execute(
        "INSERT INTO league_pairings VALUES (%s, %s, 0, 'population', 9, 'r0', 'r1', 'COMPLETE')",
        [pairing_id, generation_id],
    )
    connection.execute(
        "INSERT INTO pairing_results VALUES (%s, %s, 1, false, 1, '{}')",
        [pairing_id, margin],
    )


def test_margin_histogram_folds_mirrored_margins(connection):
    """A 400 win and a 400 loss are the same edge size, so they share a bar."""
    for index, margin in enumerate((400, -400, 100, -100)):
        _settle(connection, f"p{index}", "g", margin)

    buckets = lq.margin_histogram(connection, "g")
    assert all(float(bucket["centre"]) >= 0 for bucket in buckets)
    assert {round(float(bucket["centre"])): int(bucket["episodes"]) for bucket in buckets} == {
        100: 2,
        200: 1,  # the seeded pairing
        400: 2,
    }
    # Nothing dropped and nothing counted twice: five settled pairings, five bars' worth.
    assert sum(int(bucket["episodes"]) for bucket in buckets) == 5


def test_margin_histogram_survives_a_generation_of_draws(connection):
    """Every margin identical used to collapse the bounds and return no rows at all."""
    connection.execute(
        "INSERT INTO generations (generation_id, session_id, generation_index, status, created_at)"
        " VALUES ('gz', 's', 1, 'RUNNING', now())"
    )
    _settle(connection, "z0", "gz", 0)
    _settle(connection, "z1", "gz", 0)

    buckets = lq.margin_histogram(connection, "gz")
    assert len(buckets) == 1
    assert float(buckets[0]["centre"]) == 0
    assert int(buckets[0]["episodes"]) == 2


def test_pairing_filter_values_come_back_split_and_sorted(connection):
    values = lq.pairing_filter_values(connection, "g")
    assert values == {"status": ["COMPLETE", "FAILED"], "kind": ["population"]}


def test_pairing_sort_and_direction_are_looked_up_not_interpolated(connection):
    with pytest.raises(ValueError):
        lq.generation_pairings(connection, "g", limit=10, sort="money")
    with pytest.raises(ValueError):
        lq.generation_pairings(connection, "g", limit=10, direction="; DROP TABLE episodes")
    ordered = lq.generation_pairings(connection, "g", limit=10, sort="round", direction="desc")
    assert [row["pairing_id"] for row in ordered] == ["bad", "good"]
    filtered = lq.generation_pairings(connection, "g", limit=10, status="FAILED")
    assert [row["pairing_id"] for row in filtered] == ["bad"]


def test_empty_id_lists_never_reach_the_database(connection):
    assert lq.session_rollups(connection, []) == []
    assert lq.session_leaders(connection, []) == []
    assert lq.session_latest_generations(connection, []) == []
    assert lq.pairing_episode_summaries(connection, []) == []
