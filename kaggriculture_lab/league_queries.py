"""Read queries over the league database, shared by the dashboard and the MCP tools.

The dashboard and `progress_mcp` answer the same questions -- how far has this
session got, who is winning this generation, how much of it failed -- and used to
carry separate copies of the SQL for each. The copies drifted: they disagreed on
what a completed round is, on which pairing status means failure, and on whether a
failed episode is one whose `episodes.failed` is set or one whose seat row is
marked. This module is the single answer to each of those questions.

Functions here return plain `list[dict]` / `dict` rows in the database's own types.
They do no HTML rendering and no MCP-shaped coercion, so both callers can layer
their own presentation on top. Every rollup is scoped to the session, generation
or match it is about: an unscoped CTE aggregates the whole lab and then throws all
but one row away, which is how these pages became slower with every league run
ever recorded.
"""

from __future__ import annotations

from typing import Any

# Repeated verbatim by the pairing, episode and replay queries, all three of which
# need both sides of a pairing resolved to a display name, strategy id, checkpoint
# and params blob. Kept as one fragment so the three cannot drift.
AGENT_IDENTITY_COLUMNS = """
       coalesce(afa.name, ara.family_id, lp.revision_a) AS agent_a_name,
       ara.strategy_id AS agent_a_strategy_id, ara.checkpoint_id AS agent_a_checkpoint_id,
       ara.params_json AS agent_a_params_json,
       coalesce(afb.name, arb.family_id, lp.revision_b) AS agent_b_name,
       arb.strategy_id AS agent_b_strategy_id, arb.checkpoint_id AS agent_b_checkpoint_id,
       arb.params_json AS agent_b_params_json
"""

AGENT_IDENTITY_JOINS = """
LEFT JOIN agent_revisions ara ON ara.revision_id = lp.revision_a
LEFT JOIN agent_families afa ON afa.family_id = ara.family_id
LEFT JOIN agent_revisions arb ON arb.revision_id = lp.revision_b
LEFT JOIN agent_families afb ON afb.family_id = arb.family_id
"""

# Exposed so callers can validate a user-supplied sort key and raise their own
# error type before any SQL is built.
PAIRING_SORT_COLUMNS = {
    "round": "lp.round_index",
    "pairing": "lp.pairing_id",
    "kind": "lp.pairing_kind",
    "agents": "agent_a_name",
    "status": "lp.status",
    "margin": "pr.paired_margin",
    "episodes": "pr.episodes_completed",
}

SORT_DIRECTIONS = frozenset({"asc", "desc"})


def rows(connection: Any, sql: str, parameters: list[Any] | None = None) -> list[dict[str, Any]]:
    cursor = connection.execute(sql, parameters or [])
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def row(connection: Any, sql: str, parameters: list[Any] | None = None) -> dict[str, Any] | None:
    found = rows(connection, sql, parameters)
    return found[0] if found else None


def scalar(connection: Any, sql: str, parameters: list[Any] | None = None) -> Any:
    found = connection.execute(sql, parameters or []).fetchone()
    return None if found is None else found[0]


def watermark(connection: Any) -> int:
    """A number that changes whenever anything in the cluster is written.

    Used to answer "has anything happened?" before paying to re-render a page.
    None of the hot tables carry a timestamp -- `league_pairings`,
    `pairing_results`, `episodes`, `episode_players`, `rating_snapshots` and
    `generation_stats` have no `updated_at` -- so there is nothing to take a
    `max()` over, and a `count(*)` both scans the tables and misses the in-place
    updates that ratings are written as.

    This reads the transaction counter instead. Only a transaction that writes is
    assigned a real xid, so `xmax` stands still under any amount of reading and
    advances the moment a writer commits. It is exact and immediate, which
    `pg_stat_user_tables` is not: the statistics collector only flushes a
    backend's counters about once a second, so a stats-based watermark reports a
    write up to a second late -- long enough to serve a stale page to somebody who
    reloaded straight after a generation closed.

    Its error is in the safe direction. Aborted transactions and writes to *other*
    databases in the same cluster also move it, so at worst a page is re-rendered
    when it need not have been. It can never hold still across a write.
    """
    return int(
        scalar(connection, "SELECT pg_snapshot_xmax(pg_current_snapshot())::text::bigint") or 0
    )


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #


def sessions(
    connection: Any,
    *,
    limit: int,
    offset: int = 0,
    league: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """One page of sessions, newest first. Callers asking for `limit + 1` detect a next page."""
    conditions: list[str] = []
    parameters: list[Any] = []
    if status:
        conditions.append("status = %s")
        parameters.append(status)
    if league:
        conditions.append("league = %s")
        parameters.append(league)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return rows(
        connection,
        f"""
        SELECT session_id, league, name, status, seed, config_json, created_at
        FROM training_sessions {where}
        ORDER BY created_at DESC, session_id LIMIT %s OFFSET %s
        """,
        [*parameters, limit, offset],
    )


def session(connection: Any, session_id: str) -> dict[str, Any] | None:
    return row(
        connection,
        """
        SELECT session_id, league, name, status, seed, config_json, created_at
        FROM training_sessions WHERE session_id = %s
        """,
        [session_id],
    )


def session_rollups(connection: Any, session_ids: list[str]) -> list[dict[str, Any]]:
    """Generation, pairing, episode and completed-round counts for several sessions.

    One CTE per rollup rather than one join. Joining pairings, episodes and rounds
    in a single FROM produced their cross product and leaned on `count(DISTINCT ...)`
    -- including a string-concatenated key -- to undo it.

    A session with no generations produces no row at all, matching the shape the
    callers already handle with `.get(session_id, {})`.
    """
    if not session_ids:
        return []
    placeholders = ",".join("%s" for _ in session_ids)
    return rows(
        connection,
        f"""
        WITH session_generations AS (
            SELECT session_id, generation_id FROM generations
            WHERE session_id IN ({placeholders})
        ), generation_counts AS (
            SELECT session_id, count(*) AS generations
            FROM session_generations GROUP BY session_id
        ), pairing_counts AS (
            SELECT sg.session_id, count(*) AS pairings,
                   count(*) FILTER (WHERE lp.status = 'COMPLETE') AS completed_pairings,
                   count(*) FILTER (WHERE lp.status = 'FAILED') AS failed_pairings
            FROM session_generations sg
            JOIN league_pairings lp ON lp.generation_id = sg.generation_id
            GROUP BY sg.session_id
        ), episode_counts AS (
            SELECT sg.session_id, count(DISTINCT pe.match_id) AS episodes,
                   count(DISTINCT pe.match_id) FILTER (WHERE e.failed) AS failed_episodes,
                   coalesce(sum(e.wallclock), 0) AS wallclock_seconds
            FROM session_generations sg
            JOIN league_pairings lp ON lp.generation_id = sg.generation_id
            JOIN pairing_episodes pe ON pe.pairing_id = lp.pairing_id
            JOIN episodes e ON e.match_id = pe.match_id
            GROUP BY sg.session_id
        ), rounds AS (
            SELECT sg.session_id, mrh.generation_id, mrh.round_index
            FROM session_generations sg
            JOIN model_round_hooks mrh ON mrh.generation_id = sg.generation_id
            WHERE mrh.scope = 'round'
            GROUP BY sg.session_id, mrh.generation_id, mrh.round_index
            HAVING bool_and(mrh.status = 'COMPLETE')
        ), round_counts AS (
            SELECT session_id, count(*) AS completed_rounds
            FROM rounds GROUP BY session_id
        )
        SELECT gc.session_id, gc.generations,
               coalesce(pc.pairings, 0) AS pairings,
               coalesce(pc.completed_pairings, 0) AS completed_pairings,
               coalesce(pc.failed_pairings, 0) AS failed_pairings,
               coalesce(ec.episodes, 0) AS episodes,
               coalesce(ec.failed_episodes, 0) AS failed_episodes,
               coalesce(ec.wallclock_seconds, 0) AS wallclock_seconds,
               coalesce(rc.completed_rounds, 0) AS completed_rounds
        FROM generation_counts gc
        LEFT JOIN pairing_counts pc USING (session_id)
        LEFT JOIN episode_counts ec USING (session_id)
        LEFT JOIN round_counts rc USING (session_id)
        """,
        list(session_ids),
    )


def session_leaders(connection: Any, session_ids: list[str]) -> list[dict[str, Any]]:
    """The top-rated unfrozen revision in each session's newest rated generation."""
    if not session_ids:
        return []
    placeholders = ",".join("%s" for _ in session_ids)
    return rows(
        connection,
        f"""
        SELECT session_id, revision_id, display_name, conservative FROM (
            SELECT g.session_id, rs.revision_id,
                   coalesce(af.name, ar.family_id, rs.revision_id) AS display_name,
                   rs.rating - 2 * rs.deviation AS conservative,
                   row_number() OVER (
                       PARTITION BY g.session_id
                       ORDER BY g.generation_index DESC,
                                (rs.rating - 2 * rs.deviation) DESC,
                                rs.revision_id
                   ) AS position
            FROM generations g JOIN rating_snapshots rs USING (generation_id)
            LEFT JOIN agent_revisions ar USING (revision_id)
            LEFT JOIN agent_families af USING (family_id)
            WHERE g.session_id IN ({placeholders}) AND NOT rs.is_frozen
        ) WHERE position = 1
        """,
        list(session_ids),
    )


def session_latest_generations(connection: Any, session_ids: list[str]) -> list[dict[str, Any]]:
    """The newest generation of each session, by generation index."""
    if not session_ids:
        return []
    placeholders = ",".join("%s" for _ in session_ids)
    return rows(
        connection,
        f"""
        SELECT session_id, generation_id, generation_index, status FROM (
            SELECT session_id, generation_id, generation_index, status,
                   row_number() OVER (
                       PARTITION BY session_id ORDER BY generation_index DESC
                   ) AS position
            FROM generations WHERE session_id IN ({placeholders})
        ) WHERE position = 1
        """,
        list(session_ids),
    )


def session_generations(
    connection: Any, session_id: str, *, limit: int | None = None
) -> list[dict[str, Any]]:
    """Per-generation rollups for one session, newest generation first."""
    limit_clause = "LIMIT %s" if limit is not None else ""
    parameters: list[Any] = [session_id, session_id]
    if limit is not None:
        parameters.append(limit)
    return rows(
        connection,
        f"""
        WITH session_generations AS (
            SELECT generation_id FROM generations WHERE session_id = %s
        ), member_counts AS (
            SELECT generation_id, count(*) AS members
            FROM generation_members
            WHERE generation_id IN (SELECT generation_id FROM session_generations)
            GROUP BY generation_id
        ), completed_rounds AS (
            SELECT generation_id, round_index FROM model_round_hooks
            WHERE scope = 'round'
              AND generation_id IN (SELECT generation_id FROM session_generations)
            GROUP BY generation_id, round_index
            HAVING bool_and(status = 'COMPLETE')
        ), round_counts AS (
            SELECT generation_id, count(*) AS completed_rounds
            FROM completed_rounds GROUP BY generation_id
        ), hook_errors AS (
            SELECT generation_id, count(*) FILTER (WHERE status = 'ERROR') AS hook_errors
            FROM model_round_hooks
            WHERE generation_id IN (SELECT generation_id FROM session_generations)
            GROUP BY generation_id
        ), pairing_counts AS (
            SELECT lp.generation_id, count(DISTINCT lp.pairing_id) AS pairings,
                   count(DISTINCT CASE WHEN lp.status = 'COMPLETE' THEN lp.pairing_id END) AS completed_pairings,
                   count(DISTINCT CASE WHEN lp.status = 'FAILED' THEN lp.pairing_id END) AS failed_pairings,
                   count(DISTINCT pe.match_id) AS episodes,
                   count(DISTINCT CASE WHEN e.failed THEN e.match_id END) AS failed_episodes,
                   coalesce(sum(e.wallclock), 0) AS wallclock_seconds
            FROM league_pairings lp
            LEFT JOIN pairing_episodes pe USING (pairing_id)
            LEFT JOIN episodes e USING (match_id)
            WHERE lp.generation_id IN (SELECT generation_id FROM session_generations)
            GROUP BY lp.generation_id
        )
        SELECT g.generation_id, g.generation_index, g.status, g.created_at, g.completed_at,
               coalesce(mc.members, 0) AS members, coalesce(pc.pairings, 0) AS pairings,
               coalesce(pc.completed_pairings, 0) AS completed_pairings,
               coalesce(pc.failed_pairings, 0) AS failed_pairings,
               coalesce(pc.episodes, 0) AS episodes,
               coalesce(pc.failed_episodes, 0) AS failed_episodes,
               coalesce(pc.wallclock_seconds, 0) AS wallclock_seconds,
               coalesce(rc.completed_rounds, 0) AS completed_rounds,
               coalesce(he.hook_errors, 0) AS hook_errors
        FROM generations g
        LEFT JOIN member_counts mc USING (generation_id)
        LEFT JOIN pairing_counts pc USING (generation_id)
        LEFT JOIN round_counts rc USING (generation_id)
        LEFT JOIN hook_errors he USING (generation_id)
        WHERE g.session_id = %s
        ORDER BY g.generation_index DESC
        {limit_clause}
        """,
        parameters,
    )


def rating_trend(connection: Any, session_id: str) -> list[dict[str, Any]]:
    """Champion and median conservative rating per generation, oldest first."""
    return rows(
        connection,
        """
        SELECT g.generation_id, g.generation_index,
               max(rs.rating - 2 * rs.deviation) AS champion_conservative,
               percentile_cont(0.5) WITHIN GROUP (
                   ORDER BY rs.rating - 2 * rs.deviation
               ) AS median_conservative
        FROM generations g JOIN rating_snapshots rs USING (generation_id)
        WHERE g.session_id = %s AND NOT rs.is_frozen
        GROUP BY g.generation_id, g.generation_index
        ORDER BY g.generation_index
        """,
        [session_id],
    )


def completed_round_count(
    connection: Any, *, session_id: str | None = None, generation_id: str | None = None
) -> int:
    """Rounds where *every* model's hook completed.

    A round is not done because one model finished it. Counting distinct round
    indexes with any COMPLETE hook reports a round as finished while another
    model's hook is still erroring, which is how the two MCP tools came to
    contradict each other about the same session.
    """
    if (session_id is None) == (generation_id is None):
        raise ValueError("pass exactly one of session_id or generation_id")
    if generation_id is not None:
        return int(
            scalar(
                connection,
                """
                SELECT count(*) FROM (
                    SELECT round_index FROM model_round_hooks
                    WHERE generation_id = %s AND scope = 'round'
                    GROUP BY round_index HAVING bool_and(status = 'COMPLETE')
                ) AS complete_rounds
                """,
                [generation_id],
            )
            or 0
        )
    return int(
        scalar(
            connection,
            """
            SELECT count(*) FROM (
                SELECT mrh.generation_id, mrh.round_index
                FROM model_round_hooks mrh JOIN generations g USING (generation_id)
                WHERE g.session_id = %s AND mrh.scope = 'round'
                GROUP BY mrh.generation_id, mrh.round_index
                HAVING bool_and(mrh.status = 'COMPLETE')
            ) AS complete_rounds
            """,
            [session_id],
        )
        or 0
    )


def evaluation_rollup(
    connection: Any, *, session_id: str | None = None, generation_id: str | None = None
) -> dict[str, Any]:
    """Episode-evaluation counts for a session or a generation.

    `should_reevaluate` is only read off rows that completed: an evaluator that
    raised has no meaningful result, and counting the flag out of an ERROR row
    reports a re-evaluation nobody asked for.
    """
    if (session_id is None) == (generation_id is None):
        raise ValueError("pass exactly one of session_id or generation_id")
    empty = {"evaluations": 0, "errors": 0, "reevaluation_requests": 0}
    counts = """
        SELECT count(*) AS evaluations,
               count(*) FILTER (WHERE ee.status = 'ERROR') AS errors,
               count(*) FILTER (
                   WHERE ee.status = 'COMPLETE'
                     AND coalesce(
                           CAST(ee.result_json ->> 'should_reevaluate' AS BOOLEAN),
                           false
                         )
               ) AS reevaluation_requests
        FROM episode_evaluations ee
        JOIN pairing_episodes pe USING (match_id)
        JOIN league_pairings lp USING (pairing_id)
    """
    if generation_id is not None:
        found = row(connection, f"{counts} WHERE lp.generation_id = %s", [generation_id])
    else:
        found = row(
            connection,
            f"{counts} JOIN generations g USING (generation_id) WHERE g.session_id = %s",
            [session_id],
        )
    return found or dict(empty)


def evaluation_rollup_by_evaluator(connection: Any, generation_id: str) -> list[dict[str, Any]]:
    return rows(
        connection,
        """
        SELECT ee.status, ee.evaluator_path, count(*) AS evaluations,
               count(*) FILTER (
                   WHERE ee.status = 'COMPLETE'
                     AND coalesce(CAST(ee.result_json ->> 'should_reevaluate' AS BOOLEAN), false)
               ) AS reevaluation_requests
        FROM episode_evaluations ee JOIN pairing_episodes pe USING (match_id)
        JOIN league_pairings lp USING (pairing_id)
        WHERE lp.generation_id = %s GROUP BY ee.status, ee.evaluator_path
        ORDER BY ee.status, ee.evaluator_path
        """,
        [generation_id],
    )


# --------------------------------------------------------------------------- #
# Generations
# --------------------------------------------------------------------------- #


def generation(connection: Any, generation_id: str) -> dict[str, Any] | None:
    return row(
        connection,
        """
        SELECT g.generation_id, g.session_id, g.generation_index, g.status,
               g.model_checkpoint_id, g.created_at, g.completed_at,
               s.league, s.name AS session_name, s.config_json
        FROM generations g JOIN training_sessions s USING (session_id)
        WHERE g.generation_id = %s
        """,
        [generation_id],
    )


def generation_work(connection: Any, generation_id: str) -> dict[str, Any]:
    """Member, pairing, episode and round counts for one generation.

    `league_pairings.status` is written as COMPLETE or FAILED and never ERROR
    (`league.py`), so a pairing-error count against 'ERROR' is structurally always
    zero; `failed_pairings` is the real number.
    """
    return row(
        connection,
        """
        SELECT
            (SELECT count(*) FROM generation_members WHERE generation_id = %s) AS members,
            (SELECT count(*) FROM league_pairings WHERE generation_id = %s) AS pairings,
            (SELECT count(*) FROM league_pairings
              WHERE generation_id = %s AND status = 'COMPLETE') AS completed_pairings,
            (SELECT count(*) FROM league_pairings
              WHERE generation_id = %s AND status = 'FAILED') AS failed_pairings,
            (SELECT count(DISTINCT pe.match_id)
               FROM league_pairings lp JOIN pairing_episodes pe USING (pairing_id)
              WHERE lp.generation_id = %s) AS episodes,
            (SELECT count(DISTINCT pe.match_id)
               FROM league_pairings lp JOIN pairing_episodes pe USING (pairing_id)
               JOIN episodes e USING (match_id)
              WHERE lp.generation_id = %s AND e.failed) AS failed_episodes,
            (SELECT coalesce(sum(e.wallclock), 0)
               FROM league_pairings lp JOIN pairing_episodes pe USING (pairing_id)
               JOIN episodes e USING (match_id)
              WHERE lp.generation_id = %s) AS wallclock_seconds,
            (SELECT count(*) FROM (
                SELECT round_index FROM model_round_hooks
                WHERE generation_id = %s AND scope = 'round'
                GROUP BY round_index HAVING bool_and(status = 'COMPLETE')
            ) AS complete_rounds) AS completed_rounds
        """,
        [generation_id] * 8,
    ) or {}


def generation_leaderboard(
    connection: Any, generation_id: str, *, limit: int | None = None
) -> list[dict[str, Any]]:
    """Ranked unfrozen revisions, with rating, paired record and episode aggregates.

    One superset of what the dashboard leaderboard and the MCP leaders list each
    used to fetch separately, ordered the dashboard's way -- mean margin breaks a
    conservative-rating tie before the revision id does.
    """
    limit_clause = "LIMIT %s" if limit is not None else ""
    parameters: list[Any] = [generation_id, generation_id]
    if limit is not None:
        parameters.append(limit)
    return rows(
        connection,
        f"""
        WITH episode_rollup AS (
            SELECT ep.strategy_id AS revision_id, count(*) AS episodes,
                   avg(ep.final_money) AS mean_final_money,
                   avg(ep.money_difference) AS mean_episode_margin,
                   avg(ep.win) AS win_rate,
                   avg(ep.productive_action_percentage) AS productive_action_fraction,
                   avg(ep.final_shed_items + ep.final_carried_items) AS mean_unsold_items
            FROM league_pairings lp JOIN pairing_episodes pe USING (pairing_id)
            JOIN episodes e USING (match_id)
            JOIN episode_players ep USING (match_id)
            WHERE lp.generation_id = %s AND NOT e.failed GROUP BY ep.strategy_id
        )
        SELECT rs.revision_id, ar.strategy_id,
               coalesce(af.name, ar.family_id, rs.revision_id) AS display_name,
               gm.role, rs.rating, rs.deviation,
               rs.rating - 2 * rs.deviation AS conservative,
               rs.games AS rated_games, gs.games AS paired_games,
               gs.wins, gs.draws, gs.losses,
               gs.mean_margin, gs.median_margin, gs.margin_ci_low, gs.margin_ci_high,
               gs.opponent_diversity, er.episodes, er.mean_final_money,
               er.mean_episode_margin, er.win_rate, er.productive_action_fraction,
               er.mean_unsold_items
        FROM rating_snapshots rs
        LEFT JOIN agent_revisions ar USING (revision_id)
        LEFT JOIN agent_families af USING (family_id)
        LEFT JOIN generation_members gm USING (generation_id, revision_id)
        LEFT JOIN generation_stats gs USING (generation_id, revision_id)
        LEFT JOIN episode_rollup er USING (revision_id)
        WHERE rs.generation_id = %s AND NOT rs.is_frozen
        ORDER BY conservative DESC, gs.mean_margin DESC NULLS LAST, rs.revision_id
        {limit_clause}
        """,
        parameters,
    )


def generation_outcomes(connection: Any, generation_id: str) -> dict[str, Any]:
    """Margin distribution over settled pairings.

    Reads `pairing_results.paired_margin`, which is already seat-cancelled, so the
    zero point is a fair "no edge" reference.
    """
    return row(
        connection,
        """
        -- One pass with FILTER rather than an aggregate plus a correlated
        -- subquery for the failures: both wanted the same rows, and scanning
        -- pairing_results twice to split them was the only reason for the second.
        SELECT count(*) FILTER (WHERE NOT pr.failed) AS settled_pairings,
               avg(pr.paired_margin) FILTER (WHERE NOT pr.failed) AS mean_paired_margin,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY pr.paired_margin)
                   FILTER (WHERE NOT pr.failed) AS median_paired_margin,
               stddev_samp(pr.paired_margin) FILTER (WHERE NOT pr.failed) AS paired_margin_stddev,
               min(pr.paired_margin) FILTER (WHERE NOT pr.failed) AS min_paired_margin,
               max(pr.paired_margin) FILTER (WHERE NOT pr.failed) AS max_paired_margin,
               avg(pr.outcome_a) FILTER (WHERE NOT pr.failed) AS mean_outcome_a,
               count(*) FILTER (WHERE NOT pr.failed AND pr.paired_margin = 0) AS drawn,
               count(*) FILTER (WHERE pr.failed) AS failed_pairing_results
        FROM league_pairings lp JOIN pairing_results pr USING (pairing_id)
        WHERE lp.generation_id = %s
        """,
        [generation_id],
    ) or {}


def margin_histogram(
    connection: Any, generation_id: str, *, buckets: int = 21
) -> list[dict[str, Any]]:
    """Settled pairing margins, pre-binned by size.

    Binned in SQL rather than in the renderer: `width_bucket` is what the database
    is for, and it keeps the chart primitive from growing a statistics layer.

    Reads `paired_margin` because it is already seat-cancelled, so the zero point
    is a fair "no edge" reference. `episode_players.money_difference` would count
    each pairing twice, once from each side, and carry the seat advantage with it
    -- which shows up as a bimodal shape that is an artefact, not a finding.

    Bins `abs(margin)`, folding the two orientations of the same edge size into
    one bucket: a 4,000 win and a 4,000 loss are one finding about how decisive
    the pairings are, and splitting them halves every bar for no gain. Which way
    the edge ran is read off the mean/min/max cards, not off this shape.

    The low bound is a fixed 0 rather than `min(margin)`, so bucket 1 is always
    the level pairings instead of sliding whenever the closest pairing changes.
    Flooring the high bound keeps a generation whose margins are all identical
    -- all draws, most often -- rendering one honest bar rather than nothing.
    """
    return rows(
        connection,
        """
        WITH settled AS (
            SELECT abs(pr.paired_margin) AS margin
            FROM league_pairings lp JOIN pairing_results pr USING (pairing_id)
            WHERE lp.generation_id = %s AND NOT pr.failed AND pr.paired_margin IS NOT NULL
        ), bounds AS (
            SELECT greatest(max(margin), 1.0) AS high FROM settled
        )
        SELECT width_bucket(s.margin, 0, b.high + 1e-9, %s) AS bucket,
               count(*) AS episodes,
               min(s.margin) AS low, max(s.margin) AS high,
               avg(s.margin) AS centre
        FROM settled s CROSS JOIN bounds b
        GROUP BY bucket ORDER BY bucket
        """,
        [generation_id, buckets],
    )


def head_to_head_by_family(connection: Any, session_id: str) -> list[dict[str, Any]]:
    """Every family's record against every other, across one session.

    The `UNION ALL` with a negated margin and `1 - outcome_a` is what makes this
    honest: each pairing is counted once from each side, so the matrix is
    antisymmetric by construction and no seat or ordering bias can leak in.

    Only complete generations count. A generation still being played has whichever
    pairings happen to have finished, which is not a sample of anything.
    """
    return rows(
        connection,
        """
        WITH sided AS (
            SELECT ara.family_id AS me, arb.family_id AS opponent,
                   pr.paired_margin AS margin, pr.outcome_a AS score
            FROM league_pairings lp JOIN pairing_results pr USING (pairing_id)
            JOIN generations g USING (generation_id)
            JOIN agent_revisions ara ON ara.revision_id = lp.revision_a
            JOIN agent_revisions arb ON arb.revision_id = lp.revision_b
            WHERE g.session_id = %s AND NOT pr.failed AND g.status = 'COMPLETE'
              AND pr.paired_margin IS NOT NULL
            UNION ALL
            SELECT arb.family_id, ara.family_id, -pr.paired_margin, 1 - pr.outcome_a
            FROM league_pairings lp JOIN pairing_results pr USING (pairing_id)
            JOIN generations g USING (generation_id)
            JOIN agent_revisions ara ON ara.revision_id = lp.revision_a
            JOIN agent_revisions arb ON arb.revision_id = lp.revision_b
            WHERE g.session_id = %s AND NOT pr.failed AND g.status = 'COMPLETE'
              AND pr.paired_margin IS NOT NULL
        )
        SELECT s.me, s.opponent,
               coalesce(af.name, s.opponent) AS opponent_name,
               count(*) AS pairings, avg(s.margin) AS mean_margin,
               stddev_samp(s.margin) AS margin_stddev, avg(s.score) AS score
        FROM sided s LEFT JOIN agent_families af ON af.family_id = s.opponent
        GROUP BY s.me, s.opponent, opponent_name
        ORDER BY s.me, mean_margin
        """,
        [session_id, session_id],
    )


def failure_summary(connection: Any, session_id: str) -> dict[str, list[dict[str, Any]]]:
    """Every recorded failure in one session, grouped by what actually went wrong.

    Four separate things are called a failure and each is stored differently:
    a crashed episode, a failed pairing, an errored round hook, and an errored
    model update. Only the last two carry text, and `model_updates.error_text`
    holds a whole formatted traceback -- so it is grouped by its *first* line,
    which turns "47 failures" into "one exception, 47 times".
    """
    hooks = rows(
        connection,
        """
        SELECT mrh.model_id, mrh.generation_id, g.generation_index, mrh.round_index,
               mrh.error_text, mrh.started_at, mrh.completed_at
        FROM model_round_hooks mrh LEFT JOIN generations g USING (generation_id)
        WHERE mrh.session_id = %s AND mrh.status = 'ERROR'
        ORDER BY g.generation_index DESC NULLS LAST, mrh.round_index DESC
        """,
        [session_id],
    )
    updates = rows(
        connection,
        """
        WITH session_models AS (
            SELECT DISTINCT model_id FROM model_round_hooks WHERE session_id = %s
        )
        SELECT mu.model_id, mu.update_id, mu.cutoff_batch_seq, mu.error_text, mu.started_at
        FROM model_updates mu
        WHERE mu.model_id IN (SELECT model_id FROM session_models) AND mu.status = 'ERROR'
        ORDER BY mu.started_at DESC
        """,
        [session_id],
    )
    episodes = rows(
        connection,
        """
        SELECT g.generation_id, g.generation_index, e.match_id, e.stage, e.steps,
               lp.round_index, lp.pairing_kind
        FROM generations g JOIN league_pairings lp USING (generation_id)
        JOIN pairing_episodes pe USING (pairing_id)
        JOIN episodes e USING (match_id)
        WHERE g.session_id = %s AND e.failed
        ORDER BY g.generation_index DESC, lp.round_index DESC LIMIT 200
        """,
        [session_id],
    )
    pairings = rows(
        connection,
        """
        SELECT g.generation_id, g.generation_index, lp.pairing_id, lp.round_index,
               lp.pairing_kind
        FROM generations g JOIN league_pairings lp USING (generation_id)
        WHERE g.session_id = %s AND lp.status = 'FAILED'
        ORDER BY g.generation_index DESC, lp.round_index DESC LIMIT 200
        """,
        [session_id],
    )
    return {
        "hooks": hooks,
        "updates": updates,
        "episodes": episodes,
        "pairings": pairings,
    }


def seat_balance(connection: Any, generation_id: str) -> list[dict[str, Any]]:
    """Per-seat money and win rate. Every pairing is played in both orientations,
    so a standing seat-to-seat gap is the environment talking, not the agents.

    Filtered on `episodes.failed` rather than `episode_players.failed`: an episode
    that crashed is excluded whole, which is how failures are counted everywhere
    else. Filtering per seat let a crashed episode's surviving seat row through.
    """
    return rows(
        connection,
        """
        SELECT ep.seat, count(*) AS episodes, avg(ep.final_money) AS mean_final_money,
               avg(ep.money_difference) AS mean_margin, avg(ep.win) AS win_rate
        FROM league_pairings lp JOIN pairing_episodes pe USING (pairing_id)
        JOIN episodes e USING (match_id) JOIN episode_players ep USING (match_id)
        WHERE lp.generation_id = %s AND NOT e.failed
        GROUP BY ep.seat ORDER BY ep.seat
        """,
        [generation_id],
    )


def agent_episode_stats(connection: Any, generation_id: str) -> list[dict[str, Any]]:
    """Per-agent money and behaviour mix for one generation.

    `episode_players.strategy_id` already carries the seat-correct revision id for
    league episodes -- `league.py` swaps the pair by orientation before handing it
    to `add_match` -- so the seat needs no remapping here.
    """
    return rows(
        connection,
        """
        SELECT ep.strategy_id AS revision_id,
               coalesce(af.name, ar.family_id, ep.strategy_id) AS display_name,
               count(*) AS episodes, avg(ep.final_money) AS mean_money,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY ep.final_money) AS median_money,
               avg(ep.money_difference) AS mean_margin,
               avg(ep.reward) AS mean_reward, avg(ep.win) AS win_rate,
               avg(ep.productive_action_percentage) AS productive,
               avg(ep.movement_percentage) AS movement, avg(ep.idle_percentage) AS idle,
               avg(ep.plants_lost_inferred) AS plants_lost,
               avg(ep.final_shed_items + ep.final_carried_items) AS unsold,
               max(rs.rating - 2 * rs.deviation) AS conservative
        FROM league_pairings lp JOIN pairing_episodes pe USING (pairing_id)
        JOIN episodes e USING (match_id) JOIN episode_players ep USING (match_id)
        LEFT JOIN agent_revisions ar ON ar.revision_id = ep.strategy_id
        LEFT JOIN agent_families af USING (family_id)
        LEFT JOIN rating_snapshots rs
               ON rs.generation_id = lp.generation_id AND rs.revision_id = ep.strategy_id
        WHERE lp.generation_id = %s AND NOT e.failed
        GROUP BY ep.strategy_id, display_name
        ORDER BY conservative DESC NULLS LAST, mean_margin DESC NULLS LAST, revision_id
        """,
        [generation_id],
    )


def episode_runtime(connection: Any, generation_id: str) -> dict[str, Any]:
    """Episode wall clock and step counts.

    Per-agent-call latency is measured by `sim.runner` but never persisted by
    `storage.add_match`, so episode wall clock is all that is available.
    """
    return row(
        connection,
        """
        SELECT count(*) AS episodes, count(*) FILTER (WHERE e.failed) AS failed,
               sum(e.wallclock) AS total_wallclock, avg(e.wallclock) AS mean_wallclock,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY e.wallclock) AS median_wallclock,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY e.wallclock) AS p95_wallclock,
               max(e.wallclock) AS max_wallclock,
               avg(e.steps) AS mean_steps, max(e.steps) AS max_steps
        FROM league_pairings lp JOIN pairing_episodes pe USING (pairing_id)
        JOIN episodes e USING (match_id) WHERE lp.generation_id = %s
        """,
        [generation_id],
    ) or {}


def slowest_episodes(connection: Any, generation_id: str, *, limit: int = 5) -> list[dict[str, Any]]:
    return rows(
        connection,
        """
        SELECT e.match_id, e.wallclock, e.steps, e.failed, lp.round_index
        FROM league_pairings lp JOIN pairing_episodes pe USING (pairing_id)
        JOIN episodes e USING (match_id) WHERE lp.generation_id = %s
        ORDER BY e.wallclock DESC LIMIT %s
        """,
        [generation_id, limit],
    )


def generation_pairings(
    connection: Any,
    generation_id: str,
    *,
    limit: int,
    offset: int = 0,
    status: str | None = None,
    kind: str | None = None,
    sort: str = "round",
    direction: str = "asc",
) -> list[dict[str, Any]]:
    """One page of pairings with both agents resolved and the paired result attached.

    `sort` and `direction` are looked up rather than interpolated; validate them
    against `PAIRING_SORT_COLUMNS` / `SORT_DIRECTIONS` first if the value came from
    a request, so the caller controls the error type.
    """
    if sort not in PAIRING_SORT_COLUMNS:
        raise ValueError(f"unknown pairing sort {sort!r}")
    if direction not in SORT_DIRECTIONS:
        raise ValueError(f"unknown sort direction {direction!r}")
    conditions = ["lp.generation_id = %s"]
    parameters: list[Any] = [generation_id]
    if status is not None:
        conditions.append("lp.status = %s")
        parameters.append(status)
    if kind is not None:
        conditions.append("lp.pairing_kind = %s")
        parameters.append(kind)
    order = f"{PAIRING_SORT_COLUMNS[sort]} {direction.upper()} NULLS LAST, lp.pairing_id"
    return rows(
        connection,
        f"""
        SELECT lp.pairing_id, lp.round_index, lp.pairing_kind, lp.seed, lp.revision_a,
               lp.revision_b, lp.status, pr.paired_margin, pr.outcome_a, pr.failed,
               pr.episodes_completed,
               {AGENT_IDENTITY_COLUMNS}
        FROM league_pairings lp LEFT JOIN pairing_results pr USING (pairing_id)
        {AGENT_IDENTITY_JOINS}
        WHERE {' AND '.join(conditions)} ORDER BY {order} LIMIT %s OFFSET %s
        """,
        [*parameters, limit, offset],
    )


def pairing_episode_summaries(connection: Any, pairing_ids: list[str]) -> list[dict[str, Any]]:
    """Both seat orientations of each pairing, with each seat's final money."""
    if not pairing_ids:
        return []
    placeholders = ",".join("%s" for _ in pairing_ids)
    return rows(
        connection,
        f"""
        SELECT pe.pairing_id, pe.orientation, pe.match_id, e.failed,
               ep0.final_money AS seat0_money, ep1.final_money AS seat1_money
        FROM pairing_episodes pe JOIN episodes e USING (match_id)
        LEFT JOIN episode_players ep0 ON ep0.match_id = pe.match_id AND ep0.seat = 0
        LEFT JOIN episode_players ep1 ON ep1.match_id = pe.match_id AND ep1.seat = 1
        WHERE pe.pairing_id IN ({placeholders}) ORDER BY pe.pairing_id, pe.orientation
        """,
        list(pairing_ids),
    )


def pairing_filter_values(connection: Any, generation_id: str) -> dict[str, list[str]]:
    """The distinct statuses and kinds present, for the pairing table's filter selects."""
    found = rows(
        connection,
        """
        SELECT 'status' AS field, status AS value
        FROM league_pairings WHERE generation_id = %s GROUP BY status
        UNION ALL
        SELECT 'kind' AS field, pairing_kind AS value
        FROM league_pairings WHERE generation_id = %s GROUP BY pairing_kind
        ORDER BY field, value
        """,
        [generation_id, generation_id],
    )
    return {
        "status": [str(item["value"]) for item in found if item["field"] == "status"],
        "kind": [str(item["value"]) for item in found if item["field"] == "kind"],
    }


def generation_round_hooks(connection: Any, generation_id: str) -> list[dict[str, Any]]:
    return rows(
        connection,
        """
        SELECT round_index, model_id, status, batches_recorded, error_text,
               started_at, completed_at
        FROM model_round_hooks
        WHERE generation_id = %s AND scope = 'round'
        ORDER BY round_index, model_id
        """,
        [generation_id],
    )


def generation_hook_rollup(connection: Any, generation_id: str) -> list[dict[str, Any]]:
    return rows(
        connection,
        """
        SELECT model_id, scope, status, count(*) AS hooks,
               min(round_index) AS first_round, max(round_index) AS last_round,
               sum(batches_recorded) AS batches_recorded
        FROM model_round_hooks WHERE generation_id = %s
        GROUP BY model_id, scope, status ORDER BY model_id, scope, status
        """,
        [generation_id],
    )


# --------------------------------------------------------------------------- #
# Episodes
# --------------------------------------------------------------------------- #


def episode(connection: Any, match_id: str) -> dict[str, Any] | None:
    return row(
        connection,
        f"""
        SELECT e.match_id, e.seed, e.steps, e.wallclock, e.failed, e.stage,
               pe.orientation, lp.pairing_id, lp.generation_id, lp.revision_a, lp.revision_b,
               {AGENT_IDENTITY_COLUMNS}
        FROM episodes e JOIN pairing_episodes pe USING (match_id)
        JOIN league_pairings lp USING (pairing_id)
        {AGENT_IDENTITY_JOINS}
        WHERE e.match_id = %s
        """,
        [match_id],
    )


def episode_seats(connection: Any, match_id: str) -> list[dict[str, Any]]:
    return rows(
        connection,
        """
        SELECT seat, strategy_id, opponent_id, status, failed, reward, final_money,
               opponent_money, money_difference, win, total_unit_actions, productive_actions,
               movement_actions, idle_actions, productive_action_percentage, movement_percentage,
               idle_percentage, plants_lost_inferred, final_shed_items, final_carried_items,
               metrics_json
        FROM episode_players WHERE match_id = %s ORDER BY seat
        """,
        [match_id],
    )


def episode_evaluations(connection: Any, match_id: str) -> list[dict[str, Any]]:
    return rows(
        connection,
        """
        SELECT revision_id, seat, evaluator_path, status, result_json, error_text, created_at
        FROM episode_evaluations WHERE match_id = %s ORDER BY seat, created_at DESC
        """,
        [match_id],
    )


def replay_identities(connection: Any, match_id: str) -> dict[str, Any] | None:
    """Just the seat-order and both agent identities, for naming replay players."""
    return row(
        connection,
        f"""
        SELECT pe.orientation, lp.revision_a, lp.revision_b,
               {AGENT_IDENTITY_COLUMNS}
        FROM pairing_episodes pe JOIN league_pairings lp USING (pairing_id)
        {AGENT_IDENTITY_JOINS}
        WHERE pe.match_id = %s
        """,
        [match_id],
    )


def has_replay(connection: Any, match_id: str) -> bool:
    return scalar(connection, "SELECT 1 FROM episode_payloads WHERE match_id = %s", [match_id]) is not None


def replay_gzip(connection: Any, match_id: str) -> bytes | None:
    found = scalar(connection, "SELECT replay_gzip FROM episode_payloads WHERE match_id = %s", [match_id])
    return None if found is None else bytes(found)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def model(connection: Any, model_id: str) -> dict[str, Any] | None:
    """One trainable model with its batch watermark and a readable family name."""
    return row(
        connection,
        """
        SELECT tm.model_id, tm.family_id, tm.agent_type, tm.agent_name,
               tm.declared_model_id, tm.trainer_path, tm.min_new_batches,
               tm.config_json, tm.active_checkpoint_id, tm.last_attempted_batch_seq,
               coalesce(tb.latest_batch_seq, 0) AS latest_batch_seq,
               af.name AS family_name, tm.created_at, tm.updated_at
        FROM training_models tm
        LEFT JOIN (
            SELECT model_id, max(batch_seq) AS latest_batch_seq
            FROM training_batches WHERE model_id = %s GROUP BY model_id
        ) tb USING (model_id)
        LEFT JOIN agent_families af USING (family_id)
        WHERE tm.model_id = %s
        """,
        [model_id, model_id],
    )


def checkpoint(connection: Any, checkpoint_id: str) -> dict[str, Any] | None:
    return row(
        connection,
        """
        SELECT checkpoint_id, model_id, update_id, parent_checkpoint_id,
               trainer_artifact_path, inference_artifact_path, artifact_sha256,
               dataset_sha256, metrics_json, gate_json, promoted, created_at
        FROM model_training_checkpoints WHERE checkpoint_id = %s
        """,
        [checkpoint_id],
    )


def model_batches(
    connection: Any, model_id: str, *, limit: int = 10
) -> list[dict[str, Any]]:
    return rows(
        connection,
        """
        SELECT batch_seq, batch_id, source_kind, source_id, sample_count,
               metadata_json, created_at FROM training_batches
        WHERE model_id = %s ORDER BY batch_seq DESC LIMIT %s
        """,
        [model_id, limit],
    )


def model_batch_sources(connection: Any, model_id: str) -> list[dict[str, Any]]:
    """Batch counts by source kind, split at the model's last attempted watermark.

    `source_kind = 'model_evaluation'` is treated specially by the updater, so the
    split between consumed and pending batches is worth seeing per kind.
    """
    return rows(
        connection,
        """
        SELECT tb.source_kind,
               count(*) AS batches,
               count(*) FILTER (WHERE tb.batch_seq <= tm.last_attempted_batch_seq) AS consumed,
               count(*) FILTER (WHERE tb.batch_seq > tm.last_attempted_batch_seq) AS pending,
               sum(tb.sample_count) AS samples
        FROM training_batches tb JOIN training_models tm USING (model_id)
        WHERE tb.model_id = %s GROUP BY tb.source_kind ORDER BY tb.source_kind
        """,
        [model_id],
    )


def model_updates(
    connection: Any, model_id: str, *, limit: int | None = None, oldest_first: bool = False
) -> list[dict[str, Any]]:
    """Update attempts for one model.

    Ordered by `cutoff_batch_seq` when read oldest-first: training is bursty, so
    batch sequence is a fairer x axis than wall clock for a gate timeline.
    """
    order = "cutoff_batch_seq, started_at" if oldest_first else "started_at DESC"
    limit_clause = "LIMIT %s" if limit is not None else ""
    parameters: list[Any] = [model_id]
    if limit is not None:
        parameters.append(limit)
    return rows(
        connection,
        f"""
        SELECT update_id, parent_checkpoint_id, cutoff_batch_seq, new_batch_count,
               dataset_sha256, status, candidate_checkpoint_id, metrics_json,
               gate_json, error_text, started_at, completed_at
        FROM model_updates WHERE model_id = %s
        ORDER BY {order} {limit_clause}
        """,
        parameters,
    )


def model_checkpoints(
    connection: Any, model_id: str, *, limit: int | None = None
) -> list[dict[str, Any]]:
    limit_clause = "LIMIT %s" if limit is not None else ""
    parameters: list[Any] = [model_id]
    if limit is not None:
        parameters.append(limit)
    return rows(
        connection,
        f"""
        SELECT checkpoint_id, update_id, parent_checkpoint_id, promoted,
               metrics_json, gate_json, dataset_sha256, created_at
        FROM model_training_checkpoints WHERE model_id = %s
        ORDER BY created_at DESC {limit_clause}
        """,
        parameters,
    )


def model_evaluations(
    connection: Any, model_id: str, *, limit: int | None = None
) -> list[dict[str, Any]]:
    limit_clause = "LIMIT %s" if limit is not None else ""
    parameters: list[Any] = [model_id]
    if limit is not None:
        parameters.append(limit)
    return rows(
        connection,
        f"""
        SELECT evaluation_id, update_id, evaluation_kind, opponent_checkpoint_id,
               metrics_json, passed, created_at
        FROM model_evaluations WHERE model_id = %s
        ORDER BY created_at DESC {limit_clause}
        """,
        parameters,
    )


def model_hook_status(connection: Any, model_id: str) -> list[dict[str, Any]]:
    """Hook counts per session, ordered by the session's newest generation index.

    Ordering on `max(generation_id)` is a lexical string comparison and says very
    little; the generation index is the real sequence.
    """
    return rows(
        connection,
        """
        SELECT mrh.session_id, mrh.scope, mrh.status, count(*) AS hooks,
               max(g.generation_index) AS latest_generation_index,
               max(mrh.round_index) AS latest_round_index,
               sum(mrh.batches_recorded) AS batches_recorded
        FROM model_round_hooks mrh
        LEFT JOIN generations g USING (generation_id)
        WHERE mrh.model_id = %s
        GROUP BY mrh.session_id, mrh.scope, mrh.status
        ORDER BY max(g.generation_index) DESC NULLS LAST,
                 mrh.session_id, mrh.scope, mrh.status
        """,
        [model_id],
    )


def session_learner_roster(connection: Any, session_id: str) -> list[dict[str, Any]]:
    """One row per trainable model in a session, as of its newest update.

    A roster view: which policy currently occupies each slot, how far through its
    schedule it is, what it last measured, and whether its release has reached the
    shared pool. Answering that from `model_updates` alone means reading JSON out
    of ten rows by hand, which is why it kept being done in SQL by hand.

    `reseed_index` is the slot's generation marker. It starts at 0 and steps every
    time the slot is reseeded, so `learner_03` at index 2 is the third policy to
    have occupied that slot -- the identity that `model_id` alone cannot express,
    because the slot keeps its id across restarts.

    The `lr_*` and gate columns are read out of whatever the trainer recorded, so
    they are populated for agents that keep a slot schedule and NULL for those that
    do not. That is deliberate: this is one query for a general shape, not a
    per-agent view, and a model with no schedule simply shows blanks.
    """
    return rows(
        connection,
        """
        WITH session_models AS (
            SELECT DISTINCT model_id FROM model_round_hooks WHERE session_id = %s
        ), batches AS (
            SELECT model_id, max(batch_seq) AS latest_batch_seq
            FROM training_batches
            WHERE model_id IN (SELECT model_id FROM session_models)
            GROUP BY model_id
        ), latest AS (
            SELECT DISTINCT ON (mu.model_id)
                   mu.model_id, mu.status, mu.cutoff_batch_seq, mu.started_at,
                   mu.metrics_json, mu.gate_json
            FROM model_updates mu
            WHERE mu.model_id IN (SELECT model_id FROM session_models)
            ORDER BY mu.model_id, mu.started_at DESC
        )
        SELECT tm.model_id, tm.agent_name, tm.declared_model_id,
               tm.active_checkpoint_id, tm.last_attempted_batch_seq,
               tm.min_new_batches, coalesce(b.latest_batch_seq, 0) AS latest_batch_seq,
               l.status AS last_status, l.cutoff_batch_seq, l.started_at AS last_update_at,
               CAST(l.metrics_json ->> 'lr_reseed_index' AS INTEGER) AS reseed_index,
               CAST(l.metrics_json ->> 'lr_updates_since_reseed' AS INTEGER) AS updates_since_reseed,
               CAST(l.metrics_json ->> 'lr_decay_steps' AS INTEGER) AS decay_steps,
               CAST(l.metrics_json ->> 'lr_stall_updates' AS INTEGER) AS stall_updates,
               l.metrics_json ->> 'lr_stage' AS stage,
               CAST(l.metrics_json ->> 'lr_score' AS DOUBLE PRECISION) AS score,
               CAST(l.gate_json -> 'benchmark_ratchet' ->> 'candidate_money'
                    AS DOUBLE PRECISION) AS last_money,
               CAST(l.gate_json -> 'hard_viability' -> 'head_to_head' ->> 'trading_rate'
                    AS DOUBLE PRECISION) AS trading_rate,
               CAST(l.gate_json -> 'population' ->> 'release_age' AS INTEGER) AS release_age,
               (SELECT count(*) FROM model_updates u
                 WHERE u.model_id = tm.model_id AND u.status = 'PROMOTED') AS promotions,
               pm.member_id IS NOT NULL AND pm.retired_at IS NULL AS in_pool,
               pm.score AS pool_score
        FROM session_models sm
        JOIN training_models tm USING (model_id)
        LEFT JOIN latest l USING (model_id)
        LEFT JOIN batches b USING (model_id)
        LEFT JOIN population_members pm
               ON pm.checkpoint_id = tm.active_checkpoint_id
              AND pm.declared_model_id = tm.declared_model_id
        ORDER BY tm.model_id
        """,
        [session_id],
    )


def session_models(connection: Any, session_id: str) -> list[dict[str, Any]]:
    """Every trainable model that ran a hook in this session, with its batch watermark."""
    return rows(
        connection,
        """
        WITH session_models AS (
            SELECT DISTINCT model_id FROM model_round_hooks WHERE session_id = %s
        ), batches AS (
            SELECT model_id, max(batch_seq) AS latest_batch_seq
            FROM training_batches
            WHERE model_id IN (SELECT model_id FROM session_models)
            GROUP BY model_id
        )
        SELECT tm.model_id, tm.active_checkpoint_id, tm.min_new_batches,
               tm.last_attempted_batch_seq, coalesce(b.latest_batch_seq, 0) AS latest_batch_seq
        FROM session_models sm JOIN training_models tm USING (model_id)
        LEFT JOIN batches b USING (model_id) ORDER BY tm.model_id
        """,
        [session_id],
    )
