"""MCP tools for evaluating persistent league progress.

The dashboard owns the HTTP transport. This module owns protocol dispatch and
the PostgreSQL queries, which keeps model-facing data independent from the
HTML presentation layer.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

PROTOCOL_VERSION = "2025-11-25"
SUPPORTED_PROTOCOL_VERSIONS = (PROTOCOL_VERSION, "2025-06-18", "2025-03-26")


class MCPError(ValueError):
    """JSON-RPC/MCP request error with a protocol error code."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


def _plain(value: Any) -> Any:
    """Convert PostgreSQL/Python values into JSON-compatible values."""

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _rows(connection: Any, sql: str, parameters: list[Any] | None = None) -> list[dict[str, Any]]:
    cursor = connection.execute(sql, parameters or [])
    names = [item[0] for item in cursor.description]
    return [_plain(dict(zip(names, row))) for row in cursor.fetchall()]


def _row(connection: Any, sql: str, parameters: list[Any] | None = None) -> dict[str, Any] | None:
    values = _rows(connection, sql, parameters)
    return values[0] if values else None


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return _plain(value)
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {"unparsed": str(value)}
    return _plain(parsed) if isinstance(parsed, dict) else {"value": _plain(parsed)}


def _required_text(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > 256 or "\x00" in value:
        raise ValueError(f"{name} is invalid")
    return value


def _bounded_int(arguments: dict[str, Any], name: str, default: int, maximum: int) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return value


READ_ONLY_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

MUTATING_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": True,
    "idempotentHint": False,
    "openWorldHint": False,
}

TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "list_training_sessions",
        "title": "List training sessions",
        "description": (
            "List persistent league sessions with generation, pairing, episode, failure, "
            "and latest-leader progress. Use this first when the session ID is unknown."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Optional exact session status, such as RUNNING or COMPLETE.",
                    "maxLength": 40,
                },
                "league": {
                    "type": "string",
                    "description": "Optional exact league name.",
                    "minLength": 1,
                    "maxLength": 100,
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
            },
            "additionalProperties": False,
        },
        "annotations": READ_ONLY_ANNOTATIONS,
    },
    {
        "name": "get_session_progress",
        "title": "Get session progress",
        "description": (
            "Summarize one training session, including recent generation completion, rating "
            "trend and deltas, latest leaders, hook evaluations, and models used by the session."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "minLength": 1, "maxLength": 256},
                "generation_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                },
                "leader_limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
        "annotations": READ_ONLY_ANNOTATIONS,
    },
    {
        "name": "get_generation_progress",
        "title": "Get generation progress",
        "description": (
            "Inspect one generation's work completion, outcomes, seat balance, ranked agents, "
            "trainer hooks, post-episode evaluations, failures, and runtime."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "generation_id": {"type": "string", "minLength": 1, "maxLength": 256},
                "leader_limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": ["generation_id"],
            "additionalProperties": False,
        },
        "annotations": READ_ONLY_ANNOTATIONS,
    },
    {
        "name": "get_model_progress",
        "title": "Get model progress",
        "description": (
            "Inspect one training model's batch watermark, active checkpoint, recent update "
            "gates and evaluations, checkpoint history, and session hook status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "minLength": 1, "maxLength": 256},
                "history_limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": ["model_id"],
            "additionalProperties": False,
        },
        "annotations": READ_ONLY_ANNOTATIONS,
    },
    {
        "name": "query_database",
        "title": "Query or modify the progress database",
        "description": (
            "Run parameterized SQL against the shared PostgreSQL database. Reads, writes, "
            "and DDL are all permitted and commit immediately, so mutate deliberately. "
            "Result sets are row-capped by max_rows; statements without a result set "
            "report the number of affected rows instead. "
            "Use information_schema.tables and information_schema.columns to discover the schema."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "minLength": 1, "maxLength": 20000},
                "parameters": {
                    "type": "array",
                    "maxItems": 100,
                    "items": {"type": ["string", "number", "integer", "boolean", "null"]},
                    "default": [],
                },
                "max_rows": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 200},
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
        "annotations": MUTATING_ANNOTATIONS,
    },
)


class ProgressMCP:
    """Stateless MCP dispatcher backed by a dashboard repository."""

    def __init__(self, repository: Any):
        self.repository = repository
        self._tool_handlers = {
            "list_training_sessions": self._list_training_sessions,
            "get_session_progress": self._get_session_progress,
            "get_generation_progress": self._get_generation_progress,
            "get_model_progress": self._get_model_progress,
            "query_database": self._query_database,
        }

    def handle(self, message: Any) -> dict[str, Any] | None:
        """Handle one JSON-RPC message; notifications intentionally return no body."""

        request_id: Any = None
        has_id = isinstance(message, dict) and "id" in message
        if isinstance(message, dict):
            request_id = message.get("id")
        try:
            if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                raise MCPError(-32600, "Invalid Request")
            method = message.get("method")
            if not isinstance(method, str) or not method:
                raise MCPError(-32600, "Invalid Request")
            params = message.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise MCPError(-32602, "Invalid params")

            if not has_id:
                # The server has no notification-producing capabilities. Unknown
                # notifications are ignored as required by JSON-RPC.
                return None
            if method == "initialize":
                requested = params.get("protocolVersion")
                negotiated = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else PROTOCOL_VERSION
                result = {
                    "protocolVersion": negotiated,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": "kaggriculture-progress",
                        "title": "Kaggriculture League Progress",
                        "version": "0.1.0",
                    },
                    "instructions": (
                        "Use list_training_sessions first, then inspect a session, generation, "
                        "or model. The inspection tools are bounded and read-only; "
                        "query_database runs arbitrary SQL, including writes and DDL."
                    ),
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                if params.get("cursor") not in (None, ""):
                    raise MCPError(-32602, "This server's tool list is not paginated")
                result = {"tools": list(TOOLS)}
            elif method == "tools/call":
                result = self._call_tool(params)
            else:
                raise MCPError(-32601, "Method not found", {"method": method})
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except MCPError as error:
            payload: dict[str, Any] = {"code": error.code, "message": str(error)}
            if error.data is not None:
                payload["data"] = _plain(error.data)
            return {"jsonrpc": "2.0", "id": request_id, "error": payload}

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or name not in self._tool_handlers:
            raise MCPError(-32602, "Unknown tool", {"name": name})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise MCPError(-32602, "Tool arguments must be an object")
        declared = next(tool for tool in TOOLS if tool["name"] == name)
        allowed = set(declared["inputSchema"].get("properties", {}))
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise MCPError(-32602, "Unknown tool arguments", {"arguments": unknown})
        try:
            value = _plain(self._tool_handlers[name](arguments))
        except Exception as error:
            problem = {"error": type(error).__name__, "message": str(error)}
            return {
                "content": [{"type": "text", "text": json.dumps(problem, sort_keys=True)}],
                "structuredContent": problem,
                "isError": True,
            }
        text = json.dumps(value, indent=2, sort_keys=True)
        return {
            "content": [{"type": "text", "text": text}],
            "structuredContent": value,
            "isError": False,
        }

    def _list_training_sessions(self, arguments: dict[str, Any]) -> dict[str, Any]:
        limit = _bounded_int(arguments, "limit", 20, 100)
        status = arguments.get("status")
        if status is not None and (not isinstance(status, str) or not status or len(status) > 40):
            raise ValueError("status must be a non-empty string of at most 40 characters")
        league = arguments.get("league")
        if league is not None and (not isinstance(league, str) or not league or len(league) > 100):
            raise ValueError("league must be a non-empty string of at most 100 characters")
        conditions: list[str] = []
        parameters: list[Any] = []
        if status:
            conditions.append("s.status = %s")
            parameters.append(status)
        if league:
            conditions.append("s.league = %s")
            parameters.append(league)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        parameters.append(limit)
        with self.repository.connect() as connection:
            sessions = _rows(
                connection,
                f"""
                WITH generation_rollup AS (
                    SELECT session_id, count(*) AS generations,
                           max(generation_index) AS latest_generation_index
                    FROM generations GROUP BY session_id
                ), pairing_rollup AS (
                    SELECT g.session_id, count(*) AS pairings,
                           count(*) FILTER (WHERE lp.status = 'COMPLETE') AS completed_pairings
                    FROM generations g JOIN league_pairings lp USING (generation_id)
                    GROUP BY g.session_id
                ), episode_rollup AS (
                    SELECT g.session_id, count(DISTINCT pe.match_id) AS episodes,
                           count(DISTINCT pe.match_id) FILTER (WHERE e.failed) AS failed_episodes
                    FROM generations g JOIN league_pairings lp USING (generation_id)
                    JOIN pairing_episodes pe USING (pairing_id)
                    JOIN episodes e USING (match_id) GROUP BY g.session_id
                ), latest_generation AS (
                    SELECT session_id, generation_id,
                           generation_index AS latest_generation_index, status,
                           row_number() OVER (
                               PARTITION BY session_id ORDER BY generation_index DESC
                           ) AS position
                    FROM generations
                ), latest_leader AS (
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
                    WHERE NOT rs.is_frozen
                )
                SELECT s.session_id, s.league, s.name, s.status, s.seed, s.created_at,
                       coalesce(gr.generations, 0) AS generations,
                       lg.generation_id AS latest_generation_id,
                       lg.latest_generation_index, lg.status AS latest_generation_status,
                       coalesce(pr.pairings, 0) AS pairings,
                       coalesce(pr.completed_pairings, 0) AS completed_pairings,
                       coalesce(er.episodes, 0) AS episodes,
                       coalesce(er.failed_episodes, 0) AS failed_episodes,
                       ll.revision_id AS leader_revision_id,
                       ll.display_name AS leader_name,
                       ll.conservative AS leader_conservative
                FROM training_sessions s
                LEFT JOIN generation_rollup gr ON gr.session_id = s.session_id
                LEFT JOIN latest_generation lg
                       ON lg.session_id = s.session_id AND lg.position = 1
                LEFT JOIN pairing_rollup pr ON pr.session_id = s.session_id
                LEFT JOIN episode_rollup er ON er.session_id = s.session_id
                LEFT JOIN latest_leader ll
                       ON ll.session_id = s.session_id AND ll.position = 1
                {where}
                ORDER BY s.created_at DESC, s.session_id LIMIT %s
                """,
                parameters,
            )
        for session in sessions:
            total = int(session["pairings"] or 0)
            complete = int(session["completed_pairings"] or 0)
            session["pairing_completion_fraction"] = complete / total if total else None
        return {"sessions": sessions, "count": len(sessions)}

    def _get_session_progress(self, arguments: dict[str, Any]) -> dict[str, Any]:
        session_id = _required_text(arguments, "session_id")
        generation_limit = _bounded_int(arguments, "generation_limit", 10, 100)
        leader_limit = _bounded_int(arguments, "leader_limit", 10, 50)
        with self.repository.connect() as connection:
            session = _row(
                connection,
                """
                SELECT session_id, league, name, status, seed, config_json, created_at
                FROM training_sessions WHERE session_id = %s
                """,
                [session_id],
            )
            if session is None:
                raise LookupError(f"session not found: {session_id}")
            session["config"] = _json_object(session.pop("config_json"))
            generations = _rows(
                connection,
                """
                WITH members AS (
                    SELECT generation_id, count(*) AS members
                    FROM generation_members GROUP BY generation_id
                ), pairings AS (
                    SELECT generation_id, count(*) AS pairings,
                           count(*) FILTER (WHERE status = 'COMPLETE') AS completed_pairings,
                           count(*) FILTER (WHERE status = 'ERROR') AS pairing_errors
                    FROM league_pairings GROUP BY generation_id
                ), episode_counts AS (
                    SELECT lp.generation_id, count(DISTINCT pe.match_id) AS episodes,
                           count(DISTINCT pe.match_id) FILTER (WHERE e.failed) AS failed_episodes,
                           sum(e.wallclock) AS wallclock_seconds
                    FROM league_pairings lp JOIN pairing_episodes pe USING (pairing_id)
                    JOIN episodes e USING (match_id) GROUP BY lp.generation_id
                ), rounds AS (
                    SELECT generation_id,
                           count(DISTINCT round_index) FILTER (
                               WHERE scope = 'round' AND status = 'COMPLETE'
                           ) AS completed_rounds,
                           count(*) FILTER (WHERE status = 'ERROR') AS hook_errors
                    FROM model_round_hooks GROUP BY generation_id
                )
                SELECT g.generation_id, g.generation_index, g.status, g.created_at,
                       g.completed_at, coalesce(m.members, 0) AS members,
                       coalesce(p.pairings, 0) AS pairings,
                       coalesce(p.completed_pairings, 0) AS completed_pairings,
                       coalesce(p.pairing_errors, 0) AS pairing_errors,
                       coalesce(ec.episodes, 0) AS episodes,
                       coalesce(ec.failed_episodes, 0) AS failed_episodes,
                       coalesce(ec.wallclock_seconds, 0) AS wallclock_seconds,
                       coalesce(r.completed_rounds, 0) AS completed_rounds,
                       coalesce(r.hook_errors, 0) AS hook_errors
                FROM generations g LEFT JOIN members m USING (generation_id)
                LEFT JOIN pairings p USING (generation_id)
                LEFT JOIN episode_counts ec USING (generation_id)
                LEFT JOIN rounds r USING (generation_id)
                WHERE g.session_id = %s ORDER BY g.generation_index DESC LIMIT %s
                """,
                [session_id, generation_limit],
            )
            trend = _rows(
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
            latest_generation_id = generations[0]["generation_id"] if generations else None
            leaders = []
            if latest_generation_id is not None:
                leaders = self._generation_leaders(connection, str(latest_generation_id), leader_limit)
            evaluations = _row(
                connection,
                """
                SELECT count(*) AS evaluations,
                       count(*) FILTER (WHERE ee.status = 'ERROR') AS errors,
                       count(*) FILTER (
                           WHERE ee.status = 'COMPLETE'
                             AND coalesce(CAST(ee.result_json ->> 'should_reevaluate' AS BOOLEAN), false)
                       ) AS reevaluation_requests
                FROM episode_evaluations ee JOIN pairing_episodes pe USING (match_id)
                JOIN league_pairings lp USING (pairing_id)
                JOIN generations g USING (generation_id) WHERE g.session_id = %s
                """,
                [session_id],
            ) or {"evaluations": 0, "errors": 0, "reevaluation_requests": 0}
            models = _rows(
                connection,
                """
                WITH session_models AS (
                    SELECT DISTINCT model_id FROM model_round_hooks WHERE session_id = %s
                ), batches AS (
                    SELECT model_id, max(batch_seq) AS latest_batch_seq
                    FROM training_batches GROUP BY model_id
                )
                SELECT tm.model_id, tm.active_checkpoint_id, tm.min_new_batches,
                       tm.last_attempted_batch_seq, coalesce(b.latest_batch_seq, 0) AS latest_batch_seq
                FROM session_models sm JOIN training_models tm USING (model_id)
                LEFT JOIN batches b USING (model_id) ORDER BY tm.model_id
                """,
                [session_id],
            )
        for generation in generations:
            pairings = int(generation["pairings"] or 0)
            complete = int(generation["completed_pairings"] or 0)
            generation["pairing_completion_fraction"] = complete / pairings if pairings else None
        for model in models:
            model["new_batches"] = int(model["latest_batch_seq"]) - int(model["last_attempted_batch_seq"])
        rating_delta = None
        median_delta = None
        if len(trend) >= 2:
            rating_delta = float(trend[-1]["champion_conservative"]) - float(trend[-2]["champion_conservative"])
            median_delta = float(trend[-1]["median_conservative"]) - float(trend[-2]["median_conservative"])
        totals = {
            "generations_returned": len(generations),
            "pairings": sum(int(row["pairings"] or 0) for row in generations),
            "completed_pairings": sum(int(row["completed_pairings"] or 0) for row in generations),
            "episodes": sum(int(row["episodes"] or 0) for row in generations),
            "failed_episodes": sum(int(row["failed_episodes"] or 0) for row in generations),
        }
        return {
            "session": session,
            "totals_for_returned_generations": totals,
            "recent_generations": generations,
            "rating_trend": trend,
            "latest_vs_previous": {
                "champion_conservative_delta": rating_delta,
                "median_conservative_delta": median_delta,
            },
            "latest_leaders": leaders,
            "episode_evaluations": evaluations,
            "models": models,
        }

    def _generation_leaders(self, connection: Any, generation_id: str, limit: int) -> list[dict[str, Any]]:
        return _rows(
            connection,
            """
            WITH episode_rollup AS (
                SELECT ep.strategy_id AS revision_id, count(*) AS episodes,
                       avg(ep.final_money) AS mean_final_money,
                       avg(ep.money_difference) AS mean_episode_margin,
                       avg(ep.win) AS win_rate,
                       avg(ep.productive_action_percentage) AS productive_action_fraction,
                       avg(ep.final_shed_items + ep.final_carried_items) AS mean_unsold_items
                FROM league_pairings lp JOIN pairing_episodes pe USING (pairing_id)
                JOIN episode_players ep USING (match_id)
                WHERE lp.generation_id = %s AND NOT ep.failed GROUP BY ep.strategy_id
            )
            SELECT rs.revision_id, ar.strategy_id,
                   coalesce(af.name, ar.family_id, rs.revision_id) AS display_name,
                   rs.rating, rs.deviation, rs.rating - 2 * rs.deviation AS conservative,
                   rs.games AS rated_games, gs.games AS paired_games, gs.wins, gs.draws, gs.losses,
                   gs.mean_margin, gs.median_margin, er.episodes, er.mean_final_money,
                   er.mean_episode_margin, er.win_rate, er.productive_action_fraction,
                   er.mean_unsold_items
            FROM rating_snapshots rs
            LEFT JOIN generation_stats gs USING (generation_id, revision_id)
            LEFT JOIN agent_revisions ar USING (revision_id)
            LEFT JOIN agent_families af USING (family_id)
            LEFT JOIN episode_rollup er USING (revision_id)
            WHERE rs.generation_id = %s AND NOT rs.is_frozen
            ORDER BY conservative DESC, rs.revision_id LIMIT %s
            """,
            [generation_id, generation_id, limit],
        )

    def _get_generation_progress(self, arguments: dict[str, Any]) -> dict[str, Any]:
        generation_id = _required_text(arguments, "generation_id")
        leader_limit = _bounded_int(arguments, "leader_limit", 10, 50)
        with self.repository.connect() as connection:
            generation = _row(
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
            if generation is None:
                raise LookupError(f"generation not found: {generation_id}")
            config = _json_object(generation.pop("config_json"))
            work = _row(
                connection,
                """
                SELECT
                    (SELECT count(*) FROM generation_members WHERE generation_id = %s) AS members,
                    (SELECT count(*) FROM league_pairings WHERE generation_id = %s) AS pairings,
                    (SELECT count(*) FROM league_pairings WHERE generation_id = %s AND status = 'COMPLETE') AS completed_pairings,
                    (SELECT count(*) FROM league_pairings WHERE generation_id = %s AND status = 'ERROR') AS pairing_errors,
                    (SELECT count(DISTINCT pe.match_id) FROM league_pairings lp JOIN pairing_episodes pe USING (pairing_id) WHERE lp.generation_id = %s) AS episodes,
                    (SELECT count(DISTINCT pe.match_id) FROM league_pairings lp JOIN pairing_episodes pe USING (pairing_id) JOIN episodes e USING (match_id) WHERE lp.generation_id = %s AND e.failed) AS failed_episodes,
                    (SELECT coalesce(sum(e.wallclock), 0) FROM league_pairings lp JOIN pairing_episodes pe USING (pairing_id) JOIN episodes e USING (match_id) WHERE lp.generation_id = %s) AS wallclock_seconds,
                    (SELECT count(*) FROM (
                        SELECT round_index FROM model_round_hooks
                        WHERE generation_id = %s AND scope = 'round'
                        GROUP BY round_index HAVING bool_and(status = 'COMPLETE')
                    )) AS completed_rounds
                """,
                [generation_id] * 8,
            ) or {}
            outcomes = _row(
                connection,
                """
                SELECT count(*) FILTER (WHERE NOT failed) AS settled_pairings,
                       avg(paired_margin) FILTER (WHERE NOT failed) AS mean_paired_margin,
                       percentile_cont(0.5) WITHIN GROUP (ORDER BY paired_margin)
                           FILTER (WHERE NOT failed) AS median_paired_margin,
                       stddev_samp(paired_margin) FILTER (WHERE NOT failed) AS paired_margin_stddev,
                       count(*) FILTER (WHERE failed) AS failed_pairing_results
                FROM pairing_results pr JOIN league_pairings lp USING (pairing_id)
                WHERE lp.generation_id = %s
                """,
                [generation_id],
            ) or {}
            seats = _rows(
                connection,
                """
                SELECT ep.seat, count(*) AS episodes, avg(ep.final_money) AS mean_final_money,
                       avg(ep.money_difference) AS mean_margin, avg(ep.win) AS win_rate
                FROM league_pairings lp JOIN pairing_episodes pe USING (pairing_id)
                JOIN episode_players ep USING (match_id)
                WHERE lp.generation_id = %s AND NOT ep.failed
                GROUP BY ep.seat ORDER BY ep.seat
                """,
                [generation_id],
            )
            hooks = _rows(
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
            evaluations = _rows(
                connection,
                """
                SELECT ee.status, ee.evaluator_path, count(*) AS evaluations,
                       count(*) FILTER (
                           WHERE coalesce(CAST(ee.result_json ->> 'should_reevaluate' AS BOOLEAN), false)
                       ) AS reevaluation_requests
                FROM episode_evaluations ee JOIN pairing_episodes pe USING (match_id)
                JOIN league_pairings lp USING (pairing_id)
                WHERE lp.generation_id = %s GROUP BY ee.status, ee.evaluator_path
                ORDER BY ee.status, ee.evaluator_path
                """,
                [generation_id],
            )
            leaders = self._generation_leaders(connection, generation_id, leader_limit)
        pairings = int(work.get("pairings") or 0)
        completed = int(work.get("completed_pairings") or 0)
        rounds_planned = int(config.get("rounds_per_generation", 0) or 0)
        work["pairing_completion_fraction"] = completed / pairings if pairings else None
        work["rounds_planned"] = rounds_planned
        return {
            "generation": generation,
            "work": work,
            "outcomes": outcomes,
            "seat_balance": seats,
            "leaders": leaders,
            "trainer_hooks": hooks,
            "episode_evaluations": evaluations,
        }

    def _get_model_progress(self, arguments: dict[str, Any]) -> dict[str, Any]:
        model_id = _required_text(arguments, "model_id")
        limit = _bounded_int(arguments, "history_limit", 10, 50)
        with self.repository.connect() as connection:
            model = _row(
                connection,
                """
                SELECT tm.model_id, tm.family_id, tm.agent_type, tm.agent_name,
                       tm.declared_model_id, tm.trainer_path, tm.min_new_batches,
                       tm.config_json, tm.active_checkpoint_id, tm.last_attempted_batch_seq,
                       coalesce(tb.latest_batch_seq, 0) AS latest_batch_seq,
                       tm.created_at, tm.updated_at
                FROM training_models tm
                LEFT JOIN (
                    SELECT model_id, max(batch_seq) AS latest_batch_seq
                    FROM training_batches GROUP BY model_id
                ) tb USING (model_id)
                WHERE tm.model_id = %s
                """,
                [model_id],
            )
            if model is None:
                raise LookupError(f"model not found: {model_id}")
            model["config"] = _json_object(model.pop("config_json"))
            model["new_batches"] = int(model["latest_batch_seq"]) - int(model["last_attempted_batch_seq"])
            model["update_eligible"] = model["new_batches"] >= int(model["min_new_batches"])
            active_checkpoint = None
            if model["active_checkpoint_id"]:
                active_checkpoint = _row(
                    connection,
                    """
                    SELECT checkpoint_id, update_id, parent_checkpoint_id,
                           trainer_artifact_path, inference_artifact_path, artifact_sha256,
                           dataset_sha256, metrics_json, gate_json, promoted, created_at
                    FROM model_training_checkpoints WHERE checkpoint_id = %s
                    """,
                    [model["active_checkpoint_id"]],
                )
                if active_checkpoint:
                    active_checkpoint["metrics"] = _json_object(active_checkpoint.pop("metrics_json"))
                    active_checkpoint["gate"] = _json_object(active_checkpoint.pop("gate_json"))
            batches = _rows(
                connection,
                """
                SELECT batch_seq, batch_id, source_kind, source_id, sample_count,
                       metadata_json, created_at FROM training_batches
                WHERE model_id = %s ORDER BY batch_seq DESC LIMIT %s
                """,
                [model_id, limit],
            )
            for batch in batches:
                batch["metadata"] = _json_object(batch.pop("metadata_json"))
            updates = _rows(
                connection,
                """
                SELECT update_id, parent_checkpoint_id, cutoff_batch_seq, new_batch_count,
                       dataset_sha256, status, candidate_checkpoint_id, metrics_json,
                       gate_json, error_text, started_at, completed_at
                FROM model_updates WHERE model_id = %s
                ORDER BY started_at DESC LIMIT %s
                """,
                [model_id, limit],
            )
            for update in updates:
                update["metrics"] = _json_object(update.pop("metrics_json"))
                update["gate"] = _json_object(update.pop("gate_json"))
            checkpoints = _rows(
                connection,
                """
                SELECT checkpoint_id, update_id, parent_checkpoint_id, promoted,
                       metrics_json, gate_json, dataset_sha256, created_at
                FROM model_training_checkpoints WHERE model_id = %s
                ORDER BY created_at DESC LIMIT %s
                """,
                [model_id, limit],
            )
            for checkpoint in checkpoints:
                checkpoint["metrics"] = _json_object(checkpoint.pop("metrics_json"))
                checkpoint["gate"] = _json_object(checkpoint.pop("gate_json"))
            evaluations = _rows(
                connection,
                """
                SELECT me.evaluation_id, me.update_id, me.evaluation_kind,
                       me.opponent_checkpoint_id, me.metrics_json, me.passed, me.created_at
                FROM model_evaluations me WHERE me.model_id = %s
                ORDER BY me.created_at DESC LIMIT %s
                """,
                [model_id, limit],
            )
            for evaluation in evaluations:
                evaluation["metrics"] = _json_object(evaluation.pop("metrics_json"))
            hook_status = _rows(
                connection,
                """
                SELECT session_id, scope, status, count(*) AS hooks,
                       max(generation_id) AS latest_generation_id,
                       max(round_index) AS latest_round_index,
                       sum(batches_recorded) AS batches_recorded
                FROM model_round_hooks WHERE model_id = %s
                GROUP BY session_id, scope, status ORDER BY session_id, scope, status
                """,
                [model_id],
            )
        return {
            "model": model,
            "active_checkpoint": active_checkpoint,
            "recent_batches": batches,
            "recent_updates": updates,
            "recent_checkpoints": checkpoints,
            "recent_evaluations": evaluations,
            "hook_status": hook_status,
        }

    def _query_database(self, arguments: dict[str, Any]) -> dict[str, Any]:
        sql = arguments.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("sql must be a non-empty string")
        if len(sql) > 20_000:
            raise ValueError("sql must be at most 20000 characters")
        parameters = arguments.get("parameters", [])
        if not isinstance(parameters, list) or len(parameters) > 100:
            raise ValueError("parameters must be an array with at most 100 values")
        if any(not isinstance(value, (str, int, float, bool, type(None))) for value in parameters):
            raise ValueError("parameters may contain only JSON scalar values")
        max_rows = _bounded_int(arguments, "max_rows", 200, 1000)
        with self.repository.connect() as connection:
            cursor = connection.execute(sql, list(parameters))
            if cursor.description is None:
                # INSERT/UPDATE/DELETE/DDL: nothing to page through, so report the
                # write instead. Connections are autocommit, so it has already landed.
                return {
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "affected_rows": cursor.rowcount,
                    "truncated": False,
                    "max_rows": max_rows,
                }
            columns = [str(item[0]) for item in cursor.description]
            fetched = cursor.fetchmany(max_rows + 1)
        rows = fetched[:max_rows]
        return {
            "columns": columns,
            "rows": [_plain(row) for row in rows],
            "row_count": len(rows),
            "truncated": len(fetched) > max_rows,
            "max_rows": max_rows,
        }
