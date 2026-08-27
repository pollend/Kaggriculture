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

from kaggriculture_lab import league_queries as lq

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
            "trend and deltas, latest leaders, hook evaluations, and models used by the session. "
            "`family_trend` is per-generation money and behaviour split by family -- read the "
            "learner (`*_match_policy`) on its own, never pooled with the seat it played -- and "
            "`opponent_pool` names the weights artifact that filled each seat, which is what "
            "makes a cross-session money comparison valid or not."
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
        with self.repository.connect() as connection:
            # Resolve the page first, then roll up only those sessions.
            sessions = lq.sessions(connection, limit=limit, league=league, status=status)
            session_ids = [str(row["session_id"]) for row in sessions]
            rollups = {
                str(row["session_id"]): row
                for row in lq.session_rollups(connection, session_ids)
            }
            latest = {
                str(row["session_id"]): row
                for row in lq.session_latest_generations(connection, session_ids)
            }
            leaders = {
                str(row["session_id"]): row
                for row in lq.session_leaders(connection, session_ids)
            }
        listed = []
        for session in sessions:
            session_id = str(session["session_id"])
            rollup = rollups.get(session_id, {})
            generation = latest.get(session_id, {})
            leader = leaders.get(session_id, {})
            total = int(rollup.get("pairings") or 0)
            complete = int(rollup.get("completed_pairings") or 0)
            listed.append({
                "session_id": session_id,
                "league": session["league"],
                "name": session["name"],
                "status": session["status"],
                "seed": session["seed"],
                "created_at": session["created_at"],
                "generations": int(rollup.get("generations") or 0),
                "latest_generation_id": generation.get("generation_id"),
                "latest_generation_index": generation.get("generation_index"),
                "latest_generation_status": generation.get("status"),
                "pairings": total,
                "completed_pairings": complete,
                "failed_pairings": int(rollup.get("failed_pairings") or 0),
                "episodes": int(rollup.get("episodes") or 0),
                "failed_episodes": int(rollup.get("failed_episodes") or 0),
                "completed_rounds": int(rollup.get("completed_rounds") or 0),
                "leader_revision_id": leader.get("revision_id"),
                "leader_name": leader.get("display_name"),
                "leader_conservative": leader.get("conservative"),
                "pairing_completion_fraction": complete / total if total else None,
            })
        return {"sessions": listed, "count": len(listed)}

    def _get_session_progress(self, arguments: dict[str, Any]) -> dict[str, Any]:
        session_id = _required_text(arguments, "session_id")
        generation_limit = _bounded_int(arguments, "generation_limit", 10, 100)
        leader_limit = _bounded_int(arguments, "leader_limit", 10, 50)
        with self.repository.connect() as connection:
            session = lq.session(connection, session_id)
            if session is None:
                raise LookupError(f"session not found: {session_id}")
            session["config"] = _json_object(session.pop("config_json"))
            generations = lq.session_generations(
                connection, session_id, limit=generation_limit
            )
            trend = lq.rating_trend(connection, session_id)
            latest_generation_id = generations[0]["generation_id"] if generations else None
            leaders = []
            if latest_generation_id is not None:
                leaders = lq.generation_leaderboard(
                    connection, str(latest_generation_id), limit=leader_limit
                )
            evaluations = lq.evaluation_rollup(connection, session_id=session_id)
            models = lq.session_models(connection, session_id)
            family_trend = lq.session_family_trend(connection, session_id)
            opponent_pool = lq.session_opponent_pool(connection, session_id)
        for generation in generations:
            pairings = int(generation["pairings"] or 0)
            complete = int(generation["completed_pairings"] or 0)
            generation["pairing_completion_fraction"] = complete / pairings if pairings else None
        for model in models:
            model["new_batches"] = int(model["latest_batch_seq"]) - int(model["last_attempted_batch_seq"])
        rating_delta = None
        median_delta = None
        if len(trend) >= 2:
            last, previous = trend[-1], trend[-2]
            if last["champion_conservative"] is not None and previous["champion_conservative"] is not None:
                rating_delta = float(last["champion_conservative"]) - float(previous["champion_conservative"])
            if last["median_conservative"] is not None and previous["median_conservative"] is not None:
                median_delta = float(last["median_conservative"]) - float(previous["median_conservative"])
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
            "family_trend": family_trend,
            "opponent_pool": opponent_pool,
            "episode_evaluations": evaluations,
            "models": models,
        }

    def _get_generation_progress(self, arguments: dict[str, Any]) -> dict[str, Any]:
        generation_id = _required_text(arguments, "generation_id")
        leader_limit = _bounded_int(arguments, "leader_limit", 10, 50)
        with self.repository.connect() as connection:
            generation = lq.generation(connection, generation_id)
            if generation is None:
                raise LookupError(f"generation not found: {generation_id}")
            config = _json_object(generation.pop("config_json"))
            work = lq.generation_work(connection, generation_id)
            outcomes = lq.generation_outcomes(connection, generation_id)
            seats = lq.seat_balance(connection, generation_id)
            hooks = lq.generation_hook_rollup(connection, generation_id)
            evaluations = lq.evaluation_rollup_by_evaluator(connection, generation_id)
            leaders = lq.generation_leaderboard(connection, generation_id, limit=leader_limit)
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
            model = lq.model(connection, model_id)
            if model is None:
                raise LookupError(f"model not found: {model_id}")
            model["config"] = _json_object(model.pop("config_json"))
            model["new_batches"] = int(model["latest_batch_seq"]) - int(model["last_attempted_batch_seq"])
            model["update_eligible"] = model["new_batches"] >= int(model["min_new_batches"])
            active_checkpoint = None
            if model["active_checkpoint_id"]:
                active_checkpoint = lq.checkpoint(connection, str(model["active_checkpoint_id"]))
                if active_checkpoint:
                    active_checkpoint.pop("model_id", None)
                    active_checkpoint["metrics"] = _json_object(active_checkpoint.pop("metrics_json"))
                    active_checkpoint["gate"] = _json_object(active_checkpoint.pop("gate_json"))
            batches = lq.model_batches(connection, model_id, limit=limit)
            for batch in batches:
                batch["metadata"] = _json_object(batch.pop("metadata_json"))
            updates = lq.model_updates(connection, model_id, limit=limit)
            for update in updates:
                update["metrics"] = _json_object(update.pop("metrics_json"))
                update["gate"] = _json_object(update.pop("gate_json"))
            checkpoints = lq.model_checkpoints(connection, model_id, limit=limit)
            for checkpoint in checkpoints:
                checkpoint["metrics"] = _json_object(checkpoint.pop("metrics_json"))
                checkpoint["gate"] = _json_object(checkpoint.pop("gate_json"))
            evaluations = lq.model_evaluations(connection, model_id, limit=limit)
            for evaluation in evaluations:
                evaluation["metrics"] = _json_object(evaluation.pop("metrics_json"))
            hook_status = lq.model_hook_status(connection, model_id)
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
