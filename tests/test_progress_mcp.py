from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

from kaggriculture_lab.dashboard import DashboardRepository, make_server
from kaggriculture_lab.progress_mcp import PROTOCOL_VERSION, ProgressMCP
from kaggriculture_lab.storage import ExperimentStore
from tests.postgres_support import postgres_database, seed_reference_rows


def _seed_progress(database: Path) -> None:
    with ExperimentStore(database) as store:
        connection = store.connection
        connection.execute(
            """
            INSERT INTO training_sessions (session_id, league, name, status, seed, config_json)
            VALUES ('session', 'progress', 'Progress league', 'RUNNING', 17,
                    '{"rounds_per_generation": 4}')
            """
        )
        connection.execute(
            """
            INSERT INTO generations (
                generation_id, session_id, generation_index, status, created_at, completed_at
            ) VALUES
                ('g0', 'session', 0, 'COMPLETE', now(), now()),
                ('g1', 'session', 1, 'RUNNING', now(), NULL)
            """
        )
        seed_reference_rows(connection, run_id="session", family_id="family")
        connection.execute(
            """
            INSERT INTO strategies VALUES
                ('strategy-0', '{}', 'test', NULL),
                ('strategy-1', '{}', 'test', NULL)
            """
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
        for generation_id in ("g0", "g1"):
            connection.execute(
                "INSERT INTO generation_members VALUES (%s, 'r0', 'population', 1500, 200, 0.06), "
                "(%s, 'r1', 'population', 1500, 200, 0.06)",
                [generation_id, generation_id],
            )
        connection.execute(
            """
            INSERT INTO rating_snapshots VALUES
                ('g0', 'r0', 1500, 100, 0.06, 2, false),
                ('g0', 'r1', 1400, 100, 0.06, 2, false),
                ('g1', 'r0', 1600, 80, 0.06, 2, false),
                ('g1', 'r1', 1450, 90, 0.06, 2, false)
            """
        )
        connection.execute(
            """
            INSERT INTO generation_stats VALUES
                ('g0', 'r0', 1, 1, 0, 0, 100, 100, 100, 100, 1),
                ('g0', 'r1', 1, 0, 0, 1, -100, -100, -100, -100, 1),
                ('g1', 'r0', 1, 1, 0, 0, 200, 200, 200, 200, 1),
                ('g1', 'r1', 1, 0, 0, 1, -200, -200, -200, -200, 1)
            """
        )
        connection.execute(
            "INSERT INTO league_pairings VALUES ('pair', 'g1', 0, 'population', 17, 'r0', 'r1', 'COMPLETE')"
        )
        connection.execute(
            """
            INSERT INTO episodes (
                match_id, run_id, stage, seed, seat0_id, seat1_id, seat0_spec,
                seat1_spec, steps, wallclock, failed
            ) VALUES ('match', 'session', 'league:g1', 17, 'r0', 'r1', 'a', 'b', 720, 1.5, false)
            """
        )
        connection.execute("INSERT INTO pairing_episodes VALUES ('pair', 0, 'match')")
        connection.execute("INSERT INTO pairing_results VALUES ('pair', 200, 1, false, 1, '{}')")
        connection.execute(
            """
            INSERT INTO episode_players (
                match_id, run_id, stage, seed, seat, strategy_id, opponent_id,
                status, failed, reward, final_money, opponent_money,
                money_difference, win, productive_action_percentage,
                movement_percentage, idle_percentage, final_shed_items,
                final_carried_items, metrics_json
            ) VALUES
                ('match', 'session', 'league:g1', 17, 0, 'r0', 'r1', 'DONE', false,
                 3200, 3200, 3000, 200, 1, 0.6, 0.3, 0.1, 2, 1, '{}'),
                ('match', 'session', 'league:g1', 17, 1, 'r1', 'r0', 'DONE', false,
                 3000, 3000, 3200, -200, 0, 0.5, 0.3, 0.2, 3, 2, '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO training_models (
                model_id, family_id, trainer_path, min_new_batches, config_json,
                last_attempted_batch_seq
            ) VALUES ('model', 'family', 'agent.train:trainer', 2, '{"trees": 10}', 1)
            """
        )
        connection.execute(
            """
            INSERT INTO training_batches (
                model_id, batch_seq, batch_id, source_kind, source_id,
                sample_count, metadata_json
            ) VALUES
                ('model', 1, 'batch-1', 'league_generation', 'g0', 2, '{}'),
                ('model', 2, 'batch-2', 'league_generation', 'g1', 2, '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO model_updates (
                update_id, model_id, cutoff_batch_seq, new_batch_count,
                dataset_sha256, status, metrics_json, gate_json, completed_at
            ) VALUES ('update', 'model', 1, 1, 'data', 'COMPLETE',
                      '{"mae": 1.5}', '{"passed": true}', now())
            """
        )
        connection.execute(
            """
            INSERT INTO model_training_checkpoints VALUES (
                'checkpoint', 'model', 'update', NULL, 'trainer.joblib', 'model.joblib',
                'artifact', 'data', '{"mae": 1.5}', '{"passed": true}', true, now()
            )
            """
        )
        # The checkpoint cites its update, so both citations back to it have to
        # wait until the checkpoint row exists.
        connection.execute(
            "UPDATE training_models SET active_checkpoint_id = 'checkpoint' WHERE model_id = 'model'"
        )
        connection.execute(
            "UPDATE model_updates SET candidate_checkpoint_id = 'checkpoint' WHERE update_id = 'update'"
        )
        connection.execute(
            """
            INSERT INTO model_evaluations (
                evaluation_id, update_id, model_id, evaluation_kind,
                metrics_json, passed
            ) VALUES ('model-eval', 'update', 'model', 'held_out', '{"mae": 1.5}', true)
            """
        )
        connection.execute(
            """
            INSERT INTO model_round_hooks (
                hook_id, model_id, session_id, generation_id, scope,
                round_index, status, batches_recorded
            ) VALUES
                ('hook-round', 'model', 'session', 'g1', 'round', 0, 'COMPLETE', 0),
                ('hook-generation', 'model', 'session', 'g1', 'generation', -1, 'COMPLETE', 1)
            """
        )


@pytest.fixture
def progress_database(tmp_path: Path) -> Path:
    database = postgres_database(tmp_path, "progress")
    _seed_progress(database)
    return database


def _tool_call(backend: ProgressMCP, name: str, arguments: dict) -> dict:
    response = backend.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    assert response is not None
    return response["result"]


def test_progress_tools_summarize_sessions_generations_and_models(progress_database: Path):
    backend = ProgressMCP(DashboardRepository(progress_database))

    sessions = _tool_call(backend, "list_training_sessions", {})["structuredContent"]
    assert sessions["sessions"][0]["session_id"] == "session"
    assert sessions["sessions"][0]["league"] == "progress"
    assert sessions["sessions"][0]["pairing_completion_fraction"] == 1
    filtered = _tool_call(
        backend, "list_training_sessions", {"league": "missing"}
    )["structuredContent"]
    assert filtered == {"sessions": [], "count": 0}

    session = _tool_call(backend, "get_session_progress", {"session_id": "session"})[
        "structuredContent"
    ]
    assert session["latest_vs_previous"]["champion_conservative_delta"] == 140
    assert session["recent_generations"][0]["generation_id"] == "g1"
    assert session["models"][0]["new_batches"] == 1

    generation = _tool_call(backend, "get_generation_progress", {"generation_id": "g1"})[
        "structuredContent"
    ]
    assert generation["work"]["episodes"] == 1
    assert generation["leaders"][0]["revision_id"] == "r0"
    assert generation["outcomes"]["mean_paired_margin"] == 200

    model = _tool_call(backend, "get_model_progress", {"model_id": "model"})[
        "structuredContent"
    ]
    assert model["model"]["new_batches"] == 1
    assert model["active_checkpoint"]["gate"] == {"passed": True}
    assert model["recent_evaluations"][0]["passed"] is True


def test_query_database_accepts_parameterized_selects_and_caps_rows(progress_database: Path):
    backend = ProgressMCP(DashboardRepository(progress_database))
    result = _tool_call(
        backend,
        "query_database",
        {
            "sql": "SELECT generation_id, status FROM generations WHERE session_id = %s ORDER BY generation_index",
            "parameters": ["session"],
            "max_rows": 1,
        },
    )
    assert result["isError"] is False
    assert result["structuredContent"] == {
        "columns": ["generation_id", "status"],
        "rows": [["g0", "COMPLETE"]],
        "row_count": 1,
        "truncated": True,
        "max_rows": 1,
    }


def test_query_database_writes_are_permitted(progress_database: Path):
    backend = ProgressMCP(DashboardRepository(progress_database))
    updated = _tool_call(
        backend,
        "query_database",
        {
            "sql": "UPDATE generations SET status = %s WHERE generation_id = %s",
            "parameters": ["ARCHIVED", "g0"],
        },
    )
    assert updated["isError"] is False
    assert updated["structuredContent"]["affected_rows"] == 1
    assert updated["structuredContent"]["rows"] == []

    after = _tool_call(
        backend,
        "query_database",
        {"sql": "SELECT status FROM generations WHERE generation_id = %s", "parameters": ["g0"]},
    )
    assert after["structuredContent"]["rows"] == [["ARCHIVED"]]


@pytest.fixture
def mcp_dashboard(progress_database: Path):
    server = make_server(progress_database, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _post_mcp(server, message: dict, **headers: str) -> tuple[int, dict[str, str], bytes]:
    host, port = server.server_address[:2]
    body = json.dumps(message).encode()
    connection = HTTPConnection(host, port, timeout=3)
    connection.request(
        "POST",
        "/mcp",
        body=body,
        headers={"Content-Type": "application/json", **headers},
    )
    response = connection.getresponse()
    payload = response.read()
    result = response.status, dict(response.headers), payload
    connection.close()
    return result


def test_dashboard_exposes_stateless_mcp_json_transport(mcp_dashboard):
    status, headers, body = _post_mcp(
        mcp_dashboard,
        {
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
    )
    assert status == 200
    assert headers["MCP-Protocol-Version"] == PROTOCOL_VERSION
    initialized = json.loads(body)
    assert initialized["result"]["capabilities"] == {"tools": {"listChanged": False}}

    status, _, body = _post_mcp(
        mcp_dashboard,
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        **{"MCP-Protocol-Version": PROTOCOL_VERSION},
    )
    assert status == 200
    names = {tool["name"] for tool in json.loads(body)["result"]["tools"]}
    assert names == {
        "list_training_sessions",
        "get_session_progress",
        "get_generation_progress",
        "get_model_progress",
        "query_database",
    }

    status, _, body = _post_mcp(
        mcp_dashboard,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        **{"MCP-Protocol-Version": PROTOCOL_VERSION},
    )
    assert status == 202
    assert body == b""


def test_mcp_transport_rejects_cross_origin_and_get(mcp_dashboard):
    host, port = mcp_dashboard.server_address[:2]
    connection = HTTPConnection(host, port, timeout=3)
    connection.request("GET", "/mcp")
    response = connection.getresponse()
    response.read()
    assert response.status == 405
    assert response.headers["Allow"] == "POST"
    connection.close()

    status, _, body = _post_mcp(
        mcp_dashboard,
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        Origin="https://evil.example",
    )
    assert status == 403
    assert json.loads(body)["error"]["message"] == "Invalid Origin"
