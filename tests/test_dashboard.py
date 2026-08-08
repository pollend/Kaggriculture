import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from kaggle_environments import make

from kaggriculture_lab.cli import build_parser
from kaggriculture_lab.dashboard import (
    DashboardNotFound,
    DashboardUnavailable,
    _episode_page,
    _load_replay,
    make_server,
)
from kaggriculture_lab.storage import ExperimentStore
from sim.runner import MatchResult, SeatResult


def _seed_dashboard(database: Path) -> dict:
    environment = make("kaggriculture", configuration={"episodeSteps": 3, "seed": 17})
    environment.run(["pass", "pass"])
    replay = environment.toJSON()
    result = MatchResult(
        seed=17,
        seats=[
            SeatResult(
                "alpha", 0, 3200, "DONE", False, None, None, 24, 0.01, 0.005, 0.002,
                3200, {"productive_actions": 12, "productive_action_percentage": 0.5},
            ),
            SeatResult(
                "beta", 1, 2800, "DONE", False, None, None, 24, 0.01, 0.005, 0.002,
                2800, {"productive_actions": 10, "productive_action_percentage": 0.4},
            ),
        ],
        steps=24,
        wallclock=0.25,
        replay=replay,
    )
    with ExperimentStore(database) as store:
        connection = store.connection
        connection.execute(
            "INSERT INTO training_sessions (session_id, name, status, seed, config_json) VALUES (?, ?, 'COMPLETE', 17, ?)",
            ["session/one", "Test <script> League", json.dumps({"rounds": 1})],
        )
        connection.execute(
            "INSERT INTO generations (generation_id, session_id, generation_index, status, created_at, completed_at) VALUES (?, ?, 0, 'COMPLETE', now(), now())",
            ["session/one:g0000", "session/one"],
        )
        connection.execute(
            "INSERT INTO agent_families (family_id, name, manifest_path, entry_point, parameter_schema_version) VALUES ('family', 'Farm Stars', 'agent.toml', 'agent:agent', 1)"
        )
        for revision, strategy in (("revision-a", "alpha"), ("revision-b", "beta")):
            connection.execute(
                """
                INSERT INTO agent_revisions (
                    revision_id, family_id, strategy_id, entry_point, agent_spec,
                    code_fingerprint, parameter_schema_version, params_json, source
                ) VALUES (?, 'family', ?, 'agent:agent', ?, 'fingerprint', 1, '{}', 'test')
                """,
                [revision, strategy, strategy],
            )
            connection.execute(
                "INSERT INTO generation_members VALUES (?, ?, 'population', 1500, 200, 0.06)",
                ["session/one:g0000", revision],
            )
        connection.execute(
            "INSERT INTO league_pairings VALUES ('pair-1', ?, 0, 'population', 17, 'revision-a', 'revision-b', 'COMPLETE')",
            ["session/one:g0000"],
        )
        store.add_match("match-1", "session/one", "league:g0000", ("alpha", "beta"), ("alpha", "beta"), result)
        store.add_replay_payload("match-1", replay)
        connection.execute("INSERT INTO pairing_episodes VALUES ('pair-1', 0, 'match-1')")
        reverse_result = MatchResult(
            seed=17,
            seats=[result.seats[1], result.seats[0]],
            steps=result.steps,
            wallclock=result.wallclock,
            replay=replay,
        )
        store.add_match("match-2", "session/one", "league:g0000", ("beta", "alpha"), ("beta", "alpha"), reverse_result)
        store.add_replay_payload("match-2", replay)
        connection.execute("INSERT INTO pairing_episodes VALUES ('pair-1', 1, 'match-2')")
        connection.execute("INSERT INTO pairing_results VALUES ('pair-1', 400, 1, false, 2, '{}')")
        connection.execute(
            "INSERT INTO rating_snapshots VALUES (?, 'revision-a', 1600, 80, 0.06, 1, false), (?, 'revision-b', 1400, 80, 0.06, 1, false)",
            ["session/one:g0000", "session/one:g0000"],
        )
        connection.execute(
            "INSERT INTO generation_stats VALUES (?, 'revision-a', 1, 1, 0, 0, 400, 400, 400, 400, 1), (?, 'revision-b', 1, 0, 0, 1, -400, -400, -400, -400, 1)",
            ["session/one:g0000", "session/one:g0000"],
        )
        connection.execute(
            """
            INSERT INTO episode_evaluations (
                evaluation_id, match_id, revision_id, seat, evaluator_path,
                evaluator_fingerprint, status, result_json
            ) VALUES ('eval-1', 'match-1', 'revision-a', 0, 'agent:evaluate', 'fp', 'SUCCESS', '{"score": 1}')
            """
        )
    return replay


@pytest.fixture
def dashboard(tmp_path: Path):
    database = tmp_path / "dashboard.duckdb"
    replay = _seed_dashboard(database)
    server = make_server(database, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}", replay
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _get(url: str) -> tuple[int, dict[str, str], bytes]:
    with urlopen(url) as response:
        return response.status, dict(response.headers), response.read()


def test_dashboard_drills_from_sessions_to_episode(dashboard):
    base, _ = dashboard
    status, headers, body = _get(base + "/")
    document = body.decode()
    assert status == 200
    assert "Test &lt;script&gt; League" in document
    assert "Test <script> League" not in document
    assert "session%2Fone" in document
    assert "Content-Security-Policy" in headers

    _, _, body = _get(base + "/sessions/session%2Fone")
    assert "Rating trend" in body.decode()
    assert "session%2Fone%3Ag0000" in body.decode()

    _, _, body = _get(base + "/generations/session%2Fone%3Ag0000?status=COMPLETE&kind=population")
    document = body.decode()
    assert "Farm Stars" in document
    assert "Agent A" in document
    assert "strategy alpha" in document
    assert "Agent B" in document
    assert "strategy beta" in document
    assert "1–0–0" in document
    assert "match-1" in document

    _, _, body = _get(base + "/episodes/match-1")
    document = body.decode()
    assert "Watch replay" in document
    assert "Download replay JSON" in document
    assert "agent:evaluate" in document
    assert "3,200" in document

    _, _, body = _get(base + "/episodes/match-2")
    document = body.decode()
    beta_at = document.index("strategy beta")
    alpha_at = document.index("strategy alpha")
    assert beta_at < alpha_at


def test_dashboard_downloads_exact_replay_json(dashboard):
    base, expected = dashboard
    status, headers, body = _get(base + "/episodes/match-1/replay.json")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert headers["Content-Disposition"] == 'attachment; filename="replay.json"'
    assert json.loads(body) == expected


def test_dashboard_renders_visual_replay_with_agent_names(dashboard):
    base, _ = dashboard
    status, headers, body = _get(base + "/episodes/match-1/replay")
    document = body.decode()
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert "script-src 'unsafe-inline'" in headers["Content-Security-Policy"]
    assert "Farm Stars \\u00b7 alpha" in document
    assert "Farm Stars \\u00b7 beta" in document
    assert "window.kaggle" in document


def test_dashboard_health_head_and_error_responses(dashboard):
    base, _ = dashboard
    status, _, body = _get(base + "/healthz")
    assert status == 200
    assert json.loads(body) == {"sessions": 1, "status": "ok"}

    request = Request(base + "/", method="HEAD")
    with urlopen(request) as response:
        assert response.status == 200
        assert response.read() == b""
        assert int(response.headers["Content-Length"]) > 0

    with pytest.raises(HTTPError) as missing:
        urlopen(base + "/episodes/missing")
    assert missing.value.code == 404

    with pytest.raises(HTTPError) as invalid_page:
        urlopen(base + "/?page=zero")
    assert invalid_page.value.code == 400

    with pytest.raises(HTTPError) as method:
        urlopen(Request(base + "/", data=b"", method="POST"))
    assert method.value.code == 405
    assert method.value.headers["Allow"] == "GET, HEAD"


def test_episode_page_hides_replay_actions_when_payload_is_missing(tmp_path: Path):
    database = tmp_path / "missing-replay.duckdb"
    _seed_dashboard(database)
    with ExperimentStore(database) as store:
        store.connection.execute("DELETE FROM episode_payloads WHERE match_id = 'match-2'")
        document = _episode_page(store.connection, "match-2")
    assert "Replay unavailable" in document
    assert "Watch replay" not in document
    assert "Download replay JSON" not in document


def test_replay_loader_rejects_missing_and_corrupt_payloads(tmp_path: Path):
    database = tmp_path / "corrupt-replay.duckdb"
    with ExperimentStore(database) as store:
        with pytest.raises(DashboardNotFound, match="missing"):
            _load_replay(store.connection, "missing")
        store.connection.execute(
            "INSERT INTO episode_payloads VALUES ('broken', ?, 'not-a-digest')",
            [b"not gzip"],
        )
        with pytest.raises(DashboardUnavailable, match="invalid"):
            _load_replay(store.connection, "broken")


def test_server_refuses_to_create_a_missing_database(tmp_path: Path):
    database = tmp_path / "missing.duckdb"
    with pytest.raises(FileNotFoundError, match="create a league session first"):
        make_server(database, "127.0.0.1", 0)
    assert not database.exists()


def test_web_command_defaults_to_the_local_catalog():
    args = build_parser().parse_args(["web"])
    assert args.database == "artifacts/experiments.duckdb"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
