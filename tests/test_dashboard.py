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
    _generation_page,
    _load_replay,
    _round_panel,
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
        # `league.py` hands `add_match` the seat-correct revision ids, not strategy ids.
        store.add_match("match-1", "session/one", "league:g0000", ("revision-a", "revision-b"), ("alpha", "beta"), result)
        store.add_replay_payload("match-1", replay)
        connection.execute("INSERT INTO pairing_episodes VALUES ('pair-1', 0, 'match-1')")
        reverse_result = MatchResult(
            seed=17,
            seats=[result.seats[1], result.seats[0]],
            steps=30,
            wallclock=0.75,
            replay=replay,
        )
        store.add_match("match-2", "session/one", "league:g0000", ("revision-b", "revision-a"), ("beta", "alpha"), reverse_result)
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
            ) VALUES ('eval-1', 'match-1', 'revision-a', 0, 'agent:evaluate', 'fp', 'SUCCESS', '{"score": 1}'),
                     ('eval-2', 'match-2', 'revision-b', 1, 'agent:evaluate', 'fp', 'COMPLETE', '{"should_reevaluate": true}')
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


def test_generation_page_reports_performance_statistics(dashboard):
    base, _ = dashboard
    _, _, body = _get(base + "/generations/session%2Fone%3Ag0000")
    document = body.decode()

    for heading in ("Performance", "Match outcomes", "Rounds", "Agent money and behaviour",
                    "Runtime cost", "Evaluation hooks"):
        assert heading in document

    # pair-1 is the only settled pairing, margin +400 for revision-a.
    assert "Settled pairings" in document
    assert "Mean margin" in document

    # Both seat orders are played, so the seats must come out even.
    assert "Seat-order bias" in document
    assert "50.0%" in document

    # Per-agent rows resolve through episode_players.strategy_id -> agent_revisions.
    assert "revision-a" in document
    assert "3,200" in document and "2,800" in document
    assert "100.0%" in document and "0.0%" in document
    assert "40.0%" in document  # revision-b productive action percentage

    # 0.25s + 0.75s over two episodes, the slower one listed first.
    assert "Episode wall clock" in document
    assert "1.0s" in document
    assert "Slowest episodes" in document
    assert document.index("0.8s") < document.index("0.2s")

    assert "Re-evaluation requests" in document


def test_performance_section_renders_for_a_generation_with_no_episodes(tmp_path: Path):
    database = tmp_path / "empty-generation.duckdb"
    _seed_dashboard(database)
    with ExperimentStore(database) as store:
        store.connection.execute(
            "INSERT INTO generations (generation_id, session_id, generation_index, status) VALUES (?, 'session/one', 1, 'OPEN')",
            ["session/one:g0001"],
        )
        document = _generation_page(store.connection, "session/one:g0001", 1, "all", "all")

    assert "Performance" in document
    assert "No pairings have been settled yet." in document
    assert "No rounds have been scheduled yet." in document
    assert "Timings appear once episodes run." in document
    assert "No post-episode evaluations were recorded." in document


def test_round_panel_averages_margin_per_pairing_not_per_episode(tmp_path: Path):
    """A pairing contributes its margin once, however many episodes it produced."""
    database = tmp_path / "round-margins.duckdb"
    _seed_dashboard(database)
    with ExperimentStore(database) as store:
        connection = store.connection
        # Round 1: one pairing settled over two episodes, one over a single episode.
        for pairing, margin in (("pair-2", 400.0), ("pair-3", 1000.0)):
            connection.execute(
                "INSERT INTO league_pairings VALUES (?, ?, 1, 'population', 17, 'revision-a', 'revision-b', 'COMPLETE')",
                [pairing, "session/one:g0000"],
            )
            connection.execute(
                "INSERT INTO pairing_results VALUES (?, ?, 1, false, 2, '{}')",
                [pairing, margin],
            )
        for match_id, pairing, orientation in (
            ("match-3", "pair-2", 0), ("match-4", "pair-2", 1), ("match-5", "pair-3", 0),
        ):
            connection.execute(
                """
                INSERT INTO episodes (
                    match_id, run_id, stage, seed, seat0_id, seat1_id, seat0_spec,
                    seat1_spec, steps, wallclock, failed
                ) VALUES (?, 'session/one', 'league:g0000', 17, 'revision-a', 'revision-b',
                          'alpha', 'beta', 24, 0.5, false)
                """,
                [match_id],
            )
            connection.execute(
                "INSERT INTO pairing_episodes VALUES (?, ?, ?)", [pairing, orientation, match_id]
            )
        document = _round_panel(connection, "session/one:g0000")

    # Unweighted mean of 400 and 1000. Fanning pairings out across their episodes would
    # weight pair-2 twice and yield 600.
    assert "700.0" in document
    assert "600.0" not in document


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
