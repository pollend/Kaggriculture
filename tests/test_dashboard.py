import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
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
    _policy_role_label,
    _replay_page,
    _session_page,
    _sessions_page,
    make_server,
)
from kaggriculture_lab.database import DatabaseTarget
from kaggriculture_lab.storage import ExperimentStore
from sim.runner import MatchResult, SeatResult
from tests.postgres_support import postgres_database, seed_reference_rows


def _seed_dashboard(database: DatabaseTarget) -> dict:
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
            "INSERT INTO training_sessions (session_id, league, name, status, seed, config_json) VALUES (%s, %s, %s, 'COMPLETE', 17, %s)",
            ["session/one", "test-league", "Test <script> League", json.dumps({"rounds": 1})],
        )
        connection.execute(
            "INSERT INTO generations (generation_id, session_id, generation_index, status, created_at, completed_at) VALUES (%s, %s, 0, 'COMPLETE', now(), now())",
            ["session/one:g0000", "session/one"],
        )
        connection.execute(
            "INSERT INTO agent_families (family_id, name, manifest_path, entry_point, parameter_schema_version) VALUES ('family', 'Farm Stars', 'agent.toml', 'agent:agent', 1)"
        )
        seed_reference_rows(connection, run_id="session/one")
        for revision, strategy in (("revision-a", "alpha"), ("revision-b", "beta")):
            connection.execute(
                "INSERT INTO strategies VALUES (%s, '{}', 'test', 'session/one')",
                [strategy],
            )
            connection.execute(
                """
                INSERT INTO agent_revisions (
                    revision_id, family_id, strategy_id, entry_point, agent_spec,
                    code_fingerprint, parameter_schema_version, params_json, source
                ) VALUES (%s, 'family', %s, 'agent:agent', %s, 'fingerprint', 1, '{}', 'test')
                """,
                [revision, strategy, strategy],
            )
            connection.execute(
                "INSERT INTO generation_members VALUES (%s, %s, 'population', 1500, 200, 0.06)",
                ["session/one:g0000", revision],
            )
        connection.execute(
            "INSERT INTO league_pairings VALUES ('pair-1', %s, 0, 'population', 17, 'revision-a', 'revision-b', 'COMPLETE')",
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
            "INSERT INTO rating_snapshots VALUES (%s, 'revision-a', 1600, 80, 0.06, 1, false), (%s, 'revision-b', 1400, 80, 0.06, 1, false)",
            ["session/one:g0000", "session/one:g0000"],
        )
        connection.execute(
            "INSERT INTO generation_stats VALUES (%s, 'revision-a', 1, 1, 0, 0, 400, 400, 400, 400, 1), (%s, 'revision-b', 1, 0, 0, 1, -400, -400, -400, -400, 1)",
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
    database = postgres_database(tmp_path, "dashboard")
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


def _origin(base: str) -> str:
    return base


def test_dashboard_drills_from_sessions_to_episode(dashboard):
    base, _ = dashboard
    status, headers, body = _get(base + "/")
    document = body.decode()
    assert status == 200
    assert "Test &lt;script&gt; League" in document
    assert "test-league" in document
    assert "Test <script> League" not in document
    assert "session%2Fone" in document
    assert "Content-Security-Policy" in headers
    assert '<script src="/dashboard.js" defer></script>' in document
    assert "Refresh" not in document
    _, js_headers, javascript = _get(_origin(base) + "/dashboard.js")
    assert js_headers["Content-Type"].startswith("text/javascript")
    assert b"new EventSource(eventUrl)" not in javascript
    assert b"new EventSource(url)" in javascript
    assert b"form.filters" in javascript
    assert b"history.pushState" in javascript
    assert b"data-dashboard-nav" in javascript
    assert b"window.Alpine" not in javascript
    assert b"x-data" not in javascript
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
    assert '<div class="players">' in document
    assert '<iframe class="replay-frame" src="/episodes/match-1/replay"' in document
    assert 'scrolling="no"' in document
    assert "<th>Seat</th>" not in document
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


def test_sp_psro_policy_role_labels_are_unambiguous():
    assert _policy_role_label('{"role":"population"}', "member-000000-initial") == (
        "Frozen population — initial baseline"
    )
    assert _policy_role_label('{"role":"population"}', "live-nu") == "Live new policy — ν"
    assert _policy_role_label('{"role":"population"}', "member-000012-beta") == (
        "Frozen population — β member 12"
    )
    assert _policy_role_label('{"role":"population"}', "member-000012-nu-average") == (
        "Frozen population — averaged ν member 12"
    )
    assert _policy_role_label('{"role":"best_response"}', "beta-928dc4299cc6") == (
        "Live best response — β 928dc429"
    )
    assert _policy_role_label("{}", "revision-a") is None
    assert _policy_role_label("not-json", "revision-a") is None


def test_trainer_only_dashboard_reports_round_progress_not_pairing_fraction(tmp_path: Path):
    database = postgres_database(tmp_path, "trainer-progress")
    _seed_dashboard(database)
    with ExperimentStore(database) as store:
        connection = store.connection
        connection.execute(
            "UPDATE training_sessions SET config_json = %s WHERE session_id = 'session/one'",
            [json.dumps({"pairing_mode": "trainer_only", "rounds_per_generation": 100})],
        )
        connection.execute(
            """
            INSERT INTO training_models (
                model_id, family_id, trainer_path, min_new_batches, config_json
            ) VALUES ('model', 'family', 'trainer:train', 1, '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO model_round_hooks (
                hook_id, model_id, session_id, generation_id, scope, round_index, status
            ) VALUES
                ('hook-0', 'model', 'session/one', 'session/one:g0000', 'round', 0, 'COMPLETE'),
                ('hook-1', 'model', 'session/one', 'session/one:g0000', 'round', 1, 'COMPLETE')
            """
        )
        sessions = _sessions_page(connection, 1)
        session = _session_page(connection, "session/one")
        generation = _generation_page(connection, "session/one:g0000", 1, "all", "all")

    for document in (sessions, session, generation):
        assert "2 / 100" in document
        assert '<span class="muted">Pairings</span><b>1 / 1</b>' not in document
    assert "1 pairings recorded" in sessions
    assert "Pairings recorded" in session
    assert "Pairings recorded" in generation
    assert "Training rounds" in generation
    assert "hook-0" not in generation
    assert "<td>0</td>" in generation and "<td>1</td>" in generation


def test_sp_psro_labels_flow_to_episode_generation_and_replay(tmp_path: Path):
    database = postgres_database(tmp_path, "sp-labels")
    _seed_dashboard(database)
    with ExperimentStore(database) as store:
        connection = store.connection
        connection.execute(
            "INSERT INTO agent_families (family_id, name, manifest_path, entry_point, "
            "parameter_schema_version) VALUES ('sp', 'SP-PSRO', 'sp.toml', 'sp:agent', 1)"
        )
        connection.execute(
            """
            INSERT INTO agent_revisions (
                revision_id, family_id, entry_point, agent_spec, code_fingerprint,
                parameter_schema_version, params_json, source
            ) VALUES
                ('member-000000-initial', 'sp', 'sp:agent', 'initial', 'fp', 1,
                 '{"role":"population"}', 'test'),
                ('beta-928dc4299cc6', 'sp', 'sp:agent', 'beta', 'fp', 1,
                 '{"role":"best_response"}', 'test')
            """
        )
        connection.execute(
            "UPDATE league_pairings SET revision_a = 'member-000000-initial', "
            "revision_b = 'beta-928dc4299cc6' WHERE pairing_id = 'pair-1'"
        )
        episode = _episode_page(connection, "match-1")
        generation = _generation_page(
            connection, "session/one:g0000", 1, "all", "all"
        )
        replay = _replay_page(connection, "match-1")

    for document in (episode, generation):
        assert "Frozen population — initial baseline" in document
        assert "Live best response — β 928dc429" in document
    assert "Seat 1 \\u00b7 Frozen population \\u2014 initial baseline" in replay
    assert "Seat 2 \\u00b7 Live best response \\u2014 \\u03b2 928dc429" in replay
    assert replay.count("fixed for episode") >= 2


def test_dashboard_health_head_and_error_responses(dashboard):
    base, _ = dashboard
    status, _, body = _get(_origin(base) + "/healthz")
    assert status == 200
    health = json.loads(body)
    assert health["status"] == "ok"
    assert health["sessions"] >= 1
    assert "catalogs" not in health

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

    with pytest.raises(HTTPError) as invalid_sort:
        urlopen(base + "/generations/session%2Fone%3Ag0000?sort=money")
    assert invalid_sort.value.code == 400

    with pytest.raises(HTTPError) as method:
        urlopen(Request(_origin(base) + "/", data=b"", method="POST"))
    assert method.value.code == 405
    assert method.value.headers["Allow"] == "GET, HEAD"


def test_dashboard_event_stream_sends_current_page_fragment(dashboard):
    base, _ = dashboard
    event_url = _origin(base) + "/events/dashboard?" + urlencode(
        {"path": base.removeprefix(_origin(base)) + "/sessions/session%2Fone"}
    )
    with urlopen(event_url, timeout=3) as response:
        event_line = response.readline()
        data_line = response.readline()
        separator = response.readline()

    assert response.headers["Content-Type"].startswith("text/event-stream")
    assert event_line == b"event: dashboard.fragment\n"
    assert data_line.startswith(b"data: ")
    payload = json.loads(data_line.removeprefix(b"data: "))
    assert payload["selector"] == "main"
    assert payload["mergeMode"] == "outer"
    assert "Rating trend" in payload["fragments"]
    assert separator == b"\n"


def test_generation_page_reports_performance_statistics(dashboard):
    base, _ = dashboard
    _, _, body = _get(base + "/generations/session%2Fone%3Ag0000")
    document = body.decode()

    for heading in ("Performance", "Match outcomes", "Agent money and behaviour",
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


def test_pairings_can_be_sorted_by_clickable_headers(tmp_path: Path):
    database = postgres_database(tmp_path, "sorted-pairings")
    _seed_dashboard(database)
    with ExperimentStore(database) as store:
        store.connection.execute(
            "INSERT INTO league_pairings VALUES ('pair-2', %s, 5, 'population', 18, "
            "'revision-a', 'revision-b', 'OPEN')",
            ["session/one:g0000"],
        )
        document = _generation_page(
            store.connection,
            "session/one:g0000",
            1,
            "all",
            "all",
            "round",
            "desc",
        )

    assert document.index("pair-2") < document.index("pair-1")
    assert 'data-dashboard-nav href="/generations/session%2Fone%3Ag0000?sort=round&amp;direction=asc"' in document
    assert "Round ↓" in document
    assert 'name="sort" value="round"' in document
    assert 'name="direction" value="desc"' in document


def test_performance_section_renders_for_a_generation_with_no_episodes(tmp_path: Path):
    database = postgres_database(tmp_path, "empty-generation")
    _seed_dashboard(database)
    with ExperimentStore(database) as store:
        store.connection.execute(
            "INSERT INTO generations (generation_id, session_id, generation_index, status) VALUES (%s, 'session/one', 1, 'OPEN')",
            ["session/one:g0001"],
        )
        document = _generation_page(store.connection, "session/one:g0001", 1, "all", "all")

    assert "Performance" in document
    assert "No pairings have been settled yet." in document
    assert "Timings appear once episodes run." in document
    assert "No post-episode evaluations were recorded." in document


def test_episode_page_hides_replay_actions_when_payload_is_missing(tmp_path: Path):
    database = postgres_database(tmp_path, "missing-replay")
    _seed_dashboard(database)
    with ExperimentStore(database) as store:
        store.connection.execute("DELETE FROM episode_payloads WHERE match_id = 'match-2'")
        document = _episode_page(store.connection, "match-2")
    assert "Replay unavailable" in document
    assert "Watch replay" not in document
    assert "Download replay JSON" not in document


def test_replay_loader_rejects_missing_and_corrupt_payloads(tmp_path: Path):
    database = postgres_database(tmp_path, "corrupt-replay")
    with ExperimentStore(database) as store:
        with pytest.raises(DashboardNotFound, match="missing"):
            _load_replay(store.connection, "missing")
        seed_reference_rows(store.connection, run_id="run", match_ids=("broken",))
        store.connection.execute(
            "INSERT INTO episode_payloads VALUES ('broken', %s, 'not-a-digest')",
            [b"not gzip"],
        )
        with pytest.raises(DashboardUnavailable, match="invalid"):
            _load_replay(store.connection, "broken")


def test_server_refuses_a_missing_database(tmp_path: Path):
    database = postgres_database(tmp_path, "existing")
    missing = DatabaseTarget(database.url.rsplit("/", 1)[0] + "/missing_kaggriculture_test")
    with pytest.raises(ValueError, match="not a Kaggriculture PostgreSQL database"):
        make_server(missing, "127.0.0.1", 0)


def test_web_command_defaults_to_the_shared_postgres_service():
    args = build_parser().parse_args(["web"])
    assert args.database_url is None
    assert not hasattr(args, "catalog")
    assert args.host == "127.0.0.1"
    assert args.port == 8000
