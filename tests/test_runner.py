from pathlib import Path

from kaggriculture_lab import runner
from kaggriculture_lab.dashboard import DashboardRepository
from kaggriculture_lab.storage import ExperimentStore


def test_run_generations_reports_round_start_and_session_completion(
    monkeypatch, capsys
):
    def fake_run_generation(
        session_id: str,
        database: Path,
        artifacts: Path,
        *,
        update_models: bool,
        on_progress,
    ):
        on_progress(
            {
                "event": "round_started",
                "generation_index": 3,
                "round_number": 2,
                "round_count": 20,
            }
        )
        return {"status": "COMPLETE"}

    monkeypatch.setattr(runner, "run_generation", fake_run_generation)
    monkeypatch.setattr(
        runner,
        "session_status",
        lambda *args, **kwargs: {"status": "OPEN"},
    )
    monkeypatch.setattr(runner, "create_next_generation", lambda *args, **kwargs: None)

    runner.run_generations(
        session_id="training-session",
        database=Path("database.duckdb"),
        artifacts=Path("artifacts"),
        generations=1,
        dashboard=False,
    )

    output = capsys.readouterr().out
    assert "generation 1/1 started" in output
    assert "generation g0003 training round 2/20 started" in output
    assert "training session run completed (1 generation)" in output


def test_run_generations_starts_and_stops_embedded_dashboard(monkeypatch, capsys):
    calls = []

    class FakeServer:
        server_address = ("127.0.0.1", 8123)

        def serve_forever(self):
            calls.append("serve")

        def shutdown(self):
            calls.append("shutdown")

        def server_close(self):
            calls.append("close")

    monkeypatch.setattr(runner, "make_server", lambda *args, **kwargs: FakeServer())
    monkeypatch.setattr(runner, "session_status", lambda *args: {"status": "OPEN"})
    monkeypatch.setattr(
        runner,
        "run_generation",
        lambda *args, **kwargs: {"status": "COMPLETE"},
    )

    runner.run_generations(
        session_id="session",
        database=Path("database.duckdb"),
        artifacts=Path("artifacts"),
        generations=1,
    )

    assert calls == ["serve", "shutdown", "close"]
    assert "dashboard: http://127.0.0.1:8123/" in capsys.readouterr().out


def test_embedded_dashboard_can_read_while_training_store_is_open(tmp_path):
    database = tmp_path / "live.duckdb"
    with ExperimentStore(database) as store:
        store.connection.execute(
            "INSERT INTO training_sessions VALUES (?, ?, ?, ?, ?, current_timestamp)",
            ["live", "live", "RUNNING", 1, "{}"],
        )
        repository = DashboardRepository(database, read_only=False)
        with repository.connect() as connection:
            assert connection.execute(
                "SELECT status FROM training_sessions WHERE session_id = 'live'"
            ).fetchone()[0] == "RUNNING"
