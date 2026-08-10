import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from kaggriculture_lab import runner
from kaggriculture_lab.league import LeagueConfig
from kaggriculture_lab.storage import ExperimentStore
from tests.postgres_support import postgres_database


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
        database="postgresql://writer:test@127.0.0.1/test",
        artifacts=Path("artifacts"),
        generations=1,
    )

    output = capsys.readouterr().out
    assert "generation 1/1 started" in output
    assert "generation g0003 training round 2/20 started" in output
    assert "training session run completed (1 generation)" in output


def test_run_generations_does_not_own_dashboard_or_mcp(monkeypatch, capsys):
    monkeypatch.setattr(runner, "session_status", lambda *args: {"status": "OPEN"})
    monkeypatch.setattr(
        runner,
        "run_generation",
        lambda *args, **kwargs: {"status": "COMPLETE"},
    )

    runner.run_generations(
        session_id="session",
        database="postgresql://writer:test@127.0.0.1/test",
        artifacts=Path("artifacts"),
        generations=1,
    )

    output = capsys.readouterr().out
    assert "dashboard:" not in output
    assert "MCP:" not in output


def test_continuous_generations_are_blocking_and_sequential(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(runner, "session_status", lambda *args: {"status": "COMPLETE"})
    monkeypatch.setattr(
        runner,
        "create_next_generation",
        lambda *args, **kwargs: calls.append("create"),
    )

    def run(*args, **kwargs):
        calls.append("run")
        if calls.count("run") == 3:
            raise KeyboardInterrupt
        return {"status": "COMPLETE"}

    monkeypatch.setattr(runner, "run_generation", run)
    with pytest.raises(KeyboardInterrupt):
        runner.run_generations(
            session_id="continuous",
            database="postgresql://writer:test@127.0.0.1/test",
            artifacts=Path("artifacts"),
            generations=5,
            continuous=True,
        )

    assert calls == ["create", "run", "create", "run", "create", "run"]
    assert "generation 1 started" in capsys.readouterr().out


def test_ensure_session_upgrades_only_the_legacy_missing_league_manifest(tmp_path: Path):
    database = postgres_database(tmp_path, "legacy-run-manifest")
    config = LeagueConfig(
        league="mixed_farm_v3",
        name="legacy-manifest-test",
        population_size=2,
        rounds_per_generation=1,
        hall_of_fame_rounds=0,
        episode_steps=1,
    )
    runner.ensure_session(
        config,
        database,
        "legacy-manifest-session",
        artifacts=tmp_path,
    )
    with ExperimentStore(database) as store:
        row = store.connection.execute(
            "SELECT manifest_json FROM session_manifests WHERE session_id = %s",
            ["legacy-manifest-session"],
        ).fetchone()
        payload = json.loads(row[0])
        del payload["league"]["league"]
        legacy_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        legacy_digest = hashlib.sha256(legacy_json.encode()).hexdigest()
        store.connection.execute(
            "UPDATE session_manifests SET manifest_json = %s, manifest_sha256 = %s "
            "WHERE session_id = %s",
            [legacy_json, legacy_digest, "legacy-manifest-session"],
        )

    created, _ = runner.ensure_session(
        config,
        database,
        "legacy-manifest-session",
        artifacts=tmp_path,
    )

    assert created is False
    with ExperimentStore(database) as store:
        upgraded = store.connection.execute(
            "SELECT manifest_json FROM session_manifests WHERE session_id = %s",
            ["legacy-manifest-session"],
        ).fetchone()[0]
    assert json.loads(upgraded)["league"]["league"] == "mixed_farm_v3"
    with pytest.raises(ValueError, match="manifest mismatch"):
        runner.ensure_session(
            replace(config, seed=config.seed + 1),
            database,
            "legacy-manifest-session",
            artifacts=tmp_path,
        )
