import json
from pathlib import Path

import pytest

from agents.configurable_crop_surrogate.strategy import Strategy
from kaggriculture_lab.hooks import EpisodeHookContext, EpisodeReader
from kaggriculture_lab.identity import (
    AgentManifest,
    TrainingAgentSpec,
    build_revision,
    effective_model_id,
    training_artifact_root,
    training_instance_seed,
)
from kaggriculture_lab.league import (
    LeagueConfig,
    _primary_training_model_id,
    _run_hook,
    _training_model_ids,
    create_session,
    run_generation,
    schedule_generation,
)
from kaggriculture_lab.model_trainers import load_trainer
from kaggriculture_lab.ratings import GlickoRating, RatingGame, update_glicko2
from kaggriculture_lab.storage import SCHEMA_VERSION, ExperimentStore
from kaggriculture_lab.training import allocate_generation
from sim.runner import MatchResult, SeatResult


def test_glicko2_matches_published_example():
    updated = update_glicko2(
        GlickoRating(1500, 200, 0.06),
        [
            RatingGame(GlickoRating(1400, 30), 1.0),
            RatingGame(GlickoRating(1550, 100), 0.0),
            RatingGame(GlickoRating(1700, 300), 0.0),
        ],
    )
    assert updated.rating == pytest.approx(1464.06, abs=0.02)
    assert updated.deviation == pytest.approx(151.52, abs=0.02)
    assert updated.volatility == pytest.approx(0.059996, abs=0.000001)


def test_generation_allocation_is_exact():
    assert allocate_generation(20) == {
        "elite": 4,
        "evolution": 10,
        "surrogate": 4,
        "random": 2,
    }
    assert sum(allocate_generation(7).values()) == 7


def test_revision_identity_tracks_behavior_code(tmp_path: Path):
    code = tmp_path / "agent.py"
    code.write_text("def agent(obs): return {}\n")
    manifest = AgentManifest(
        family_id="test",
        name="Test",
        entry_point="agent:agent",
        parameter_schema_version=1,
        code_paths=("agent.py",),
    )
    first = build_revision(manifest, Strategy(), workspace=tmp_path)
    same = build_revision(manifest, Strategy(), workspace=tmp_path)
    code.write_text("def agent(obs): return {'market': []}\n")
    changed = build_revision(manifest, Strategy(), workspace=tmp_path)
    assert first.revision_id == same.revision_id
    assert first.revision_id != changed.revision_id


def test_replay_is_compressed_in_duckdb_and_read_through_narrow_reader(tmp_path: Path):
    result = MatchResult(
        seed=7,
        seats=[
            SeatResult("a", 0, 20, "DONE", False, None, None, 1, 0, 0, 0, 20, {"productive_actions": 1}),
            SeatResult("b", 1, 10, "DONE", False, None, None, 1, 0, 0, 0, 10, {"productive_actions": 0}),
        ],
        steps=1,
        wallclock=0.1,
        replay={"steps": [[{"observation": {"step": 0}}]]},
    )
    with ExperimentStore(tmp_path / "episodes.duckdb") as store:
        store.add_match("match", "run", "league:test", ("a", "b"), ("a", "b"), result)
        store.add_replay_payload("match", result.replay)
        reader = EpisodeReader(store.connection, "match", 0)
        assert reader.summary()["money_difference"] == 10
        assert reader.replay() == result.replay
        assert store.connection.execute(
            "SELECT max(version) FROM schema_migrations"
        ).fetchone()[0] == SCHEMA_VERSION

        context = EpisodeHookContext("run", "generation", "pair", "match", "a", "b", 0)
        _run_hook(store, context, "missing_episode_evaluator:hook")
        status = store.connection.execute(
            "SELECT status FROM episode_evaluations WHERE match_id = 'match'"
        ).fetchone()[0]
        assert status == "ERROR"


def test_scheduler_gives_every_member_each_round_and_avoids_self_play(tmp_path: Path):
    database = tmp_path / "league.duckdb"
    config = LeagueConfig(
        population_size=6,
        rounds_per_generation=4,
        hall_of_fame_rounds=1,
        episode_steps=24,
    )
    session_id = create_session(config, database, session_id="schedule")
    with ExperimentStore(database) as store:
        assert schedule_generation(store, session_id) == 15
        generation_id = "schedule:g0000"
        members = {
            row[0]
            for row in store.connection.execute(
                "SELECT revision_id FROM generation_members WHERE generation_id = ?", [generation_id]
            ).fetchall()
        }
        rows = store.connection.execute(
            "SELECT revision_a, revision_b, pairing_kind FROM league_pairings WHERE generation_id = ?",
            [generation_id],
        ).fetchall()
        assert all(left != right for left, right, _ in rows)
        population_pairs = {
            tuple(sorted((left, right))) for left, right, kind in rows if kind == "population"
        }
        assert len(population_pairs) == 9
        appearances = {member: 0 for member in members}
        for left, right, _ in rows:
            if left in appearances:
                appearances[left] += 1
            if right in appearances:
                appearances[right] += 1
        assert set(appearances.values()) == {4}


def test_running_generation_resumes_without_being_scheduled_again(
    tmp_path: Path, monkeypatch
):
    from kaggriculture_lab import league

    database = tmp_path / "resume.duckdb"
    config = LeagueConfig(
        population_size=4,
        rounds_per_generation=1,
        hall_of_fame_rounds=0,
        episode_steps=2,
        jobs=1,
    )
    session_id = create_session(config, database, session_id="resume")
    with ExperimentStore(database) as store:
        scheduled = schedule_generation(store, session_id)
        store.connection.execute(
            "UPDATE generations SET status = 'RUNNING' WHERE generation_id = 'resume:g0000'"
        )
    monkeypatch.setattr(
        league,
        "schedule_generation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rescheduled")),
    )
    monkeypatch.setattr(league, "_run_round", lambda *args, **kwargs: None)
    monkeypatch.setattr(league, "_close_generation", lambda *args, **kwargs: None)
    report = run_generation(session_id, database, tmp_path / "artifacts")
    assert report["status"] == "COMPLETE"
    with ExperimentStore(database) as store:
        assert store.connection.execute(
            "SELECT count(*) FROM league_pairings WHERE generation_id = 'resume:g0000'"
        ).fetchone()[0] == scheduled


def test_trainer_only_generation_stops_on_round_hook_error(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    from kaggriculture_lab import league

    database = tmp_path / "trainer-error.duckdb"
    config = LeagueConfig(
        pairing_mode="trainer_only",
        population_size=0,
        rounds_per_generation=1,
        hall_of_fame_rounds=0,
        episode_steps=2,
        jobs=1,
    )
    session_id = create_session(config, database, session_id="trainer-error")
    monkeypatch.setattr(league, "_run_round", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        league,
        "run_round_hooks",
        lambda *args, **kwargs: [
            SimpleNamespace(model_id="broken", status="error", error="boom")
        ],
    )
    with pytest.raises(RuntimeError, match="generation remains resumable"):
        run_generation(session_id, database, tmp_path / "artifacts")
    with ExperimentStore(database) as store:
        assert store.connection.execute(
            "SELECT status FROM generations WHERE generation_id = 'trainer-error:g0000'"
        ).fetchone()[0] == "RUNNING"


def test_manifest_file_is_valid():
    manifest = AgentManifest.load("agents/configurable_crop_surrogate/agent.toml")
    assert manifest.family_id == "configurable_crop_surrogate"
    assert manifest.episode_evaluator
    assert manifest.models[0].min_new_batches == 2
    assert json.loads(build_revision(manifest, Strategy()).params_json) == Strategy().to_dict()


def test_named_training_identity_is_stable_and_scopes_artifacts(tmp_path: Path):
    first = training_instance_seed("dqn_job_scorer", "dqn05")
    assert first == training_instance_seed("dqn_job_scorer", "dqn05")
    assert first != training_instance_seed("dqn_job_scorer", "dqn06")
    assert effective_model_id("dqn_job_scorer.q_network", "default") == (
        "dqn_job_scorer.q_network"
    )
    assert effective_model_id("dqn_job_scorer.q_network", "dqn05") == (
        "dqn_job_scorer.q_network@dqn05"
    )
    assert training_artifact_root(
        tmp_path, {"agent_type": "dqn_job_scorer", "name": "dqn05"}
    ) == tmp_path / "dqn_job_scorer" / "dqn05"


def test_named_roster_selects_exact_trainable_models_and_locked_primary():
    config = LeagueConfig(
        training_manifests=("agents/configurable_crop_surrogate/agent.toml",),
        training_agents=(
            TrainingAgentSpec(
                "configurable_crop_surrogate", "crop01", primary=True, train=False
            ),
            TrainingAgentSpec("configurable_crop_surrogate", "crop02"),
        ),
    )
    config.validate()
    assert _training_model_ids(config) == (
        "configurable_crop_surrogate.surrogate@crop02",
    )
    assert _primary_training_model_id(config, "configurable_crop_surrogate") == (
        "configurable_crop_surrogate.surrogate@crop01"
    )


def test_duplicate_named_surrogates_require_one_primary():
    config = LeagueConfig(
        training_agents=(
            TrainingAgentSpec("configurable_crop_surrogate", "one"),
            TrainingAgentSpec("configurable_crop_surrogate", "two"),
        )
    )
    with pytest.raises(ValueError, match="exactly one primary"):
        config.validate()

    dqn = AgentManifest.load("agents/dqn_job_scorer/agent.toml")
    # The DQN collects per closed round, so its update threshold is expressed in
    # rounds: one update per generation means matching the league's round count.
    league = LeagueConfig.from_toml("configs/league.toml")
    model = dqn.models[0]
    assert model.min_new_batches == (
        league.rounds_per_generation * int(model.config["batches_per_round"])
    )


def test_every_trainable_agent_declares_a_loadable_trainer():
    """A stale manifest should fail here, not part-way through a training run."""

    found = 0
    for path in sorted(Path("agents").glob("*/agent.toml")):
        for model in AgentManifest.load(path).models:
            trainer = load_trainer(model.trainer)
            assert callable(getattr(trainer, "on_batch", None)), model.model_id
            found += 1
    assert found >= 3


def test_second_generation_schedules_with_a_hall_of_fame(tmp_path: Path):
    """Generation 1 is the first to query the hall of fame.

    Generation 0 returns before that query, so a smoke test that ran only the
    first generation passed while `league run` crashed on every subsequent one
    with an ambiguous `generation_id`. This exercises the path that broke.
    """
    from kaggriculture_lab.league import _hall_of_fame, create_next_generation, run_generation

    database = tmp_path / "generations.duckdb"
    config = LeagueConfig(
        population_size=4,
        rounds_per_generation=2,
        hall_of_fame_rounds=1,
        episode_steps=24,
        seed=11,
    )
    session_id = create_session(config, database, session_id="gen2")

    run_generation(session_id, database, tmp_path / "artifacts")
    assert create_next_generation(session_id, database) == "gen2:g0001"

    with ExperimentStore(database) as store:
        # The query that used to raise BinderException.
        hall = _hall_of_fame(store, session_id, generation_index=1, limit=3)
        assert hall, "generation 1 must have a hall of fame to draw opponents from"
        assert schedule_generation(store, session_id) > 0

    result = run_generation(session_id, database, tmp_path / "artifacts")
    assert result["status"] == "COMPLETE"
    assert result["generation_index"] == 1
