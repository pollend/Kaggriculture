"""The round hook: fires once per closed round, and never twice."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggriculture_lab.league import (
    LeagueConfig,
    _training_family_ids,
    create_session,
    run_generation,
)
from kaggriculture_lab.model_trainers import (
    GENERATION_ROUND_INDEX,
    TrainingBatchWriter,
    run_round_hooks,
)
from kaggriculture_lab.model_updates import record_training_batch
from kaggriculture_lab.storage import ExperimentStore

MODEL_ID = "test.rounds"
FAMILY_ID = "test_family"

#: Every hook call this module's trainers see, as (model_id, scope, round_index).
CALLS: list[tuple[str, str, int]] = []
#: Rating-snapshot counts observed by generation-scope calls.
RATING_COUNTS: list[int] = []
#: Rounds whose hook should raise, consumed one failure per entry.
FAIL_ROUNDS: set[int] = set()


class RecordingTrainer:
    def on_round(self, context, reader, writer) -> None:
        CALLS.append((context.model_id, context.scope, context.round_index))
        if context.scope == "generation":
            RATING_COUNTS.append(
                int(
                    reader.fetchone(
                        "SELECT count(*) FROM rating_snapshots WHERE generation_id = ?",
                        [context.generation_id],
                    )[0]
                )
            )
        if context.round_index in FAIL_ROUNDS:
            FAIL_ROUNDS.discard(context.round_index)
            raise RuntimeError("trainer exploded")
        writer.record(
            source_kind="unit",
            source_id=f"{context.scope}-{context.round_index}",
            sample_count=1,
        )

    def on_batch(self, context, reader):  # pragma: no cover - never reached here
        raise AssertionError("on_batch must not run during round hooks")


class BatchOnlyTrainer:
    def on_batch(self, context, reader):  # pragma: no cover - never reached here
        raise AssertionError("on_batch must not run during round hooks")


recording = RecordingTrainer()
batch_only = BatchOnlyTrainer()


@pytest.fixture(autouse=True)
def _reset_recorders():
    CALLS.clear()
    RATING_COUNTS.clear()
    FAIL_ROUNDS.clear()
    yield
    CALLS.clear()
    RATING_COUNTS.clear()
    FAIL_ROUNDS.clear()


def _register(store: ExperimentStore, trainer: str, model_id: str = MODEL_ID) -> None:
    store.connection.execute(
        """
        INSERT INTO training_models (
            model_id, family_id, trainer_path, min_new_batches, config_json
        ) VALUES (?, ?, ?, 1, '{}')
        """,
        [model_id, FAMILY_ID, trainer],
    )


def _fire(store: ExperimentStore, scope: str, round_index: int, **kwargs):
    return run_round_hooks(
        store,
        session_id="s",
        generation_id="g0",
        generation_index=0,
        scope=scope,
        round_index=round_index,
        family_ids=[FAMILY_ID],
        **kwargs,
    )


def test_hook_runs_once_and_is_not_reoffered(tmp_path: Path):
    """The ledger is what makes a hook safe to re-enter after an interruption."""

    with ExperimentStore(tmp_path / "rounds.duckdb") as store:
        _register(store, "test_model_trainers:recording")
        first = _fire(store, "round", 0)
        assert [item.status for item in first] == ["complete"]
        assert first[0].batches and len(first[0].batches) == 1

        second = _fire(store, "round", 0)
        assert [item.status for item in second] == ["complete"]
        assert second[0].batches == ()
        assert CALLS == [(MODEL_ID, "round", 0)]

        forced = _fire(store, "round", 0, force=True)
        assert forced[0].status == "complete"
        assert len(CALLS) == 2


def test_round_and_generation_scopes_are_tracked_separately(tmp_path: Path):
    with ExperimentStore(tmp_path / "scopes.duckdb") as store:
        _register(store, "test_model_trainers:recording")
        _fire(store, "round", 0)
        _fire(store, "generation", GENERATION_ROUND_INDEX)
        _fire(store, "round", 0)
        _fire(store, "generation", GENERATION_ROUND_INDEX)
        assert CALLS == [
            (MODEL_ID, "round", 0),
            (MODEL_ID, "generation", GENERATION_ROUND_INDEX),
        ]
        rows = store.connection.execute(
            "SELECT scope, round_index, status FROM model_round_hooks ORDER BY scope"
        ).fetchall()
        assert rows == [
            ("generation", GENERATION_ROUND_INDEX, "COMPLETE"),
            ("round", 0, "COMPLETE"),
        ]


def test_a_raising_trainer_is_recorded_and_retried(tmp_path: Path):
    with ExperimentStore(tmp_path / "error.duckdb") as store:
        _register(store, "test_model_trainers:recording")
        FAIL_ROUNDS.add(0)
        failed = _fire(store, "round", 0)
        assert failed[0].status == "error"
        assert "trainer exploded" in (failed[0].error or "")
        status, text = store.connection.execute(
            "SELECT status, error_text FROM model_round_hooks"
        ).fetchone()
        assert status == "ERROR" and "trainer exploded" in text

        # ERROR is retried; COMPLETE and SKIPPED are not.
        retried = _fire(store, "round", 0)
        assert retried[0].status == "complete"


def test_a_trainer_without_on_round_is_skipped_permanently(tmp_path: Path):
    with ExperimentStore(tmp_path / "skip.duckdb") as store:
        _register(store, "test_model_trainers:batch_only")
        assert _fire(store, "round", 0)[0].status == "skipped"
        assert _fire(store, "round", 0)[0].status == "skipped"
        assert store.connection.execute(
            "SELECT status FROM model_round_hooks"
        ).fetchone()[0] == "SKIPPED"


def test_hooks_only_fire_for_declared_families(tmp_path: Path):
    with ExperimentStore(tmp_path / "families.duckdb") as store:
        _register(store, "test_model_trainers:recording")
        _register(store, "test_model_trainers:recording", "other.model")
        store.connection.execute(
            "UPDATE training_models SET family_id = 'other_family' WHERE model_id = 'other.model'"
        )
        reports = _fire(store, "round", 0)
        assert [item.model_id for item in reports] == [MODEL_ID]
        assert CALLS == [(MODEL_ID, "round", 0)]


def test_writer_is_bound_to_its_own_model(tmp_path: Path):
    """A trainer cannot record training data against a model it does not own."""

    with ExperimentStore(tmp_path / "writer.duckdb") as store:
        _register(store, "test_model_trainers:recording")
        _register(store, "test_model_trainers:recording", "other.model")
        writer = TrainingBatchWriter(store, MODEL_ID)
        assert writer.model_id == MODEL_ID
        batch = writer.record(source_kind="unit", source_id="one", sample_count=3)
        assert batch.model_id == MODEL_ID
        assert writer.recorded == (batch.batch_id,)
        owners = store.connection.execute(
            "SELECT DISTINCT model_id FROM training_batches"
        ).fetchall()
        assert owners == [(MODEL_ID,)]


def test_round_reader_rejects_mutating_sql(tmp_path: Path):
    with ExperimentStore(tmp_path / "reader.duckdb") as store:
        _register(store, "test_model_trainers:mutating")
        report = _fire(store, "round", 0)[0]
        assert report.status == "error"
        assert "only permits SELECT/WITH" in (report.error or "")


class MutatingTrainer:
    def on_round(self, context, reader, writer) -> None:
        reader.fetchall("DELETE FROM training_models")

    def on_batch(self, context, reader):  # pragma: no cover - never reached here
        raise AssertionError("on_batch must not run during round hooks")


mutating = MutatingTrainer()


def _league(tmp_path: Path) -> tuple[str, Path]:
    config = LeagueConfig(
        name="hooks",
        population_size=4,
        rounds_per_generation=2,
        hall_of_fame_rounds=0,
        episode_steps=6,
        seed=5,
        jobs=1,
        training_manifests=("agents/configurable_crop_surrogate/agent.toml",),
    )
    database = tmp_path / "league.duckdb"
    return create_session(config, database, session_id="hooks"), database


def test_generation_hook_sees_closed_ratings_and_never_refires(tmp_path: Path):
    """Pins hook ordering: `_close_generation` writes ratings before the hook."""

    session_id, database = _league(tmp_path)
    with ExperimentStore(database) as store:
        store.connection.execute(
            """
            INSERT INTO training_models (
                model_id, family_id, trainer_path, min_new_batches, config_json
            ) VALUES (?, 'configurable_crop_surrogate',
                      'test_model_trainers:recording', 1, '{}')
            """,
            [MODEL_ID],
        )
    run_generation(session_id, database, tmp_path / "artifacts")

    scopes = [(scope, index) for _, scope, index in CALLS]
    assert scopes == [("round", 0), ("round", 1), ("generation", GENERATION_ROUND_INDEX)]
    assert RATING_COUNTS and RATING_COUNTS[0] > 0

    before = list(CALLS)
    run_generation(session_id, database, tmp_path / "artifacts")
    assert CALLS == before, "a resumed generation must not re-fire its hooks"


def test_training_family_ids_defaults_to_the_bred_manifest():
    config = LeagueConfig(manifest="agents/configurable_crop_surrogate/agent.toml")
    assert _training_family_ids(config) == ("configurable_crop_surrogate",)

    both = LeagueConfig(
        manifest="agents/configurable_crop_surrogate/agent.toml",
        training_manifests=(
            "agents/configurable_crop_surrogate/agent.toml",
            "agents/dqn_job_scorer/agent.toml",
        ),
    )
    assert _training_family_ids(both) == ("configurable_crop_surrogate", "dqn_job_scorer")


def test_session_config_round_trips_training_manifests(tmp_path: Path):
    session_id, database = _league(tmp_path)
    with ExperimentStore(database) as store:
        stored = json.loads(
            store.connection.execute(
                "SELECT config_json FROM training_sessions WHERE session_id = ?",
                [session_id],
            ).fetchone()[0]
        )
    assert stored["training_manifests"] == [
        "agents/configurable_crop_surrogate/agent.toml"
    ]


def test_record_training_batch_still_rejects_unknown_models(tmp_path: Path):
    with ExperimentStore(tmp_path / "unknown.duckdb") as store:
        with pytest.raises(ValueError, match="unknown training model"):
            record_training_batch(
                store, "nope.model", source_kind="unit", source_id="one", sample_count=1
            )
