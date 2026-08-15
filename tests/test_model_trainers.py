"""The round hook: fires once per closed round, and never twice."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from kaggriculture_lab.identity import AgentManifest, TrainingAgentSpec
from kaggriculture_lab.league import (
    LeagueConfig,
    _training_family_ids,
    create_session,
    run_generation,
)
from kaggriculture_lab.model_trainers import (
    GENERATION_ROUND_INDEX,
    EvaluationPolicy,
    RoundContext,
    TrainingBatchWriter,
    run_round_hooks,
)
from kaggriculture_lab.model_updates import record_training_batch
from kaggriculture_lab.storage import ExperimentStore
from tests.postgres_support import postgres_database, seed_reference_rows

MODEL_ID = "test.rounds"
FAMILY_ID = "test_family"

#: Every hook call this module's trainers see, as (model_id, scope, round_index).
CALLS: list[tuple[str, str, int]] = []
#: Rating-snapshot counts observed by generation-scope calls.
RATING_COUNTS: list[int] = []
#: Rounds whose hook should raise, consumed one failure per entry.
FAIL_ROUNDS: set[int] = set()
#: Rendezvous for the concurrency tests. Set to a `threading.Barrier` sized to the
#: number of models expected to be in flight at once; a hook that reaches it and
#: waits can only be released by its peers, so passing it *is* the proof they ran
#: together rather than a guess from timing.
BARRIER: threading.Barrier | None = None
#: Model ids whose hook should raise, for error isolation under fan-out.
FAIL_MODELS: set[str] = set()


class RecordingTrainer:
    def on_round(self, context, reader, writer) -> None:
        CALLS.append((context.model_id, context.scope, context.round_index))
        if context.scope == "generation":
            RATING_COUNTS.append(
                int(
                    reader.fetchone(
                        "SELECT count(*) FROM rating_snapshots WHERE generation_id = %s",
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


class RendezvousTrainer:
    """Waits for every peer to arrive, then records and returns.

    The wait is the assertion. Run serially, the first hook blocks forever and the
    barrier times out; the only way all of them get through is if they were in
    flight at the same time.
    """

    def on_round(self, context, reader, writer) -> None:
        if BARRIER is not None:
            BARRIER.wait(timeout=60)
        # After the rendezvous, so a query proves each thread has a working store
        # of its own rather than sharing the caller's connection.
        reader.fetchone("SELECT count(*) FROM training_models")
        CALLS.append((context.model_id, context.scope, context.round_index))
        if context.model_id in FAIL_MODELS:
            raise RuntimeError(f"{context.model_id} exploded")
        writer.record(
            source_kind="unit",
            source_id=f"{context.scope}-{context.round_index}",
            sample_count=1,
        )

    def on_batch(self, context, reader):  # pragma: no cover - never reached here
        raise AssertionError("on_batch must not run during round hooks")


recording = RecordingTrainer()
batch_only = BatchOnlyTrainer()
rendezvous = RendezvousTrainer()


@pytest.fixture(autouse=True)
def _reset_recorders():
    global BARRIER
    CALLS.clear()
    RATING_COUNTS.clear()
    FAIL_ROUNDS.clear()
    FAIL_MODELS.clear()
    BARRIER = None
    yield
    CALLS.clear()
    RATING_COUNTS.clear()
    FAIL_ROUNDS.clear()
    FAIL_MODELS.clear()
    BARRIER = None


def _register(store: ExperimentStore, trainer: str, model_id: str = MODEL_ID) -> None:
    seed_reference_rows(
        store.connection, session_id="s", generation_id="g0", family_id=FAMILY_ID
    )
    seed_reference_rows(store.connection, family_id="other_family")
    store.connection.execute(
        """
        INSERT INTO training_models (
            model_id, family_id, trainer_path, min_new_batches, config_json
        ) VALUES (%s, %s, %s, 1, '{}')
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


def test_models_run_their_hooks_at_the_same_time(tmp_path: Path):
    """The serial walk was the whole overhead on a population run.

    Measured on `market-free-population-v23`, ten learners on a 32-core box: max
    concurrent model updates 1, load average 9, 77 minutes a generation of which
    96% was simulation nothing was waiting on. The models are independent
    lineages, so there was never a reason to walk them.

    A barrier rather than a stopwatch: every hook waits for its peers, so the call
    only returns if they genuinely overlapped.
    """

    global BARRIER
    models = [f"{MODEL_ID}.{index}" for index in range(3)]
    with ExperimentStore(postgres_database(tmp_path, "concurrent")) as store:
        for model in models:
            _register(store, "tests.test_model_trainers:rendezvous", model_id=model)
        BARRIER = threading.Barrier(len(models))

        reports = _fire(store, "round", 0, concurrency=len(models))

    # Order is by model, not by completion, so the two paths are indistinguishable
    # from the return value.
    assert [item.model_id for item in reports] == models
    assert [item.status for item in reports] == ["complete"] * len(models)
    assert sorted(call[0] for call in CALLS) == models


def test_fan_out_workers_do_not_re_bootstrap_the_schema(tmp_path: Path):
    """`market-free-population-v25` died on generation zero for this.

    `ExperimentStore` ensures the schema on open, which is right for the first
    connection and a deadlock for the fourth: `_initialize_schema` executes the
    whole of `schema.sql`, and two of those running at once take the same objects'
    locks in different orders.

        psycopg.errors.DeadlockDetected: deadlock detected
        Process 154699 waits for ShareLock on relation 16444; blocked by 154301.

    Asserted on the flag rather than by racing stores, because the deadlock needs
    a particular interleaving to show up -- the concurrency tests above open three
    stores at once and got through it on luck.
    """

    global BARRIER
    models = [f"{MODEL_ID}.{index}" for index in range(3)]
    initialized: list[bool] = []
    original = ExperimentStore.__init__

    def recording_init(self, database, *, initialize=True):
        initialized.append(initialize)
        original(self, database, initialize=initialize)

    with ExperimentStore(postgres_database(tmp_path, "bootstrap")) as store:
        for model in models:
            _register(store, "tests.test_model_trainers:rendezvous", model_id=model)
        BARRIER = threading.Barrier(len(models))
        ExperimentStore.__init__ = recording_init
        try:
            reports = _fire(store, "round", 0, concurrency=len(models))
        finally:
            ExperimentStore.__init__ = original

    assert [item.status for item in reports] == ["complete"] * len(models)
    # One store per model, none of them re-running the bootstrap.
    assert initialized == [False] * len(models)


def test_a_failing_model_does_not_take_its_peers_down_under_fan_out(tmp_path: Path):
    """Error isolation is the one property the fan-out must not quietly drop.

    One broken trainer being recorded and skipped rather than propagated is the
    existing contract; running the models on threads gives an exception a second
    way to escape, and `pool.map` re-raising the first failure would abort the rest
    of the generation.
    """

    global BARRIER
    models = [f"{MODEL_ID}.{index}" for index in range(3)]
    with ExperimentStore(postgres_database(tmp_path, "isolation")) as store:
        for model in models:
            _register(store, "tests.test_model_trainers:rendezvous", model_id=model)
        BARRIER = threading.Barrier(len(models))
        FAIL_MODELS.add(models[1])

        reports = _fire(store, "round", 0, concurrency=len(models))

        statuses = {item.model_id: item.status for item in reports}
        assert statuses == {
            models[0]: "complete",
            models[1]: "error",
            models[2]: "complete",
        }
        # The ledger has to agree, because that is what a resume reads.
        recorded = dict(
            store.connection.execute(
                "SELECT model_id, status FROM model_round_hooks WHERE round_index = 0"
            ).fetchall()
        )
        assert recorded == {
            models[0]: "COMPLETE",
            models[1]: "ERROR",
            models[2]: "COMPLETE",
        }


def test_concurrency_of_one_uses_the_callers_own_store(tmp_path: Path):
    """The default path must not open connections it never used to.

    A store is a connection, and a generation fires this once per round per model.
    Silently doubling that on the single-model case would be a regression paid for
    by every existing league.
    """

    with ExperimentStore(postgres_database(tmp_path, "serial")) as store:
        _register(store, "tests.test_model_trainers:recording")
        opened: list[object] = []
        original = ExperimentStore.__init__

        def counting_init(self, database):
            opened.append(database)
            original(self, database)

        ExperimentStore.__init__ = counting_init
        try:
            reports = _fire(store, "round", 0)
        finally:
            ExperimentStore.__init__ = original

    assert [item.status for item in reports] == ["complete"]
    assert opened == []


def test_hook_runs_once_and_is_not_reoffered(tmp_path: Path):
    """The ledger is what makes a hook safe to re-enter after an interruption."""

    with ExperimentStore(postgres_database(tmp_path, "rounds")) as store:
        _register(store, "tests.test_model_trainers:recording")
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
    with ExperimentStore(postgres_database(tmp_path, "scopes")) as store:
        _register(store, "tests.test_model_trainers:recording")
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
    with ExperimentStore(postgres_database(tmp_path, "error")) as store:
        _register(store, "tests.test_model_trainers:recording")
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
    with ExperimentStore(postgres_database(tmp_path, "skip")) as store:
        _register(store, "tests.test_model_trainers:batch_only")
        assert _fire(store, "round", 0)[0].status == "skipped"
        assert _fire(store, "round", 0)[0].status == "skipped"
        assert store.connection.execute(
            "SELECT status FROM model_round_hooks"
        ).fetchone()[0] == "SKIPPED"


def test_hooks_only_fire_for_declared_families(tmp_path: Path):
    with ExperimentStore(postgres_database(tmp_path, "families")) as store:
        _register(store, "tests.test_model_trainers:recording")
        _register(store, "tests.test_model_trainers:recording", "other.model")
        store.connection.execute(
            "UPDATE training_models SET family_id = 'other_family' WHERE model_id = 'other.model'"
        )
        reports = _fire(store, "round", 0)
        assert [item.model_id for item in reports] == [MODEL_ID]
        assert CALLS == [(MODEL_ID, "round", 0)]


def test_writer_is_bound_to_its_own_model(tmp_path: Path):
    """A trainer cannot record training data against a model it does not own."""

    with ExperimentStore(postgres_database(tmp_path, "writer")) as store:
        _register(store, "tests.test_model_trainers:recording")
        _register(store, "tests.test_model_trainers:recording", "other.model")
        writer = TrainingBatchWriter(store, MODEL_ID)
        assert writer.model_id == MODEL_ID
        batch = writer.record(source_kind="unit", source_id="one", sample_count=3)
        assert batch.model_id == MODEL_ID
        assert writer.recorded == (batch.batch_id,)
        owners = store.connection.execute(
            "SELECT DISTINCT model_id FROM training_batches"
        ).fetchall()
        assert owners == [(MODEL_ID,)]


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
    database = postgres_database(tmp_path, "league")
    return create_session(config, database, session_id="hooks"), database


def test_generation_hook_sees_closed_ratings_and_never_refires(tmp_path: Path):
    """Pins hook ordering: `_close_generation` writes ratings before the hook."""

    session_id, database = _league(tmp_path)
    with ExperimentStore(database) as store:
        store.connection.execute(
            """
            UPDATE training_models
            SET trainer_path = 'tests.test_model_trainers:recording', min_new_batches = 1
            WHERE model_id = 'configurable_crop_surrogate.surrogate'
            """
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
                "SELECT config_json FROM training_sessions WHERE session_id = %s",
                [session_id],
            ).fetchone()[0]
        )
    assert stored["training_manifests"] == [
        "agents/configurable_crop_surrogate/agent.toml"
    ]


def test_record_training_batch_still_rejects_unknown_models(tmp_path: Path):
    with ExperimentStore(postgres_database(tmp_path, "unknown")) as store:
        with pytest.raises(ValueError, match="unknown training model"):
            record_training_batch(
                store, "nope.model", source_kind="unit", source_id="one", sample_count=1
            )


def test_trainer_evaluation_round_is_paired_recorded_and_resumable(tmp_path: Path):
    with ExperimentStore(postgres_database(tmp_path, "evaluation")) as store:
        seed_reference_rows(store.connection, run_id="session")
        store.connection.execute(
            "INSERT INTO training_sessions VALUES (%s, %s, %s, %s, %s, %s, current_timestamp)",
            ["session", "test", "session", "RUNNING", 1, "{}"],
        )
        store.connection.execute(
            """
            INSERT INTO generations (generation_id, session_id, generation_index, status)
            VALUES (%s, %s, %s, %s)
            """,
            ["session:g0000", "session", 0, "RUNNING"],
        )
        _register(store, "tests.test_model_trainers:recording")
        writer = TrainingBatchWriter(store, MODEL_ID)
        context = RoundContext(
            MODEL_ID,
            FAMILY_ID,
            "round",
            "session",
            "session:g0000",
            0,
            0,
            None,
            {},
            tmp_path,
        )
        matchups = [
            (
                EvaluationPolicy("builtin:pass", "pass"),
                EvaluationPolicy("builtin:starter", "starter"),
                17,
            )
        ]
        first = writer.run_evaluation_round(
            context=context,
            request_id="request-a",
            pairing_kind="unit_evaluation",
            matchups=matchups,
            jobs=1,
            episode_steps=2,
        )
        second = writer.run_evaluation_round(
            context=context,
            request_id="request-a",
            pairing_kind="unit_evaluation",
            matchups=matchups,
            jobs=1,
            episode_steps=2,
        )
        assert first == second
        assert first["pairs"] == 1
        assert store.connection.execute(
            "SELECT count(*) FROM league_pairings WHERE pairing_kind = 'unit_evaluation'"
        ).fetchone()[0] == 1
        assert store.connection.execute("SELECT count(*) FROM pairing_episodes").fetchone()[0] == 2
        assert store.connection.execute("SELECT count(*) FROM episode_payloads").fetchone()[0] == 2


def test_meeting_a_manifest_does_not_rewrite_its_models_configuration(tmp_path: Path):
    """Recording an opponent must not reset the model a runner is training.

    A rollout pairing registers whatever manifest its opponent came from, and a
    self-play opponent's manifest is the learner's own. `register_family` would
    rewrite that model's `config_json` back to the manifest defaults, silently
    discarding the runner's overlay mid-generation.
    """
    from kaggriculture_lab.model_trainers import _register_policies

    manifest_path = "agents/dqn_job_scorer/agent.toml"
    manifest = AgentManifest.load(manifest_path)
    with ExperimentStore(postgres_database(tmp_path, "family-only")) as store:
        store.register_training_agent(
            manifest,
            manifest_path,
            TrainingAgentSpec(
                "dqn_job_scorer", "default", model_config={"jobs": 3, "opponents": ["frozen"]}
            ),
        )
        _register_policies(
            store,
            [
                EvaluationPolicy(
                    "opponent-a",
                    "agents/dqn_job_scorer/main.py?weights=snapshot.npz",
                    manifest_path,
                    "rollout_opponent",
                )
            ],
            source="policy_matchup",
            register_families=True,
        )
        config = json.loads(
            store.connection.execute(
                "SELECT config_json FROM training_models WHERE model_id = %s",
                ["dqn_job_scorer.adaptive_hybrid_q_v2"],
            ).fetchone()[0]
        )
        family = store.connection.execute(
            "SELECT manifest_path FROM agent_families WHERE family_id = %s",
            ["dqn_job_scorer"],
        ).fetchone()

    assert config["jobs"] == 3
    assert config["opponents"] == ["frozen"]
    assert family == (manifest_path,)


def test_same_agent_type_registers_independent_named_models(tmp_path: Path):
    manifest = AgentManifest.load("agents/dqn_job_scorer/agent.toml")
    with ExperimentStore(postgres_database(tmp_path, "named")) as store:
        first = store.register_training_agent(
            manifest,
            "agents/dqn_job_scorer/agent.toml",
            TrainingAgentSpec("dqn_job_scorer", "dqn05"),
        )
        second = store.register_training_agent(
            manifest,
            "agents/dqn_job_scorer/agent.toml",
            TrainingAgentSpec(
                "dqn_job_scorer",
                "dqn06",
                model_config={"jobs": 3, "opponents": ["frozen-dqn"]},
            ),
        )
        rows = store.connection.execute(
            """
            SELECT model_id, agent_type, agent_name, declared_model_id, config_json
            FROM training_models ORDER BY model_id
            """
        ).fetchall()

    assert first == ("dqn_job_scorer.adaptive_hybrid_q_v2@dqn05",)
    assert second == ("dqn_job_scorer.adaptive_hybrid_q_v2@dqn06",)
    assert [(row[1], row[2], row[3]) for row in rows] == [
        ("dqn_job_scorer", "dqn05", "dqn_job_scorer.adaptive_hybrid_q_v2"),
        ("dqn_job_scorer", "dqn06", "dqn_job_scorer.adaptive_hybrid_q_v2"),
    ]
    configs = [json.loads(row[4]) for row in rows]
    assert [item["name"] for item in configs] == ["dqn05", "dqn06"]
    assert configs[0]["seed"] != configs[1]["seed"]
    assert configs[1]["jobs"] == 3
    assert configs[1]["opponents"] == ["frozen-dqn"]
