from __future__ import annotations

import hashlib
from pathlib import Path

from agents.configurable_crop_surrogate.strategy import generate_population
from kaggriculture_lab.identity import AgentManifest, build_revision
from kaggriculture_lab.model_updates import (
    ModelTrainingReader,
    ModelUpdateContext,
    ModelUpdateResult,
    record_training_batch,
    update_model_in_store,
)
from kaggriculture_lab.storage import SCHEMA_VERSION, ExperimentStore
from scripts.rebuild_experiments_db import rebuild


class FakeTrainer:
    def on_batch(self, context: ModelUpdateContext, reader) -> ModelUpdateResult:
        destination = context.artifacts / f"{context.update_id}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(context.dataset_sha256)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        promote = bool(context.config.get("promote", True))
        checkpoint_id = f"checkpoint-{context.cutoff_batch_seq}"
        return ModelUpdateResult(
            checkpoint_id,
            str(destination),
            str(destination),
            digest,
            {"batch_count": len(context.all_batches)},
            {"passed": promote},
            promote,
        )


trainer = FakeTrainer()


def test_model_training_reader_rejects_multi_statement_bypass(tmp_path: Path):
    with ExperimentStore(tmp_path / "reader.duckdb") as store:
        reader = ModelTrainingReader(store)
        assert reader.fetchone("WITH value AS (SELECT 1 AS n) SELECT n FROM value") == (1,)
        for sql in (
            "DELETE FROM training_models",
            "SELECT 1; DELETE FROM training_models",
        ):
            try:
                reader.fetchall(sql)
            except ValueError as error:
                assert "only permits SELECT/WITH" in str(error)
            else:
                raise AssertionError(f"mutating SQL was accepted: {sql}")


def _register(store: ExperimentStore, model_id: str, promote: bool = True) -> None:
    store.connection.execute(
        """
        INSERT INTO training_models (
            model_id, family_id, trainer_path, min_new_batches, config_json
        ) VALUES (?, 'test', 'test_model_updates:trainer', 2, ?)
        """,
        [model_id, '{"promote": true}' if promote else '{"promote": false}'],
    )


def test_thresholds_idempotency_and_promotion(tmp_path: Path):
    with ExperimentStore(tmp_path / "models.duckdb") as store:
        _register(store, "test.model")
        first = record_training_batch(
            store,
            "test.model",
            source_kind="unit",
            source_id="one",
            sample_count=4,
        )
        duplicate = record_training_batch(
            store,
            "test.model",
            source_kind="unit",
            source_id="one",
            sample_count=4,
        )
        assert duplicate.batch_id == first.batch_id
        assert update_model_in_store(store, "test.model", artifacts=tmp_path).status == "deferred"

        record_training_batch(
            store,
            "test.model",
            source_kind="unit",
            source_id="two",
            sample_count=5,
        )
        report = update_model_in_store(store, "test.model", artifacts=tmp_path)
        assert report.status == "promoted"
        assert report.active_checkpoint_id == "checkpoint-2"
        assert update_model_in_store(store, "test.model", artifacts=tmp_path, force=True).status == "deferred"


def test_rejection_advances_only_its_models_watermark(tmp_path: Path):
    with ExperimentStore(tmp_path / "models.duckdb") as store:
        _register(store, "test.rejected", promote=False)
        _register(store, "test.independent")
        for model_id in ("test.rejected", "test.independent"):
            for source in ("one", "two"):
                record_training_batch(
                    store,
                    model_id,
                    source_kind="unit",
                    source_id=source,
                    sample_count=1,
                )
        rejected = update_model_in_store(store, "test.rejected", artifacts=tmp_path)
        assert rejected.status == "rejected"
        assert rejected.active_checkpoint_id is None
        assert update_model_in_store(store, "test.rejected", artifacts=tmp_path).status == "deferred"
        assert update_model_in_store(store, "test.independent", artifacts=tmp_path).status == "promoted"


def test_rebuild_uses_schema_and_backs_up_existing_database(tmp_path: Path):
    database = tmp_path / "experiments.duckdb"
    with ExperimentStore(database) as store:
        store.connection.execute(
            "INSERT INTO schema_migrations (version) VALUES (99)"
        )
    rebuilt, backup = rebuild(database, force=True)
    assert rebuilt == database
    assert backup is not None and backup.exists()
    with ExperimentStore(database) as store:
        assert store.connection.execute(
            "SELECT max(version) FROM schema_migrations"
        ).fetchone()[0] == SCHEMA_VERSION


def test_crop_surrogate_trains_after_two_generation_batches(tmp_path: Path):
    manifest_path = Path("agents/configurable_crop_surrogate/agent.toml")
    manifest = AgentManifest.load(manifest_path)
    strategies = generate_population(12, seed=91)
    revisions = [build_revision(manifest, strategy) for strategy in strategies]
    with ExperimentStore(tmp_path / "crop.duckdb") as store:
        store.register_family(manifest, manifest_path)
        store.connection.execute(
            """
            UPDATE training_models
            SET config_json = '{"n_estimators": 5, "min_samples_leaf": 1}'
            WHERE model_id = 'configurable_crop_surrogate.surrogate'
            """
        )
        store.add_agent_revisions(revisions)
        for generation_index in range(2):
            generation_id = f"crop:g{generation_index:04d}"
            store.connection.execute(
                "INSERT INTO generations VALUES (?, 'crop', ?, 'COMPLETE', NULL, now(), now())",
                [generation_id, generation_index],
            )
            selected = revisions[generation_index * 6 : (generation_index + 1) * 6]
            store.connection.executemany(
                "INSERT INTO rating_snapshots VALUES (?, ?, ?, 80, 0.06, 4, false)",
                [
                    [generation_id, revision.revision_id, 1400 + index * 25]
                    for index, revision in enumerate(selected)
                ],
            )
            record_training_batch(
                store,
                "configurable_crop_surrogate.surrogate",
                source_kind="league_generation",
                source_id=generation_id,
                sample_count=6,
                metadata={"session_id": "crop"},
            )
        report = update_model_in_store(
            store,
            "configurable_crop_surrogate.surrogate",
            artifacts=tmp_path,
        )
        assert report.status == "promoted"
        assert report.active_checkpoint_id is not None
        artifact = store.connection.execute(
            "SELECT inference_artifact_path FROM model_training_checkpoints WHERE checkpoint_id = ?",
            [report.active_checkpoint_id],
        ).fetchone()[0]
        assert Path(artifact).exists()
