"""Incremental, model-scoped training orchestration."""

from __future__ import annotations

import hashlib
import importlib
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from kaggriculture_lab.identity import training_artifact_root
from kaggriculture_lab.read_only import fetchall_select
from kaggriculture_lab.storage import ExperimentStore


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "\0".join(str(part) for part in parts).encode()
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:20]}"


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    model_id: str
    sequence: int
    batch_id: str
    source_kind: str
    source_id: str
    artifact_path: str | None
    artifact_sha256: str | None
    sample_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ActiveCheckpoint:
    checkpoint_id: str
    trainer_artifact_path: str
    inference_artifact_path: str
    artifact_sha256: str
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelUpdateContext:
    model_id: str
    family_id: str
    update_id: str
    cutoff_batch_seq: int
    new_batches: tuple[TrainingBatch, ...]
    all_batches: tuple[TrainingBatch, ...]
    active_checkpoint: ActiveCheckpoint | None
    config: dict[str, Any]
    artifacts: Path
    dataset_sha256: str


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    kind: str
    metrics: dict[str, Any]
    passed: bool
    opponent_checkpoint_id: str | None = None


@dataclass(frozen=True, slots=True)
class ModelUpdateResult:
    checkpoint_id: str
    trainer_artifact_path: str
    inference_artifact_path: str
    artifact_sha256: str
    metrics: dict[str, Any]
    gate: dict[str, Any]
    promote: bool
    evaluations: tuple[ModelEvaluation, ...] = ()


class ModelUpdater(Protocol):
    """The batch hook: one call, one candidate checkpoint.

    Everything around it -- which batches are new, whether there are enough of
    them, and whether the result is promoted -- stays here. The agent decides
    only what training on them means. `kaggriculture_lab.model_trainers` defines
    the round hook that feeds it.
    """

    def on_batch(
        self, context: ModelUpdateContext, reader: ModelTrainingReader
    ) -> ModelUpdateResult: ...


@dataclass(frozen=True, slots=True)
class UpdateReport:
    model_id: str
    status: str
    new_batches: int
    required_batches: int
    cutoff_batch_seq: int
    update_id: str | None = None
    checkpoint_id: str | None = None
    active_checkpoint_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelTrainingReader:
    """Read-only-by-convention view supplied to family-specific updaters."""

    def __init__(self, store: ExperimentStore):
        self._connection = store.connection

    def fetchall(self, sql: str, parameters: list[Any] | tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        return fetchall_select(
            self._connection, sql, parameters, reader_name="model training reader"
        )

    def fetchone(self, sql: str, parameters: list[Any] | tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
        rows = self.fetchall(sql, parameters)
        return rows[0] if rows else None


def _load_object(path: str) -> Any:
    module_name, separator, attribute = path.partition(":")
    if not separator:
        raise ValueError(f"callable path must use module:attribute syntax: {path}")
    return getattr(importlib.import_module(module_name), attribute)


def _batch_from_row(row: tuple[Any, ...]) -> TrainingBatch:
    return TrainingBatch(
        model_id=str(row[0]),
        sequence=int(row[1]),
        batch_id=str(row[2]),
        source_kind=str(row[3]),
        source_id=str(row[4]),
        artifact_path=row[5],
        artifact_sha256=row[6],
        sample_count=int(row[7]),
        metadata=json.loads(row[8]),
    )


def record_training_batch(
    store: ExperimentStore,
    model_id: str,
    *,
    source_kind: str,
    source_id: str,
    sample_count: int,
    artifact_path: str | Path | None = None,
    artifact_sha256: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TrainingBatch:
    """Idempotently register one model-defined unit of new training data."""

    if sample_count < 0:
        raise ValueError("sample_count cannot be negative")
    known = store.connection.execute(
        """
        SELECT model_id, batch_seq, batch_id, source_kind, source_id,
               artifact_path, artifact_sha256, sample_count, metadata_json
        FROM training_batches
        WHERE model_id = ? AND source_kind = ? AND source_id = ?
        """,
        [model_id, source_kind, source_id],
    ).fetchone()
    if known:
        return _batch_from_row(known)
    if store.connection.execute(
        "SELECT 1 FROM training_models WHERE model_id = ?", [model_id]
    ).fetchone() is None:
        raise ValueError(f"unknown training model: {model_id}")
    path_text = str(artifact_path) if artifact_path is not None else None
    payload = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"))
    batch_id = _stable_id(
        "batch", model_id, source_kind, source_id, artifact_sha256 or "", payload
    )
    store.connection.execute("BEGIN")
    try:
        sequence = int(
            store.connection.execute(
                "SELECT coalesce(max(batch_seq), 0) + 1 FROM training_batches WHERE model_id = ?",
                [model_id],
            ).fetchone()[0]
        )
        store.connection.execute(
            """
            INSERT INTO training_batches (
                model_id, batch_seq, batch_id, source_kind, source_id,
                artifact_path, artifact_sha256, sample_count, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                model_id,
                sequence,
                batch_id,
                source_kind,
                source_id,
                path_text,
                artifact_sha256,
                sample_count,
                payload,
            ],
        )
        store.connection.execute("COMMIT")
    except Exception:
        store.connection.execute("ROLLBACK")
        raise
    return TrainingBatch(
        model_id,
        sequence,
        batch_id,
        source_kind,
        source_id,
        path_text,
        artifact_sha256,
        sample_count,
        metadata or {},
    )


def _active_checkpoint(store: ExperimentStore, checkpoint_id: str | None) -> ActiveCheckpoint | None:
    if not checkpoint_id:
        return None
    row = store.connection.execute(
        """
        SELECT checkpoint_id, trainer_artifact_path, inference_artifact_path,
               artifact_sha256, metrics_json
        FROM model_training_checkpoints WHERE checkpoint_id = ?
        """,
        [checkpoint_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"active checkpoint is missing: {checkpoint_id}")
    return ActiveCheckpoint(str(row[0]), str(row[1]), str(row[2]), str(row[3]), json.loads(row[4]))


def update_model_in_store(
    store: ExperimentStore,
    model_id: str,
    *,
    artifacts: str | Path = "artifacts",
    force: bool = False,
) -> UpdateReport:
    row = store.connection.execute(
        """
        SELECT family_id, trainer_path, min_new_batches, config_json,
               active_checkpoint_id, last_attempted_batch_seq
        FROM training_models WHERE model_id = ?
        """,
        [model_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown training model: {model_id}")
    family_id, trainer_path, minimum, config_json, active_id, watermark = row
    batch_rows = store.connection.execute(
        """
        SELECT model_id, batch_seq, batch_id, source_kind, source_id,
               artifact_path, artifact_sha256, sample_count, metadata_json
        FROM training_batches WHERE model_id = ? ORDER BY batch_seq
        """,
        [model_id],
    ).fetchall()
    batches = tuple(_batch_from_row(item) for item in batch_rows)
    new_batches = tuple(item for item in batches if item.sequence > int(watermark))
    cutoff = new_batches[-1].sequence if new_batches else int(watermark)
    if not new_batches or (len(new_batches) < int(minimum) and not force):
        return UpdateReport(
            model_id,
            "deferred",
            len(new_batches),
            int(minimum),
            cutoff,
            active_checkpoint_id=active_id,
        )

    selected = tuple(item for item in batches if item.sequence <= cutoff)
    digest_payload = [
        [item.batch_id, item.artifact_sha256, item.sample_count] for item in selected
    ]
    dataset_digest = hashlib.sha256(
        json.dumps(digest_payload, separators=(",", ":")).encode()
    ).hexdigest()
    update_id = _stable_id("update", model_id, active_id or "", cutoff, dataset_digest)
    existing = store.connection.execute(
        """
        SELECT status, candidate_checkpoint_id, error_text,
               started_at < now() - INTERVAL '6 hours' AS stale
        FROM model_updates WHERE update_id = ?
        """,
        [update_id],
    ).fetchone()
    if existing and existing[0] == "RUNNING" and not existing[3]:
        return UpdateReport(
            model_id,
            "busy",
            len(new_batches),
            int(minimum),
            cutoff,
            update_id,
            existing[1],
            active_id,
        )
    if existing and existing[0] not in {"ERROR", "RUNNING"}:
        return UpdateReport(
            model_id,
            str(existing[0]).lower(),
            len(new_batches),
            int(minimum),
            cutoff,
            update_id,
            existing[1],
            active_id,
            existing[2],
        )
    if existing:
        store.connection.execute(
            """
            UPDATE model_updates SET status = 'RUNNING', error_text = NULL,
                started_at = current_timestamp, completed_at = NULL
            WHERE update_id = ?
            """,
            [update_id],
        )
    else:
        store.connection.execute(
            """
            INSERT INTO model_updates (
                update_id, model_id, parent_checkpoint_id, cutoff_batch_seq,
                new_batch_count, dataset_sha256, status
            ) VALUES (?, ?, ?, ?, ?, ?, 'RUNNING')
            """,
            [update_id, model_id, active_id, cutoff, len(new_batches), dataset_digest],
        )

    context = ModelUpdateContext(
        model_id=str(model_id),
        family_id=str(family_id),
        update_id=update_id,
        cutoff_batch_seq=cutoff,
        new_batches=new_batches,
        all_batches=selected,
        active_checkpoint=_active_checkpoint(store, active_id),
        config=json.loads(config_json),
        artifacts=training_artifact_root(artifacts, json.loads(config_json)),
        dataset_sha256=dataset_digest,
    )
    try:
        trainer = _load_object(str(trainer_path))
        result = trainer.on_batch(context, ModelTrainingReader(store))
        if not isinstance(result, ModelUpdateResult):
            raise TypeError("trainer.on_batch must return ModelUpdateResult")
        status = "PROMOTED" if result.promote else "REJECTED"
        store.connection.execute("BEGIN")
        try:
            store.connection.execute(
                """
                INSERT OR REPLACE INTO model_training_checkpoints VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp
                )
                """,
                [
                    result.checkpoint_id,
                    model_id,
                    update_id,
                    active_id,
                    result.trainer_artifact_path,
                    result.inference_artifact_path,
                    result.artifact_sha256,
                    dataset_digest,
                    json.dumps(result.metrics, sort_keys=True),
                    json.dumps(result.gate, sort_keys=True),
                    result.promote,
                ],
            )
            for evaluation in result.evaluations:
                evaluation_id = _stable_id(
                    "evaluation", update_id, evaluation.kind, evaluation.opponent_checkpoint_id or ""
                )
                store.connection.execute(
                    """
                    INSERT OR REPLACE INTO model_evaluations VALUES (
                        ?, ?, ?, ?, ?, ?, ?, current_timestamp
                    )
                    """,
                    [
                        evaluation_id,
                        update_id,
                        model_id,
                        evaluation.kind,
                        evaluation.opponent_checkpoint_id,
                        json.dumps(evaluation.metrics, sort_keys=True),
                        evaluation.passed,
                    ],
                )
            store.connection.execute(
                """
                UPDATE model_updates SET status = ?, candidate_checkpoint_id = ?,
                    metrics_json = ?, gate_json = ?, completed_at = current_timestamp
                WHERE update_id = ?
                """,
                [
                    status,
                    result.checkpoint_id,
                    json.dumps(result.metrics, sort_keys=True),
                    json.dumps(result.gate, sort_keys=True),
                    update_id,
                ],
            )
            store.connection.execute(
                """
                UPDATE training_models SET last_attempted_batch_seq = ?,
                    active_checkpoint_id = CASE WHEN ? THEN ? ELSE active_checkpoint_id END,
                    updated_at = current_timestamp WHERE model_id = ?
                """,
                [cutoff, result.promote, result.checkpoint_id, model_id],
            )
            store.connection.execute("COMMIT")
        except Exception:
            store.connection.execute("ROLLBACK")
            raise
        return UpdateReport(
            model_id,
            status.lower(),
            len(new_batches),
            int(minimum),
            cutoff,
            update_id,
            result.checkpoint_id,
            result.checkpoint_id if result.promote else active_id,
        )
    except Exception as error:
        store.connection.execute(
            """
            UPDATE model_updates SET status = 'ERROR', error_text = ?,
                completed_at = current_timestamp WHERE update_id = ?
            """,
            ["".join(traceback.format_exception(error)), update_id],
        )
        return UpdateReport(
            model_id,
            "error",
            len(new_batches),
            int(minimum),
            cutoff,
            update_id,
            active_checkpoint_id=active_id,
            error=str(error),
        )


def update_model(
    model_id: str,
    database: str | Path = "artifacts/experiments.duckdb",
    *,
    artifacts: str | Path = "artifacts",
    force: bool = False,
) -> UpdateReport:
    with ExperimentStore(database) as store:
        return update_model_in_store(store, model_id, artifacts=artifacts, force=force)


def update_models(
    database: str | Path = "artifacts/experiments.duckdb",
    *,
    model_id: str | None = None,
    artifacts: str | Path = "artifacts",
    force: bool = False,
) -> list[UpdateReport]:
    with ExperimentStore(database) as store:
        ids = [model_id] if model_id else [
            str(row[0])
            for row in store.connection.execute(
                "SELECT model_id FROM training_models ORDER BY model_id"
            ).fetchall()
        ]
        return [
            update_model_in_store(store, item, artifacts=artifacts, force=force)
            for item in ids
        ]
