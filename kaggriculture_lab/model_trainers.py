"""The round hook: the league's clock, offered to each model's trainer.

`model_updates` owns the batch hook -- "here is new training data, produce a
candidate checkpoint". This module owns the hook that *produces* that data:
after every closed round and again after the closed generation, each declared
model's trainer is offered a chance to act.

The two hooks together are the whole lab-to-agent training surface. A trainer is
one object named by `[[models]] trainer` in an `agent.toml`:

    class Trainer:
        def on_round(self, context, reader, writer) -> None: ...   # optional
        def on_batch(self, context, reader) -> ModelUpdateResult: ...

What the lab keeps: which rounds exist, when they closed, whether a hook already
ran, and every decision about promotion. What the agent decides: whether this
round is worth acting on, and what training means.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from kaggriculture_lab.identity import training_artifact_root
from kaggriculture_lab.model_updates import (
    ActiveCheckpoint,
    ModelTrainingReader,
    ModelUpdateContext,
    ModelUpdateResult,
    TrainingBatch,
    _active_checkpoint,
    record_training_batch,
)
from kaggriculture_lab.storage import ExperimentStore

RoundScope = Literal["round", "generation"]

#: `round_index` for generation-scope hooks. A sentinel rather than NULL because
#: DuckDB's UNIQUE does not dedupe NULL-bearing keys, so a nullable column would
#: quietly admit a second generation row.
GENERATION_ROUND_INDEX = -1


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "\0".join(str(part) for part in parts).encode()
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:20]}"


@dataclass(frozen=True, slots=True)
class RoundContext:
    """Where in the league this hook fired, and what the model knows.

    `config` and `active_checkpoint` are resolved here rather than left for the
    trainer to query, so the round hook needs no privileged access to reach the
    two things every trainer wants first.
    """

    model_id: str
    family_id: str
    scope: RoundScope
    session_id: str
    generation_id: str
    generation_index: int
    round_index: int
    active_checkpoint: ActiveCheckpoint | None
    config: dict[str, Any]
    artifacts: Path


class TrainingBatchWriter:
    """The single write a round hook is granted.

    `model_id` is bound at construction, so a trainer cannot record data against
    another model. Everything else a hook needs is a read, and reads go through
    `ModelTrainingReader` -- the same SELECT-only boundary the batch hook and
    every episode evaluator already live under.
    """

    def __init__(self, store: ExperimentStore, model_id: str):
        self._store = store
        self._model_id = model_id
        self._recorded: list[str] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def recorded(self) -> tuple[str, ...]:
        return tuple(self._recorded)

    def record(
        self,
        *,
        source_kind: str,
        source_id: str,
        sample_count: int,
        artifact_path: str | Path | None = None,
        artifact_sha256: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrainingBatch:
        batch = record_training_batch(
            self._store,
            self._model_id,
            source_kind=source_kind,
            source_id=source_id,
            sample_count=sample_count,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            metadata=metadata,
        )
        self._recorded.append(batch.batch_id)
        return batch


class Trainer(Protocol):
    """`on_round` is optional and probed by `getattr`, like `TrainingAdapter.propose`."""

    def on_batch(
        self, context: ModelUpdateContext, reader: ModelTrainingReader
    ) -> ModelUpdateResult: ...


@dataclass(frozen=True, slots=True)
class RoundReport:
    model_id: str
    scope: str
    round_index: int
    status: str
    batches: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_trainer(path: str) -> Any:
    module_name, separator, attribute = path.partition(":")
    if not separator:
        raise ValueError(f"trainer path must use module:attribute syntax: {path}")
    return getattr(importlib.import_module(module_name), attribute)


def session_training_models(
    store: ExperimentStore,
    family_ids: Sequence[str] = (),
    *,
    model_ids: Sequence[str] | None = None,
) -> list[tuple[str, str, str, dict[str, Any], str | None]]:
    """Every registered model belonging to one of `family_ids`.

    The session declares which families it trains (`LeagueConfig.training_manifests`)
    so that a league can host another family as an anchor opponent without also
    driving its trainer.
    """

    selected = tuple(model_ids) if model_ids is not None else tuple(family_ids)
    if not selected:
        return []
    placeholders = ",".join("?" for _ in selected)
    column = "model_id" if model_ids is not None else "family_id"
    rows = store.connection.execute(
        f"""
        SELECT model_id, family_id, trainer_path, config_json, active_checkpoint_id
        FROM training_models WHERE {column} IN ({placeholders}) ORDER BY model_id
        """,
        list(selected),
    ).fetchall()
    return [
        (str(row[0]), str(row[1]), str(row[2]), json.loads(row[3]), row[4])
        for row in rows
    ]


def _ledger_status(
    store: ExperimentStore,
    model_id: str,
    scope: str,
    generation_id: str,
    round_index: int,
) -> str | None:
    row = store.connection.execute(
        """
        SELECT status FROM model_round_hooks
        WHERE model_id = ? AND scope = ? AND generation_id = ? AND round_index = ?
        """,
        [model_id, scope, generation_id, round_index],
    ).fetchone()
    return None if row is None else str(row[0])


def _ledger_begin(
    store: ExperimentStore,
    *,
    model_id: str,
    session_id: str,
    generation_id: str,
    scope: str,
    round_index: int,
    known: str | None,
) -> None:
    if known is None:
        store.connection.execute(
            """
            INSERT INTO model_round_hooks (
                hook_id, model_id, session_id, generation_id, scope, round_index, status
            ) VALUES (?, ?, ?, ?, ?, ?, 'RUNNING')
            """,
            [
                _stable_id("hook", model_id, scope, generation_id, round_index),
                model_id,
                session_id,
                generation_id,
                scope,
                round_index,
            ],
        )
        return
    store.connection.execute(
        """
        UPDATE model_round_hooks SET status = 'RUNNING', error_text = NULL,
            started_at = current_timestamp, completed_at = NULL
        WHERE model_id = ? AND scope = ? AND generation_id = ? AND round_index = ?
        """,
        [model_id, scope, generation_id, round_index],
    )


def _ledger_finish(
    store: ExperimentStore,
    *,
    model_id: str,
    generation_id: str,
    scope: str,
    round_index: int,
    status: str,
    batches: int = 0,
    error: str | None = None,
) -> None:
    store.connection.execute(
        """
        UPDATE model_round_hooks SET status = ?, batches_recorded = ?, error_text = ?,
            completed_at = current_timestamp
        WHERE model_id = ? AND scope = ? AND generation_id = ? AND round_index = ?
        """,
        [status, batches, error, model_id, scope, generation_id, round_index],
    )


def run_round_hooks(
    store: ExperimentStore,
    *,
    session_id: str,
    generation_id: str,
    generation_index: int,
    scope: RoundScope,
    round_index: int,
    family_ids: Sequence[str] = (),
    model_ids: Sequence[str] | None = None,
    artifacts: str | Path = "artifacts",
    model_id: str | None = None,
    force: bool = False,
) -> list[RoundReport]:
    """Offer one round to every declared model's trainer, at most once.

    The ledger is what makes this safe to call on every resume: a hook may roll
    out episodes for minutes, and "already done" cannot be re-derived from
    pairing status -- resuming a half-finished generation would re-fire every
    round that closed before the interruption.

    A trainer that raises is recorded and skipped, never propagated: one agent's
    broken trainer must not abort another family's league, the same posture the
    episode-evaluator hook already takes.
    """

    reports: list[RoundReport] = []
    for identifier, family_id, trainer_path, config, active_id in session_training_models(
        store, family_ids, model_ids=model_ids
    ):
        if model_id is not None and identifier != model_id:
            continue
        known = _ledger_status(store, identifier, scope, generation_id, round_index)
        if known in {"COMPLETE", "SKIPPED"} and not force:
            reports.append(RoundReport(identifier, scope, round_index, known.lower()))
            continue
        _ledger_begin(
            store,
            model_id=identifier,
            session_id=session_id,
            generation_id=generation_id,
            scope=scope,
            round_index=round_index,
            known=known,
        )
        writer = TrainingBatchWriter(store, identifier)
        try:
            trainer = load_trainer(trainer_path)
            hook = getattr(trainer, "on_round", None)
            if hook is None:
                _ledger_finish(
                    store,
                    model_id=identifier,
                    generation_id=generation_id,
                    scope=scope,
                    round_index=round_index,
                    status="SKIPPED",
                )
                reports.append(RoundReport(identifier, scope, round_index, "skipped"))
                continue
            context = RoundContext(
                model_id=identifier,
                family_id=family_id,
                scope=scope,
                session_id=session_id,
                generation_id=generation_id,
                generation_index=generation_index,
                round_index=round_index,
                active_checkpoint=_active_checkpoint(store, active_id),
                config=config,
                artifacts=training_artifact_root(artifacts, config),
            )
            hook(context, ModelTrainingReader(store), writer)
        except Exception as error:
            _ledger_finish(
                store,
                model_id=identifier,
                generation_id=generation_id,
                scope=scope,
                round_index=round_index,
                status="ERROR",
                batches=len(writer.recorded),
                error="".join(traceback.format_exception(error)),
            )
            reports.append(
                RoundReport(
                    identifier,
                    scope,
                    round_index,
                    "error",
                    writer.recorded,
                    str(error),
                )
            )
            continue
        _ledger_finish(
            store,
            model_id=identifier,
            generation_id=generation_id,
            scope=scope,
            round_index=round_index,
            status="COMPLETE",
            batches=len(writer.recorded),
        )
        reports.append(
            RoundReport(identifier, scope, round_index, "complete", writer.recorded)
        )
    return reports
