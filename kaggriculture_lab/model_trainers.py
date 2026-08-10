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
import math
import multiprocessing
import statistics
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from kaggriculture_lab.identity import (
    AgentManifest,
    AgentRevision,
    code_fingerprint,
    training_artifact_root,
)
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
#: A non-null sentinel keeps hook identity unambiguous across SQL engines.
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


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    """One immutable policy identity used by a trainer-requested league round."""

    revision_id: str
    agent_spec: str
    manifest_path: str | None = None
    role: str = "evaluation"


class TrainingBatchWriter:
    """The batch-recording surface handed to a round hook.

    `model_id` is bound at construction, so a trainer records batches against
    its own model without having to name it on every call. Everything else a
    hook needs is a read, and reads go through `ModelTrainingReader`.
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

    def record_policy_episode(
        self,
        *,
        context: RoundContext,
        manifest_path: str | Path,
        pairing_index: int,
        beta_seat: int,
        population_revision_id: str,
        beta_revision_id: str,
        mixture_id: str,
        result: Any,
    ) -> tuple[str, ...]:
        """Commit one already-run manifest episode to the league ledger.

        The trainer supplies policy identity and the match result; the lab owns
        stable pairing/match IDs, revision registration, and artifact linkage.
        Calling this again during resume is idempotent.
        """
        from kaggriculture_lab.league import (
            LeagueEpisodeTask,
            _finish_pairing,
            _store_orientation_result,
        )
        from kaggriculture_lab.league import (
            _stable_id as league_id,
        )

        manifest = AgentManifest.load(manifest_path)
        self._store.register_family(manifest, manifest_path)
        fingerprint = code_fingerprint(manifest)
        population_seat = 1 - beta_seat
        orientation = 1 - beta_seat
        specs = [seat.spec for seat in result.seats]

        def revision(revision_id: str, spec: str, role: str) -> AgentRevision:
            return AgentRevision(
                revision_id=revision_id,
                family_id=manifest.family_id,
                strategy_id=None,
                entry_point=manifest.entry_point,
                agent_spec=spec,
                code_fingerprint=fingerprint,
                parameter_schema_version=manifest.parameter_schema_version,
                params_json=json.dumps({"role": role}, sort_keys=True),
                source="policy_matchup",
            )

        population = revision(population_revision_id, specs[population_seat], "population")
        beta = revision(beta_revision_id, specs[beta_seat], "best_response")
        self._store.add_agent_revisions((population, beta))
        pairing_id = league_id(
            "policy-pair",
            context.generation_id,
            context.round_index,
            pairing_index,
            population_revision_id,
            beta_revision_id,
            result.seed,
        )
        self._store.connection.execute(
            """
            INSERT INTO league_pairings VALUES (
                %s, %s, %s, 'sp_psro_training', %s, %s, %s, 'PENDING'
            )
            ON CONFLICT DO NOTHING
            """,
            [
                pairing_id,
                context.generation_id,
                context.round_index,
                result.seed,
                population_revision_id,
                beta_revision_id,
            ],
        )
        task = LeagueEpisodeTask(
            pairing_id,
            orientation,
            result.seed,
            population,
            beta,
            result.steps,
        )
        _store_orientation_result(
            self._store, context.session_id, context.generation_id, task, result
        )
        self._store.connection.execute(
            """
            INSERT INTO pairing_policy_assignments VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            [
                pairing_id,
                mixture_id,
                f"{mixture_id}:composite",
                population_revision_id,
                beta_revision_id,
            ],
        )
        completed = self._store.connection.execute(
            "SELECT count(*) FROM pairing_episodes WHERE pairing_id = %s", [pairing_id]
        ).fetchone()[0]
        if int(completed) == 2:
            _finish_pairing(self._store, pairing_id, population_revision_id)
        return tuple(
            str(row[0])
            for row in self._store.connection.execute(
                """
                SELECT eta.artifact_id FROM episode_training_artifacts eta
                JOIN pairing_episodes pe USING (match_id)
                WHERE pe.pairing_id = %s AND pe.orientation = %s
                ORDER BY eta.artifact_id
                """,
                [pairing_id, orientation],
            ).fetchall()
        )

    def run_evaluation_round(
        self,
        *,
        context: RoundContext,
        request_id: str,
        pairing_kind: str,
        matchups: Sequence[tuple[EvaluationPolicy, EvaluationPolicy, int]],
        jobs: int,
        episode_steps: int,
    ) -> dict[str, Any]:
        """Schedule, resume, and record a seat-swapped model evaluation.

        Trainers describe immutable policy matchups; the lab owns their pairing
        identities, execution, replay storage, and paired results.  Completed
        orientations are reused after interruption instead of being replayed.
        """

        from kaggriculture_lab.identity import builtin_revision
        from kaggriculture_lab.league import (
            LeagueEpisodeTask,
            _finish_pairing,
            _get_revision,
            _league_worker,
            _store_orientation_result,
        )
        from kaggriculture_lab.league import _stable_id as league_id

        if not request_id:
            raise ValueError("evaluation request_id cannot be empty")
        if not pairing_kind:
            raise ValueError("evaluation pairing_kind cannot be empty")
        if jobs < 1:
            raise ValueError("evaluation jobs must be positive")
        if episode_steps < 1:
            raise ValueError("evaluation episode_steps must be positive")

        policies = {
            policy.revision_id: policy
            for left, right, _ in matchups
            for policy in (left, right)
        }
        revisions: list[AgentRevision] = []
        manifests: dict[str, tuple[AgentManifest, str]] = {}
        for policy in policies.values():
            if policy.manifest_path is None and policy.revision_id.startswith("builtin:"):
                revisions.append(
                    builtin_revision(policy.revision_id.removeprefix("builtin:"), policy.agent_spec)
                )
                continue
            if policy.manifest_path is not None:
                cached = manifests.get(policy.manifest_path)
                if cached is None:
                    manifest = AgentManifest.load(policy.manifest_path)
                    cached = (manifest, code_fingerprint(manifest))
                    manifests[policy.manifest_path] = cached
                manifest, fingerprint = cached
                family_id = manifest.family_id
                entry_point = manifest.entry_point
                schema_version = manifest.parameter_schema_version
                evaluator = manifest.episode_evaluator
            else:
                family_id = "evaluation_opponent"
                entry_point = policy.agent_spec.partition("?")[0]
                fingerprint = hashlib.sha256(policy.agent_spec.encode()).hexdigest()
                schema_version = 1
                evaluator = None
            revisions.append(
                AgentRevision(
                    revision_id=policy.revision_id,
                    family_id=family_id,
                    strategy_id=None,
                    entry_point=entry_point,
                    agent_spec=policy.agent_spec,
                    code_fingerprint=fingerprint,
                    parameter_schema_version=schema_version,
                    params_json=json.dumps({"role": policy.role}, sort_keys=True),
                    source="model_evaluation",
                    episode_evaluator=evaluator,
                )
            )
        self._store.add_agent_revisions(revisions)

        pairing_ids: list[str] = []
        tasks: list[Any] = []
        for index, (left, right, seed) in enumerate(matchups):
            pairing_id = league_id(
                "model-eval-pair",
                self._model_id,
                request_id,
                index,
                left.revision_id,
                right.revision_id,
                seed,
            )
            pairing_ids.append(pairing_id)
            self._store.connection.execute(
                """
                INSERT INTO league_pairings VALUES (
                    %s, %s, %s, %s, %s, %s, %s, 'PENDING'
                )
                ON CONFLICT DO NOTHING
                """,
                [
                    pairing_id,
                    context.generation_id,
                    context.round_index,
                    pairing_kind,
                    int(seed),
                    left.revision_id,
                    right.revision_id,
                ],
            )
            left_revision = _get_revision(self._store, left.revision_id)
            right_revision = _get_revision(self._store, right.revision_id)
            for orientation in (0, 1):
                known = self._store.connection.execute(
                    """
                    SELECT 1 FROM pairing_episodes
                    WHERE pairing_id = %s AND orientation = %s
                    """,
                    [pairing_id, orientation],
                ).fetchone()
                if known is None:
                    tasks.append(
                        LeagueEpisodeTask(
                            pairing_id,
                            orientation,
                            int(seed),
                            left_revision,
                            right_revision,
                            episode_steps,
                        )
                    )

        if jobs == 1:
            outcomes = map(_league_worker, tasks)
            for task, result in outcomes:
                _store_orientation_result(
                    self._store, context.session_id, context.generation_id, task, result
                )
        elif tasks:
            with ProcessPoolExecutor(
                max_workers=jobs, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                for task, result in pool.map(_league_worker, tasks, chunksize=1):
                    _store_orientation_result(
                        self._store, context.session_id, context.generation_id, task, result
                    )

        for pairing_id in pairing_ids:
            revision_a = self._store.connection.execute(
                "SELECT revision_a FROM league_pairings WHERE pairing_id = %s",
                [pairing_id],
            ).fetchone()[0]
            _finish_pairing(self._store, pairing_id, str(revision_a))

        if not pairing_ids:
            return {
                "pairs": 0,
                "errors": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "win_rate": 0.0,
                "margin": 0.0,
                "margin_std": 0.0,
                "money": 0.0,
                "pairing_ids": [],
            }
        placeholders = ",".join("%s" for _ in pairing_ids)
        rows = self._store.connection.execute(
            f"""
            SELECT pr.paired_margin, pr.outcome_a, pr.failed,
                   avg(ep.final_money) AS candidate_money
            FROM league_pairings lp
            JOIN pairing_results pr USING (pairing_id)
            LEFT JOIN pairing_episodes pe USING (pairing_id)
            LEFT JOIN episode_players ep
              ON ep.match_id = pe.match_id AND ep.strategy_id = lp.revision_a
            WHERE lp.pairing_id IN ({placeholders})
            GROUP BY lp.pairing_id, pr.paired_margin, pr.outcome_a, pr.failed
            ORDER BY lp.pairing_id
            """,
            pairing_ids,
        ).fetchall()
        valid = [row for row in rows if not row[2] and row[0] is not None]
        margins = [float(row[0]) for row in valid]
        outcomes = [float(row[1]) for row in valid]
        money = [float(row[3]) for row in valid if row[3] is not None]
        return {
            "pairs": len(valid),
            "errors": len(pairing_ids) - len(valid),
            "wins": sum(value > 0 for value in margins),
            "losses": sum(value < 0 for value in margins),
            "draws": sum(value == 0 for value in margins),
            "win_rate": (sum(outcomes) / len(outcomes) if outcomes else 0.0),
            "margin": (sum(margins) / len(margins) if margins else 0.0),
            "margin_std": (statistics.stdev(margins) if len(margins) > 1 else 0.0),
            "money": (sum(money) / len(money) if money else 0.0),
            "pairing_ids": pairing_ids,
        }

    def record_policy_mixture(
        self,
        context: RoundContext,
        revision_ids: Sequence[str],
        *,
        max_members: int,
        weights: Sequence[float] | None = None,
    ) -> str:
        """Persist one immutable policy mixture; weights default to uniform."""
        if not revision_ids:
            raise ValueError("policy mixture cannot be empty")
        if max_members < 1:
            raise ValueError("max_members must be positive")
        members = tuple(revision_ids[-max_members:])
        if weights is None:
            normalized = [1.0 / len(members)] * len(members)
        else:
            selected = list(weights[-max_members:])
            if len(selected) != len(members) or any(
                value < 0 or not math.isfinite(value) for value in selected
            ):
                raise ValueError("policy mixture weights must match members and be nonnegative")
            total = sum(selected)
            if total <= 0:
                raise ValueError("policy mixture weights must have positive mass")
            normalized = [value / total for value in selected]
        mixture_id = _stable_id(
            "mixture",
            self._model_id,
            context.generation_id,
            *(f"{member}:{weight:.17g}" for member, weight in zip(members, normalized)),
        )
        self._store.connection.execute(
            "INSERT INTO policy_mixtures VALUES (%s, %s, %s, %s, current_timestamp) ON CONFLICT DO NOTHING",
            [mixture_id, self._model_id, context.generation_index, max_members],
        )
        self._store.connection.executemany(
            "INSERT INTO policy_mixture_members VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                [mixture_id, revision_id, ordinal, normalized[ordinal]]
                for ordinal, revision_id in enumerate(members)
            ],
        )
        return mixture_id

    def link_artifacts(self, batch_id: str, artifact_ids: Sequence[str]) -> None:
        """Associate immutable episode sidecars with a consolidated batch."""
        if artifact_ids:
            self._store.connection.executemany(
                "INSERT INTO training_batch_artifacts VALUES (%s, %s) ON CONFLICT DO NOTHING",
                [[batch_id, artifact_id] for artifact_id in artifact_ids],
            )


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
    placeholders = ",".join("%s" for _ in selected)
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
        WHERE model_id = %s AND scope = %s AND generation_id = %s AND round_index = %s
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
            ) VALUES (%s, %s, %s, %s, %s, %s, 'RUNNING')
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
        WHERE model_id = %s AND scope = %s AND generation_id = %s AND round_index = %s
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
        UPDATE model_round_hooks SET status = %s, batches_recorded = %s, error_text = %s,
            completed_at = current_timestamp
        WHERE model_id = %s AND scope = %s AND generation_id = %s AND round_index = %s
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
