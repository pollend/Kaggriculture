"""League-facing trainer for the SP-PSRO job scorer."""

from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import numpy as np

from agents.sp_psro_job_scorer.policy import PopulationArtifact
from agents.sp_psro_job_scorer.training.collect import collect_batch, digest, save_batch
from agents.sp_psro_job_scorer.training.updater import (
    SELF_SPEC,
    Z95,
    PopulationUpdater,
    _network_from_state,
    load_state,
    mixture_probabilities,
    population_networks,
    validation_opponent_specs,
)
from kaggriculture_lab.model_trainers import EvaluationPolicy

if TYPE_CHECKING:
    from kaggriculture_lab.model_trainers import RoundContext, TrainingBatchWriter
    from kaggriculture_lab.model_updates import (
        ModelTrainingReader,
        ModelUpdateContext,
        ModelUpdateResult,
    )


class PopulationCollector:
    def collect(self, context, reader, writer) -> list[str]:
        config = context.config
        count = int(config["batches_per_round"])
        episodes = int(config["episodes_per_batch"])
        if count < 1:
            raise ValueError("batches_per_round must be positive")
        if episodes < 2 or episodes % 2:
            raise ValueError("episodes_per_batch must be a positive even number")
        seed = int(config["seed"])
        state = load_state(
            context.active_checkpoint.trainer_artifact_path if context.active_checkpoint is not None else None,
            seed,
        )
        rollout_root = Path(context.artifacts) / "rollout_policies" / context.generation_id
        rollout_root.mkdir(parents=True, exist_ok=True)
        population = []
        for member_id, network in population_networks(state):
            member_path = rollout_root / f"{member_id}.npz"
            if not member_path.is_file():
                network.save(member_path)
            population.append((member_id, member_path))
        arm_ids, probabilities = mixture_probabilities(state, config)
        mixture_id = writer.record_policy_mixture(
            context,
            arm_ids,
            max_members=len(arm_ids),
            weights=probabilities,
        )
        nu = _network_from_state(state["live"]["nu"])
        nu_path = rollout_root / f"nu-round-{context.round_index:04d}.npz"
        nu.save(nu_path)
        beta = _network_from_state(state["live"]["beta"])
        beta_path = rollout_root / f"beta-round-{context.round_index:04d}.npz"
        beta.save(beta_path)
        exploration_start = int(state["exploration_start_batch_seq"])
        exploration_transitions = int(
            reader.fetchone(
                """
                SELECT coalesce(sum(CAST(metadata_json ->> 'transitions' AS BIGINT)), 0)
                FROM training_batches WHERE model_id = %s AND batch_seq > %s
                """,
                [context.model_id, exploration_start],
            )[0]
        )
        batch_index = int(
            reader.fetchone(
                "SELECT count(*) FROM training_batches WHERE model_id = %s",
                [context.model_id],
            )[0]
        )
        rng = np.random.default_rng(seed + batch_index)
        written = []
        for local_index in range(count):
            fraction = min(
                1.0,
                exploration_transitions / max(1, int(config["epsilon_decay_transitions"])),
            )
            epsilon = float(config["epsilon_start"]) + fraction * (
                float(config["epsilon_end"]) - float(config["epsilon_start"])
            )
            market_epsilon = float(config["market_epsilon_start"]) + fraction * (
                float(config["market_epsilon_end"]) - float(config["market_epsilon_start"])
            )
            paired_seeds = [int(value) for value in rng.integers(0, 2**31 - 1, size=episodes // 2)]
            seeds = [value for value in paired_seeds for _ in range(2)]
            records = collect_batch(
                population,
                nu_path,
                beta_path,
                seeds,
                epsilon,
                float(config["gamma"]),
                int(config["jobs"]),
                offset=(batch_index + local_index) * episodes,
                config={
                    "episodeSteps": int(config["episode_steps"]),
                    "trace_root": str(
                        Path(context.artifacts)
                        / "episode_traces"
                        / context.generation_id
                        / f"round-{context.round_index:04d}"
                    ),
                },
                market_epsilon=market_epsilon,
                mixture_weights=probabilities,
                sampling_seed=seed + (batch_index + local_index) * 1_000_003,
            )
            if len(records) != episodes:
                raise RuntimeError(f"SP-PSRO collected {len(records)} of {episodes} rollout episodes")
            artifact_ids: list[str] = []
            beta_revision_id = f"beta-{digest(beta_path)[:20]}"
            for episode_index, record in enumerate(records):
                if record.match is None:
                    continue
                artifact_ids.extend(
                    writer.record_policy_episode(
                        context=context,
                        manifest_path="agents/sp_psro_job_scorer/match_agent.toml",
                        pairing_index=(batch_index + local_index) * (episodes // 2) + episode_index // 2,
                        beta_seat=record.beta_seat,
                        population_revision_id=str(record.behavior_member_id),
                        beta_revision_id=beta_revision_id,
                        mixture_id=mixture_id,
                        result=record.match,
                    )
                )
            source_id = f"population-rollout-{batch_index + local_index:08d}"
            path = Path(context.artifacts) / "training_batches" / f"{source_id}.npz"
            save_batch(records, path)
            transitions = sum(
                int(record.new_policy.unit_mask.sum()) + int(record.best_response.unit_mask.sum()) for record in records
            )
            batch = writer.record(
                source_kind="sp_psro_rollout",
                source_id=source_id,
                sample_count=len(records),
                artifact_path=path,
                artifact_sha256=digest(path),
                metadata={
                    "episodes": len(records),
                    "transitions": transitions,
                    "epsilon": epsilon,
                    "market_epsilon": market_epsilon,
                    "seeds": seeds,
                    "population_episodes": len(records),
                    "behavior_members": [record.behavior_member_id for record in records],
                    "behavior_kinds": [record.behavior_kind for record in records],
                    "mixture_weights": dict(zip(arm_ids, probabilities)),
                    "session_id": context.session_id,
                    "generation_id": context.generation_id,
                    "round_index": context.round_index,
                    "behavior_checkpoint_id": (
                        context.active_checkpoint.checkpoint_id if context.active_checkpoint is not None else None
                    ),
                },
            )
            writer.link_artifacts(batch.batch_id, artifact_ids)
            exploration_transitions += transitions
            written.append(source_id)
        return written


class PopulationEvaluator:
    """Submit pending SP-PSRO evaluations as durable league pairings."""

    manifest_path = "agents/sp_psro_job_scorer/agent.toml"

    @staticmethod
    def _policy(path: str | Path, *, role: str, **params: int) -> EvaluationPolicy:
        source = Path(path)
        query = {"weights": str(source), **params}
        spec = f"{SELF_SPEC}?{urlencode(query)}"
        identity = hashlib.sha256(spec.encode()).hexdigest()[:20]
        return EvaluationPolicy(
            f"sp-psro-eval:{identity}",
            spec,
            PopulationEvaluator.manifest_path,
            role,
        )

    @staticmethod
    def _opponent(spec: str) -> EvaluationPolicy:
        if spec in {"starter", "pass", "random"}:
            return EvaluationPolicy(f"builtin:{spec}", spec, role="validation_opponent")
        identity = hashlib.sha256(spec.encode()).hexdigest()[:20]
        manifest_path = None
        plain = Path(spec.partition("?")[0])
        candidate = plain.parent / "agent.toml"
        if plain.name == "main.py" and candidate.is_file():
            manifest_path = str(candidate)
        return EvaluationPolicy(
            f"validation-opponent:{identity}",
            spec,
            manifest_path,
            "validation_opponent",
        )

    @staticmethod
    def _without_pairings(summary: dict) -> dict:
        return {key: value for key, value in summary.items() if key != "pairing_ids"}

    def _admission(self, context, writer, pending: dict, config: dict) -> dict:
        candidates = [Path(pending["beta_path"]), Path(pending["nu_path"])]
        population = [Path(member["path"]) for member in pending["population"]]
        seeds = int(pending["seeds"])
        seed_start = int(pending["seed_start"])
        rng = np.random.default_rng(seed_start)
        left_choices = rng.integers(0, len(candidates), size=seeds)
        right_choices = rng.integers(0, len(population), size=seeds)
        matchups = []
        for index in range(seeds):
            episode_seed = seed_start + index
            left = self._policy(
                candidates[int(left_choices[index])],
                role="admission_candidate",
                policy_seed=episode_seed * 2 + 1,
            )
            right = self._policy(
                population[int(right_choices[index])],
                role="population_member",
                policy_seed=episode_seed * 2 + 2,
            )
            matchups.append((left, right, episode_seed))
        summary = writer.run_evaluation_round(
            context=context,
            request_id=str(pending["request_id"]),
            pairing_kind="sp_psro_admission",
            matchups=matchups,
            jobs=int(config["jobs"]),
            episode_steps=int(config["episode_steps"]),
        )
        pairs = int(summary["pairs"])
        se = float(summary["margin_std"]) / math.sqrt(pairs) if pairs > 1 else 0.0
        result = self._without_pairings(summary)
        result["margin_ci"] = [
            float(summary["margin"]) - Z95 * se,
            float(summary["margin"]) + Z95 * se,
        ]
        result["pairs"] = pairs
        return result

    def _member(
        self,
        context,
        writer,
        *,
        request_id: str,
        path: Path,
        opponents: list[str],
        seeds: int,
        seed_start: int,
        jobs: int,
        episode_steps: int,
        role: str,
    ) -> dict:
        artifact = PopulationArtifact.load(path)
        results = []
        for opponent_index, opponent in enumerate(opponents):
            print(
                f"[sp-psro evaluation] {role} opponent "
                f"{opponent_index + 1}/{len(opponents)}: {opponent}",
                flush=True,
            )
            start = seed_start + opponent_index * 100_000
            episode_seeds = list(range(start, start + seeds))
            rng = np.random.default_rng(start)
            choices = rng.choice(len(artifact.components), size=seeds, p=artifact.weights)
            matchups = [
                (
                    self._policy(path, role=role, component_index=int(component)),
                    self._opponent(opponent),
                    episode_seed,
                )
                for episode_seed, component in zip(episode_seeds, choices)
            ]
            summary = writer.run_evaluation_round(
                context=context,
                request_id=f"{request_id}:opponent-{opponent_index}",
                pairing_kind="sp_psro_validation",
                matchups=matchups,
                jobs=jobs,
                episode_steps=episode_steps,
            )
            results.append(
                {
                    "opponent": opponent,
                    "pairs": int(summary["pairs"]),
                    "win_rate": float(summary["win_rate"]),
                    "margin": float(summary["margin"]),
                    "money": float(summary["money"]),
                    "errors": int(summary["errors"]),
                }
            )
        return {
            "opponents": results,
            "mean_win_rate": statistics.fmean(item["win_rate"] for item in results),
            "mean_margin": statistics.fmean(item["margin"] for item in results),
            "mean_money": statistics.fmean(item["money"] for item in results),
            "errors": sum(bool(item["errors"]) for item in results),
        }

    def collect(self, context, reader, writer) -> bool:
        if context.active_checkpoint is None:
            return False
        config = context.config
        state = load_state(context.active_checkpoint.trainer_artifact_path, int(config["seed"]))
        pending_admission = state.get("pending_admission")
        if pending_admission is not None:
            print(
                f"[sp-psro evaluation] submitting admission round {pending_admission['request_id']}",
                flush=True,
            )
            result = self._admission(context, writer, pending_admission, config)
            writer.record(
                source_kind="model_evaluation",
                source_id=str(pending_admission["request_id"]),
                sample_count=int(result["pairs"]) * 2,
                metadata={
                    "family_id": "sp_psro_job_scorer",
                    "evaluation_kind": "admission",
                    "request_id": str(pending_admission["request_id"]),
                    "result": result,
                },
            )
            print("[sp-psro evaluation] admission round recorded", flush=True)
            return True

        pending_validation = state.get("pending_validation")
        if pending_validation is None:
            return False
        opponents = validation_opponent_specs(context, reader, config)
        suite = hashlib.sha256("\0".join(opponents).encode()).hexdigest()
        evaluate_all = state.get("validation_suite") != suite
        member_results = {}
        jobs = int(config["jobs"])
        episode_steps = int(config["episode_steps"])
        seeds = int(pending_validation["seeds"])
        base_seed = int(config["seed"])
        print(
            f"[sp-psro evaluation] submitting validation round {pending_validation['request_id']}",
            flush=True,
        )
        for member in state["population"]:
            if not evaluate_all and member.get("validation") is not None:
                continue
            member_id = str(member["member_id"])
            member_results[member_id] = self._member(
                context,
                writer,
                request_id=f"{pending_validation['request_id']}:{member_id}",
                path=Path(member["path"]),
                opponents=opponents,
                seeds=seeds,
                seed_start=base_seed + 4_000_000,
                jobs=jobs,
                episode_steps=episode_steps,
                role="population_validation",
            )
        beta_validation = self._member(
            context,
            writer,
            request_id=f"{pending_validation['request_id']}:beta",
            path=Path(pending_validation["beta_path"]),
            opponents=opponents,
            seeds=seeds,
            seed_start=(
                base_seed
                + 5_000_000
                + int(pending_validation["cutoff_batch_seq"]) * 10_000
            ),
            jobs=jobs,
            episode_steps=episode_steps,
            role="beta_validation",
        )
        result = {
            "suite": suite,
            "opponents": opponents,
            "members": member_results,
            "beta": beta_validation,
        }
        pair_count = sum(
            int(item["pairs"])
            for validation in [*member_results.values(), beta_validation]
            for item in validation["opponents"]
        )
        writer.record(
            source_kind="model_evaluation",
            source_id=str(pending_validation["request_id"]),
            sample_count=pair_count * 2,
            metadata={
                "family_id": "sp_psro_job_scorer",
                "evaluation_kind": "validation",
                "request_id": str(pending_validation["request_id"]),
                "result": result,
            },
        )
        print("[sp-psro evaluation] validation round recorded", flush=True)
        return True


class SPPSROTrainer:
    def __init__(self) -> None:
        self._collector = PopulationCollector()
        self._evaluator = PopulationEvaluator()
        self._updater = PopulationUpdater()

    def on_round(
        self,
        context: RoundContext,
        reader: ModelTrainingReader,
        writer: TrainingBatchWriter,
    ) -> None:
        if context.scope == "round":
            if not self._evaluator.collect(context, reader, writer):
                self._collector.collect(context, reader, writer)

    def on_batch(self, context: ModelUpdateContext, reader: ModelTrainingReader) -> ModelUpdateResult:
        return self._updater.update(context, reader)


trainer = SPPSROTrainer()
