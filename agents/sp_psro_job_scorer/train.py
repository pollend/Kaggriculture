"""League-facing trainer for the SP-PSRO job scorer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from agents.sp_psro_job_scorer.training.collect import collect_batch, digest, save_batch
from agents.sp_psro_job_scorer.training.updater import (
    PopulationUpdater,
    _network_from_state,
    load_state,
    mixture_probabilities,
    population_networks,
)

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
                SELECT coalesce(sum(CAST(json_extract(metadata_json, '$.transitions') AS BIGINT)), 0)
                FROM training_batches WHERE model_id = ? AND batch_seq > ?
                """,
                [context.model_id, exploration_start],
            )[0]
        )
        batch_index = int(
            reader.fetchone(
                "SELECT count(*) FROM training_batches WHERE model_id = ?",
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


class SPPSROTrainer:
    def __init__(self) -> None:
        self._collector = PopulationCollector()
        self._updater = PopulationUpdater()

    def on_round(
        self,
        context: RoundContext,
        reader: ModelTrainingReader,
        writer: TrainingBatchWriter,
    ) -> None:
        if context.scope == "round":
            self._collector.collect(context, reader, writer)

    def on_batch(self, context: ModelUpdateContext, reader: ModelTrainingReader) -> ModelUpdateResult:
        return self._updater.update(context, reader)


trainer = SPPSROTrainer()
