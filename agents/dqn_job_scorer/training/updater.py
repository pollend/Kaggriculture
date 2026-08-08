"""Persistent collector and incremental updater for the DQN job scorer."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from agents.dqn_job_scorer.policy import QNetwork
from kaggriculture_lab.model_updates import (
    ModelEvaluation,
    ModelTrainingReader,
    ModelUpdateContext,
    ModelUpdateResult,
    record_training_batch,
)
from agents.dqn_job_scorer.training import network as net_module
from agents.dqn_job_scorer.training.collect import collect_batch, load_episode_batch, save_episode_batch
from agents.dqn_job_scorer.training.replay import ReplayBuffer

SELF_SPEC = "agents/dqn_job_scorer/main.py"


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DQNCollector:
    """Collect one or more model-defined rollout batches into the shared store."""

    def collect(self, store, model_id: str, count: int, artifacts: str | Path) -> list[str]:
        if count < 1:
            raise ValueError("collection count must be positive")
        row = store.connection.execute(
            "SELECT config_json, active_checkpoint_id FROM training_models WHERE model_id = ?",
            [model_id],
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown training model: {model_id}")
        import json

        config = json.loads(row[0])
        checkpoint = None
        if row[1]:
            checkpoint = store.connection.execute(
                "SELECT inference_artifact_path FROM model_training_checkpoints WHERE checkpoint_id = ?",
                [row[1]],
            ).fetchone()
        if checkpoint:
            network = QNetwork.load(checkpoint[0])
        else:
            network = net_module.to_numpy(net_module.build())

        total_transitions = int(
            store.connection.execute(
                """
                SELECT coalesce(sum(CAST(json_extract(metadata_json, '$.transitions') AS BIGINT)), 0)
                FROM training_batches WHERE model_id = ?
                """,
                [model_id],
            ).fetchone()[0]
        )
        seed = int(config.get("seed", 0))
        rng = np.random.default_rng(seed + total_transitions)
        written: list[str] = []
        for _ in range(count):
            fraction = min(
                1.0,
                total_transitions / max(1, int(config.get("epsilon_decay_transitions", 200_000))),
            )
            epsilon = float(config.get("epsilon_start", 1.0)) + fraction * (
                float(config.get("epsilon_end", 0.05)) - float(config.get("epsilon_start", 1.0))
            )
            batch_index = int(
                store.connection.execute(
                    "SELECT count(*) FROM training_batches WHERE model_id = ?", [model_id]
                ).fetchone()[0]
            )
            seeds = [
                int(value)
                for value in rng.integers(
                    0,
                    2**31 - 1,
                    size=int(config.get("episodes_per_batch", 16)),
                )
            ]
            benchmark = str(config.get("benchmark", "starter"))
            records = collect_batch(
                network.layers,
                [benchmark if row[1] else "starter"],
                seeds,
                epsilon,
                float(config.get("gamma", 0.997)),
                int(config.get("jobs", 1)),
            )
            if not records:
                raise RuntimeError("every DQN rollout episode failed")
            source_id = f"rollout-{batch_index:08d}"
            path = Path(artifacts) / "training_batches" / model_id / f"{source_id}.npz"
            save_episode_batch(records, path)
            transitions = sum(int(record.unit_mask.sum()) for record in records)
            record_training_batch(
                store,
                model_id,
                source_kind="dqn_rollout",
                source_id=source_id,
                sample_count=len(records),
                artifact_path=path,
                artifact_sha256=_file_digest(path),
                metadata={
                    "episodes": len(records),
                    "transitions": transitions,
                    "epsilon": epsilon,
                    "seeds": seeds,
                    "behavior_checkpoint_id": row[1],
                },
            )
            total_transitions += transitions
            written.append(source_id)
        return written


def _arena_evaluate(weights_path: Path, opponent: str, seeds: int, jobs: int) -> dict[str, Any]:
    from sim.arena import play_series
    from sim.metrics import summarize

    spec = f"{SELF_SPEC}?weights={weights_path}"
    results = play_series(spec, opponent, range(seeds), {}, jobs)
    summary = summarize(spec, opponent, results)
    return {
        "opponent": opponent,
        "pairs": summary.pairs,
        "win_rate": summary.win_rate,
        "margin": summary.margin_mean,
        "margin_ci": list(summary.margin_ci),
        "significant": summary.significant,
        "errors": summary.errors,
    }


class DQNUpdater:
    def update(
        self, context: ModelUpdateContext, reader: ModelTrainingReader
    ) -> ModelUpdateResult:
        config = context.config
        torch.manual_seed(int(config.get("seed", 0)))
        buffer = ReplayBuffer(
            int(config.get("buffer_episodes", 200)),
            float(config.get("gamma", 0.997)),
            int(config.get("n_step", 3)),
            int(config.get("seed", 0)),
        )
        for batch in context.all_batches:
            if batch.source_kind != "dqn_rollout" or not batch.artifact_path:
                continue
            path = Path(batch.artifact_path)
            if batch.artifact_sha256 and _file_digest(path) != batch.artifact_sha256:
                raise ValueError(f"rollout artifact digest mismatch: {path}")
            buffer.add(load_episode_batch(path))
        warmup = int(config.get("warmup_episodes", 16))
        if len(buffer.episodes) < warmup:
            raise ValueError(f"DQN requires at least {warmup} replay episodes")

        online = net_module.build()
        target = net_module.build()
        optimizer = torch.optim.Adam(
            online.parameters(), lr=float(config.get("learning_rate", 1e-4))
        )
        gradient_steps = 0
        if context.active_checkpoint is not None:
            state = torch.load(
                context.active_checkpoint.trainer_artifact_path,
                map_location="cpu",
                weights_only=False,
            )
            online.load_state_dict(state["online"])
            target.load_state_dict(state["target"])
            optimizer.load_state_dict(state["optimizer"])
            gradient_steps = int(state.get("gradient_steps", 0))
            if "torch_rng_state" in state:
                torch.set_rng_state(state["torch_rng_state"])
            if "replay_rng_state" in state:
                buffer.rng.bit_generator.state = state["replay_rng_state"]
        else:
            target.load_state_dict(online.state_dict())

        loss_fn = nn.SmoothL1Loss()
        losses: list[float] = []
        steps_to_run = int(config.get("gradient_steps_per_batch", 400)) * len(
            context.new_batches
        )
        batch_size = int(config.get("batch_size", 512))
        target_sync = int(config.get("target_sync", 2000))
        started = time.perf_counter()
        for _ in range(steps_to_run):
            batch = buffer.sample(batch_size)
            features = torch.from_numpy(batch.features)
            next_features = torch.from_numpy(batch.next_features)
            next_mask = torch.from_numpy(batch.next_mask)
            reward = torch.from_numpy(batch.reward)
            discount = torch.from_numpy(batch.discount)
            with torch.no_grad():
                flat = next_features.reshape(-1, next_features.shape[-1])
                online_q = online(flat).reshape(next_features.shape[:2])
                best = net_module.masked_argmax(online_q, next_mask)
                target_q = target(flat).reshape(next_features.shape[:2])
                bootstrap = target_q.gather(1, best.unsqueeze(1)).squeeze(1)
                bootstrap = torch.where(
                    next_mask.any(dim=1), bootstrap, torch.zeros_like(bootstrap)
                )
                goal = reward + discount * bootstrap
            predicted = online(features).squeeze(1)
            loss = loss_fn(predicted, goal)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            gradient_steps += 1
            if gradient_steps % target_sync == 0:
                target.load_state_dict(online.state_dict())
        train_seconds = time.perf_counter() - started

        checkpoint_id = hashlib.sha256(
            f"{context.model_id}\0{context.update_id}\0{context.dataset_sha256}".encode()
        ).hexdigest()[:24]
        destination = context.artifacts / "model_checkpoints" / context.model_id
        destination.mkdir(parents=True, exist_ok=True)
        trainer_path = destination / f"{checkpoint_id}.pt"
        inference_path = destination / f"{checkpoint_id}.npz"
        torch.save(
            {
                "online": online.state_dict(),
                "target": target.state_dict(),
                "optimizer": optimizer.state_dict(),
                "gradient_steps": gradient_steps,
                "torch_rng_state": torch.get_rng_state(),
                "replay_rng_state": buffer.rng.bit_generator.state,
                "dataset_sha256": context.dataset_sha256,
            },
            trainer_path,
        )
        net_module.to_numpy(online).save(inference_path)

        eval_seeds = int(config.get("eval_seeds", 24))
        jobs = int(config.get("jobs", 1))
        benchmark = str(config.get("benchmark", "agents/configurable_crop_surrogate/main.py"))
        challenger_benchmark = _arena_evaluate(inference_path, benchmark, eval_seeds, jobs)
        evaluations: list[ModelEvaluation] = []
        promote = not challenger_benchmark["errors"]
        gate: dict[str, Any] = {
            "kind": "non_inferior_incumbent_and_benchmark",
            "challenger_benchmark": challenger_benchmark,
        }
        if context.active_checkpoint is not None:
            incumbent_spec = f"{SELF_SPEC}?weights={context.active_checkpoint.inference_artifact_path}"
            head_to_head = _arena_evaluate(inference_path, incumbent_spec, eval_seeds, jobs)
            incumbent_benchmark = _arena_evaluate(
                Path(context.active_checkpoint.inference_artifact_path),
                benchmark,
                eval_seeds,
                jobs,
            )
            # A candidate is non-inferior unless the paired interval proves it
            # loses, and the fixed benchmark proves a regression.
            head_to_head_ok = not head_to_head["errors"] and float(head_to_head["margin_ci"][1]) >= 0.0
            benchmark_ok = (
                not incumbent_benchmark["errors"]
                and float(challenger_benchmark["margin_ci"][1])
                >= float(incumbent_benchmark["margin_ci"][0])
            )
            promote = promote and head_to_head_ok and benchmark_ok
            gate.update(
                {
                    "head_to_head": head_to_head,
                    "incumbent_benchmark": incumbent_benchmark,
                    "head_to_head_non_inferior": head_to_head_ok,
                    "benchmark_non_regression": benchmark_ok,
                }
            )
            evaluations.append(
                ModelEvaluation(
                    "incumbent_head_to_head",
                    head_to_head,
                    head_to_head_ok,
                    context.active_checkpoint.checkpoint_id,
                )
            )
        gate["passed"] = promote
        evaluations.append(
            ModelEvaluation("fixed_benchmark", challenger_benchmark, promote)
        )
        metrics = {
            "episodes_in_buffer": len(buffer.episodes),
            "transitions_in_buffer": buffer.transitions,
            "gradient_steps": gradient_steps,
            "gradient_steps_this_update": steps_to_run,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "train_seconds": train_seconds,
        }
        combined_digest = hashlib.sha256(
            trainer_path.read_bytes() + inference_path.read_bytes()
        ).hexdigest()
        return ModelUpdateResult(
            checkpoint_id,
            str(trainer_path),
            str(inference_path),
            combined_digest,
            metrics,
            gate,
            promote,
            tuple(evaluations),
        )


collector = DQNCollector()
updater = DQNUpdater()
