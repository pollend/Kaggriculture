"""Persistent collector and incremental updater for the DQN job scorer."""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from agents.dqn_job_scorer.market_policy import MARKET_FEATURE_DIM
from agents.dqn_job_scorer.policy import WEIGHTS_PATH, PolicyBundle, QNetwork
from agents.dqn_job_scorer.training import network as net_module
from agents.dqn_job_scorer.training.collect import (
    EpisodeFormatError,
    collect_batch,
    load_episode_batch,
    save_episode_batch,
)
from agents.dqn_job_scorer.training.device import move_optimizer, resolve_device
from agents.dqn_job_scorer.training.peers import (
    active_checkpoint_opponents,
    historical_checkpoint_opponents,
)
from agents.dqn_job_scorer.training.replay import ReplayBuffer
from kaggriculture_lab.model_updates import (
    ModelEvaluation,
    ModelTrainingReader,
    ModelUpdateContext,
    ModelUpdateResult,
)

SELF_SPEC = "agents/dqn_job_scorer/main.py"


def _incumbent_path(config: dict[str, Any]) -> Path:
    return Path(str(config.get("incumbent_weights", WEIGHTS_PATH)))


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def league_opponents(context, reader) -> list[str]:
    """Agent specs of everyone competing in the round that just closed.

    The league breeds a fresh population every generation, so these specs carry
    parameters that the checked-in `main.py` files do not. Training only against
    those files would pin the opponent distribution to whatever happened to be
    committed, while the league is busy generating variation right next door.

    The learner's own family is excluded -- facing itself is self-play, which is
    mixed in separately and under an explicit ratio.
    """

    rows = reader.fetchall(
        """
        SELECT DISTINCT ar.agent_spec
        FROM generation_members gm
        JOIN agent_revisions ar USING (revision_id)
        WHERE gm.generation_id = ? AND ar.agent_spec IS NOT NULL
          AND ar.family_id != ?
        ORDER BY ar.agent_spec
        """,
        [context.generation_id, context.family_id],
    )
    return [str(row[0]) for row in rows]


def self_play_pool(context, reader, limit: int) -> list[str]:
    """The most recent promoted checkpoints, addressed as agent specs.

    A snapshot is reached through the agent's `weights` config key rather than as
    a bare `.npz` path: the registry resolves agent modules, not weight files.
    """

    if limit < 1:
        return []
    rows = reader.fetchall(
        """
        SELECT inference_artifact_path FROM model_training_checkpoints
        WHERE model_id = ? AND promoted
        ORDER BY created_at DESC, checkpoint_id DESC LIMIT ?
        """,
        [context.model_id, limit],
    )
    return [
        f"{SELF_SPEC}?weights={row[0]}" for row in rows if Path(str(row[0])).is_file()
    ]


def compose_opponents(base: list[str], pool: list[str], ratio: float) -> list[str]:
    """Mix self-play into `base` so it lands on `ratio` of the episodes.

    `collect_batch` walks the returned list round-robin, so its composition *is*
    the sampling distribution. This used to be `[*base, *pool]`, which with a
    single benchmark and a pool of five made self-play 5 of every 6 episodes
    while evaluation stayed on the benchmark; dqn04 fell from +2,269 to -22,154
    on the round that switched on. Solve for the slot count instead:
    `slots / (len(base) + slots) == ratio`.
    """

    ratio = min(max(ratio, 0.0), 0.9)
    if not pool or not base or ratio <= 0.0:
        return list(base) or list(pool)
    # A narrow base cannot express a small ratio in whole slots -- one static
    # opponent at ratio 0.25 wants a third of a slot, which rounds to zero and
    # silently disables self-play. Repeat the base until one slot is
    # representable, then solve as usual.
    repeats = max(1, math.ceil((1 - ratio) / (ratio * len(base))))
    widened = base * repeats
    slots = max(1, round(len(widened) * ratio / (1 - ratio)))
    return [*widened, *(pool[index % len(pool)] for index in range(slots))]


def rotate_opponents(opponents: list[str], offset: int) -> list[str]:
    """Start each batch where the previous one stopped.

    `collect_batch` plays `episodes_per_batch` episodes against
    `opponents[i % len(opponents)]`. Once the list is longer than a batch --
    which it is as soon as league revisions join -- a fixed start would mean the
    tail never plays at all.
    """

    if not opponents:
        return opponents
    start = offset % len(opponents)
    return [*opponents[start:], *opponents[:start]]


class DQNCollector:
    """Roll out exploration episodes for one closed league round.

    Off-policy, so the league's population is irrelevant -- only its clock is.
    The round hook supplies the resolved config and active checkpoint, reads go
    through the SELECT-only reader, and the one write goes through the writer,
    which is already bound to this model.
    """

    def collect(self, context, reader, writer) -> list[str]:
        config = context.config
        count = int(config.get("batches_per_round", 1))
        if count < 1:
            raise ValueError("batches_per_round must be positive")
        checkpoint = context.active_checkpoint
        if checkpoint is not None:
            bundle = PolicyBundle.load(checkpoint.inference_artifact_path)
            if bundle.market is None:
                bundle = PolicyBundle(
                    bundle.job,
                    QNetwork.random(
                        seed=int(config.get("seed", 0)) + 1,
                        feature_dim=MARKET_FEATURE_DIM,
                    ),
                )
        else:
            bundle = PolicyBundle.random(seed=int(config.get("seed", 0)))

        total_transitions = int(
            reader.fetchone(
                """
                SELECT coalesce(sum(CAST(json_extract(metadata_json, '$.transitions') AS BIGINT)), 0)
                FROM training_batches WHERE model_id = ?
                """,
                [context.model_id],
            )[0]
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
                reader.fetchone(
                    "SELECT count(*) FROM training_batches WHERE model_id = ?",
                    [context.model_id],
                )[0]
            )
            seeds = [
                int(value)
                for value in rng.integers(
                    0,
                    2**31 - 1,
                    size=int(config.get("episodes_per_batch", 16)),
                )
            ]
            configured_opponents = config.get("opponents")
            base = (
                [str(item) for item in configured_opponents]
                if configured_opponents
                else [
                    str(config.get("benchmark", "starter"))
                    if checkpoint is not None
                    else "starter"
                ]
            )
            if config.get("league_opponents", True):
                base += league_opponents(context, reader)
            base += active_checkpoint_opponents(
                context,
                reader,
                [str(item) for item in config.get("active_peer_types", [])],
            )
            # Static, league, and active-peer sources may resolve to the same
            # specification; a duplicate would unintentionally double its weight.
            base = list(dict.fromkeys(base))
            pool: list[str] = []
            if batch_index >= int(config.get("self_play_after", 20)):
                history_types = [
                    str(item) for item in config.get("history_peer_types", [])
                ]
                pool = (
                    historical_checkpoint_opponents(
                        context,
                        reader,
                        history_types,
                        int(config.get("self_play_pool", 5)),
                    )
                    if history_types
                    else self_play_pool(
                        context, reader, int(config.get("self_play_pool", 5))
                    )
                )
            opponents = rotate_opponents(
                compose_opponents(base, pool, float(config.get("self_play_ratio", 0.3))),
                batch_index * len(seeds),
            )
            records = collect_batch(
                bundle.job.layers,
                opponents,
                seeds,
                epsilon,
                float(config.get("gamma", 0.997)),
                int(config.get("jobs", 1)),
                market_weights=bundle.market.layers if bundle.market is not None else None,
                market_epsilon=float(config.get("market_epsilon_start", 1.0)) + fraction * (
                    float(config.get("market_epsilon_end", config.get("epsilon_end", 0.15)))
                    - float(config.get("market_epsilon_start", 1.0))
                ),
            )
            if not records:
                raise RuntimeError("every DQN rollout episode failed")
            source_id = f"rollout-{batch_index:08d}"
            path = (
                Path(context.artifacts)
                / "training_batches"
                / f"{source_id}.npz"
            )
            save_episode_batch(records, path)
            transitions = sum(int(record.unit_mask.sum()) for record in records)
            market_transitions = sum(0 if record.market_step is None else len(record.market_step) for record in records)
            writer.record(
                source_kind="dqn_rollout",
                source_id=source_id,
                sample_count=len(records),
                artifact_path=path,
                artifact_sha256=_file_digest(path),
                metadata={
                    "episodes": len(records),
                    "transitions": transitions,
                    "market_transitions": market_transitions,
                    "epsilon": epsilon,
                    "seeds": seeds,
                    "behavior_checkpoint_id": (
                        checkpoint.checkpoint_id if checkpoint is not None else None
                    ),
                    "session_id": context.session_id,
                    "generation_id": context.generation_id,
                    "round_index": context.round_index,
                    # Which opponents actually played, so a collapse like dqn04's
                    # can be traced to the mix rather than guessed at.
                    "opponents": [
                        opponents[index % len(opponents)] for index in range(len(seeds))
                    ],
                    "self_play_episodes": sum(
                        1
                        for index in range(len(seeds))
                        if "?weights=" in opponents[index % len(opponents)]
                    ),
                },
            )
            total_transitions += transitions
            written.append(source_id)
        return written


def _arena_evaluate(
    weights_path: Path,
    opponent: str,
    seeds: int,
    jobs: int,
    seed_start: int = 0,
) -> dict[str, Any]:
    from sim.arena import play_series
    from sim.metrics import summarize

    spec = f"{SELF_SPEC}?weights={weights_path}"
    results = play_series(spec, opponent, range(seed_start, seed_start + seeds), {}, jobs)
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
        device = resolve_device(str(config.get("device", "auto")))
        torch.manual_seed(int(config.get("seed", 0)))
        buffer = ReplayBuffer(
            int(config.get("buffer_episodes", 200)),
            float(config.get("gamma", 0.997)),
            int(config.get("n_step", 3)),
            int(config.get("seed", 0)),
            float(config.get("reservoir_fraction", 0.35)),
        )
        skipped_legacy_batches = 0
        for batch in context.all_batches:
            if batch.source_kind != "dqn_rollout" or not batch.artifact_path:
                continue
            path = Path(batch.artifact_path)
            if batch.artifact_sha256 and _file_digest(path) != batch.artifact_sha256:
                raise ValueError(f"rollout artifact digest mismatch: {path}")
            try:
                records = load_episode_batch(path)
            except EpisodeFormatError:
                # Format 1 did not record the live action mask. Treating every
                # job as legal would reintroduce the corruption this migration
                # fixes, so old batches remain auditable but are not trainable.
                skipped_legacy_batches += 1
                continue
            buffer.add(records)
        warmup = int(config.get("warmup_episodes", 16))
        if len(buffer.episodes) < warmup:
            raise ValueError(f"DQN requires at least {warmup} replay episodes")

        online = net_module.build().to(device)
        target = net_module.build().to(device)
        market_online = net_module.build(feature_dim=MARKET_FEATURE_DIM).to(device)
        market_target = net_module.build(feature_dim=MARKET_FEATURE_DIM).to(device)
        optimizer = torch.optim.Adam(
            online.parameters(), lr=float(config.get("learning_rate", 1e-4))
        )
        market_optimizer = torch.optim.Adam(
            market_online.parameters(), lr=float(config.get("market_learning_rate", 1e-4))
        )
        gradient_steps = 0
        market_gradient_steps = 0
        if context.active_checkpoint is not None:
            state = torch.load(
                context.active_checkpoint.trainer_artifact_path,
                map_location="cpu",
                weights_only=False,
            )
            online.load_state_dict(state["online"])
            target.load_state_dict(state["target"])
            optimizer.load_state_dict(state["optimizer"])
            move_optimizer(optimizer, device)
            gradient_steps = int(state.get("gradient_steps", 0))
            if "market_online" in state:
                market_online.load_state_dict(state["market_online"])
                market_target.load_state_dict(state["market_target"])
                market_optimizer.load_state_dict(state["market_optimizer"])
                move_optimizer(market_optimizer, device)
                market_gradient_steps = int(state.get("market_gradient_steps", 0))
            else:
                market_target.load_state_dict(market_online.state_dict())
            if "torch_rng_state" in state:
                torch.set_rng_state(state["torch_rng_state"])
            if "replay_rng_state" in state:
                buffer.rng.bit_generator.state = state["replay_rng_state"]
        else:
            target.load_state_dict(online.state_dict())
            market_target.load_state_dict(market_online.state_dict())

        loss_fn = nn.SmoothL1Loss()
        losses: list[float] = []
        market_losses: list[float] = []
        steps_to_run = int(config.get("gradient_steps_per_batch", 400)) * len(
            context.new_batches
        )
        batch_size = int(config.get("batch_size", 512))
        target_sync = int(config.get("target_sync", 2000))
        started = time.perf_counter()
        for _ in range(steps_to_run):
            batch = buffer.sample(batch_size)
            features = torch.from_numpy(batch.features).to(device)
            next_features = torch.from_numpy(batch.next_features).to(device)
            next_mask = torch.from_numpy(batch.next_mask).to(device)
            reward = torch.from_numpy(batch.reward).to(device)
            discount = torch.from_numpy(batch.discount).to(device)
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

        market_steps_to_run = (
            int(config.get("market_gradient_steps_per_batch", 100)) * len(context.new_batches)
            if buffer.market_transitions
            else 0
        )
        market_batch_size = int(config.get("market_batch_size", 128))
        market_target_sync = int(config.get("market_target_sync", target_sync))
        for _ in range(market_steps_to_run):
            batch = buffer.sample_market(market_batch_size)
            features = torch.from_numpy(batch.features).to(device)
            next_features = torch.from_numpy(batch.next_features).to(device)
            next_mask = torch.from_numpy(batch.next_mask).to(device)
            reward = torch.from_numpy(batch.reward).to(device)
            discount = torch.from_numpy(batch.discount).to(device)
            with torch.no_grad():
                flat = next_features.reshape(-1, next_features.shape[-1])
                shape = next_features.shape[:3]
                online_q = market_online(flat).reshape(shape)
                best = online_q.masked_fill(~next_mask, float("-inf")).argmax(dim=2)
                target_q = market_target(flat).reshape(shape)
                factor_values = target_q.gather(2, best.unsqueeze(2)).squeeze(2)
                factor_values = torch.where(next_mask.any(dim=2), factor_values, torch.zeros_like(factor_values))
                goal = reward + discount * factor_values.sum(dim=1)
            predicted = market_online(features.reshape(-1, features.shape[-1])).reshape(features.shape[:2]).sum(dim=1)
            loss = loss_fn(predicted, goal)
            market_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(market_online.parameters(), 10.0)
            market_optimizer.step()
            market_losses.append(float(loss.detach()))
            market_gradient_steps += 1
            if market_gradient_steps % market_target_sync == 0:
                market_target.load_state_dict(market_online.state_dict())
        train_seconds = time.perf_counter() - started

        checkpoint_id = hashlib.sha256(
            f"{context.model_id}\0{context.update_id}\0{context.dataset_sha256}".encode()
        ).hexdigest()[:24]
        destination = context.artifacts / "checkpoints"
        destination.mkdir(parents=True, exist_ok=True)
        trainer_path = destination / f"{checkpoint_id}.pt"
        inference_path = destination / f"{checkpoint_id}.npz"
        torch.save(
            {
                "online": online.state_dict(),
                "target": target.state_dict(),
                "optimizer": optimizer.state_dict(),
                "gradient_steps": gradient_steps,
                "market_online": market_online.state_dict(),
                "market_target": market_target.state_dict(),
                "market_optimizer": market_optimizer.state_dict(),
                "market_gradient_steps": market_gradient_steps,
                "torch_rng_state": torch.get_rng_state(),
                "replay_rng_state": buffer.rng.bit_generator.state,
                "dataset_sha256": context.dataset_sha256,
                "device": str(device),
            },
            trainer_path,
        )
        PolicyBundle(
            net_module.to_numpy(online),
            net_module.to_numpy(market_online),
        ).save(inference_path)

        eval_seeds = int(config.get("eval_seeds", 40))
        promotion_seeds = int(config.get("promotion_seeds", 60))
        seed_base = int(config.get("promotion_seed_base", 1_000_000))
        seed_stride = int(config.get("promotion_seed_stride", 10_000))
        seed_start = seed_base + context.cutoff_batch_seq * seed_stride
        jobs = int(config.get("jobs", 1))
        benchmark = str(config.get("benchmark", "agents/configurable_crop_surrogate/main.py"))
        challenger_benchmark = _arena_evaluate(
            inference_path, benchmark, eval_seeds, jobs, seed_start
        )
        evaluations: list[ModelEvaluation] = []
        promote = False
        incumbent_path = Path(context.active_checkpoint.inference_artifact_path) if context.active_checkpoint is not None else None
        gate: dict[str, Any] = {
            "kind": "significant_incumbent_improvement_and_benchmark_non_regression",
            "seed_start": seed_start,
            "challenger_benchmark": challenger_benchmark,
        }
        if incumbent_path is None:
            promote = not challenger_benchmark["errors"]
            gate.update({"initial_checkpoint": True, "passed": promote})
            evaluations.append(ModelEvaluation("fixed_benchmark", challenger_benchmark, promote))
            head_to_head = None
        else:
            incumbent_spec = f"{SELF_SPEC}?weights={incumbent_path}"
            head_to_head = _arena_evaluate(inference_path, incumbent_spec, promotion_seeds, jobs, seed_start)
            incumbent_benchmark = _arena_evaluate(incumbent_path, benchmark, eval_seeds, jobs, seed_start)
        # Promotion is a release decision, not an absence-of-proof-of-harm
        # decision. The paired lower bound must clear zero.
            head_to_head_ok = not head_to_head["errors"] and float(head_to_head["margin_ci"][0]) > 0.0
            benchmark_ok = not challenger_benchmark["errors"] and not incumbent_benchmark["errors"] and float(challenger_benchmark["margin_ci"][1]) >= float(incumbent_benchmark["margin_ci"][0])
            promote = head_to_head_ok and benchmark_ok
            gate.update({"head_to_head": head_to_head, "incumbent_benchmark": incumbent_benchmark, "head_to_head_significantly_better": head_to_head_ok, "benchmark_non_regression": benchmark_ok, "passed": promote})
            evaluations.append(ModelEvaluation("incumbent_head_to_head", head_to_head, head_to_head_ok, context.active_checkpoint.checkpoint_id))
            evaluations.append(ModelEvaluation("fixed_benchmark", challenger_benchmark, promote))
        metrics = {
            "episodes_in_buffer": len(buffer.episodes),
            "skipped_legacy_batches": skipped_legacy_batches,
            "transitions_in_buffer": buffer.transitions,
            "market_transitions_in_buffer": buffer.market_transitions,
            "gradient_steps": gradient_steps,
            "gradient_steps_this_update": steps_to_run,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "market_gradient_steps": market_gradient_steps,
            "market_gradient_steps_this_update": market_steps_to_run,
            "market_mean_loss": float(np.mean(market_losses)) if market_losses else None,
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


# The manifest names `train.py`, which constructs these itself. Nothing outside
# this agent resolves `training.updater` by path any more.
