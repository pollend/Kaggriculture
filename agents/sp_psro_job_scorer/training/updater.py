"""Persistent recent-policy population and best-response DDQN updates."""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from agents.dqn_job_scorer.market_policy import MARKET_FEATURE_DIM
from agents.dqn_job_scorer.policy import PolicyBundle, QNetwork
from agents.dqn_job_scorer.training import network as net_module
from agents.dqn_job_scorer.training.device import move_optimizer, resolve_device
from agents.dqn_job_scorer.training.peers import active_checkpoint_opponents
from agents.dqn_job_scorer.training.replay import ReplayBuffer
from agents.sp_psro_job_scorer.policy import PopulationArtifact
from agents.sp_psro_job_scorer.training.collect import digest, load_batch
from kaggriculture_lab.model_updates import (
    ModelEvaluation,
    ModelTrainingReader,
    ModelUpdateContext,
    ModelUpdateResult,
)

SELF_SPEC = "agents/sp_psro_job_scorer/main.py"
STATE_VERSION = 4
Z95 = 1.959963985


def fresh_live(seed: int) -> dict[str, Any]:
    """Create the legacy nu slot and the trainable beta learner state."""
    states: dict[str, Any] = {}
    for offset, role in enumerate(("nu", "beta"), start=1):
        torch.manual_seed(seed + offset)
        online = net_module.build()
        target = net_module.build()
        market_online = net_module.build(feature_dim=MARKET_FEATURE_DIM)
        market_target = net_module.build(feature_dim=MARKET_FEATURE_DIM)
        target.load_state_dict(online.state_dict())
        market_target.load_state_dict(market_online.state_dict())
        states[role] = {
            "online": online.state_dict(),
            "target": target.state_dict(),
            "optimizer": None,
            "gradient_steps": 0,
            "market_online": market_online.state_dict(),
            "market_target": market_target.state_dict(),
            "market_optimizer": None,
            "market_gradient_steps": 0,
            "replay_rng_state": None,
        }
    return states


def initial_state(seed: int, initial_population_weights: str | Path | None = None) -> dict[str, Any]:
    initial = (
        PopulationArtifact.load(initial_population_weights).components[0]
        if initial_population_weights is not None
        else PolicyBundle.random(seed=seed)
    )
    if initial.market is None:
        raise ValueError("SP-PSRO initial population weights require a market network")
    return {
        "version": STATE_VERSION,
        "iteration": 0,
        "population": [],
        "initial_job_network": initial.job.layers,
        "initial_market_network": initial.market.layers,
        "initial_source": str(initial_population_weights) if initial_population_weights else "random",
        "live": fresh_live(seed + 10_000),
        "champion_id": "member-000000-initial",
        "last_admission": None,
        "restricted": {"arm_ids": ["member-000000-initial", "live-nu"], "log_weights": [0.0, 0.0]},
        "nu_snapshots": [],
        "nu_snapshots_seen": 0,
        "nu_snapshot_rng_state": None,
        "validation_suite": None,
        "inference_champion": None,
    }


def load_state(
    path: str | Path | None,
    seed: int,
    initial_population_weights: str | Path | None = None,
) -> dict[str, Any]:
    if path is None:
        return initial_state(seed, initial_population_weights)
    state = torch.load(path, map_location="cpu", weights_only=False)
    version = int(state.get("version", 0))
    if version == 3:
        state["version"] = STATE_VERSION
        state.setdefault("restricted", {"arm_ids": [], "log_weights": []})
        state.setdefault("nu_snapshots", [])
        state.setdefault("nu_snapshots_seen", 0)
        state.setdefault("nu_snapshot_rng_state", None)
        state.setdefault("validation_suite", None)
    elif version != STATE_VERSION:
        raise ValueError(f"unsupported SP-PSRO trainer state {state.get('version')}")
    state.setdefault("initial_source", "random")
    state.setdefault("inference_champion", None)
    return state


def _network_from_state(state: dict[str, Any]) -> PolicyBundle:
    module = net_module.build()
    module.load_state_dict(state["online"])
    market = net_module.build(feature_dim=MARKET_FEATURE_DIM)
    market.load_state_dict(state["market_online"])
    return PolicyBundle(net_module.to_numpy(module), net_module.to_numpy(market))


def population_networks(state: dict[str, Any]) -> list[tuple[str, PopulationArtifact]]:
    result: list[tuple[str, PopulationArtifact]] = []
    for member in state["population"]:
        path = Path(member["path"])
        if digest(path) != member["sha256"]:
            raise ValueError(f"population member digest mismatch: {path}")
        result.append((str(member["member_id"]), PopulationArtifact.load(path)))
    if not result:
        result.append(
            (
                "member-000000-initial",
                PopulationArtifact.single(
                    PolicyBundle(
                        QNetwork(state["initial_job_network"]),
                        QNetwork(state["initial_market_network"]),
                    )
                ),
            )
        )
    return result


def mixture_probabilities(
    state: dict[str, Any], config: dict[str, Any]
) -> tuple[list[str], list[float]]:
    """Return the current Exp3 distribution over frozen members plus live nu."""
    arm_ids = [str(member["member_id"]) for member in state["population"]]
    if not arm_ids:
        arm_ids = ["member-000000-initial"]
    arm_ids.append("live-nu")
    restricted = state.setdefault("restricted", {"arm_ids": [], "log_weights": []})
    old = dict(zip(restricted.get("arm_ids", []), restricted.get("log_weights", [])))
    log_weights = np.asarray([float(old.get(arm, 0.0)) for arm in arm_ids], dtype=np.float64)
    shifted = log_weights - float(log_weights.max())
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    exploration = float(config.get("meta_exploration", 0.1))
    if not 0.0 <= exploration <= 1.0:
        raise ValueError("meta_exploration must be in [0, 1]")
    probabilities = (1.0 - exploration) * probabilities + exploration / len(arm_ids)
    restricted["arm_ids"] = arm_ids
    restricted["log_weights"] = log_weights.tolist()
    return arm_ids, probabilities.tolist()


def _update_restricted_distribution(
    state: dict[str, Any], records: list, config: dict[str, Any]
) -> None:
    arm_ids, _ = mixture_probabilities(state, config)
    log_weights = np.asarray(state["restricted"]["log_weights"], dtype=np.float64)
    index_of = {arm: index for index, arm in enumerate(arm_ids)}
    eta = float(config.get("meta_learning_rate", 0.05))
    pairs: dict[tuple[str, int], list] = {}
    for record in records:
        arm = str(record.behavior_member_id or "")
        if arm in index_of:
            pairs.setdefault((arm, int(record.seed)), []).append(record)
    for (arm, _), pair in pairs.items():
        payoff = statistics.fmean(float(record.population_payoff) for record in pair)
        probability = statistics.fmean(float(record.sampling_probability) for record in pair)
        if probability > 0:
            log_weights[index_of[arm]] += eta * payoff / probability
    log_weights -= float(log_weights.max())
    state["restricted"]["log_weights"] = np.clip(log_weights, -50.0, 0.0).tolist()


def _train_role(
    role_state: dict[str, Any],
    buffer: ReplayBuffer,
    config: dict[str, Any],
    steps: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[float], list[float]]:
    online = net_module.build().to(device)
    target = net_module.build().to(device)
    online.load_state_dict(role_state["online"])
    target.load_state_dict(role_state["target"])
    optimizer = torch.optim.Adam(online.parameters(), lr=float(config.get("learning_rate", 1e-4)))
    if role_state.get("optimizer") is not None:
        optimizer.load_state_dict(role_state["optimizer"])
        move_optimizer(optimizer, device)
    gradient_steps = int(role_state.get("gradient_steps", 0))
    target_sync = int(config.get("target_sync", 2000))
    batch_size = int(config.get("batch_size", 512))
    loss_fn = nn.SmoothL1Loss()
    losses: list[float] = []
    market_steps = int(config.get("market_gradient_steps_per_batch", 100)) * max(
        1, steps // max(1, int(config.get("gradient_steps_per_batch", 400)))
    )
    planned_market_steps = market_steps if buffer.market_transitions else 0
    total_steps = steps + planned_market_steps
    completed_steps = 0
    next_percentage = 10

    def report_progress() -> None:
        nonlocal next_percentage
        percentage = 100 if total_steps == 0 else int(100 * completed_steps / total_steps)
        while percentage >= next_percentage:
            print(f"[sp-psro training] gradient progress {next_percentage}%", flush=True)
            next_percentage += 10

    for _ in range(steps):
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
            bootstrap = torch.where(next_mask.any(dim=1), bootstrap, torch.zeros_like(bootstrap))
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
        completed_steps += 1
        report_progress()

    market_online = net_module.build(feature_dim=MARKET_FEATURE_DIM).to(device)
    market_target = net_module.build(feature_dim=MARKET_FEATURE_DIM).to(device)
    market_online.load_state_dict(role_state["market_online"])
    market_target.load_state_dict(role_state["market_target"])
    market_optimizer = torch.optim.Adam(
        market_online.parameters(), lr=float(config.get("market_learning_rate", 1e-4))
    )
    if role_state.get("market_optimizer") is not None:
        market_optimizer.load_state_dict(role_state["market_optimizer"])
        move_optimizer(market_optimizer, device)
    market_gradient_steps = int(role_state.get("market_gradient_steps", 0))
    market_target_sync = int(config.get("market_target_sync", 1000))
    market_batch_size = int(config.get("market_batch_size", 128))
    market_losses: list[float] = []
    if buffer.market_transitions:
        for _ in range(market_steps):
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
                factor_values = torch.where(
                    next_mask.any(dim=2), factor_values, torch.zeros_like(factor_values)
                )
                goal = reward + discount * factor_values.sum(dim=1)
            predicted = market_online(
                features.reshape(-1, features.shape[-1])
            ).reshape(features.shape[:2]).sum(dim=1)
            loss = loss_fn(predicted, goal)
            market_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(market_online.parameters(), 10.0)
            market_optimizer.step()
            market_losses.append(float(loss.detach()))
            market_gradient_steps += 1
            if market_gradient_steps % market_target_sync == 0:
                market_target.load_state_dict(market_online.state_dict())
            completed_steps += 1
            report_progress()
    return (
        {
            "online": {k: v.detach().cpu() for k, v in online.state_dict().items()},
            "target": {k: v.detach().cpu() for k, v in target.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "gradient_steps": gradient_steps,
            "market_online": {
                k: v.detach().cpu() for k, v in market_online.state_dict().items()
            },
            "market_target": {
                k: v.detach().cpu() for k, v in market_target.state_dict().items()
            },
            "market_optimizer": market_optimizer.state_dict(),
            "market_gradient_steps": market_gradient_steps,
        },
        losses,
        market_losses,
    )


def _mixture_summary(left: list[Path], right: list[Path], seeds: int, seed_start: int) -> dict[str, Any]:
    """Seat-swapped deterministic mixture evaluation with paired margins."""
    from sim.runner import run_match

    margins: list[float] = []
    errors = 0
    for index in range(seeds):
        seed = seed_start + index
        left_spec = (
            f"{SELF_SPEC}?weights={left[index % len(left)]}&policy_seed={seed * 2 + 1}"
        )
        right_spec = (
            f"{SELF_SPEC}?weights={right[index % len(right)]}&policy_seed={seed * 2 + 2}"
        )
        forward = run_match((left_spec, right_spec), seed)
        reverse = run_match((right_spec, left_spec), seed)
        if forward.failed or reverse.failed:
            errors += 1
            continue
        margins.append((float(forward.margin) - float(reverse.margin)) / 2.0)
    mean = statistics.fmean(margins) if margins else 0.0
    se = statistics.stdev(margins) / math.sqrt(len(margins)) if len(margins) > 1 else 0.0
    return {
        "pairs": len(margins),
        "errors": errors,
        "wins": sum(value > 0 for value in margins),
        "losses": sum(value < 0 for value in margins),
        "margin": mean,
        "margin_ci": [mean - Z95 * se, mean + Z95 * se],
    }


def evaluate_admission(candidates: list[Path], population: list[Path], seeds: int, seed_start: int) -> dict[str, Any]:
    return _mixture_summary(candidates, population, seeds, seed_start)


def evaluate_member(path: Path, opponents: list[str], seeds: int, seed_start: int, jobs: int) -> dict[str, Any]:
    from sim.arena import play_series
    from sim.metrics import summarize

    results = []
    for opponent_index, opponent in enumerate(opponents):
        start = seed_start + opponent_index * 100_000
        spec = f"{SELF_SPEC}?weights={path}&policy_seed={start}"
        summary = summarize(
            spec,
            opponent,
            play_series(spec, opponent, range(start, start + seeds), {}, jobs),
        )
        effective_win_rate = (summary.wins + summary.draws / 2) / summary.pairs if summary.pairs else 0.0
        results.append(
            {
                "opponent": opponent,
                "pairs": summary.pairs,
                "win_rate": effective_win_rate,
                "margin": summary.margin_mean,
                "money": summary.a_money_mean,
                "errors": summary.errors,
            }
        )
    return {
        "opponents": results,
        "mean_win_rate": statistics.fmean(item["win_rate"] for item in results),
        "mean_margin": statistics.fmean(item["margin"] for item in results),
        "mean_money": statistics.fmean(item["money"] for item in results),
        "errors": sum(bool(item["errors"]) for item in results),
    }


def passes_release_gate(validation: dict[str, Any], config: dict[str, Any]) -> bool:
    """Require absolute external quality, not merely relative self-play progress."""

    return (
        not validation.get("errors")
        and float(validation.get("mean_money", 0.0))
        >= float(config.get("release_min_money", 3000.0))
        and float(validation.get("mean_win_rate", 0.0))
        >= float(config.get("release_min_win_rate", 0.5))
        and float(validation.get("mean_margin", float("-inf")))
        >= float(config.get("release_min_margin", 0.0))
    )


def validation_opponent_specs(context, reader, config: dict[str, Any]) -> list[str]:
    """Fixed DQN reference plus the current active DQN training instances."""

    opponents = [str(item) for item in config.get("validation_opponents", ["starter"])]
    opponents += active_checkpoint_opponents(
        context,
        reader,
        [str(item) for item in config.get("validation_peer_types", [])],
    )
    return list(dict.fromkeys(opponents))


def _record_nu_snapshot(
    state: dict[str, Any], path: Path, config: dict[str, Any], seed: int
) -> None:
    capacity = int(config.get("nu_average_snapshots", 16))
    if capacity < 1:
        raise ValueError("nu_average_snapshots must be positive")
    snapshots = state.setdefault("nu_snapshots", [])
    seen = int(state.get("nu_snapshots_seen", 0)) + 1
    rng = np.random.default_rng(seed + 7001)
    if state.get("nu_snapshot_rng_state") is not None:
        rng.bit_generator.state = state["nu_snapshot_rng_state"]
    entry = {"path": str(path), "sha256": digest(path)}
    if len(snapshots) < capacity:
        snapshots.append(entry)
    else:
        index = int(rng.integers(0, seen))
        if index < capacity:
            snapshots[index] = entry
    state["nu_snapshots_seen"] = seen
    state["nu_snapshot_rng_state"] = rng.bit_generator.state


def _save_nu_average(state: dict[str, Any], path: Path) -> None:
    components = []
    for snapshot in state["nu_snapshots"]:
        source = Path(snapshot["path"])
        if digest(source) != snapshot["sha256"]:
            raise ValueError(f"nu snapshot digest mismatch: {source}")
        components.append(PolicyBundle.load(source))
    PopulationArtifact(
        tuple(components), np.full(len(components), 1.0 / len(components), dtype=np.float64)
    ).save(path)


class PopulationUpdater:
    def update(self, context: ModelUpdateContext, reader: ModelTrainingReader) -> ModelUpdateResult:
        config = context.config
        seed = int(config.get("seed", 0))
        state = load_state(
            context.active_checkpoint.trainer_artifact_path if context.active_checkpoint is not None else None,
            seed,
            config.get("initial_population_weights"),
        )
        def replay(role_seed: int) -> ReplayBuffer:
            return ReplayBuffer(
                int(config.get("buffer_episodes", 800)),
                float(config.get("gamma", 0.997)),
                int(config.get("n_step", 3)),
                role_seed,
                float(config.get("reservoir_fraction", 0.35)),
            )

        nu_buffer = replay(seed + 1)
        beta_buffer = replay(seed + 2)
        new_records = []
        new_batch_ids = {batch.batch_id for batch in context.new_batches}
        for batch in context.all_batches:
            if batch.source_kind != "sp_psro_rollout" or not batch.artifact_path:
                continue
            if batch.artifact_sha256 and digest(batch.artifact_path) != batch.artifact_sha256:
                raise ValueError(f"rollout artifact digest mismatch: {batch.artifact_path}")
            records = load_batch(batch.artifact_path)
            nu_buffer.add([record.new_policy for record in records])
            beta_buffer.add([record.best_response for record in records])
            if batch.batch_id in new_batch_ids:
                new_records.extend(records)
        if state["live"]["nu"].get("replay_rng_state") is not None:
            nu_buffer.rng.bit_generator.state = state["live"]["nu"]["replay_rng_state"]
        if state["live"]["beta"].get("replay_rng_state") is not None:
            beta_buffer.rng.bit_generator.state = state["live"]["beta"]["replay_rng_state"]
        warmup = int(config.get("warmup_episodes", 16))
        if min(len(nu_buffer.episodes), len(beta_buffer.episodes)) < warmup:
            raise ValueError(f"SP-PSRO requires at least {warmup} dual-policy episodes")

        device = resolve_device(str(config.get("device", "auto")))
        steps = int(config.get("gradient_steps_per_batch", 400)) * len(context.new_batches)
        print(
            f"[sp-psro training] update started: {len(context.new_batches)} new batches, "
            f"{steps} gradient steps per role on {device}",
            flush=True,
        )
        started = time.perf_counter()
        print("[sp-psro training] training nu policy", flush=True)
        state["live"]["nu"], nu_losses, nu_market_losses = _train_role(
            state["live"]["nu"], nu_buffer, config, steps, device
        )
        print("[sp-psro training] nu policy complete; training beta policy", flush=True)
        state["live"]["beta"], beta_losses, beta_market_losses = _train_role(
            state["live"]["beta"], beta_buffer, config, steps, device
        )
        print("[sp-psro training] beta policy complete; evaluating candidates", flush=True)
        state["live"]["nu"]["replay_rng_state"] = nu_buffer.rng.bit_generator.state
        state["live"]["beta"]["replay_rng_state"] = beta_buffer.rng.bit_generator.state
        _update_restricted_distribution(state, new_records, config)

        artifact_root = context.artifacts
        members_dir = artifact_root / "population_members"
        checkpoints_dir = artifact_root / "checkpoints"
        members_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        if not state["population"]:
            initial_path = members_dir / "member-000000-initial.npz"
            PolicyBundle(
                QNetwork(state.pop("initial_job_network")),
                QNetwork(state.pop("initial_market_network")),
            ).save(initial_path)
            state["population"].append(
                {
                    "member_id": "member-000000-initial",
                    "role": "initial",
                    "path": str(initial_path),
                    "sha256": digest(initial_path),
                    "validation": None,
                }
            )

        iteration = int(state["iteration"])
        nu_snapshot_path = members_dir / (
            f"nu-snapshot-{iteration:06d}-{context.cutoff_batch_seq:08d}.npz"
        )
        _network_from_state(state["live"]["nu"]).save(nu_snapshot_path)
        _record_nu_snapshot(state, nu_snapshot_path, config, seed)
        beta_candidate_path = members_dir / (
            f"candidate-{iteration:06d}-beta-{context.cutoff_batch_seq}.npz"
        )
        nu_candidate_path = members_dir / (
            f"candidate-{iteration:06d}-nu-average-{context.cutoff_batch_seq}.npz"
        )
        _network_from_state(state["live"]["beta"]).save(beta_candidate_path)
        _save_nu_average(state, nu_candidate_path)
        population_paths = [Path(member["path"]) for member in state["population"]]
        admission = evaluate_admission(
            [beta_candidate_path, nu_candidate_path],
            population_paths,
            int(config.get("admission_seeds", 60)),
            seed + 2_000_000 + context.cutoff_batch_seq * 10_000,
        )
        admitted = admission["errors"] == 0 and admission["pairs"] > 1 and float(admission["margin_ci"][0]) > 0.0
        print(
            f"[sp-psro training] admission evaluation complete: admitted={admitted}; "
            "running validation",
            flush=True,
        )
        new_members: list[dict[str, Any]] = []
        if admitted:
            for role, candidate_path in (
                ("beta", beta_candidate_path),
                ("nu-average", nu_candidate_path),
            ):
                member_id = f"member-{iteration + 1:06d}-{role}"
                final_path = members_dir / f"{member_id}.npz"
                PopulationArtifact.load(candidate_path).save(final_path)
                new_members.append(
                    {
                        "member_id": member_id,
                        "role": role,
                        "path": str(final_path),
                        "sha256": digest(final_path),
                        "validation": None,
                    }
                )
            state["population"].extend(new_members)
            state["population"] = state["population"][-int(config.get("population_size", 10)):]
            state["iteration"] = iteration + 1
            state["live"] = fresh_live(seed + 10_000 + (iteration + 1) * 100)
            state["restricted"] = {"arm_ids": [], "log_weights": []}
            state["nu_snapshots"] = []
            state["nu_snapshots_seen"] = 0
            state["nu_snapshot_rng_state"] = None
        state["last_admission"] = {**admission, "admitted": admitted}

        opponents = validation_opponent_specs(context, reader, config)
        suite = hashlib.sha256("\0".join(opponents).encode()).hexdigest()
        if state.get("validation_suite") != suite:
            for member in state["population"]:
                member["validation"] = None
            state["validation_suite"] = suite
            state["inference_champion"] = None
        validation_seeds = int(config.get("validation_seeds", 40))
        jobs = int(config.get("jobs", 1))
        for member in state["population"]:
            if member.get("validation") is None:
                member["validation"] = evaluate_member(
                    Path(member["path"]), opponents, validation_seeds, seed + 4_000_000, jobs
                )
        population_eligible = [
            member for member in state["population"] if not member["validation"].get("errors")
        ]
        if population_eligible:
            state["champion_id"] = max(
                population_eligible,
                key=lambda member: (
                    member["validation"]["mean_win_rate"],
                    member["validation"]["mean_margin"],
                    member["member_id"],
                ),
            )["member_id"]
        beta_validation = evaluate_member(
            beta_candidate_path,
            opponents,
            validation_seeds,
            seed + 5_000_000 + context.cutoff_batch_seq * 10_000,
            jobs,
        )
        print(
            f"[sp-psro training] validation complete after {time.perf_counter() - started:.1f}s",
            flush=True,
        )
        release_candidates = [
            {
                "candidate_id": str(member["member_id"]),
                "path": str(member["path"]),
                "validation": member["validation"],
            }
            for member in state["population"]
            if passes_release_gate(member["validation"], config)
        ]
        if passes_release_gate(beta_validation, config):
            release_candidates.append(
                {
                    "candidate_id": f"beta-{context.cutoff_batch_seq:08d}",
                    "path": str(beta_candidate_path),
                    "validation": beta_validation,
                }
            )
        if state.get("inference_champion") is not None:
            release_candidates.append(state["inference_champion"])
        if release_candidates:
            selected = max(
                release_candidates,
                key=lambda item: (
                    item["validation"]["mean_win_rate"],
                    item["validation"]["mean_margin"],
                    item["validation"]["mean_money"],
                    item["candidate_id"],
                ),
            )
            if Path(selected["path"]) == beta_candidate_path:
                released_path = members_dir / f"release-{selected['candidate_id']}.npz"
                PopulationArtifact.load(beta_candidate_path).save(released_path)
                selected = {**selected, "path": str(released_path)}
            state["inference_champion"] = selected

        retained_ids = {member["member_id"] for member in state["population"]}
        if state["champion_id"] not in retained_ids:
            state["champion_id"] = state["population"][-1]["member_id"]

        population_champion = next(
            member for member in state["population"] if member["member_id"] == state["champion_id"]
        )
        checkpoint_id = hashlib.sha256(
            f"{context.model_id}\0{context.update_id}\0{context.dataset_sha256}".encode()
        ).hexdigest()[:24]
        trainer_path = checkpoints_dir / f"{checkpoint_id}.pt"
        inference_path = checkpoints_dir / f"{checkpoint_id}.npz"
        state["dataset_sha256"] = context.dataset_sha256
        torch.save(state, trainer_path)
        inference_champion = state.get("inference_champion")
        inference_source = (
            Path(inference_champion["path"])
            if inference_champion is not None
            else Path(population_champion["path"])
        )
        PopulationArtifact.load(inference_source).save(inference_path)
        combined_digest = hashlib.sha256(trainer_path.read_bytes() + inference_path.read_bytes()).hexdigest()
        beta_candidate_path.unlink(missing_ok=True)
        nu_candidate_path.unlink(missing_ok=True)
        metrics = {
            "population_size": len(state["population"]),
            "iteration": state["iteration"],
            "admitted": admitted,
            "champion_id": state["champion_id"],
            "inference_champion_id": (
                inference_champion["candidate_id"] if inference_champion is not None else None
            ),
            "beta_validation": beta_validation,
            "beta_loss": float(np.mean(beta_losses)) if beta_losses else None,
            "nu_loss": float(np.mean(nu_losses)) if nu_losses else None,
            "beta_market_loss": (
                float(np.mean(beta_market_losses)) if beta_market_losses else None
            ),
            "nu_market_loss": (
                float(np.mean(nu_market_losses)) if nu_market_losses else None
            ),
            "beta_market_transitions": beta_buffer.market_transitions,
            "nu_market_transitions": nu_buffer.market_transitions,
            "restricted_distribution": dict(zip(*mixture_probabilities(state, config))),
            "gradient_steps_this_update": steps,
            "train_seconds": time.perf_counter() - started,
        }
        gate = {
            "kind": "external_submission_quality",
            "passed": inference_champion is not None,
            "population_admission": state["last_admission"],
            "beta_validation": beta_validation,
            "thresholds": {
                "money": float(config.get("release_min_money", 3000.0)),
                "win_rate": float(config.get("release_min_win_rate", 0.5)),
                "margin": float(config.get("release_min_margin", 0.0)),
            },
        }
        evaluations = (
            ModelEvaluation("population_pair_admission", admission, admitted),
            ModelEvaluation(
                "submission_beta_validation",
                beta_validation,
                passes_release_gate(beta_validation, config),
            ),
        )
        # The active inference artifact is always the already validated champion;
        # activating every successful checkpoint is what lets rejected live
        # learners retain optimizer state and continue training.
        return ModelUpdateResult(
            checkpoint_id,
            str(trainer_path),
            str(inference_path),
            combined_digest,
            metrics,
            gate,
            True,
            evaluations,
        )
