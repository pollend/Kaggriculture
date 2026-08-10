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
STATE_VERSION = 6
Z95 = 1.959963985


def fresh_live(seed: int) -> dict[str, Any]:
    """Create independent trainable nu and beta learner states."""
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


def initial_state(seed: int) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "iteration": 0,
        "population": [],
        "initial_job_network": QNetwork.random(seed=seed).layers,
        "live": fresh_live(seed + 10_000),
        "population_champion_id": "member-000000-initial",
        "last_admission": None,
        "restricted": {"arm_ids": ["member-000000-initial", "live-nu"], "log_weights": [0.0, 0.0]},
        "nu_snapshots": [],
        "nu_snapshots_seen": 0,
        "nu_snapshot_rng_state": None,
        "exploration_start_batch_seq": 0,
        "validation_suite": None,
        "release_champion": None,
        "pending_admission": None,
        "pending_validation": None,
        "last_training_batch_seq": 0,
    }


def load_state(path: str | Path | None, seed: int) -> dict[str, Any]:
    if path is None:
        return initial_state(seed)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("version") == 5:
        # Version 5 performed evaluation synchronously inside on_batch.  Its
        # completed checkpoints migrate cleanly into the recorded two-phase
        # evaluator; an in-flight v5 update has no checkpoint to migrate.
        state["version"] = STATE_VERSION
        state.setdefault("pending_admission", None)
        state.setdefault("pending_validation", None)
        state.setdefault("last_training_batch_seq", 0)
        state["_needs_training_watermark"] = True
    if state["version"] != STATE_VERSION:
        raise ValueError(f"unsupported SP-PSRO trainer state {state.get('version')}")
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
                PopulationArtifact.single(PolicyBundle(QNetwork(state["initial_job_network"]), None)),
            )
        )
    return result


def mixture_probabilities(state: dict[str, Any], config: dict[str, Any]) -> tuple[list[str], list[float]]:
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
    exploration = float(config["meta_exploration"])
    if not 0.0 <= exploration <= 1.0:
        raise ValueError("meta_exploration must be in [0, 1]")
    probabilities = (1.0 - exploration) * probabilities + exploration / len(arm_ids)
    restricted["arm_ids"] = arm_ids
    restricted["log_weights"] = log_weights.tolist()
    return arm_ids, probabilities.tolist()


def _update_restricted_distribution(state: dict[str, Any], records: list, config: dict[str, Any]) -> None:
    arm_ids, _ = mixture_probabilities(state, config)
    log_weights = np.asarray(state["restricted"]["log_weights"], dtype=np.float64)
    index_of = {arm: index for index, arm in enumerate(arm_ids)}
    eta = float(config["meta_learning_rate"])
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
    optimizer = torch.optim.Adam(online.parameters(), lr=float(config["learning_rate"]))
    if role_state.get("optimizer") is not None:
        optimizer.load_state_dict(role_state["optimizer"])
        move_optimizer(optimizer, device)
    gradient_steps = int(role_state.get("gradient_steps", 0))
    target_sync = int(config["target_sync"])
    batch_size = int(config["batch_size"])
    loss_fn = nn.SmoothL1Loss()
    losses: list[float] = []
    market_steps = int(config["market_gradient_steps_per_batch"]) * max(
        1, steps // max(1, int(config["gradient_steps_per_batch"]))
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
    market_optimizer = torch.optim.Adam(market_online.parameters(), lr=float(config["market_learning_rate"]))
    if role_state.get("market_optimizer") is not None:
        market_optimizer.load_state_dict(role_state["market_optimizer"])
        move_optimizer(market_optimizer, device)
    market_gradient_steps = int(role_state.get("market_gradient_steps", 0))
    market_target_sync = int(config["market_target_sync"])
    market_batch_size = int(config["market_batch_size"])
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
            completed_steps += 1
            report_progress()
    return (
        {
            "online": {k: v.detach().cpu() for k, v in online.state_dict().items()},
            "target": {k: v.detach().cpu() for k, v in target.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "gradient_steps": gradient_steps,
            "market_online": {k: v.detach().cpu() for k, v in market_online.state_dict().items()},
            "market_target": {k: v.detach().cpu() for k, v in market_target.state_dict().items()},
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
    if not left or not right:
        raise ValueError("admission evaluation requires two non-empty populations")
    rng = np.random.default_rng(seed_start)
    left_choices = rng.integers(0, len(left), size=seeds)
    right_choices = rng.integers(0, len(right), size=seeds)
    for index in range(seeds):
        seed = seed_start + index
        left_spec = f"{SELF_SPEC}?weights={left[int(left_choices[index])]}&policy_seed={seed * 2 + 1}"
        right_spec = f"{SELF_SPEC}?weights={right[int(right_choices[index])]}&policy_seed={seed * 2 + 2}"
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

    artifact = PopulationArtifact.load(path)
    results = []
    for opponent_index, opponent in enumerate(opponents):
        start = seed_start + opponent_index * 100_000
        episode_seeds = list(range(start, start + seeds))
        rng = np.random.default_rng(start)
        component_choices = rng.choice(len(artifact.components), size=seeds, p=artifact.weights)
        outcomes = {}
        for component_index in range(len(artifact.components)):
            selected_seeds = [
                episode_seed
                for episode_seed, choice in zip(episode_seeds, component_choices)
                if int(choice) == component_index
            ]
            if not selected_seeds:
                continue
            component_spec = f"{SELF_SPEC}?weights={path}&component_index={component_index}"
            outcomes.update(play_series(component_spec, opponent, selected_seeds, {}, jobs))
        spec = f"{SELF_SPEC}?weights={path}"
        summary = summarize(
            spec,
            opponent,
            outcomes,
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
        and float(validation.get("mean_money", 0.0)) >= float(config["release_min_money"])
        and float(validation.get("mean_win_rate", 0.0)) >= float(config["release_min_win_rate"])
        and float(validation.get("mean_margin", float("-inf"))) >= float(config["release_min_margin"])
    )


def _candidate_rank(candidate: dict[str, Any]) -> tuple[float, float, float, str]:
    validation = candidate["validation"]
    return (
        float(validation["mean_win_rate"]),
        float(validation["mean_margin"]),
        float(validation.get("mean_money", 0.0)),
        str(candidate.get("candidate_id", candidate.get("member_id", ""))),
    )


def validation_opponent_specs(context, reader, config: dict[str, Any]) -> list[str]:
    """Fixed DQN reference plus the current active DQN training instances."""

    opponents = [str(item) for item in config["validation_opponents"]]
    opponents += active_checkpoint_opponents(
        context,
        reader,
        [str(item) for item in config["validation_peer_types"]],
    )
    return list(dict.fromkeys(opponents))


def _record_nu_snapshot(state: dict[str, Any], path: Path, config: dict[str, Any], seed: int) -> None:
    capacity = int(config["nu_average_snapshots"])
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
    PopulationArtifact(tuple(components), np.full(len(components), 1.0 / len(components), dtype=np.float64)).save(path)


def _load_training_data(
    context: ModelUpdateContext,
    state: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> tuple[ReplayBuffer, ReplayBuffer, list]:
    def make_buffer(role_seed: int) -> ReplayBuffer:
        return ReplayBuffer(
            int(config["buffer_episodes"]),
            float(config["gamma"]),
            int(config["n_step"]),
            role_seed,
            float(config["reservoir_fraction"]),
        )

    nu_buffer = make_buffer(seed + 1)
    beta_buffer = make_buffer(seed + 2)
    new_records = []
    training_watermark = int(state.get("last_training_batch_seq", 0))
    for batch in context.all_batches:
        if batch.source_kind != "sp_psro_rollout" or not batch.artifact_path:
            continue
        if batch.artifact_sha256 and digest(batch.artifact_path) != batch.artifact_sha256:
            raise ValueError(f"rollout artifact digest mismatch: {batch.artifact_path}")
        records = load_batch(batch.artifact_path)
        nu_buffer.add([record.new_policy for record in records])
        beta_buffer.add([record.best_response for record in records])
        if batch.sequence > training_watermark:
            new_records.extend(records)
    for role, buffer in (("nu", nu_buffer), ("beta", beta_buffer)):
        rng_state = state["live"][role].get("replay_rng_state")
        if rng_state is not None:
            buffer.rng.bit_generator.state = rng_state
    warmup = int(config["warmup_episodes"])
    if min(len(nu_buffer.episodes), len(beta_buffer.episodes)) < warmup:
        raise ValueError(f"SP-PSRO requires at least {warmup} dual-policy episodes")
    return nu_buffer, beta_buffer, new_records


def _prepare_admission(
    context: ModelUpdateContext,
    state: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    members_dir: Path,
) -> tuple[Path, Path]:
    if not state["population"]:
        initial_path = members_dir / "member-000000-initial.npz"
        PolicyBundle(QNetwork(state.pop("initial_job_network")), None).save(initial_path)
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
    nu_snapshot_path = members_dir / (f"nu-snapshot-{iteration:06d}-{context.cutoff_batch_seq:08d}.npz")
    _network_from_state(state["live"]["nu"]).save(nu_snapshot_path)
    _record_nu_snapshot(state, nu_snapshot_path, config, seed)
    beta_path = members_dir / (f"candidate-{iteration:06d}-beta-{context.cutoff_batch_seq}.npz")
    nu_path = members_dir / (f"candidate-{iteration:06d}-nu-average-{context.cutoff_batch_seq}.npz")
    _network_from_state(state["live"]["beta"]).save(beta_path)
    _save_nu_average(state, nu_path)
    request_id = f"admission-{context.update_id}"
    state["pending_admission"] = {
        "request_id": request_id,
        "iteration": iteration,
        "cutoff_batch_seq": context.cutoff_batch_seq,
        "beta_path": str(beta_path),
        "nu_path": str(nu_path),
        "population": [
            {"member_id": str(member["member_id"]), "path": str(member["path"])}
            for member in state["population"]
        ],
        "seeds": int(config["admission_seeds"]),
        "seed_start": seed + 2_000_000 + context.cutoff_batch_seq * 10_000,
    }
    return beta_path, nu_path


def _apply_recorded_evaluations(
    context: ModelUpdateContext,
    state: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    members_dir: Path,
) -> list[ModelEvaluation]:
    """Advance pending evaluation requests from league-recorded control batches."""

    evaluations: list[ModelEvaluation] = []
    for batch in context.new_batches:
        if batch.source_kind != "model_evaluation":
            continue
        metadata = batch.metadata
        if metadata.get("family_id") != "sp_psro_job_scorer":
            continue
        kind = str(metadata.get("evaluation_kind", ""))
        request_id = str(metadata.get("request_id", ""))
        result = dict(metadata.get("result") or {})
        if kind == "admission":
            pending = state.get("pending_admission")
            if pending is None or request_id != pending.get("request_id"):
                continue
            admitted = (
                int(result.get("errors", 0)) == 0
                and int(result.get("pairs", 0)) > 1
                and float((result.get("margin_ci") or [0.0, 0.0])[0]) > 0.0
            )
            state["last_admission"] = {**result, "admitted": admitted}
            iteration = int(pending["iteration"])
            if admitted:
                new_members = []
                for role, key in (("beta", "beta_path"), ("nu-average", "nu_path")):
                    candidate_path = Path(pending[key])
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
                state["population"] = state["population"][-int(config["population_size"]) :]
                state["iteration"] = iteration + 1
                state["live"] = fresh_live(seed + 10_000 + (iteration + 1) * 100)
                state["exploration_start_batch_seq"] = int(pending["cutoff_batch_seq"])
                state["restricted"] = {"arm_ids": [], "log_weights": []}
                state["nu_snapshots"] = []
                state["nu_snapshots_seen"] = 0
                state["nu_snapshot_rng_state"] = None
            Path(pending["nu_path"]).unlink(missing_ok=True)
            state["pending_validation"] = {
                "request_id": f"validation-{request_id}",
                "candidate_id": f"beta-{int(pending['cutoff_batch_seq']):08d}",
                "beta_path": str(pending["beta_path"]),
                "cutoff_batch_seq": int(pending["cutoff_batch_seq"]),
                "seeds": int(config["validation_seeds"]),
            }
            state["pending_admission"] = None
            evaluations.append(ModelEvaluation("population_pair_admission", result, admitted))
            continue

        if kind != "validation":
            continue
        pending = state.get("pending_validation")
        if pending is None or request_id != pending.get("request_id"):
            continue
        suite = str(result.get("suite", ""))
        if state.get("validation_suite") != suite:
            for member in state["population"]:
                member["validation"] = None
            state["validation_suite"] = suite
            state["release_champion"] = None
        member_results = dict(result.get("members") or {})
        for member in state["population"]:
            if str(member["member_id"]) in member_results:
                member["validation"] = member_results[str(member["member_id"])]
        eligible = [
            member
            for member in state["population"]
            if member.get("validation") is not None and not member["validation"].get("errors")
        ]
        if eligible:
            state["population_champion_id"] = max(eligible, key=_candidate_rank)["member_id"]
        beta_validation = dict(result.get("beta") or {})
        release_candidates = [
            {
                "candidate_id": str(member["member_id"]),
                "path": str(member["path"]),
                "validation": member["validation"],
            }
            for member in state["population"]
            if member.get("validation") is not None
            and passes_release_gate(member["validation"], config)
        ]
        beta_path = Path(pending["beta_path"])
        if passes_release_gate(beta_validation, config):
            release_candidates.append(
                {
                    "candidate_id": str(pending["candidate_id"]),
                    "path": str(beta_path),
                    "validation": beta_validation,
                }
            )
        if state.get("release_champion") is not None:
            release_candidates.append(state["release_champion"])
        if release_candidates:
            selected = max(release_candidates, key=_candidate_rank)
            if Path(selected["path"]) == beta_path:
                released_path = members_dir / f"release-{selected['candidate_id']}.npz"
                PopulationArtifact.load(beta_path).save(released_path)
                selected = {**selected, "path": str(released_path)}
            state["release_champion"] = selected
        beta_path.unlink(missing_ok=True)
        state["pending_validation"] = None
        evaluations.append(
            ModelEvaluation(
                "submission_beta_validation",
                beta_validation,
                passes_release_gate(beta_validation, config),
            )
        )
    return evaluations


class PopulationUpdater:
    def update(self, context: ModelUpdateContext, reader: ModelTrainingReader) -> ModelUpdateResult:
        config = context.config
        seed = int(config["seed"])
        state = load_state(
            context.active_checkpoint.trainer_artifact_path if context.active_checkpoint is not None else None,
            seed,
        )
        if state.pop("_needs_training_watermark", False) and context.active_checkpoint is not None:
            row = reader.fetchone(
                """
                SELECT mu.cutoff_batch_seq
                FROM model_training_checkpoints mtc
                JOIN model_updates mu USING (update_id)
                WHERE mtc.checkpoint_id = %s
                """,
                [context.active_checkpoint.checkpoint_id],
            )
            if row is not None:
                state["last_training_batch_seq"] = int(row[0])
        started = time.perf_counter()
        artifact_root = context.artifacts
        members_dir = artifact_root / "population_members"
        checkpoints_dir = artifact_root / "checkpoints"
        members_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        evaluations = _apply_recorded_evaluations(context, state, config, seed, members_dir)
        untrained_batches = [
            batch
            for batch in context.all_batches
            if batch.source_kind == "sp_psro_rollout"
            and batch.sequence > int(state.get("last_training_batch_seq", 0))
        ]
        required = int(config.get("training_batches_per_update", 20))
        should_train = (
            len(untrained_batches) >= required
            and state.get("pending_admission") is None
            and state.get("pending_validation") is None
        )
        nu_losses: list[float] = []
        beta_losses: list[float] = []
        nu_market_losses: list[float] = []
        beta_market_losses: list[float] = []
        nu_market_transitions = 0
        beta_market_transitions = 0
        steps = 0
        updated_arm_ids, updated_probabilities = mixture_probabilities(state, config)

        if should_train:
            nu_buffer, beta_buffer, new_records = _load_training_data(
                context, state, config, seed
            )
            device = resolve_device(str(config.get("device", "auto")))
            steps = int(config["gradient_steps_per_batch"]) * len(untrained_batches)
            print(
                f"[sp-psro training] update started: {len(untrained_batches)} rollout batches, "
                f"{steps} gradient steps per role on {device}",
                flush=True,
            )
            print("[sp-psro training] training nu policy", flush=True)
            state["live"]["nu"], nu_losses, nu_market_losses = _train_role(
                state["live"]["nu"], nu_buffer, config, steps, device
            )
            print("[sp-psro training] nu policy complete; training beta policy", flush=True)
            state["live"]["beta"], beta_losses, beta_market_losses = _train_role(
                state["live"]["beta"], beta_buffer, config, steps, device
            )
            state["live"]["nu"]["replay_rng_state"] = nu_buffer.rng.bit_generator.state
            state["live"]["beta"]["replay_rng_state"] = beta_buffer.rng.bit_generator.state
            _update_restricted_distribution(state, new_records, config)
            updated_arm_ids, updated_probabilities = mixture_probabilities(state, config)
            state["last_training_batch_seq"] = max(
                batch.sequence for batch in untrained_batches
            )
            nu_market_transitions = nu_buffer.market_transitions
            beta_market_transitions = beta_buffer.market_transitions
            _prepare_admission(context, state, config, seed, members_dir)
            print(
                "[sp-psro training] candidate checkpoint complete; "
                "admission evaluation queued for the next league round",
                flush=True,
            )
        elif evaluations:
            print(
                f"[sp-psro training] applied {len(evaluations)} recorded evaluation result(s)",
                flush=True,
            )

        retained_ids = {member["member_id"] for member in state["population"]}
        if state["population_champion_id"] not in retained_ids:
            state["population_champion_id"] = state["population"][-1]["member_id"]

        population_champion = next(
            member for member in state["population"] if member["member_id"] == state["population_champion_id"]
        )
        checkpoint_id = hashlib.sha256(
            f"{context.model_id}\0{context.update_id}\0{context.dataset_sha256}".encode()
        ).hexdigest()[:24]
        trainer_path = checkpoints_dir / f"{checkpoint_id}.pt"
        inference_path = checkpoints_dir / f"{checkpoint_id}.npz"
        state["dataset_sha256"] = context.dataset_sha256
        torch.save(state, trainer_path)
        release_champion = state.get("release_champion")
        inference_source = (
            Path(release_champion["path"]) if release_champion is not None else Path(population_champion["path"])
        )
        PopulationArtifact.load(inference_source).save(inference_path)
        combined_digest = hashlib.sha256(trainer_path.read_bytes() + inference_path.read_bytes()).hexdigest()
        beta_validation = next(
            (item.metrics for item in evaluations if item.kind == "submission_beta_validation"),
            None,
        )
        metrics = {
            "population_size": len(state["population"]),
            "iteration": state["iteration"],
            "admitted": (
                None
                if state.get("last_admission") is None
                else bool(state["last_admission"].get("admitted"))
            ),
            "champion_id": state["population_champion_id"],
            "inference_champion_id": (release_champion["candidate_id"] if release_champion is not None else None),
            "beta_validation": beta_validation,
            "beta_loss": float(np.mean(beta_losses)) if beta_losses else None,
            "nu_loss": float(np.mean(nu_losses)) if nu_losses else None,
            "beta_market_loss": (float(np.mean(beta_market_losses)) if beta_market_losses else None),
            "nu_market_loss": (float(np.mean(nu_market_losses)) if nu_market_losses else None),
            "beta_market_transitions": beta_market_transitions,
            "nu_market_transitions": nu_market_transitions,
            "restricted_distribution": dict(zip(updated_arm_ids, updated_probabilities)),
            "gradient_steps_this_update": steps,
            "pending_admission": state.get("pending_admission") is not None,
            "pending_validation": state.get("pending_validation") is not None,
            "train_seconds": time.perf_counter() - started,
        }
        gate = {
            "kind": "external_submission_quality",
            "passed": release_champion is not None,
            "population_admission": state["last_admission"],
            "beta_validation": beta_validation,
            "thresholds": {
                "money": float(config["release_min_money"]),
                "win_rate": float(config["release_min_win_rate"]),
                "margin": float(config["release_min_margin"]),
            },
        }
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
            tuple(evaluations),
        )
