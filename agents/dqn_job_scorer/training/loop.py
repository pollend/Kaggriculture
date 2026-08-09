"""Double DQN training loop for the job-scoring agent.

    python -m agents.dqn_job_scorer.training.loop --rounds 200 --jobs 8

Progress is judged by the arena, not by training return: every `--eval-every`
rounds the current weights are exported and played against a fixed baseline over
seat-swapped seeds, and the paired margin is what gets logged. Training return
moves for reasons that have nothing to do with winning.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from agents.dqn_job_scorer.market_policy import MARKET_FEATURE_DIM
from agents.dqn_job_scorer.policy import WEIGHTS_PATH, PolicyBundle
from agents.dqn_job_scorer.training import network as net_module
from agents.dqn_job_scorer.training.collect import MONEY_SCALE, collect_batch
from agents.dqn_job_scorer.training.device import resolve_device
from agents.dqn_job_scorer.training.replay import ReplayBuffer

HEURISTIC = "agents/configurable_crop_surrogate/main.py"
SELF_SPEC = "agents/dqn_job_scorer/main.py"


@dataclass
class Config:
    device: str = "auto"
    rounds: int = 200
    episodes_per_round: int = 16
    gradient_steps: int = 400
    batch_size: int = 512
    learning_rate: float = 1e-4
    market_learning_rate: float = 1e-4
    market_gradient_steps: int = 100
    market_batch_size: int = 128
    market_target_sync: int = 1000
    gamma: float = 0.997
    n_step: int = 3
    # 800 episodes at 16 per round is ~50 rounds of history (~1.3 GB), against
    # the 12.5 rounds that let dqn03 forget its own past between checkpoints.
    buffer_episodes: int = 800
    # Share of the buffer held as a uniform sample of all older episodes
    # rather than the most recent ones.
    reservoir_fraction: float = 0.35
    target_sync: int = 2000
    epsilon_start: float = 1.0
    # Floor deliberately well above 0: the previous run was strong at 0.40 and
    # degenerate once epsilon reached 0.05, the greedy policy having narrowed
    # onto a slice of the state space it no longer explored.
    epsilon_end: float = 0.15
    epsilon_decay_transitions: int = 200_000
    warmup_episodes: int = 16
    eval_every: int = 10
    eval_seeds: int = 40  # selection noise directly costs us the best checkpoint
    # Seeds for the head-to-head against the incumbent, which is what actually
    # decides promotion. Larger than eval_seeds because it is the deciding test.
    duel_seeds: int = 60
    promotion_seed_base: int = 1_000_000
    promotion_seed_stride: int = 10_000
    # Skip the duel when the benchmark margin is this far below the incumbent's.
    # Purely a cost filter -- a checkpoint this far behind will not win a duel.
    challenge_slack: float = 1500.0
    jobs: int = 8
    seed: int = 0
    # Opponent curriculum: round fraction -> spec. The heuristic is the real
    # bar; `starter` only exists to get gradients flowing early.
    curriculum: tuple[tuple[float, str], ...] = ((0.0, "starter"), (0.15, HEURISTIC))
    self_play_after: float = 0.5
    self_play_pool: int = 5
    # Share of episodes played against frozen past selves once self-play
    # starts. Evaluation is always against the benchmark, so a high share
    # trains on a distribution the objective never measures.
    self_play_ratio: float = 0.25
    # The reviewed checkpoint is both the training initialization and the bar a
    # challenger must beat. A new output directory never means a blank slate.
    # Off by default: a training run must not overwrite the live submission
    # weights just by existing. A smoke run would otherwise ship its own
    # half-trained net over a good checkpoint.
    ship: bool = False


def epsilon_for(transitions: int, config: Config) -> float:
    fraction = min(1.0, transitions / max(1, config.epsilon_decay_transitions))
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def opponents_for(round_index: int, config: Config, snapshots: list[Path]) -> list[str]:
    progress = round_index / max(1, config.rounds)
    spec = config.curriculum[0][1]
    for threshold, candidate in config.curriculum:
        if progress >= threshold:
            spec = candidate
    if progress >= config.self_play_after and snapshots:
        # Mix frozen snapshots with the benchmark opponent. A snapshot is
        # addressed through the agent's `weights` config key, not as a bare .npz
        # path -- the registry resolves agent modules, not weight files.
        #
        # The ratio matters and used to be an accident: returning [spec, *pool]
        # with a pool of 5 made `collect_batch`'s round-robin play 5 of every 6
        # episodes against past selves, while evaluation is always against the
        # benchmark. Round 110 of dqn04 fell from +2,269 to -22,154 on exactly
        # the round this switched on. The benchmark is now repeated to hold the
        # intended share.
        pool = [f"{SELF_SPEC}?weights={path}" for path in snapshots[-config.self_play_pool:]]
        ratio = min(max(config.self_play_ratio, 0.0), 0.9)
        if ratio <= 0.0:
            return [spec]
        # Repeat the benchmark so self-play episodes are `ratio` of the round.
        benchmark_slots = max(1, round(len(pool) * (1 - ratio) / ratio))
        return [*([spec] * benchmark_slots), *pool]
    return [spec]


def evaluate(
    weights_path: Path,
    opponent: str,
    seeds: int,
    jobs: int,
    seed_start: int = 0,
) -> dict:
    """Run the real arena at epsilon=0 and return the paired summary."""
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
        "margin_ci": summary.margin_ci,
        "significant": summary.significant,
        "errors": summary.errors,
    }


def evaluate_head_to_head(
    challenger: Path,
    incumbent: Path,
    seeds: int,
    jobs: int,
    seed_start: int = 0,
) -> dict:
    """Play the challenger directly against the incumbent checkpoint.

    This is the deciding test. Two checkpoints can post the same margin against
    the benchmark and still be far apart against each other, so a benchmark-only
    comparison picks the wrong one.
    """
    from sim.arena import play_series
    from sim.metrics import summarize

    left = f"{SELF_SPEC}?weights={challenger}"
    right = f"{SELF_SPEC}?weights={incumbent}"
    summary = summarize(
        left,
        right,
        play_series(left, right, range(seed_start, seed_start + seeds), {}, jobs),
    )
    return {
        "pairs": summary.pairs,
        "wins": summary.wins,
        "losses": summary.losses,
        "margin": summary.margin_mean,
        "margin_ci": list(summary.margin_ci),
        "significant": summary.significant,
    }


def train(config: Config, output: Path) -> list[dict]:
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    online = net_module.build().to(device)
    target = net_module.build().to(device)
    market_online = net_module.build(feature_dim=MARKET_FEATURE_DIM).to(device)
    market_target = net_module.build(feature_dim=MARKET_FEATURE_DIM).to(device)
    target.load_state_dict(online.state_dict())
    market_target.load_state_dict(market_online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    market_optimizer = torch.optim.Adam(market_online.parameters(), lr=config.market_learning_rate)
    loss_fn = nn.SmoothL1Loss()

    buffer = ReplayBuffer(
        config.buffer_episodes, config.gamma, config.n_step, config.seed,
        reservoir_fraction=config.reservoir_fraction,
    )
    rng = np.random.default_rng(config.seed)
    history: list[dict] = []
    snapshots: list[Path] = []
    gradient_steps = 0
    market_gradient_steps = 0
    seen = 0
    best_round: int | None = None
    best_margin = float("-inf")

    for round_index in range(config.rounds):
        weights = net_module.to_numpy(online).layers
        epsilon = epsilon_for(seen, config)
        opponents = opponents_for(round_index, config, snapshots)
        seeds = [int(value) for value in rng.integers(0, 2**31 - 1, size=config.episodes_per_round)]

        started = time.perf_counter()
        records = collect_batch(
            weights,
            opponents,
            seeds,
            epsilon,
            config.gamma,
            config.jobs,
            market_weights=net_module.to_numpy(market_online).layers,
            market_epsilon=epsilon,
        )
        collect_time = time.perf_counter() - started
        if not records:
            print(f"round {round_index}: every episode errored -- stopping")
            break

        buffer.add(records)
        seen += sum(int(record.unit_mask.sum()) for record in records)

        losses: list[float] = []
        market_losses: list[float] = []
        if len(buffer.episodes) >= config.warmup_episodes:
            started = time.perf_counter()
            for _ in range(config.gradient_steps):
                batch = buffer.sample(config.batch_size)
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
                    # Double DQN: online picks the job, target prices it.
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
                if gradient_steps % config.target_sync == 0:
                    target.load_state_dict(online.state_dict())
            for _ in range(config.market_gradient_steps):
                batch = buffer.sample_market(config.market_batch_size)
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
                    values = target_q.gather(2, best.unsqueeze(2)).squeeze(2)
                    values = torch.where(next_mask.any(dim=2), values, torch.zeros_like(values))
                    goal = reward + discount * values.sum(dim=1)
                predicted = market_online(features.reshape(-1, features.shape[-1])).reshape(features.shape[:2]).sum(dim=1)
                loss = loss_fn(predicted, goal)
                market_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(market_online.parameters(), 10.0)
                market_optimizer.step()
                market_losses.append(float(loss.detach()))
                market_gradient_steps += 1
                if market_gradient_steps % config.market_target_sync == 0:
                    market_target.load_state_dict(market_online.state_dict())
            train_time = time.perf_counter() - started
        else:
            train_time = 0.0

        money = float(np.mean([record.money[-1] for record in records]))
        truncated = int(sum(int(record.truncated.sum()) for record in records))
        entry = {
            "round": round_index,
            "epsilon": round(epsilon, 4),
            "opponents": opponents,
            "episodes": len(records),
            "transitions": seen,
            "buffer_mb": round(buffer.nbytes() / 1e6, 1),
            "buffer_recent": len(buffer.recent),
            "buffer_reservoir": len(buffer.reservoir),
            "loss": round(float(np.mean(losses)), 6) if losses else None,
            "market_loss": round(float(np.mean(market_losses)), 6) if market_losses else None,
            "market_transitions": buffer.market_transitions,
            "final_money": round(money, 1),
            "raw_return": round(float(np.mean([record.raw_reward.sum() for record in records])) * MONEY_SCALE, 1),
            "job_truncations": truncated,
            "collect_s": round(collect_time, 1),
            "train_s": round(train_time, 1),
        }

        if round_index and round_index % config.eval_every == 0:
            snapshot = snapshot_dir / f"round{round_index:04d}.npz"
            exported = PolicyBundle(net_module.to_numpy(online), net_module.to_numpy(market_online))
            exported.save(snapshot)
            snapshots.append(snapshot)

            seed_start = (
                config.promotion_seed_base + round_index * config.promotion_seed_stride
            )
            result = evaluate(
                snapshot,
                HEURISTIC,
                config.eval_seeds,
                config.jobs,
                seed_start,
            )
            entry["eval"] = result

            # Promote by measured result, never by recency. DQN training is not
            # monotonic: an earlier run peaked at round 45 (+2,298) and collapsed
            # to -21,496 by round 75, so shipping the last checkpoint ships the
            # collapse.
            #
            # The benchmark margin alone is not enough to choose between two good
            # checkpoints. dqn04's round 100 and dqn03's round 60 scored +2,558
            # and +2,545 against the heuristic over 80 seeds -- indistinguishable,
            # same 78-2 record -- yet round 60 beat round 100 65-15 head to head.
            # Equal margin against a common opponent is not equal strength. So the
            # benchmark is only a cheap filter, and the incumbent decides.
            incumbent_path = output / "best.npz"
            if not incumbent_path.is_file():
                promote, reason = not result["errors"], "initial valid checkpoint"
            else:
                incumbent_result = evaluate(incumbent_path, HEURISTIC, config.eval_seeds, config.jobs, seed_start)
                entry["incumbent_eval"] = incumbent_result
                if result["margin"] < incumbent_result["margin"] - config.challenge_slack:
                    promote, reason = False, "benchmark margin far below incumbent"
                else:
                    duel = evaluate_head_to_head(snapshot, incumbent_path, config.duel_seeds, config.jobs, seed_start)
                    entry["duel"] = duel
                    promote = duel["significant"] and duel["margin"] > 0
                    reason = "beat incumbent head to head" if promote else "did not beat incumbent"

            entry["promotion"] = reason
            if promote:
                best_margin, best_round = result["margin"], round_index
                exported.save(incumbent_path)
                if config.ship:
                    exported.save(WEIGHTS_PATH)
                entry["promoted"] = True
            entry["best_round"] = best_round
            entry["best_margin"] = best_margin

        history.append(entry)
        print(json.dumps(entry))
        (output / "history.json").write_text(json.dumps(history, indent=2))

    # The final net is saved for resuming, but the shipped weights stay whichever
    # checkpoint actually evaluated best.
    PolicyBundle(net_module.to_numpy(online), net_module.to_numpy(market_online)).save(output / "final.npz")
    torch.save({"job": online.state_dict(), "market": market_online.state_dict()}, output / "online.pt")
    print(json.dumps({
        "best_round": best_round,
        "best_margin": best_margin,
        "best_weights": str(output / "best.npz"),
        "shipped": str(WEIGHTS_PATH) if config.ship else None,
    }))
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = Config()
    for name, value in asdict(defaults).items():
        if name == "curriculum":
            continue
        flag = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):
            # `type=bool` would make `--ship false` evaluate true, since
            # bool("false") is True.
            parser.add_argument(flag, action="store_true", default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    parser.add_argument("--output", default="runs/dqn", type=str)
    args = parser.parse_args(argv)

    values = {name: getattr(args, name) for name in asdict(defaults) if name != "curriculum"}
    config = Config(**values)
    train(config, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
