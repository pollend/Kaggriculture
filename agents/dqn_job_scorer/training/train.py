"""Double DQN training loop for the job-scoring agent.

    python -m agents.dqn_job_scorer.training.train --rounds 200 --jobs 8

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

from agents.dqn_job_scorer.policy import WEIGHTS_PATH
from agents.dqn_job_scorer.training import network as net_module
from agents.dqn_job_scorer.training.collect import MONEY_SCALE, collect_batch
from agents.dqn_job_scorer.training.replay import ReplayBuffer

HEURISTIC = "agents/configurable_crop_surrogate/main.py"
SELF_SPEC = "agents/dqn_job_scorer/main.py"


@dataclass
class Config:
    rounds: int = 200
    episodes_per_round: int = 16
    gradient_steps: int = 400
    batch_size: int = 512
    learning_rate: float = 1e-4
    gamma: float = 0.997
    n_step: int = 3
    buffer_episodes: int = 200
    target_sync: int = 2000
    epsilon_start: float = 1.0
    # Floor deliberately well above 0: the previous run was strong at 0.40 and
    # degenerate once epsilon reached 0.05, the greedy policy having narrowed
    # onto a slice of the state space it no longer explored.
    epsilon_end: float = 0.15
    epsilon_decay_transitions: int = 200_000
    warmup_episodes: int = 16
    eval_every: int = 10
    eval_seeds: int = 24  # selection noise directly costs us the best checkpoint
    jobs: int = 8
    seed: int = 0
    # Opponent curriculum: round fraction -> spec. The heuristic is the real
    # bar; `starter` only exists to get gradients flowing early.
    curriculum: tuple[tuple[float, str], ...] = ((0.0, "starter"), (0.15, HEURISTIC))
    self_play_after: float = 0.5
    self_play_pool: int = 5
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
        # Mix frozen snapshots with the heuristic so the policy cannot drift into
        # beating only its own past self. A snapshot is addressed through the
        # agent's `weights` config key, not as a bare .npz path -- the registry
        # resolves agent modules, not weight files.
        pool = [f"{SELF_SPEC}?weights={path}" for path in snapshots[-config.self_play_pool:]]
        return [spec, *pool]
    return [spec]


def evaluate(weights_path: Path, opponent: str, seeds: int, jobs: int) -> dict:
    """Run the real arena at epsilon=0 and return the paired summary."""
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
        "margin_ci": summary.margin_ci,
        "significant": summary.significant,
        "errors": summary.errors,
    }


def train(config: Config, output: Path) -> list[dict]:
    torch.manual_seed(config.seed)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    online = net_module.build()
    target = net_module.build()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    loss_fn = nn.SmoothL1Loss()

    buffer = ReplayBuffer(config.buffer_episodes, config.gamma, config.n_step, config.seed)
    rng = np.random.default_rng(config.seed)
    history: list[dict] = []
    snapshots: list[Path] = []
    gradient_steps = 0
    seen = 0
    best_margin: float | None = None
    best_round: int | None = None

    for round_index in range(config.rounds):
        weights = net_module.to_numpy(online).layers
        epsilon = epsilon_for(seen, config)
        opponents = opponents_for(round_index, config, snapshots)
        seeds = [int(value) for value in rng.integers(0, 2**31 - 1, size=config.episodes_per_round)]

        started = time.perf_counter()
        records = collect_batch(weights, opponents, seeds, epsilon, config.gamma, config.jobs)
        collect_time = time.perf_counter() - started
        if not records:
            print(f"round {round_index}: every episode errored -- stopping")
            break

        buffer.add(records)
        seen += sum(int(record.unit_mask.sum()) for record in records)

        losses: list[float] = []
        if len(buffer.episodes) >= config.warmup_episodes:
            started = time.perf_counter()
            for _ in range(config.gradient_steps):
                batch = buffer.sample(config.batch_size)
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
            "loss": round(float(np.mean(losses)), 6) if losses else None,
            "final_money": round(money, 1),
            "raw_return": round(float(np.mean([record.raw_reward.sum() for record in records])) * MONEY_SCALE, 1),
            "job_truncations": truncated,
            "collect_s": round(collect_time, 1),
            "train_s": round(train_time, 1),
        }

        if round_index and round_index % config.eval_every == 0:
            snapshot = snapshot_dir / f"round{round_index:04d}.npz"
            exported = net_module.to_numpy(online)
            exported.save(snapshot)
            snapshots.append(snapshot)

            result = evaluate(snapshot, HEURISTIC, config.eval_seeds, config.jobs)
            entry["eval"] = result

            # Promote by measured result, never by recency. DQN training is not
            # monotonic: the previous run peaked at round 45 (+2,298) and
            # collapsed to -21,496 by round 75, so shipping the last checkpoint
            # ships the collapse.
            if best_margin is None or result["margin"] > best_margin:
                best_margin, best_round = result["margin"], round_index
                exported.save(output / "best.npz")
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
    net_module.to_numpy(online).save(output / "final.npz")
    torch.save(online.state_dict(), output / "online.pt")
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
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    parser.add_argument("--output", default="runs/dqn", type=str)
    args = parser.parse_args(argv)

    values = {name: getattr(args, name) for name in asdict(defaults) if name != "curriculum"}
    config = Config(**values)
    train(config, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
