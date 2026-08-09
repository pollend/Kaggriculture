"""Head-to-head evaluation for kaggriculture agents.

    python -m sim.arena match  main.py starter --seeds 100
    python -m sim.arena league main.py v2.py starter --seeds 50
    python -m sim.arena sweep  main.py --vs starter --param hire_cap=8,12,16 --seeds 100

Every seed is played in both seat orientations; see `sim.metrics` for why.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from sim import registry
from sim.metrics import Summary, render, summarize
from sim.runner import MatchResult, run_match


def _job(args: tuple[tuple[str, str], int, dict[str, Any]]) -> MatchResult:
    """Worker entry point. Specs stay strings so nothing unpicklable crosses over."""
    specs, seed, config = args
    return run_match(specs, seed, config)


def play_series(
    a: str,
    b: str,
    seeds: Iterable[int],
    config: dict[str, Any],
    jobs: int,
) -> dict[int, tuple[MatchResult, MatchResult]]:
    """Play every seed in both orientations. Returns {seed: (a_first, b_first)}."""
    seeds = list(seeds)
    tasks = []
    for seed in seeds:
        tasks.append(((a, b), seed, config))
        tasks.append(((b, a), seed, config))

    if jobs == 1:
        outcomes = [_job(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=jobs, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            outcomes = list(pool.map(_job, tasks, chunksize=1))

    return {seed: (outcomes[2 * i], outcomes[2 * i + 1]) for i, seed in enumerate(seeds)}


def _save_replays(
    results: dict[int, tuple[MatchResult, MatchResult]], a: str, directory: Path
) -> int:
    """Dump replays for seeds where A did not win -- the ones worth debugging."""
    directory.mkdir(parents=True, exist_ok=True)
    saved = 0
    for seed, (forward, reverse) in results.items():
        if forward.failed or reverse.failed:
            interesting = True
        else:
            interesting = forward.margin is not None and (forward.margin + -(reverse.margin)) / 2 <= 0
        if not interesting:
            continue
        for match, label in ((forward, "fwd"), (reverse, "rev")):
            if match.replay is None:
                continue
            path = directory / f"seed{seed}_{label}.json"
            path.write_text(json.dumps(match.replay))
            saved += 1
    return saved


def _warn_nondeterministic(specs: Iterable[str]) -> None:
    bad = [s for s in specs if not registry.is_deterministic(s)]
    if bad:
        print(
            f"warning: {', '.join(bad)} is not reproducible across runs "
            f"(builds an unseeded RNG per call). Use `kaggriculture_lab.opponents:make_seeded_random` for a "
            f"repeatable stochastic opponent.\n",
            file=sys.stderr,
        )


def _run_head_to_head(a: str, b: str, args: argparse.Namespace, config: dict) -> Summary:
    results = play_series(a, b, range(args.seeds), config, args.jobs)
    summary = summarize(a, b, results, act_timeout=config.get("actTimeout", 1.0))
    if args.replay_dir:
        saved = _save_replays(results, a, Path(args.replay_dir))
        print(f"  replays      : {saved} file(s) -> {args.replay_dir}")
    return summary


def cmd_match(args: argparse.Namespace) -> int:
    config = _parse_config(args)
    _warn_nondeterministic([args.a, args.b])
    summary = _run_head_to_head(args.a, args.b, args, config)
    print(render(summary))
    return 1 if summary.errors else 0


def cmd_league(args: argparse.Namespace) -> int:
    config = _parse_config(args)
    _warn_nondeterministic(args.agents)

    summaries: list[Summary] = []
    for a, b in itertools.combinations(args.agents, 2):
        summary = _run_head_to_head(a, b, args, config)
        summaries.append(summary)
        print(render(summary))
        print()

    print(_render_table(args.agents, summaries))
    return 1 if any(s.errors for s in summaries) else 0


def cmd_sweep(args: argparse.Namespace) -> int:
    config = _parse_config(args)
    name, _, raw_values = args.param.partition("=")
    if not raw_values:
        print("--param must look like key=v1,v2,v3", file=sys.stderr)
        return 2

    variants = [f"{args.agent}?{name}={value}" for value in raw_values.split(",")]
    _warn_nondeterministic(variants + [args.vs])

    summaries = []
    for variant in variants:
        summary = _run_head_to_head(variant, args.vs, args, config)
        summaries.append(summary)
        print(render(summary))
        print()

    print(f"sweep of {name} against {args.vs}, ranked by paired margin:")
    for summary in sorted(summaries, key=lambda s: s.margin_mean, reverse=True):
        value = summary.a.rsplit("=", 1)[-1]
        marker = "" if summary.significant else "  (not resolved)"
        print(f"  {name}={value:<10} margin {summary.margin_mean:+10,.0f}"
              f"  win {summary.win_rate:5.1%}{marker}")
    return 1 if any(s.errors for s in summaries) else 0


def _render_table(agents: list[str], summaries: list[Summary]) -> str:
    """Round-robin win-rate matrix (row agent's rate against column agent)."""
    rates: dict[tuple[str, str], float] = {}
    for summary in summaries:
        rate = summary.win_rate + summary.draws / (2 * summary.pairs) if summary.pairs else 0.0
        rates[(summary.a, summary.b)] = rate
        rates[(summary.b, summary.a)] = 1 - rate

    width = max(len(a) for a in agents) + 2
    header = " " * width + "".join(f"{a[:10]:>12}" for a in agents)
    lines = [header]
    for row in agents:
        cells = "".join(
            f"{'--':>12}" if row == col else f"{rates.get((row, col), 0.0):>11.1%} "
            for col in agents
        )
        lines.append(f"{row:<{width}}{cells}")
    return "\n".join(lines)


def _parse_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for item in args.config or []:
        key, _, value = item.partition("=")
        config[key] = registry._coerce(value)
    if args.steps:
        config["episodeSteps"] = args.steps
    config.setdefault("actTimeout", 1.0)
    return config


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seeds", type=int, default=50, help="number of seeds; each is played in both seats")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--steps", type=int, default=None, help="override episodeSteps for fast coarse sweeps")
    parser.add_argument("--replay-dir", default=None, help="dump replay JSON for non-wins and errors")
    parser.add_argument("--config", action="append", metavar="KEY=VALUE", help="env configuration override")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sim.arena", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    match = subparsers.add_parser("match", help="one head-to-head")
    match.add_argument("a")
    match.add_argument("b")
    _add_common(match)
    match.set_defaults(func=cmd_match)

    league = subparsers.add_parser("league", help="round-robin over 3+ agents")
    league.add_argument("agents", nargs="+")
    _add_common(league)
    league.set_defaults(func=cmd_league)

    sweep = subparsers.add_parser("sweep", help="one agent across parameter values")
    sweep.add_argument("agent")
    sweep.add_argument("--vs", required=True)
    sweep.add_argument("--param", required=True, metavar="KEY=V1,V2,V3")
    _add_common(sweep)
    sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
