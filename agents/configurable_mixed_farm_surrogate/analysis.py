"""Paired acceptance analysis for the checked-in mixed-farm profile."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from sim.arena import play_series

ROOT = Path(__file__).resolve().parents[2]
MIXED = "agents/configurable_mixed_farm_surrogate/main.py"
CROP = "agents/configurable_crop_surrogate/main.py"
MOON = "agents/competitor_prvsiyan_kaggriculture_frontier_the_moon_counts_melons/main.py"
COMPETITORS = tuple(
    str(path.relative_to(ROOT))
    for path in sorted((ROOT / "agents").glob("competitor_*/main.py"))
)


def _margins(results) -> dict[int, float]:
    values: dict[int, float] = {}
    for seed, (forward, reverse) in results.items():
        if forward.failed or reverse.failed or forward.margin is None or reverse.margin is None:
            raise RuntimeError(f"failed paired episode for seed {seed}")
        values[int(seed)] = (float(forward.margin) - float(reverse.margin)) / 2.0
    return values


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def _series(agent: str, opponent: str, seeds: range, jobs: int) -> dict[int, float]:
    return _margins(play_series(agent, opponent, seeds, {}, jobs))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--broad", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds < 1 or args.jobs < 1:
        parser.error("--seeds and --jobs must be positive")

    seeds = range(args.seeds)
    mixed_moon = _series(MIXED, MOON, seeds, args.jobs)
    crop_moon = _series(CROP, MOON, seeds, args.jobs)
    improvement = [mixed_moon[seed] - crop_moon[seed] for seed in seeds]
    improvement_mean, improvement_low, improvement_high = _mean_ci(improvement)
    moon_passed = improvement_mean >= 25_000 and improvement_low > 0
    report: dict[str, object] = {
        "moon": {
            "seeds": args.seeds,
            "mixed_margin": statistics.fmean(mixed_moon.values()),
            "crop_margin": statistics.fmean(crop_moon.values()),
            "improvement": improvement_mean,
            "improvement_ci95": [improvement_low, improvement_high],
            "passed": moon_passed,
        }
    }

    broad_passed = True
    if args.broad:
        broad_seeds = range(min(10, args.seeds))
        opponent_differences: list[float] = []
        rows: dict[str, float] = {}
        for opponent in COMPETITORS:
            mixed = _series(MIXED, opponent, broad_seeds, args.jobs)
            crop = _series(CROP, opponent, broad_seeds, args.jobs)
            difference = statistics.fmean(mixed[seed] - crop[seed] for seed in broad_seeds)
            rows[opponent] = difference
            opponent_differences.append(difference)
        broad_mean, broad_low, broad_high = _mean_ci(opponent_differences)
        broad_passed = broad_low > -5_000
        report["broad"] = {
            "seeds_per_opponent": len(broad_seeds),
            "opponents": rows,
            "macro_improvement": broad_mean,
            "macro_improvement_ci95": [broad_low, broad_high],
            "passed": broad_passed,
        }

    report["passed"] = moon_passed and broad_passed
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
