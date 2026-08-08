"""CLI for this family's surrogate-assisted tournament pipeline.

These commands are specific to the configurable crop strategy family -- they
sample its parameter space, train its Random Forest surrogate, and promote its
profiles. They live here rather than in `kaglab` so that changing this family's
pipeline cannot break another agent.

    python -m agents.configurable_crop_surrogate.cli tournament --config configs/research.toml
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from agents.configurable_crop_surrogate.analysis import analyze_run, propose_strategies
from agents.configurable_crop_surrogate.experiment import ExperimentConfig, run_experiment
from agents.configurable_crop_surrogate.strategy import generate_population
from kaggriculture_lab.storage import ExperimentStore

DEFAULT_DATABASE = "artifacts/experiments.duckdb"
DEFAULT_ARTIFACTS = "artifacts"
STRATEGY_FOLDER = Path("agents/configurable_crop_surrogate")
SAFE_NAME = re.compile(r"[a-z][a-z0-9_]*")


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)


def _cmd_strategies(args: argparse.Namespace) -> int:
    strategies = generate_population(args.count, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            [{"strategy_id": item.strategy_id, "strategy": item.to_dict()} for item in strategies],
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {len(strategies)} strategies to {output}")
    return 0


def _cmd_tournament(args: argparse.Namespace) -> int:
    config = ExperimentConfig.from_toml(args.config)
    run_id = run_experiment(config, args.database, args.artifacts, args.run_id)
    print(run_id)
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    with ExperimentStore(args.database) as store:
        reports = analyze_run(store, args.run_id, args.artifacts)
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


def _cmd_propose(args: argparse.Namespace) -> int:
    with ExperimentStore(args.database) as store:
        strategies = propose_strategies(
            store,
            args.run_id,
            args.artifacts,
            args.candidate_pool,
            args.count,
        )
    print(f"proposed {len(strategies)} strategies")
    return 0


def _cmd_cycle(args: argparse.Namespace) -> int:
    config = ExperimentConfig.from_toml(args.config)
    run_id = run_experiment(config, args.database, args.artifacts, args.run_id)
    with ExperimentStore(args.database) as store:
        reports = analyze_run(store, run_id, args.artifacts)
        proposed = propose_strategies(
            store,
            run_id,
            args.artifacts,
            config.candidate_pool,
            config.proposal_count,
        )
        store.export(run_id, Path(args.artifacts) / "parquet" / run_id)
    print(json.dumps({"run_id": run_id, "models": reports, "proposals": len(proposed)}, indent=2))
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    if not SAFE_NAME.fullmatch(args.name):
        raise ValueError("profile name must use lowercase letters, digits, and underscores")
    with ExperimentStore(args.database) as store:
        strategies = store.load_strategies([args.strategy_id])
    if args.strategy_id not in strategies:
        raise ValueError(f"unknown strategy_id: {args.strategy_id}")
    profile_dir = Path(args.strategy_folder) / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    destination = profile_dir / f"{args.name}.json"
    if destination.exists() and not args.force:
        raise FileExistsError(f"{destination} already exists; pass --force to replace it")
    destination.write_text(json.dumps(strategies[args.strategy_id], indent=2, sort_keys=True) + "\n")
    variants = Path(args.strategy_folder) / "variants"
    variants.mkdir(parents=True, exist_ok=True)
    entry_point = variants / f"{args.name}.py"
    if entry_point.exists() and not args.force:
        raise FileExistsError(f"{entry_point} already exists; pass --force to replace it")
    import_path = ".".join(Path(args.strategy_folder).parts)
    entry_point.write_text(
        f'"""Promoted strategy profile {args.strategy_id}."""\n\n'
        f"from {import_path}._profile import named_agent\n\n"
        f'agent = named_agent("{args.name}")\n'
    )
    print(f"{destination}\n{entry_point}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="configurable-crop", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    strategies = commands.add_parser("strategies", help="generate a JSON strategy population")
    strategies.add_argument("--count", type=int, default=20)
    strategies.add_argument("--seed", type=int, default=42)
    strategies.add_argument("--output", default="artifacts/strategies.json")
    strategies.set_defaults(func=_cmd_strategies)

    tournament = commands.add_parser("tournament", help="run the configured three-stage tournament")
    tournament.add_argument("--config", required=True)
    tournament.add_argument("--run-id")
    _add_paths(tournament)
    tournament.set_defaults(func=_cmd_tournament)

    analyze = commands.add_parser("analyze", help="calculate Elo and train surrogate models")
    analyze.add_argument("--run-id", required=True)
    _add_paths(analyze)
    analyze.set_defaults(func=_cmd_analyze)

    propose = commands.add_parser("propose", help="generate a model-recommended strategy batch")
    propose.add_argument("--run-id", required=True)
    propose.add_argument("--candidate-pool", type=int)
    propose.add_argument("--count", type=int)
    _add_paths(propose)
    propose.set_defaults(func=_cmd_propose)

    cycle = commands.add_parser("cycle", help="tournament, analyze, and propose in one command")
    cycle.add_argument("--config", required=True)
    cycle.add_argument("--run-id")
    _add_paths(cycle)
    cycle.set_defaults(func=_cmd_cycle)

    promote = commands.add_parser("promote", help="save a database strategy as a named profile")
    promote.add_argument("--strategy-id", required=True)
    promote.add_argument("--name", required=True)
    promote.add_argument("--strategy-folder", default=str(STRATEGY_FOLDER))
    promote.add_argument("--force", action="store_true")
    promote.add_argument("--database", default=DEFAULT_DATABASE)
    promote.set_defaults(func=_cmd_promote)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
