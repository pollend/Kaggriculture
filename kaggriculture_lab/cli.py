"""Command-line interface for the Kaggriculture strategy lab."""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from dataclasses import asdict
from pathlib import Path

from kaggriculture_lab.dashboard import serve as serve_dashboard
from kaggriculture_lab.identity import AgentManifest
from kaggriculture_lab.league import (
    LeagueConfig,
    create_next_generation,
    create_session,
    leaderboard,
    reevaluate_hooks,
    run_generation,
    session_status,
    set_training_agents,
    set_training_manifests,
)
from kaggriculture_lab.storage import ExperimentStore

DEFAULT_DATABASE = "artifacts/experiments.duckdb"
DEFAULT_ARTIFACTS = "artifacts"
SAFE_NAME = re.compile(r"[a-z][a-z0-9_]*")


def _cmd_export(args: argparse.Namespace) -> int:
    destination = (
        Path(args.output) if args.output else Path(args.artifacts) / ("csv" if args.csv else "parquet") / args.run_id
    )
    with ExperimentStore(args.database) as store:
        paths = store.export(args.run_id, destination, csv=args.csv)
    print("\n".join(str(path) for path in paths))
    return 0


def _cmd_new_attempt(args: argparse.Namespace) -> int:
    if not SAFE_NAME.fullmatch(args.name):
        raise ValueError("strategy folder name must use lowercase letters, digits, and underscores")
    destination = Path("agents") / args.name
    if destination.exists():
        raise FileExistsError(f"strategy folder already exists: {destination}")
    destination.mkdir(parents=True)
    (destination / "README.md").write_text(
        f"# {args.title}\n\n"
        "Independent strategy attempt. Keep its implementation, profiles, variants, "
        "results, and design notes inside this folder.\n"
    )
    (destination / "__init__.py").write_text(f'"""{args.title} strategy attempt."""\n')
    (destination / "main.py").write_text(
        f'"""Entry point for {args.title}."""\n\n'
        "from typing import TYPE_CHECKING, Any\n\n"
        "if TYPE_CHECKING:\n"
        "    from kaggriculture_lab.hooks import EpisodeHookContext, EpisodeReader\n\n"
        "def agent(obs, configuration=None):\n"
        '    farms = obs.get("farms", [])\n'
        '    player = obs.get("player", 0)\n'
        '    hands = farms[player].get("hands", []) if farms else []\n'
        '    return {"farmer": ["PASS"], "hands": [["PASS"] for _ in hands], '
        '"market": []}\n\n\n'
        "def evaluate_episode(\n"
        '    context: "EpisodeHookContext", reader: "EpisodeReader"\n'
        ") -> dict[str, Any]:\n"
        '    """Derive training features after this episode is stored in DuckDB."""\n'
        "    summary = reader.summary()\n"
        '    return {"schema_version": 1, "money_difference": summary["money_difference"]}\n'
    )
    (destination / "agent.toml").write_text(
        f'family_id = "{args.name}"\n'
        f'name = "{args.title}"\n'
        f'entry_point = "agents.{args.name}.main:agent"\n'
        "parameter_schema_version = 1\n"
        f'episode_evaluator = "agents.{args.name}.main:evaluate_episode"\n'
        f'code_paths = ["agents/{args.name}/main.py"]\n'
    )
    print(destination)
    return 0


def _cmd_bundle(args: argparse.Namespace) -> int:
    """Bundle exactly what an agent's manifest declares.

    `code_paths` is an allowlist, so anything the agent does not name -- its
    trainer, its sklearn or torch dependencies, its research pipeline -- is
    excluded by construction rather than by a hand-maintained deny list. The
    previous version hardcoded one agent's files and could not bundle any other.
    """
    manifest = AgentManifest.load(args.agent)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    included = [Path(path) for path in manifest.code_paths]
    missing = [path for path in included if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{args.agent} declares code_paths that do not exist: " + ", ".join(str(path) for path in missing)
        )

    # `entry_point` is the *league's* entry and may be a factory that takes
    # parameters (`agent:make_agent`). A submission needs a ready-to-call
    # `agent`, which by convention is the agent package's own `main.py` -- the
    # same module Kaggle would run.
    package = ".".join(Path(args.agent).parent.parts)
    entry_module = f"{package}.main"
    if not any(path.as_posix() == f"{package.replace('.', '/')}/main.py" for path in included):
        raise ValueError(f"{args.agent} must list its main.py in code_paths to be bundleable")

    with tarfile.open(output, "w:gz") as archive:
        for path in included:
            archive.add(path, arcname=str(path))
        # Kaggle requires a top-level main.py.
        shim = f"from {entry_module} import agent as agent\n"
        info = tarfile.TarInfo("main.py")
        payload = shim.encode()
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    print(output)
    return 0


def _cmd_league_init(args: argparse.Namespace) -> int:
    config = LeagueConfig.from_toml(args.config)
    session_id = create_session(config, args.database, session_id=args.session_id)
    print(session_id)
    return 0


def _cmd_league_run(args: argparse.Namespace) -> int:
    # Registration lives here rather than inside `run_generation` so the library
    # function never discovers the working tree's agents on its own.
    with ExperimentStore(args.database) as store:
        _register_training_models(store)
    result = run_generation(args.session_id, args.database, args.artifacts, update_models=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_league_status(args: argparse.Namespace) -> int:
    print(json.dumps(session_status(args.session_id, args.database), indent=2, sort_keys=True))
    return 0


def _cmd_league_leaderboard(args: argparse.Namespace) -> int:
    print(json.dumps(leaderboard(args.session_id, args.database, args.limit), indent=2, sort_keys=True))
    return 0


def _cmd_league_next(args: argparse.Namespace) -> int:
    print(create_next_generation(args.session_id, args.database))
    return 0


def _cmd_league_evaluate(args: argparse.Namespace) -> int:
    result = reevaluate_hooks(
        args.session_id,
        args.database,
        generation_id=args.generation_id,
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_league_trainers(args: argparse.Namespace) -> int:
    """Repoint a session's trainers without discarding its history."""

    if args.config:
        config = LeagueConfig.from_toml(args.config)
        identifiers = set_training_agents(args.session_id, args.database, config)
        payload = {
            "training_agents": [asdict(item) for item in config.training_agents],
            "model_ids": list(identifiers),
        }
    else:
        manifests = list(args.manifest)
        families = set_training_manifests(args.session_id, args.database, manifests)
        payload = {"training_manifests": manifests, "families": list(families)}
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_web(args: argparse.Namespace) -> int:
    serve_dashboard(args.database, args.host, args.port)
    return 0


def _register_training_models(store: ExperimentStore) -> None:
    for path in sorted(Path("agents").glob("*/agent.toml")):
        manifest = AgentManifest.load(path)
        if manifest.models:
            store.register_family(manifest, path)


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kaglab", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="export one run to Parquet or CSV")
    export.add_argument("--run-id", required=True)
    export.add_argument("--output")
    export.add_argument("--csv", action="store_true")
    _add_paths(export)
    export.set_defaults(func=_cmd_export)

    attempt = commands.add_parser("new-attempt", help="create a descriptive folder for another approach")
    attempt.add_argument("name", help="lowercase descriptive folder name")
    attempt.add_argument("--title", required=True)
    attempt.set_defaults(func=_cmd_new_attempt)

    bundle = commands.add_parser("bundle", help="build a multi-file Kaggle submission archive")
    bundle.add_argument("--agent", required=True, help="path to the agent's agent.toml")
    bundle.add_argument("--output", default="artifacts/submission.tar.gz")
    bundle.set_defaults(func=_cmd_bundle)

    web = commands.add_parser("web", help="review persistent league sessions in a local dashboard")
    web.add_argument("--database", default=DEFAULT_DATABASE)
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    web.set_defaults(func=_cmd_web)

    league = commands.add_parser("league", help="manage persistent paired training leagues")
    league_commands = league.add_subparsers(dest="league_command", required=True)

    league_init = league_commands.add_parser("init", help="create generation zero")
    league_init.add_argument("--config", required=True)
    league_init.add_argument("--session-id")
    league_init.add_argument("--database", default=DEFAULT_DATABASE)
    league_init.set_defaults(func=_cmd_league_init)

    for name, help_text in (
        ("run", "run or resume the current generation"),
        ("resume", "resume the current generation"),
    ):
        league_run = league_commands.add_parser(name, help=help_text)
        league_run.add_argument("--session-id", required=True)
        _add_paths(league_run)
        league_run.set_defaults(func=_cmd_league_run)

    league_trainers = league_commands.add_parser(
        "trainers",
        help="repoint an existing session's training_manifests (the only stored config field that is safe to rewrite)",
    )
    league_trainers.add_argument("--session-id", required=True)
    league_trainers.add_argument("--config", help="read training_manifests from this league config")
    league_trainers.add_argument("--manifest", action="append", default=[], help="path to an agent.toml; repeatable")
    league_trainers.add_argument("--database", default=DEFAULT_DATABASE)
    league_trainers.set_defaults(func=_cmd_league_trainers)

    league_status = league_commands.add_parser("status", help="show persisted generation progress")
    league_status.add_argument("--session-id", required=True)
    league_status.add_argument("--database", default=DEFAULT_DATABASE)
    league_status.set_defaults(func=_cmd_league_status)

    league_board = league_commands.add_parser("leaderboard", help="show Glicko-2 standings")
    league_board.add_argument("--session-id", required=True)
    league_board.add_argument("--limit", type=int, default=20)
    league_board.add_argument("--database", default=DEFAULT_DATABASE)
    league_board.set_defaults(func=_cmd_league_leaderboard)

    league_next = league_commands.add_parser("next-generation", help="breed the next immutable population")
    league_next.add_argument("--session-id", required=True)
    league_next.add_argument("--database", default=DEFAULT_DATABASE)
    league_next.set_defaults(func=_cmd_league_next)

    league_evaluate = league_commands.add_parser(
        "evaluate", help="run main.py episode hooks against stored DuckDB replays"
    )
    league_evaluate.add_argument("--session-id", required=True)
    league_evaluate.add_argument("--generation-id")
    league_evaluate.add_argument("--force", action="store_true")
    league_evaluate.add_argument("--database", default=DEFAULT_DATABASE)
    league_evaluate.set_defaults(func=_cmd_league_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
