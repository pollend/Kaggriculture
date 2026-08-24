"""Shared implementation for agent-owned model administration commands."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from kaggriculture_lab.constants import default_database
from kaggriculture_lab.database import DatabaseTarget, database_target
from kaggriculture_lab.identity import (
    DEFAULT_AGENT_NAME,
    AgentManifest,
    TrainingAgentSpec,
    effective_model_id,
)
from kaggriculture_lab.model_updates import UpdateReport, update_model_in_store
from kaggriculture_lab.storage import ExperimentStore


def _register_manifest(
    store: ExperimentStore,
    manifest_path: str | Path,
    name: str = DEFAULT_AGENT_NAME,
) -> AgentManifest:
    path = Path(manifest_path)
    manifest = AgentManifest.load(path)
    if not manifest.models:
        raise ValueError(f"agent declares no trainable models: {path}")
    instance = TrainingAgentSpec(type=manifest.family_id, name=name)
    instance.validate()
    identifiers = [effective_model_id(model.model_id, name) for model in manifest.models]
    placeholders = ",".join("%s" for _ in identifiers)
    existing = int(
        store.connection.execute(
            f"SELECT count(*) FROM training_models WHERE model_id IN ({placeholders})",
            identifiers,
        ).fetchone()[0]
    )
    if existing != len(identifiers):
        store.register_training_agent(manifest, path, instance)
    return manifest


def _single_model_id(
    manifest: AgentManifest, name: str = DEFAULT_AGENT_NAME
) -> str:
    if len(manifest.models) != 1:
        raise ValueError(
            f"agent-owned model commands require exactly one model; "
            f"{manifest.family_id} declares {len(manifest.models)}"
        )
    return effective_model_id(manifest.models[0].model_id, name)


def model_status(
    manifest_path: str | Path,
    database: str | DatabaseTarget | None = None,
    *,
    name: str = DEFAULT_AGENT_NAME,
) -> list[dict[str, object]]:
    """Return batch progress and the active checkpoint for one agent's models."""

    with ExperimentStore(database or default_database()) as store:
        manifest = _register_manifest(store, manifest_path, name)
        identifiers = [effective_model_id(model.model_id, name) for model in manifest.models]
        placeholders = ",".join("%s" for _ in identifiers)
        rows = store.connection.execute(
            f"""
            SELECT tm.model_id, tm.min_new_batches, tm.active_checkpoint_id,
                   tm.last_attempted_batch_seq, coalesce(max(tb.batch_seq), 0)
            FROM training_models tm
            LEFT JOIN training_batches tb USING (model_id)
            WHERE tm.model_id IN ({placeholders})
            GROUP BY tm.model_id, tm.min_new_batches, tm.active_checkpoint_id,
                     tm.last_attempted_batch_seq
            ORDER BY tm.model_id
            """,
            identifiers,
        ).fetchall()
    return [
        {
            "model_id": str(row[0]),
            "min_new_batches": int(row[1]),
            "active_checkpoint_id": row[2],
            "last_attempted_batch_seq": int(row[3]),
            "latest_batch_seq": int(row[4]),
            "new_batches": int(row[4]) - int(row[3]),
        }
        for row in rows
    ]


def update_agent_model(
    manifest_path: str | Path,
    database: str | DatabaseTarget | None = None,
    *,
    artifacts: str | Path,
    force: bool = False,
    name: str = DEFAULT_AGENT_NAME,
) -> UpdateReport:
    """Try the sole model declared by an agent manifest."""

    with ExperimentStore(database or default_database()) as store:
        manifest = _register_manifest(store, manifest_path, name)
        return update_model_in_store(
            store,
            _single_model_id(manifest, name),
            artifacts=artifacts,
            force=force,
        )


def export_active_model(
    manifest_path: str | Path,
    database: str | DatabaseTarget,
    output: str | Path,
    *,
    confirm_release: bool,
    name: str = DEFAULT_AGENT_NAME,
) -> Path:
    """Copy an agent's promoted inference artifact into a reviewed release path."""

    if not confirm_release:
        raise ValueError("export-active is a release action; pass --confirm-release after reviewing the gate")
    with ExperimentStore(database) as store:
        manifest = _register_manifest(store, manifest_path, name)
        model_id = _single_model_id(manifest, name)
        row = store.connection.execute(
            """
            SELECT mtc.inference_artifact_path, mtc.gate_json
            FROM training_models tm
            JOIN model_training_checkpoints mtc
              ON mtc.checkpoint_id = tm.active_checkpoint_id
            WHERE tm.model_id = %s
            """,
            [model_id],
        ).fetchone()
    if row is None:
        raise ValueError(f"model has no active checkpoint: {model_id}")
    gate = json.loads(str(row[1] or "{}"))
    if gate.get("passed") is not True:
        raise ValueError(f"active checkpoint has not passed its release gate: {model_id}")
    source = Path(str(row[0]))
    _copy_inference_artifact(source, output)
    return Path(output)


_BEST_CHECKPOINT_SCORE_SQL = {
    "benchmark-money": (
        "COALESCE("
        "CAST(mtc.gate_json -> 'benchmark_score' ->> 'candidate_money' "
        "AS DOUBLE PRECISION), "
        "CAST(mtc.gate_json -> 'benchmark_ratchet' ->> 'candidate_money' "
        "AS DOUBLE PRECISION))"
    ),
    "recent-mean-final-money": (
        "CAST(mtc.metrics_json ->> 'recent_mean_final_money' AS DOUBLE PRECISION)"
    ),
}


def export_best_model(
    manifest_path: str | Path,
    database: str | DatabaseTarget,
    output: str | Path,
    *,
    confirm_release: bool,
    name: str = DEFAULT_AGENT_NAME,
    metric: str = "benchmark-money",
) -> dict[str, object]:
    """Copy the highest-scoring completed checkpoint into a release path."""

    if not confirm_release:
        raise ValueError("export-best is a release action; pass --confirm-release after reviewing the run")
    score_sql = _BEST_CHECKPOINT_SCORE_SQL.get(metric)
    if score_sql is None:
        raise ValueError(
            f"metric must be one of {sorted(_BEST_CHECKPOINT_SCORE_SQL)}: {metric}"
        )
    with ExperimentStore(database) as store:
        manifest = _register_manifest(store, manifest_path, name)
        model_id = _single_model_id(manifest, name)
        row = store.connection.execute(
            f"""
            SELECT checkpoint_id, inference_artifact_path, score
            FROM (
                SELECT mtc.checkpoint_id, mtc.inference_artifact_path,
                       mu.cutoff_batch_seq, mtc.created_at, {score_sql} AS score
                FROM model_training_checkpoints mtc
                JOIN model_updates mu ON mu.update_id = mtc.update_id
                WHERE mtc.model_id = %s
                  AND mu.status IN ('PROMOTED', 'REJECTED')
            ) ranked
            WHERE score IS NOT NULL
            ORDER BY score DESC, cutoff_batch_seq DESC, created_at DESC
            LIMIT 1
            """,
            [model_id],
        ).fetchone()
    if row is None:
        raise ValueError(f"model has no completed checkpoints with {metric}: {model_id}")
    source = Path(str(row[1]))
    _copy_inference_artifact(source, output)
    return {
        "checkpoint_id": str(row[0]),
        "metric": metric,
        "output": str(Path(output)),
        "score": float(row[2]),
    }


def _copy_inference_artifact(source: Path, output: str | Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"inference artifact is missing: {source}")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)


def add_model_commands(
    commands: argparse._SubParsersAction,
    *,
    manifest_path: str | Path,
    default_artifacts: str,
    allow_export: bool = False,
) -> None:
    """Attach a standard agent-owned ``model`` command group."""

    model = commands.add_parser("model", help="inspect or update this agent's model")
    actions = model.add_subparsers(dest="model_command", required=True)

    status = actions.add_parser("status", help="show batch progress and active checkpoint")
    status.add_argument("--name", default=DEFAULT_AGENT_NAME)
    _add_database_options(status)
    status.set_defaults(
        func=_cmd_status,
        model_manifest=str(manifest_path),
    )

    update = actions.add_parser("update", help="retry this agent's eligible model update")
    update.add_argument("--name", default=DEFAULT_AGENT_NAME)
    update.add_argument("--force", action="store_true")
    _add_database_options(update)
    update.add_argument("--artifacts", default=default_artifacts)
    update.set_defaults(
        func=_cmd_update,
        model_manifest=str(manifest_path),
    )

    if allow_export:
        export = actions.add_parser("export-active", help="copy the promoted inference artifact for release")
        export.add_argument("--name", default=DEFAULT_AGENT_NAME)
        export.add_argument("--output", required=True)
        _add_database_options(export)
        export.add_argument("--confirm-release", action="store_true")
        export.set_defaults(
            func=_cmd_export_active,
            model_manifest=str(manifest_path),
        )

        export_best = actions.add_parser(
            "export-best",
            help="copy the highest-scoring completed inference artifact for release",
        )
        export_best.add_argument("--name", default=DEFAULT_AGENT_NAME)
        export_best.add_argument("--output", required=True)
        export_best.add_argument(
            "--metric",
            choices=sorted(_BEST_CHECKPOINT_SCORE_SQL),
            default="benchmark-money",
        )
        _add_database_options(export_best)
        export_best.add_argument("--confirm-release", action="store_true")
        export_best.set_defaults(
            func=_cmd_export_best,
            model_manifest=str(manifest_path),
        )


def _add_database_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database-url")


def _target(args: argparse.Namespace) -> DatabaseTarget:
    return database_target(url=args.database_url)


def _cmd_status(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            model_status(args.model_manifest, _target(args), name=args.name),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_update(args: argparse.Namespace) -> int:
    report = update_agent_model(
        args.model_manifest,
        _target(args),
        artifacts=args.artifacts,
        force=args.force,
        name=args.name,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 1 if report.status == "error" else 0


def _cmd_export_active(args: argparse.Namespace) -> int:
    output = export_active_model(
        args.model_manifest,
        _target(args),
        args.output,
        confirm_release=args.confirm_release,
        name=args.name,
    )
    print(output)
    return 0


def _cmd_export_best(args: argparse.Namespace) -> int:
    result = export_best_model(
        args.model_manifest,
        _target(args),
        args.output,
        confirm_release=args.confirm_release,
        name=args.name,
        metric=args.metric,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
