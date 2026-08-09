"""Shared implementation for agent-owned model administration commands."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

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
    placeholders = ",".join("?" for _ in identifiers)
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
    database: str | Path,
    *,
    name: str = DEFAULT_AGENT_NAME,
) -> list[dict[str, object]]:
    """Return batch progress and the active checkpoint for one agent's models."""

    with ExperimentStore(database) as store:
        manifest = _register_manifest(store, manifest_path, name)
        identifiers = [effective_model_id(model.model_id, name) for model in manifest.models]
        placeholders = ",".join("?" for _ in identifiers)
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
    database: str | Path,
    *,
    artifacts: str | Path,
    force: bool = False,
    name: str = DEFAULT_AGENT_NAME,
) -> UpdateReport:
    """Try the sole model declared by an agent manifest."""

    with ExperimentStore(database) as store:
        manifest = _register_manifest(store, manifest_path, name)
        return update_model_in_store(
            store,
            _single_model_id(manifest, name),
            artifacts=artifacts,
            force=force,
        )


def export_active_model(
    manifest_path: str | Path,
    database: str | Path,
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
            WHERE tm.model_id = ?
            """,
            [model_id],
        ).fetchone()
    if row is None:
        raise ValueError(f"model has no active checkpoint: {model_id}")
    gate = json.loads(str(row[1] or "{}"))
    if gate.get("passed") is not True:
        raise ValueError(f"active checkpoint has not passed its release gate: {model_id}")
    source = Path(str(row[0]))
    if not source.is_file():
        raise FileNotFoundError(f"active inference artifact is missing: {source}")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    return destination


def add_model_commands(
    commands: argparse._SubParsersAction,
    *,
    manifest_path: str | Path,
    default_database: str,
    default_artifacts: str,
    allow_export: bool = False,
) -> None:
    """Attach a standard agent-owned ``model`` command group."""

    model = commands.add_parser("model", help="inspect or update this agent's model")
    actions = model.add_subparsers(dest="model_command", required=True)

    status = actions.add_parser("status", help="show batch progress and active checkpoint")
    status.add_argument("--name", default=DEFAULT_AGENT_NAME)
    status.add_argument("--database", default=default_database)
    status.set_defaults(
        func=_cmd_status,
        model_manifest=str(manifest_path),
    )

    update = actions.add_parser("update", help="retry this agent's eligible model update")
    update.add_argument("--name", default=DEFAULT_AGENT_NAME)
    update.add_argument("--force", action="store_true")
    update.add_argument("--database", default=default_database)
    update.add_argument("--artifacts", default=default_artifacts)
    update.set_defaults(
        func=_cmd_update,
        model_manifest=str(manifest_path),
    )

    if allow_export:
        export = actions.add_parser("export-active", help="copy the promoted inference artifact for release")
        export.add_argument("--name", default=DEFAULT_AGENT_NAME)
        export.add_argument("--output", required=True)
        export.add_argument("--database", default=default_database)
        export.add_argument("--confirm-release", action="store_true")
        export.set_defaults(
            func=_cmd_export_active,
            model_manifest=str(manifest_path),
        )


def _cmd_status(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            model_status(args.model_manifest, args.database, name=args.name),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_update(args: argparse.Namespace) -> int:
    report = update_agent_model(
        args.model_manifest,
        args.database,
        artifacts=args.artifacts,
        force=args.force,
        name=args.name,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 1 if report.status == "error" else 0


def _cmd_export_active(args: argparse.Namespace) -> int:
    output = export_active_model(
        args.model_manifest,
        args.database,
        args.output,
        confirm_release=args.confirm_release,
        name=args.name,
    )
    print(output)
    return 0
