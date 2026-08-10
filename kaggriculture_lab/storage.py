"""Experiment storage and portable exports.

PostgreSQL is the sole backend.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from kaggriculture_lab.database import DatabaseTarget, connect_postgres
from kaggriculture_lab.identity import (
    DEFAULT_AGENT_NAME,
    AgentManifest,
    AgentRevision,
    TrainingAgentSpec,
    effective_model_id,
    training_instance_seed,
)
from sim.runner import MatchResult


class ExperimentStore:
    def __init__(
        self,
        database: str | DatabaseTarget,
    ):
        self.target = (
            database
            if isinstance(database, DatabaseTarget)
            else DatabaseTarget(database)
        )
        self.connection = connect_postgres(self.target)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ExperimentStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def register_family(
        self, manifest: AgentManifest, manifest_path: str | Path
    ) -> None:
        """Register a family and its default model instance."""

        self.register_training_agent(
            manifest,
            manifest_path,
            TrainingAgentSpec(type=manifest.family_id, name=DEFAULT_AGENT_NAME),
        )

    def register_training_agent(
        self,
        manifest: AgentManifest,
        manifest_path: str | Path,
        instance: TrainingAgentSpec,
    ) -> tuple[str, ...]:
        """Register one independently trained instance of a manifest's models."""

        instance.validate()
        if instance.type != manifest.family_id:
            raise ValueError(
                f"training agent type {instance.type!r} does not match manifest "
                f"family_id {manifest.family_id!r}"
            )
        self.connection.execute(
            """
            INSERT INTO agent_families VALUES (
                %s, %s, %s, %s, %s, %s, %s, current_timestamp
            )
            ON CONFLICT (family_id) DO UPDATE SET
                name = excluded.name,
                manifest_path = excluded.manifest_path,
                entry_point = excluded.entry_point,
                training_adapter = excluded.training_adapter,
                episode_evaluator = excluded.episode_evaluator,
                parameter_schema_version = excluded.parameter_schema_version,
                created_at = excluded.created_at
            """,
            [
                manifest.family_id,
                manifest.name,
                str(manifest_path),
                manifest.entry_point,
                manifest.training_adapter,
                manifest.episode_evaluator,
                manifest.parameter_schema_version,
            ],
        )
        identifiers: list[str] = []
        for model in manifest.models:
            identifier = effective_model_id(model.model_id, instance.name)
            identifiers.append(identifier)
            config = dict(model.config or {})
            config.update(instance.model_config or {})
            config["agent_type"] = instance.type
            config["name"] = instance.name
            if instance.seed is not None:
                config["seed"] = instance.seed
            elif instance.name != DEFAULT_AGENT_NAME:
                config["seed"] = training_instance_seed(instance.type, instance.name)
            self.connection.execute(
                """
                INSERT INTO training_models (
                    model_id, family_id, agent_type, agent_name, declared_model_id,
                    trainer_path, min_new_batches, config_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_id) DO UPDATE SET
                    family_id = excluded.family_id,
                    agent_type = excluded.agent_type,
                    agent_name = excluded.agent_name,
                    declared_model_id = excluded.declared_model_id,
                    trainer_path = excluded.trainer_path,
                    min_new_batches = excluded.min_new_batches,
                    config_json = excluded.config_json,
                    updated_at = now()
                """,
                [
                    identifier,
                    manifest.family_id,
                    instance.type,
                    instance.name,
                    model.model_id,
                    model.trainer,
                    model.min_new_batches,
                    json.dumps(config, sort_keys=True),
                ],
            )
        return tuple(identifiers)

    def ensure_agent_families(self, revisions: Iterable[AgentRevision]) -> None:
        """Guarantee a parent row for every family a revision cites.

        Built-in anchors (`identity.builtin_revision`) and ad-hoc evaluation
        opponents name families -- `builtin`, `evaluation_opponent` -- that no
        manifest ever registers, and `agent_revisions.family_id` references
        `agent_families`. Placeholders are written with `DO NOTHING`, so a real
        `register_family` still overwrites them whenever one arrives."""

        first: dict[str, AgentRevision] = {}
        for item in revisions:
            first.setdefault(item.family_id, item)
        if not first:
            return
        self.connection.executemany(
            """
            INSERT INTO agent_families
            VALUES (%s, %s, '', %s, NULL, %s, %s, current_timestamp)
            ON CONFLICT (family_id) DO NOTHING
            """,
            [
                [
                    family_id,
                    family_id,
                    item.entry_point,
                    item.episode_evaluator,
                    item.parameter_schema_version,
                ]
                for family_id, item in first.items()
            ],
        )

    def add_agent_revisions(self, revisions: Iterable[AgentRevision]) -> None:
        items = list(revisions)
        rows = [
            [
                item.revision_id,
                item.family_id,
                item.strategy_id,
                item.entry_point,
                item.agent_spec,
                item.code_fingerprint,
                item.parameter_schema_version,
                item.params_json,
                item.checkpoint_id,
                item.checkpoint_digest,
                item.source,
                item.episode_evaluator,
            ]
            for item in items
        ]
        if rows:
            self.ensure_agent_families(items)
            self.connection.executemany(
                """
                INSERT INTO agent_revisions (
                    revision_id, family_id, strategy_id, entry_point, agent_spec,
                    code_fingerprint, parameter_schema_version, params_json,
                    checkpoint_id, checkpoint_digest, source, episode_evaluator
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                rows,
            )

    def add_replay_payload(self, match_id: str, replay: Any) -> str:
        raw = json.dumps(replay, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()
        payload = gzip.compress(raw, compresslevel=6)
        self.connection.execute(
            """INSERT INTO episode_payloads VALUES (%s, %s, %s)
            ON CONFLICT (match_id) DO UPDATE SET
                replay_gzip = excluded.replay_gzip,
                payload_sha256 = excluded.payload_sha256""",
            [match_id, payload, digest],
        )
        return digest

    def begin_run(
        self,
        run_id: str,
        name: str,
        seed: int,
        config: dict[str, Any],
        versions: dict[str, str],
        git_commit: str | None,
    ) -> None:
        self.connection.execute(
            """INSERT INTO runs VALUES (%s, %s, current_timestamp, 'RUNNING', %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                name = excluded.name,
                created_at = excluded.created_at,
                status = excluded.status,
                seed = excluded.seed,
                config_json = excluded.config_json,
                versions_json = excluded.versions_json,
                git_commit = excluded.git_commit""",
            [run_id, name, seed, json.dumps(config), json.dumps(versions), git_commit],
        )

    def finish_run(self, run_id: str, status: str = "COMPLETE") -> None:
        self.connection.execute("UPDATE runs SET status = %s WHERE run_id = %s", [status, run_id])

    def add_strategies(
        self, strategies: Iterable[Any], source: str, run_id: str | None = None
    ) -> None:
        """Persist family parameter sets. Only `strategy_id` and `canonical_json()`
        are read, so the store never needs to know the parameter type."""
        rows = [
            [strategy.strategy_id, strategy.canonical_json(), source, run_id]
            for strategy in strategies
        ]
        if rows:
            self.connection.executemany(
                "INSERT INTO strategies VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING", rows
            )

    def completed_match_ids(self, run_id: str) -> set[str]:
        rows = self.connection.execute(
            "SELECT match_id FROM episodes WHERE run_id = %s", [run_id]
        ).fetchall()
        return {row[0] for row in rows}

    def add_match(
        self,
        match_id: str,
        run_id: str,
        stage: str,
        strategy_ids: tuple[str, str],
        specs: tuple[str, str],
        result: MatchResult,
        replay_path: str | None = None,
    ) -> None:
        self.connection.execute("BEGIN")
        try:
            self.connection.execute(
                "INSERT INTO episodes VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [
                    match_id,
                    run_id,
                    stage,
                    result.seed,
                    strategy_ids[0],
                    strategy_ids[1],
                    specs[0],
                    specs[1],
                    result.steps,
                    result.wallclock,
                    result.failed,
                    replay_path,
                ],
            )
            money = [seat.final_money for seat in result.seats]
            for seat_index, seat in enumerate(result.seats):
                opponent_money = money[1 - seat_index]
                difference = (
                    seat.final_money - opponent_money
                    if seat.final_money is not None and opponent_money is not None
                    else None
                )
                win = None if difference is None else (1.0 if difference > 0 else 0.0 if difference < 0 else 0.5)
                metrics = seat.metrics
                self.connection.execute(
                    """
                    INSERT INTO episode_players VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT DO NOTHING
                    """,
                    [
                        match_id,
                        run_id,
                        stage,
                        result.seed,
                        seat_index,
                        strategy_ids[seat_index],
                        strategy_ids[1 - seat_index],
                        seat.status,
                        seat.failed,
                        seat.reward,
                        seat.final_money,
                        opponent_money,
                        difference,
                        win,
                        metrics.get("total_unit_actions", 0),
                        metrics.get("productive_actions", 0),
                        metrics.get("movement_actions", 0),
                        metrics.get("idle_actions", 0),
                        metrics.get("productive_action_percentage", 0.0),
                        metrics.get("movement_percentage", 0.0),
                        metrics.get("idle_percentage", 0.0),
                        metrics.get("plants_lost_inferred", 0),
                        metrics.get("final_shed_items", 0),
                        metrics.get("final_carried_items", 0),
                        json.dumps(metrics, sort_keys=True),
                    ],
                )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def ranked_strategy_ids(self, run_id: str, stage: str, limit: int) -> list[str]:
        rows = self.connection.execute(
            """
            SELECT strategy_id, avg(money_difference) AS score
            FROM episode_players
            WHERE run_id = %s AND stage = %s AND strategy_id NOT LIKE 'builtin:%' AND NOT failed
            GROUP BY strategy_id
            ORDER BY score DESC, strategy_id
            LIMIT %s
            """,
            [run_id, stage, limit],
        ).fetchall()
        return [row[0] for row in rows]

    def load_strategies(self, strategy_ids: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
        """Return raw parameter dicts keyed by strategy id.

        Deliberately does not decode into a family type -- that would make the
        generic store depend on one agent's dataclass. Callers that own the
        family decode with their own `from_dict`."""
        if strategy_ids is None:
            rows = self.connection.execute("SELECT strategy_id, params_json FROM strategies").fetchall()
        else:
            ids = list(strategy_ids)
            if not ids:
                return {}
            placeholders = ",".join("%s" for _ in ids)
            rows = self.connection.execute(
                f"SELECT strategy_id, params_json FROM strategies WHERE strategy_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def save_ratings(self, run_id: str, ratings: dict[str, tuple[float, int]]) -> None:
        self.connection.execute("DELETE FROM ratings WHERE run_id = %s", [run_id])
        rows = [[run_id, key, value[0], value[1]] for key, value in ratings.items()]
        if rows:
            self.connection.executemany("INSERT INTO ratings VALUES (%s, %s, %s, %s)", rows)

    def save_model(
        self,
        model_id: str,
        run_id: str,
        kind: str,
        artifact_path: str,
        report: dict[str, Any],
    ) -> None:
        self.connection.execute(
            """INSERT INTO models VALUES (%s, %s, %s, %s, %s, current_timestamp)
            ON CONFLICT (model_id) DO UPDATE SET
                run_id = excluded.run_id,
                kind = excluded.kind,
                artifact_path = excluded.artifact_path,
                report_json = excluded.report_json,
                created_at = excluded.created_at""",
            [model_id, run_id, kind, artifact_path, json.dumps(report, sort_keys=True)],
        )

    def save_proposals(self, run_id: str, rows: list[dict[str, Any]]) -> None:
        self.connection.execute("DELETE FROM proposals WHERE run_id = %s", [run_id])
        self.connection.executemany(
            "INSERT INTO proposals VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [
                [
                    run_id,
                    row["strategy_id"],
                    row["rank"],
                    row["acquisition"],
                    row.get("predicted_quality"),
                    row.get("predicted_matchup"),
                    row.get("uncertainty"),
                ]
                for row in rows
            ],
        )

    def export(self, run_id: str, directory: str | Path, csv: bool = False) -> list[Path]:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        tables = ("runs", "strategies", "episodes", "episode_players", "ratings", "models", "proposals")
        written: list[Path] = []
        for table in tables:
            run_filter = "run_id = %s" if table != "strategies" else "created_run_id = %s"
            extension = "csv" if csv else "parquet"
            path = destination / f"{table}.{extension}"
            import pyarrow as pa
            import pyarrow.csv as pa_csv
            import pyarrow.parquet as pa_parquet

            cursor = self.connection.execute(
                f"SELECT * FROM {table} WHERE {run_filter}", [run_id]
            )
            names = [column.name for column in cursor.description]
            arrow_table = pa.Table.from_pylist(
                [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
            )
            if csv:
                pa_csv.write_csv(arrow_table, path)
            else:
                pa_parquet.write_table(arrow_table, path, compression="zstd")
            written.append(path)
        return written
