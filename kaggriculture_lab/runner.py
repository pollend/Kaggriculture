"""Direct Python orchestration shared by constants-driven training runners."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kaggriculture_lab.constants import default_database
from kaggriculture_lab.database import DatabaseTarget
from kaggriculture_lab.league import (
    LeagueConfig,
    create_next_generation,
    create_session,
    run_generation,
    session_status,
    set_training_agents,
    set_training_manifests,
)
from kaggriculture_lab.storage import ExperimentStore
from sim.arena import play_series
from sim.metrics import Summary, summarize


def ensure_session(
    config: LeagueConfig,
    session_id: str,
    *,
    database: str | DatabaseTarget | None = None,
    workspace: str | Path = ".",
    artifacts: str | Path | None = None,
    runner: str = "league",
    experiment: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Create or synchronize a session without probing a CLI return code.

    Returns ``(created, status)``. A session row without a generation is an
    interrupted initialization, not an unknown session, and is reported
    explicitly so a runner never compounds the partial state.

    ``database`` defaults to the project-local target, so a runner that uses the
    one lab database never has to name it.
    """

    target = database if database is not None else default_database()
    config.validate()
    manifest = {
        "schema_version": 1,
        "runner": runner,
        "session_id": session_id,
        "league": asdict(config),
        "experiment": experiment or {},
    }
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest_digest = hashlib.sha256(manifest_json.encode()).hexdigest()
    manifest_path = Path(artifacts or "artifacts") / f"{session_id}.run.json"
    with ExperimentStore(target) as store:
        session = store.connection.execute(
            "SELECT league FROM training_sessions WHERE session_id = %s", [session_id]
        ).fetchone()
        generations = int(
            store.connection.execute(
                "SELECT count(*) FROM generations WHERE session_id = %s", [session_id]
            ).fetchone()[0]
        )
        stored_manifest = store.connection.execute(
            "SELECT manifest_sha256, manifest_json FROM session_manifests WHERE session_id = %s",
            [session_id],
        ).fetchone()
    if session is not None and generations == 0:
        raise RuntimeError(
            f"session {session_id!r} was only partially initialized; "
            "use a new session ID or repair the orphaned session row"
        )
    if stored_manifest is not None and str(stored_manifest[0]) != manifest_digest:
        stored_payload = json.loads(str(stored_manifest[1]))
        stored_league_config = stored_payload.get("league")
        legacy_league_only = (
            session is not None
            and isinstance(stored_league_config, dict)
            and "league" not in stored_league_config
        )
        if legacy_league_only:
            stored_league_config["league"] = str(session[0])
            upgraded_json = json.dumps(
                stored_payload, sort_keys=True, separators=(",", ":")
            )
            upgraded_digest = hashlib.sha256(upgraded_json.encode()).hexdigest()
        else:
            upgraded_json = ""
            upgraded_digest = ""
        if upgraded_digest != manifest_digest:
            raise ValueError(
                f"run manifest mismatch for {session_id!r}; resume with the original "
                "experiment settings or choose a new session ID"
            )
        with ExperimentStore(target) as store:
            store.connection.execute(
                """
                UPDATE session_manifests
                SET manifest_sha256 = %s, manifest_json = %s
                WHERE session_id = %s AND manifest_sha256 = %s
                """,
                [upgraded_digest, upgraded_json, session_id, stored_manifest[0]],
            )
    if session is None:
        create_session(config, target, workspace=workspace, session_id=session_id)
        created = True
    else:
        created = False
        if config.training_agents:
            set_training_agents(session_id, target, config, workspace=workspace)
        else:
            set_training_manifests(session_id, target, config.training_manifests)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with ExperimentStore(target) as store:
        store.connection.execute(
            """
            INSERT INTO session_manifests (
                session_id, manifest_path, manifest_sha256, manifest_json
            ) VALUES (%s, %s, %s, %s)
            ON CONFLICT (session_id) DO NOTHING
            """,
            [session_id, str(manifest_path), manifest_digest, manifest_json],
        )
    return created, session_status(session_id, target)


def run_generations(
    *,
    session_id: str,
    database: str | DatabaseTarget | None = None,
    artifacts: str | Path,
    generations: int,
    continuous: bool = False,
    workspace: str | Path = ".",
    on_generation: Callable[[int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run a finite block, or sequential generations until interrupted.

    The shared dashboard is a separate service backed by PostgreSQL. This
    runner only orchestrates training and never binds an HTTP or MCP endpoint.
    Continuous mode never queues generations: the next generation is created
    only after ``run_generation`` has closed the current one.

    ``database`` defaults to the project-local target, so a runner that uses the
    one lab database never has to name it.
    """

    if generations < 1:
        raise ValueError("generations must be positive")
    target = database if database is not None else default_database()
    reports: list[dict[str, Any]] = []
    indices = itertools.count(1) if continuous else range(1, generations + 1)
    for index in indices:
        current = session_status(session_id, target)
        if current["status"] == "COMPLETE":
            create_next_generation(session_id, target, workspace=workspace)
        progress_label = f"{index}" if continuous else f"{index}/{generations}"
        print(f"[{session_id}] generation {progress_label} started", flush=True)

        def progress(event: dict[str, Any]) -> None:
            if event.get("event") == "round_started":
                print(
                    f"[{session_id}] generation g{int(event['generation_index']):04d} "
                    f"training round {event['round_number']}/{event['round_count']} started",
                    flush=True,
                )

        report = run_generation(
            session_id,
            target,
            artifacts,
            update_models=True,
            on_progress=progress,
        )
        if not continuous:
            reports.append(report)
        if on_generation is not None:
            on_generation(index, report)
    if not continuous:
        print(
            f"[{session_id}] training session run completed "
            f"({generations} generation{'s' if generations != 1 else ''})",
            flush=True,
        )
    return reports


def run_round_robin(
    agent_specs: list[str], *, seeds: int, jobs: int, config: dict[str, Any] | None = None
) -> list[Summary]:
    """Evaluate every distinct pair directly, without invoking the arena CLI."""

    summaries: list[Summary] = []
    for left_index, left in enumerate(agent_specs):
        for right in agent_specs[left_index + 1 :]:
            results = play_series(left, right, range(seeds), config or {}, jobs)
            summaries.append(summarize(left, right, results))
    return summaries
