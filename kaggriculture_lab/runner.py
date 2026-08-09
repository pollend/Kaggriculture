"""Direct Python orchestration shared by constants-driven training runners."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kaggriculture_lab.dashboard import make_server
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
    database: str | Path,
    session_id: str,
    *,
    workspace: str | Path = ".",
    artifacts: str | Path | None = None,
    runner: str = "league",
    experiment: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Create or synchronize a session without probing a CLI return code.

    Returns ``(created, status)``. A session row without a generation is an
    interrupted initialization, not an unknown session, and is reported
    explicitly so a runner never compounds the partial state.
    """

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
    manifest_path = Path(artifacts or Path(database).parent) / f"{session_id}.run.json"
    with ExperimentStore(database) as store:
        session = store.connection.execute(
            "SELECT 1 FROM training_sessions WHERE session_id = ?", [session_id]
        ).fetchone()
        generations = int(
            store.connection.execute(
                "SELECT count(*) FROM generations WHERE session_id = ?", [session_id]
            ).fetchone()[0]
        )
        stored_manifest = store.connection.execute(
            "SELECT manifest_sha256, manifest_json FROM session_manifests WHERE session_id = ?",
            [session_id],
        ).fetchone()
    if session is not None and generations == 0:
        raise RuntimeError(
            f"session {session_id!r} was only partially initialized; "
            "use a new session ID or repair the orphaned session row"
        )
    if stored_manifest is not None and str(stored_manifest[0]) != manifest_digest:
        raise ValueError(
            f"run manifest mismatch for {session_id!r}; resume with the original "
            "experiment settings or choose a new session ID"
        )
    if session is None:
        create_session(config, database, workspace=workspace, session_id=session_id)
        created = True
    else:
        created = False
        if config.training_agents:
            set_training_agents(session_id, database, config, workspace=workspace)
        else:
            set_training_manifests(session_id, database, config.training_manifests)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with ExperimentStore(database) as store:
        store.connection.execute(
            """
            INSERT INTO session_manifests (
                session_id, manifest_path, manifest_sha256, manifest_json
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT (session_id) DO NOTHING
            """,
            [session_id, str(manifest_path), manifest_digest, manifest_json],
        )
    return created, session_status(session_id, database)


def run_generations(
    *,
    session_id: str,
    database: str | Path,
    artifacts: str | Path,
    generations: int,
    workspace: str | Path = ".",
    on_generation: Callable[[int, dict[str, Any]], None] | None = None,
    dashboard: bool = True,
    dashboard_host: str = "127.0.0.1",
    dashboard_port: int = 8000,
) -> list[dict[str, Any]]:
    """Run exactly ``generations`` updates and leave the following one open."""

    if generations < 1:
        raise ValueError("generations must be positive")
    server = None
    server_thread = None
    if dashboard:
        try:
            server = make_server(
                database,
                dashboard_host,
                dashboard_port,
                read_only=False,
            )
        except OSError:
            if dashboard_port == 0:
                raise
            server = make_server(database, dashboard_host, 0, read_only=False)
        actual_host, actual_port = server.server_address[:2]
        server_thread = threading.Thread(
            target=server.serve_forever,
            name=f"dashboard-{session_id}",
            daemon=True,
        )
        server_thread.start()
        print(
            f"[{session_id}] dashboard: http://{actual_host}:{actual_port}/",
            flush=True,
        )

    reports: list[dict[str, Any]] = []
    try:
        for index in range(1, generations + 1):
            current = session_status(session_id, database)
            if current["status"] == "COMPLETE":
                create_next_generation(session_id, database, workspace=workspace)
            print(
                f"[{session_id}] generation {index}/{generations} started",
                flush=True,
            )

            def progress(event: dict[str, Any]) -> None:
                if event.get("event") == "round_started":
                    print(
                        f"[{session_id}] generation g{int(event['generation_index']):04d} "
                        f"training round {event['round_number']}/{event['round_count']} started",
                        flush=True,
                    )

            report = run_generation(
                session_id,
                database,
                artifacts,
                update_models=True,
                on_progress=progress,
            )
            reports.append(report)
            if on_generation is not None:
                on_generation(index, report)
        print(
            f"[{session_id}] training session run completed "
            f"({generations} generation{'s' if generations != 1 else ''})",
            flush=True,
        )
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=5)
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
