"""Persistent, resumable paired-match training league."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import random
import statistics
import tomllib
import traceback
import uuid
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kaggriculture_lab.episode_metrics import episode_metrics
from kaggriculture_lab.hooks import EpisodeHookContext, EpisodeReader, load_evaluator
from kaggriculture_lab.identity import (
    AgentManifest,
    AgentRevision,
    build_revision,
    builtin_revision,
)
from kaggriculture_lab.model_trainers import (
    GENERATION_ROUND_INDEX,
    run_round_hooks,
    session_training_models,
)
from kaggriculture_lab.model_updates import ModelTrainingReader, update_model_in_store
from kaggriculture_lab.ratings import GlickoRating, RatingGame, update_glicko2
from kaggriculture_lab.storage import ExperimentStore
from kaggriculture_lab.training import allocate_generation, load_training_adapter, propose
from sim.runner import MatchResult, run_match


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    name: str = "configurable-crop-league"
    manifest: str = "agents/configurable_crop_surrogate/agent.toml"
    anchor_manifests: tuple[str, ...] = ()
    #: Manifests whose models this session drives round hooks for. Empty means
    #: just `manifest`. Declaring it separately is what lets a league host
    #: another family as an anchor opponent without also running its trainer --
    #: and what lets a session drive a model, like the DQN, that no league breeds.
    training_manifests: tuple[str, ...] = ()
    population_size: int = 20
    rounds_per_generation: int = 20
    hall_of_fame_rounds: int = 4
    episode_steps: int = 720
    seed: int = 42
    jobs: int = 1
    tau: float = 0.5
    candidate_pool: int = 2_000

    @classmethod
    def from_toml(cls, path: str | Path) -> LeagueConfig:
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        values = dict(raw.get("league", raw))
        values["anchor_manifests"] = tuple(values.get("anchor_manifests", ()))
        values["training_manifests"] = tuple(values.get("training_manifests", ()))
        return cls(**values)

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.rounds_per_generation < 1:
            raise ValueError("rounds_per_generation must be positive")
        if not 0 <= self.hall_of_fame_rounds <= self.rounds_per_generation:
            raise ValueError("hall_of_fame_rounds must be between 0 and rounds_per_generation")
        if self.episode_steps < 1:
            raise ValueError("episode_steps must be positive")
        if self.jobs < 0:
            raise ValueError("jobs must be zero (automatic) or positive")


@dataclass(frozen=True, slots=True)
class LeagueEpisodeTask:
    pairing_id: str
    orientation: int
    seed: int
    left: AgentRevision
    right: AgentRevision
    episode_steps: int


def _league_worker(task: LeagueEpisodeTask):
    seats = (task.left, task.right) if task.orientation == 0 else (task.right, task.left)
    result = run_match(
        (seats[0].agent_spec, seats[1].agent_spec),
        task.seed,
        {"episodeSteps": task.episode_steps},
        keep_replay=True,
        # The league persists per-episode metrics for agents' evaluation hooks
        # to read back, so it opts into the kaggriculture analyzer explicitly.
        analyzer=episode_metrics,
    )
    return task, result


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "\0".join(str(part) for part in parts).encode()
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:20]}"


def _revision_from_row(row: tuple[Any, ...]) -> AgentRevision:
    return AgentRevision(
        revision_id=row[0],
        family_id=row[1],
        strategy_id=row[2],
        entry_point=row[3],
        agent_spec=row[4],
        code_fingerprint=row[5],
        parameter_schema_version=row[6],
        params_json=row[7],
        checkpoint_id=row[8],
        checkpoint_digest=row[9],
        source=row[10],
        episode_evaluator=row[11],
    )


def _get_revision(store: ExperimentStore, revision_id: str) -> AgentRevision:
    row = store.connection.execute(
        """
        SELECT revision_id, family_id, strategy_id, entry_point, agent_spec,
               code_fingerprint, parameter_schema_version, params_json,
               checkpoint_id, checkpoint_digest, source, episode_evaluator
        FROM agent_revisions WHERE revision_id = ?
        """,
        [revision_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown agent revision: {revision_id}")
    return _revision_from_row(row)


def create_session(
    config: LeagueConfig,
    database: str | Path,
    *,
    workspace: str | Path = ".",
    session_id: str | None = None,
) -> str:
    """Create generation zero and its immutable initial population."""

    config.validate()
    session_id = session_id or f"league_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    manifest_path = (Path(workspace) / config.manifest).resolve()
    manifest = AgentManifest.load(manifest_path)
    anchor_manifest_paths = [
        (Path(workspace) / path).resolve() for path in config.anchor_manifests
    ]
    anchor_manifests = [AgentManifest.load(path) for path in anchor_manifest_paths]
    adapter = load_training_adapter(manifest)
    if adapter is None:
        raise ValueError(f"agent family has no training_adapter: {manifest.family_id}")
    population_rng = random.Random(config.seed)
    strategies = [adapter.sample(population_rng) for _ in range(config.population_size)]
    revisions = [
        build_revision(
            manifest,
            strategy,
            workspace=workspace,
            episode_steps=config.episode_steps,
            source="random",
        )
        for strategy in strategies
    ]
    anchors = [
        builtin_revision("starter"),
        builtin_revision("seeded_random", f"kaggriculture_lab.opponents:make_seeded_random?seed={config.seed}"),
        builtin_revision("pass"),
    ]
    competitor_anchors = [
        build_revision(item, workspace=workspace, source="competitor")
        for item in anchor_manifests
    ]
    generation_id = f"{session_id}:g0000"
    with ExperimentStore(database) as store:
        if store.connection.execute(
            "SELECT 1 FROM training_sessions WHERE session_id = ?", [session_id]
        ).fetchone():
            raise ValueError(f"training session already exists: {session_id}")
        store.register_family(manifest, manifest_path)
        for anchor_manifest, anchor_path in zip(anchor_manifests, anchor_manifest_paths):
            store.register_family(anchor_manifest, anchor_path)
        store.add_agent_revisions([*revisions, *anchors, *competitor_anchors])
        store.add_strategies(strategies, "league-random", session_id)
        config_json = json.dumps(asdict(config), sort_keys=True)
        store.connection.execute(
            "INSERT INTO training_sessions VALUES (?, ?, 'ACTIVE', ?, ?, current_timestamp)",
            [session_id, config.name, config.seed, config_json],
        )
        store.begin_run(session_id, config.name, config.seed, asdict(config), {}, None)
        store.connection.execute(
            "INSERT INTO generations (generation_id, session_id, generation_index, status) VALUES (?, ?, 0, 'OPEN')",
            [generation_id, session_id],
        )
        store.connection.executemany(
            "INSERT INTO generation_members VALUES (?, ?, 'population', 1500, 350, 0.06)",
            [[generation_id, revision.revision_id] for revision in revisions],
        )
        if competitor_anchors:
            store.connection.executemany(
                "INSERT INTO generation_members VALUES (?, ?, 'anchor', 1500, 350, 0.06)",
                [[generation_id, revision.revision_id] for revision in competitor_anchors],
            )
    return session_id


def _load_session(store: ExperimentStore, session_id: str) -> LeagueConfig:
    row = store.connection.execute(
        "SELECT config_json FROM training_sessions WHERE session_id = ?", [session_id]
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown training session: {session_id}")
    values = json.loads(row[0])
    values["anchor_manifests"] = tuple(values.get("anchor_manifests", ()))
    values["training_manifests"] = tuple(values.get("training_manifests", ()))
    return LeagueConfig(**values)


def set_training_manifests(
    session_id: str, database: str | Path, manifests: Sequence[str]
) -> tuple[str, ...]:
    """Repoint an existing session at a different set of trainers.

    Only this one field is writable. Everything else in a stored session config
    determines how generations were scheduled, and rewriting it would silently
    invalidate the history already computed under it.
    """

    families = _training_family_ids(
        LeagueConfig(training_manifests=tuple(manifests))
    ) if manifests else ()
    with ExperimentStore(database) as store:
        row = store.connection.execute(
            "SELECT config_json FROM training_sessions WHERE session_id = ?", [session_id]
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown training session: {session_id}")
        values = json.loads(row[0])
        values["training_manifests"] = list(manifests)
        store.connection.execute(
            "UPDATE training_sessions SET config_json = ? WHERE session_id = ?",
            [json.dumps(values, sort_keys=True), session_id],
        )
    return families


def _current_generation(store: ExperimentStore, session_id: str) -> tuple[str, int, str]:
    row = store.connection.execute(
        """
        SELECT generation_id, generation_index, status FROM generations
        WHERE session_id = ? ORDER BY generation_index DESC LIMIT 1
        """,
        [session_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"session has no generations: {session_id}")
    return row[0], row[1], row[2]


def _member_ids(store: ExperimentStore, generation_id: str) -> list[str]:
    return [
        row[0]
        for row in store.connection.execute(
            """
            SELECT revision_id FROM generation_members
            WHERE generation_id = ? AND role != 'anchor' ORDER BY revision_id
            """,
            [generation_id],
        ).fetchall()
    ]


def _hall_of_fame(
    store: ExperimentStore, session_id: str, generation_index: int, limit: int
) -> list[str]:
    anchors = [
        row[0]
        for row in store.connection.execute(
            """
            SELECT gm.revision_id FROM generation_members gm
            JOIN generations g USING (generation_id)
            WHERE g.session_id = ? AND gm.role = 'anchor'
            ORDER BY gm.revision_id
            """,
            [session_id],
        ).fetchall()
    ]
    if generation_index == 0:
        return ["builtin:starter", "builtin:seeded_random", "builtin:pass", *anchors]
    rows = store.connection.execute(
        """
        SELECT rs.revision_id, max(rs.rating - 2 * rs.deviation) AS conservative,
               max(gs.mean_margin) AS margin
        FROM rating_snapshots rs
        JOIN generations g ON g.generation_id = rs.generation_id
        -- Explicit ON, not USING: the join above leaves both rs.generation_id and
        -- g.generation_id in scope, so `USING (generation_id, ...)` is ambiguous
        -- and DuckDB rejects it. Only reachable from generation 1 onward, since
        -- generation 0 returns before this query.
        LEFT JOIN generation_stats gs
          ON gs.generation_id = rs.generation_id AND gs.revision_id = rs.revision_id
        WHERE g.session_id = ? AND g.generation_index < ? AND NOT rs.is_frozen
        GROUP BY rs.revision_id
        ORDER BY conservative DESC, margin DESC NULLS LAST, rs.revision_id
        LIMIT ?
        """,
        [session_id, generation_index, limit],
    ).fetchall()
    historical = [row[0] for row in rows]
    return [*anchors, *historical] or ["builtin:starter"]


def schedule_generation(store: ExperimentStore, session_id: str) -> int:
    """Create deterministic random pairings, including both seat orientations."""

    config = _load_session(store, session_id)
    generation_id, generation_index, status = _current_generation(store, session_id)
    existing = store.connection.execute(
        "SELECT count(*) FROM league_pairings WHERE generation_id = ?", [generation_id]
    ).fetchone()[0]
    if existing:
        return int(existing)
    if status != "OPEN":
        raise ValueError(f"generation is not open: {generation_id} ({status})")

    members = _member_ids(store, generation_id)
    rng = random.Random(config.seed + generation_index * 1_000_003)
    rows: list[list[Any]] = []
    intra_rounds = config.rounds_per_generation - config.hall_of_fame_rounds
    rotation: list[str | None] = members[:]
    rng.shuffle(rotation)
    if len(rotation) % 2:
        rotation.append(None)
    cycle_length = len(rotation) - 1
    for round_index in range(intra_rounds):
        if round_index and round_index % cycle_length == 0:
            rotation = members[:]
            rng.shuffle(rotation)
            if len(rotation) % 2:
                rotation.append(None)
        pairs = [
            (rotation[index], rotation[-1 - index])
            for index in range(len(rotation) // 2)
        ]
        for left, right in pairs:
            if left is None or right is None:
                continue
            left, right = sorted((left, right))
            seed = rng.randrange(0, 2**31)
            pairing_id = _stable_id("pair", generation_id, round_index, left, right, seed)
            rows.append([pairing_id, generation_id, round_index, "population", seed, left, right, "PENDING"])
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]

    hall = _hall_of_fame(store, session_id, generation_index, max(3, math.ceil(len(members) * 0.2)))
    for offset in range(config.hall_of_fame_rounds):
        round_index = intra_rounds + offset
        shuffled = members[:]
        rng.shuffle(shuffled)
        for index, member in enumerate(shuffled):
            opponent = hall[(index + offset) % len(hall)]
            if member == opponent:
                opponent = hall[(index + offset + 1) % len(hall)]
            if member == opponent:
                continue
            seed = rng.randrange(0, 2**31)
            pairing_id = _stable_id("pair", generation_id, round_index, member, opponent, seed)
            rows.append([pairing_id, generation_id, round_index, "hall_of_fame", seed, member, opponent, "PENDING"])

    if rows:
        store.connection.executemany(
            "INSERT INTO league_pairings VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
    store.connection.execute(
        "UPDATE generations SET status = 'SCHEDULED' WHERE generation_id = ?", [generation_id]
    )
    return len(rows)


def _run_hook(
    store: ExperimentStore,
    context: EpisodeHookContext,
    evaluator_path: str | None,
    *,
    force: bool = False,
) -> None:
    if not evaluator_path:
        return
    try:
        evaluator, fingerprint = load_evaluator(evaluator_path)
    except Exception:
        fingerprint = hashlib.sha256(f"unloadable:{evaluator_path}".encode()).hexdigest()
        evaluation_id = _stable_id(
            "eval", context.match_id, context.revision_id, context.seat, fingerprint
        )
        store.connection.execute(
            "DELETE FROM episode_evaluations WHERE evaluation_id = ?", [evaluation_id]
        )
        store.connection.execute(
            """
            INSERT INTO episode_evaluations (
                evaluation_id, match_id, revision_id, seat, evaluator_path,
                evaluator_fingerprint, status, result_json, error_text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'ERROR', NULL, ?, current_timestamp)
            """,
            [
                evaluation_id,
                context.match_id,
                context.revision_id,
                context.seat,
                evaluator_path,
                fingerprint,
                traceback.format_exc(),
            ],
        )
        return
    evaluation_id = _stable_id(
        "eval", context.match_id, context.revision_id, context.seat, fingerprint
    )
    existing = store.connection.execute(
        "SELECT status FROM episode_evaluations WHERE evaluation_id = ?", [evaluation_id]
    ).fetchone()
    if existing and existing[0] == "COMPLETE" and not force:
        return
    try:
        result = evaluator(
            context, EpisodeReader(store.connection, context.match_id, context.seat)
        )
        store.connection.execute(
            "DELETE FROM episode_evaluations WHERE evaluation_id = ?", [evaluation_id]
        )
        store.connection.execute(
            """
            INSERT INTO episode_evaluations (
                evaluation_id, match_id, revision_id, seat, evaluator_path,
                evaluator_fingerprint, status, result_json, error_text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'COMPLETE', ?, NULL, current_timestamp)
            """,
            [
                evaluation_id,
                context.match_id,
                context.revision_id,
                context.seat,
                evaluator_path,
                fingerprint,
                json.dumps(result, sort_keys=True),
            ],
        )
    except Exception:
        store.connection.execute(
            "DELETE FROM episode_evaluations WHERE evaluation_id = ?", [evaluation_id]
        )
        store.connection.execute(
            """
            INSERT INTO episode_evaluations (
                evaluation_id, match_id, revision_id, seat, evaluator_path,
                evaluator_fingerprint, status, result_json, error_text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'ERROR', NULL, ?, current_timestamp)
            """,
            [
                evaluation_id,
                context.match_id,
                context.revision_id,
                context.seat,
                evaluator_path,
                fingerprint,
                traceback.format_exc(),
            ],
        )


def _persist_orientation(
    store: ExperimentStore,
    session_id: str,
    generation_id: str,
    pairing_id: str,
    orientation: int,
    seed: int,
    left: AgentRevision,
    right: AgentRevision,
    episode_steps: int,
) -> None:
    match_id = _stable_id("match", pairing_id, orientation)
    existing = store.connection.execute(
        "SELECT match_id FROM pairing_episodes WHERE pairing_id = ? AND orientation = ?",
        [pairing_id, orientation],
    ).fetchone()
    seats = (left, right) if orientation == 0 else (right, left)
    if existing:
        match_id = existing[0]
        for seat, revision in enumerate(seats):
            _run_hook(
                store,
                EpisodeHookContext(
                    session_id=session_id,
                    generation_id=generation_id,
                    pairing_id=pairing_id,
                    match_id=match_id,
                    revision_id=revision.revision_id,
                    opponent_revision_id=seats[1 - seat].revision_id,
                    seat=seat,
                ),
                revision.episode_evaluator,
            )
        return
    task = LeagueEpisodeTask(
        pairing_id, orientation, seed, left, right, episode_steps
    )
    _, result = _league_worker(task)
    _store_orientation_result(
        store, session_id, generation_id, task, result
    )


def _store_orientation_result(
    store: ExperimentStore,
    session_id: str,
    generation_id: str,
    task: LeagueEpisodeTask,
    result: MatchResult,
) -> None:
    pairing_id = task.pairing_id
    orientation = task.orientation
    match_id = _stable_id("match", pairing_id, orientation)
    seats = (task.left, task.right) if orientation == 0 else (task.right, task.left)
    store.add_match(
        match_id,
        session_id,
        f"league:{generation_id}",
        (seats[0].revision_id, seats[1].revision_id),
        (seats[0].agent_spec, seats[1].agent_spec),
        result,
    )
    if result.replay is not None:
        store.add_replay_payload(match_id, result.replay)
    store.connection.execute(
        "INSERT OR IGNORE INTO pairing_episodes VALUES (?, ?, ?)",
        [pairing_id, orientation, match_id],
    )
    for seat, revision in enumerate(seats):
        _run_hook(
            store,
            EpisodeHookContext(
                session_id=session_id,
                generation_id=generation_id,
                pairing_id=pairing_id,
                match_id=match_id,
                revision_id=revision.revision_id,
                opponent_revision_id=seats[1 - seat].revision_id,
                seat=seat,
            ),
            revision.episode_evaluator,
        )


def _finish_pairing(store: ExperimentStore, pairing_id: str, revision_a: str) -> None:
    rows = store.connection.execute(
        """
        SELECT pe.orientation, ep.strategy_id, ep.money_difference, ep.failed
        FROM pairing_episodes pe
        JOIN episode_players ep ON ep.match_id = pe.match_id
        WHERE pe.pairing_id = ? AND ep.strategy_id = ?
        ORDER BY pe.orientation
        """,
        [pairing_id, revision_a],
    ).fetchall()
    failed = len(rows) != 2 or any(row[3] or row[2] is None for row in rows)
    margins = [float(row[2]) for row in rows if row[2] is not None]
    paired_margin = statistics.fmean(margins) if len(margins) == 2 and not failed else None
    outcome = None if paired_margin is None else (1.0 if paired_margin > 0 else 0.0 if paired_margin < 0 else 0.5)
    store.connection.execute(
        """
        INSERT OR REPLACE INTO pairing_results VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            pairing_id,
            paired_margin,
            outcome,
            failed,
            len(rows),
            json.dumps({"orientation_margins": margins}),
        ],
    )
    store.connection.execute(
        "UPDATE league_pairings SET status = ? WHERE pairing_id = ?",
        ["FAILED" if failed else "COMPLETE", pairing_id],
    )


def _training_family_ids(config: LeagueConfig) -> tuple[str, ...]:
    """Families whose trainers this session drives, from its declared manifests.

    Reading them from the config rather than from the generation's membership is
    deliberate: a session may train a family it does not breed (an off-policy
    learner riding the league's clock), and it must *not* train a family that is
    only present as an anchor opponent.
    """

    paths = config.training_manifests or (config.manifest,)
    return tuple(
        dict.fromkeys(AgentManifest.load(path).family_id for path in paths)
    )


def _run_round(
    store: ExperimentStore,
    session_id: str,
    generation_id: str,
    config: LeagueConfig,
    round_index: int,
    pool: ProcessPoolExecutor | None,
) -> None:
    """Play every outstanding pairing in one round and settle its results."""

    pairings = store.connection.execute(
        """
        SELECT pairing_id, seed, revision_a, revision_b FROM league_pairings
        WHERE generation_id = ? AND round_index = ?
          AND status NOT IN ('COMPLETE', 'FAILED')
        ORDER BY pairing_id
        """,
        [generation_id, round_index],
    ).fetchall()
    if not pairings:
        return
    tasks: list[LeagueEpisodeTask] = []
    for pairing_id, seed, revision_a, revision_b in pairings:
        left = _get_revision(store, revision_a)
        right = _get_revision(store, revision_b)
        for orientation in (0, 1):
            existing = store.connection.execute(
                "SELECT 1 FROM pairing_episodes WHERE pairing_id = ? AND orientation = ?",
                [pairing_id, orientation],
            ).fetchone()
            if existing:
                _persist_orientation(
                    store,
                    session_id,
                    generation_id,
                    pairing_id,
                    orientation,
                    seed,
                    left,
                    right,
                    config.episode_steps,
                )
                continue
            tasks.append(
                LeagueEpisodeTask(
                    pairing_id, orientation, seed, left, right, config.episode_steps
                )
            )
    outcomes = (
        map(_league_worker, tasks)
        if pool is None
        else pool.map(_league_worker, tasks, chunksize=1)
    )
    for task, result in outcomes:
        _store_orientation_result(store, session_id, generation_id, task, result)
    for pairing_id, _, revision_a, _ in pairings:
        _finish_pairing(store, pairing_id, revision_a)


def run_generation(
    session_id: str,
    database: str | Path,
    artifacts: str | Path = "artifacts",
    *,
    update_models: bool = False,
) -> dict[str, Any]:
    """Resume all missing paired episodes and close ratings as one batch.

    Rounds are played and settled one at a time so each closed round can be
    offered to the declared models' round hooks. The process pool is created
    once for the whole generation -- the per-round barrier is inherent to the
    hook's contract, but paying spawn cost per round is not.

    `update_models` decides whether a closed round also attempts a model update.
    It defaults off so that calling this as a library never pulls an agent's
    training dependencies in; the CLI turns it on.
    """

    with ExperimentStore(database) as store:
        config = _load_session(store, session_id)
        family_ids = _training_family_ids(config)
        generation_id, generation_index, status = _current_generation(store, session_id)
        if status == "COMPLETE":
            _close_generation_training(
                store,
                session_id,
                generation_id,
                generation_index,
                family_ids,
                artifacts,
                update_models,
            )
            return generation_status(store, session_id)
        schedule_generation(store, session_id)
        store.connection.execute(
            "UPDATE generations SET status = 'RUNNING' WHERE generation_id = ?",
            [generation_id],
        )
        rounds = [
            int(row[0])
            for row in store.connection.execute(
                """
                SELECT DISTINCT round_index FROM league_pairings
                WHERE generation_id = ? ORDER BY round_index
                """,
                [generation_id],
            ).fetchall()
        ]
        jobs = config.jobs if config.jobs > 0 else max(1, (os.cpu_count() or 2) - 1)
        pool = (
            None
            if jobs == 1
            else ProcessPoolExecutor(
                max_workers=jobs, mp_context=multiprocessing.get_context("spawn")
            )
        )
        try:
            for round_index in rounds:
                _run_round(
                    store, session_id, generation_id, config, round_index, pool
                )
                run_round_hooks(
                    store,
                    session_id=session_id,
                    generation_id=generation_id,
                    generation_index=generation_index,
                    scope="round",
                    round_index=round_index,
                    family_ids=family_ids,
                    artifacts=artifacts,
                )
                if update_models:
                    _update_session_models(store, family_ids, artifacts)
        finally:
            if pool is not None:
                pool.shutdown()
        _close_generation(store, session_id, generation_id, config)
        _close_generation_training(
            store,
            session_id,
            generation_id,
            generation_index,
            family_ids,
            artifacts,
            update_models,
        )
        store.connection.execute(
            """
            UPDATE generations SET status = 'COMPLETE', completed_at = current_timestamp
            WHERE generation_id = ?
            """,
            [generation_id],
        )
        return generation_status(store, session_id)


def _close_generation_training(
    store: ExperimentStore,
    session_id: str,
    generation_id: str,
    generation_index: int,
    family_ids: Sequence[str],
    artifacts: str | Path,
    update_models: bool,
) -> None:
    """Offer the closed generation to each declared model, then let it train.

    Ordering matters: `_close_generation` must already have written this
    generation's `rating_snapshots`, because a surrogate's round hook counts them
    to size its batch.
    """

    run_round_hooks(
        store,
        session_id=session_id,
        generation_id=generation_id,
        generation_index=generation_index,
        scope="generation",
        round_index=GENERATION_ROUND_INDEX,
        family_ids=family_ids,
        artifacts=artifacts,
    )
    if update_models:
        _update_session_models(store, family_ids, artifacts)


def _update_session_models(
    store: ExperimentStore, family_ids: Sequence[str], artifacts: str | Path
) -> None:
    for model in session_training_models(store, family_ids):
        update_model_in_store(store, model[0], artifacts=artifacts)


def reevaluate_hooks(
    session_id: str,
    database: str | Path,
    *,
    generation_id: str | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Run each revision's current evaluator against already-persisted episodes."""

    with ExperimentStore(database) as store:
        if generation_id is None:
            generation_id, _, _ = _current_generation(store, session_id)
        rows = store.connection.execute(
            """
            SELECT lp.pairing_id, pe.orientation, pe.match_id,
                   lp.revision_a, lp.revision_b
            FROM league_pairings lp
            JOIN pairing_episodes pe USING (pairing_id)
            WHERE lp.generation_id = ?
            ORDER BY lp.pairing_id, pe.orientation
            """,
            [generation_id],
        ).fetchall()
        attempted = 0
        for pairing_id, orientation, match_id, revision_a, revision_b in rows:
            revisions = (
                (_get_revision(store, revision_a), _get_revision(store, revision_b))
                if orientation == 0
                else (_get_revision(store, revision_b), _get_revision(store, revision_a))
            )
            for seat, revision in enumerate(revisions):
                if not revision.episode_evaluator:
                    continue
                attempted += 1
                _run_hook(
                    store,
                    EpisodeHookContext(
                        session_id=session_id,
                        generation_id=generation_id,
                        pairing_id=pairing_id,
                        match_id=match_id,
                        revision_id=revision.revision_id,
                        opponent_revision_id=revisions[1 - seat].revision_id,
                        seat=seat,
                    ),
                    revision.episode_evaluator,
                    force=force,
                )
        status_rows = store.connection.execute(
            """
            SELECT ee.status, count(*) FROM episode_evaluations ee
            JOIN pairing_episodes pe USING (match_id)
            JOIN league_pairings lp USING (pairing_id)
            WHERE lp.generation_id = ? GROUP BY ee.status
            """,
            [generation_id],
        ).fetchall()
        return {"attempted": attempted, **{str(key).lower(): int(value) for key, value in status_rows}}


def _previous_rating(
    store: ExperimentStore, session_id: str, revision_id: str
) -> GlickoRating:
    row = store.connection.execute(
        """
        SELECT rs.rating, rs.deviation, rs.volatility, rs.games
        FROM rating_snapshots rs
        JOIN generations g ON g.generation_id = rs.generation_id
        WHERE g.session_id = ? AND rs.revision_id = ?
        ORDER BY g.generation_index DESC LIMIT 1
        """,
        [session_id, revision_id],
    ).fetchone()
    return GlickoRating(*row) if row else GlickoRating()


def _close_generation(
    store: ExperimentStore,
    session_id: str,
    generation_id: str,
    config: LeagueConfig,
) -> None:
    if store.connection.execute(
        "SELECT 1 FROM rating_snapshots WHERE generation_id = ? LIMIT 1", [generation_id]
    ).fetchone():
        return
    member_rows = store.connection.execute(
        """
        SELECT revision_id, initial_rating, initial_deviation, initial_volatility
        FROM generation_members WHERE generation_id = ? AND role != 'anchor'
        """,
        [generation_id],
    ).fetchall()
    initial = {
        row[0]: GlickoRating(float(row[1]), float(row[2]), float(row[3]), _previous_rating(store, session_id, row[0]).games)
        for row in member_rows
    }
    result_rows = store.connection.execute(
        """
        SELECT p.revision_a, p.revision_b, r.outcome_a, r.paired_margin
        FROM league_pairings p JOIN pairing_results r USING (pairing_id)
        WHERE p.generation_id = ? AND NOT r.failed
        """,
        [generation_id],
    ).fetchall()
    games: dict[str, list[RatingGame]] = {revision_id: [] for revision_id in initial}
    margins: dict[str, list[float]] = {revision_id: [] for revision_id in initial}
    opponents: dict[str, set[str]] = {revision_id: set() for revision_id in initial}
    records = {revision_id: [0, 0, 0] for revision_id in initial}
    for left, right, outcome, margin in result_rows:
        left_rating = initial.get(left) or _previous_rating(store, session_id, left)
        right_rating = initial.get(right) or _previous_rating(store, session_id, right)
        if left in initial:
            games[left].append(RatingGame(right_rating, float(outcome)))
            margins[left].append(float(margin))
            opponents[left].add(right)
            records[left][0 if outcome > 0.5 else 1 if outcome == 0.5 else 2] += 1
        if right in initial:
            games[right].append(RatingGame(left_rating, 1.0 - float(outcome)))
            margins[right].append(-float(margin))
            opponents[right].add(left)
            right_outcome = 1.0 - float(outcome)
            records[right][0 if right_outcome > 0.5 else 1 if right_outcome == 0.5 else 2] += 1

    snapshot_rows: list[list[Any]] = []
    stat_rows: list[list[Any]] = []
    for revision_id, rating in initial.items():
        updated = update_glicko2(rating, games[revision_id], tau=config.tau)
        snapshot_rows.append(
            [generation_id, revision_id, updated.rating, updated.deviation, updated.volatility, updated.games, False]
        )
        values = margins[revision_id]
        mean = statistics.fmean(values) if values else None
        median = statistics.median(values) if values else None
        if len(values) > 1:
            half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
            low, high = mean - half_width, mean + half_width
        else:
            low = high = mean
        wins, draws, losses = records[revision_id]
        stat_rows.append(
            [generation_id, revision_id, len(values), wins, draws, losses, mean, median, low, high, len(opponents[revision_id])]
        )
    store.connection.executemany(
        "INSERT INTO rating_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)", snapshot_rows
    )
    store.connection.executemany(
        "INSERT INTO generation_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", stat_rows
    )


def create_next_generation(
    session_id: str,
    database: str | Path,
    *,
    workspace: str | Path = ".",
) -> str:
    """Build a 20/50/20/10 elite/evolution/surrogate/random population."""

    with ExperimentStore(database) as store:
        config = _load_session(store, session_id)
        current_id, current_index, status = _current_generation(store, session_id)
        if status != "COMPLETE":
            raise ValueError("current generation must be complete before breeding")
        manifest_row = store.connection.execute(
            """
            SELECT af.manifest_path FROM generation_members gm
            JOIN agent_revisions ar USING (revision_id)
            JOIN agent_families af USING (family_id)
            WHERE gm.generation_id = ? AND gm.role != 'anchor' AND af.family_id != 'builtin'
            LIMIT 1
            """,
            [current_id],
        ).fetchone()
        if manifest_row is None:
            raise ValueError("session has no trainable agent family")
        manifest = AgentManifest.load(manifest_row[0])
        adapter = load_training_adapter(manifest)
        if adapter is None:
            raise ValueError("agent family does not define a training adapter")
        ranked_rows = store.connection.execute(
            """
            SELECT ar.revision_id, ar.family_id, ar.strategy_id, ar.entry_point,
                   ar.agent_spec, ar.code_fingerprint, ar.parameter_schema_version,
                   ar.params_json, ar.checkpoint_id, ar.checkpoint_digest, ar.source,
                   ar.episode_evaluator, rs.rating, rs.deviation, rs.volatility, rs.games
            FROM rating_snapshots rs JOIN agent_revisions ar USING (revision_id)
            LEFT JOIN generation_stats gs USING (generation_id, revision_id)
            WHERE rs.generation_id = ?
            ORDER BY rs.rating - 2 * rs.deviation DESC,
                     gs.mean_margin DESC NULLS LAST, ar.revision_id
            """,
            [current_id],
        ).fetchall()
        parents = [_revision_from_row(row[:12]) for row in ranked_rows]
        ratings = {
            row[0]: GlickoRating(float(row[12]), float(row[13]), float(row[14]), int(row[15]))
            for row in ranked_rows
        }
        allocation = allocate_generation(config.population_size)
        rng = random.Random(config.seed + (current_index + 1) * 1_000_003)
        selected: list[tuple[AgentRevision, str, str | None, str | None]] = []
        for parent in parents[: allocation["elite"]]:
            selected.append((parent, "elite", parent.revision_id, None))

        def add_strategy(strategy: Any, method: str, left: str | None, right: str | None) -> None:
            revision = build_revision(
                manifest,
                strategy,
                workspace=workspace,
                episode_steps=config.episode_steps,
                source=method,
            )
            if revision.revision_id not in {item[0].revision_id for item in selected}:
                selected.append((revision, method, left, right))

        elite_pool = parents[: max(2, math.ceil(len(parents) * 0.4))]
        while sum(item[1] == "evolution" for item in selected) < allocation["evolution"]:
            left, right = rng.sample(elite_pool, 2)
            add_strategy(
                adapter.crossover(adapter.decode(left), adapter.decode(right), rng),
                "evolution",
                left.revision_id,
                right.revision_id,
            )

        # Model-guided candidates are the family's business: which model, how it
        # scores, and whether one exists at all. An adapter with no `propose`
        # returns nothing and this falls through to mutation below.
        proposed = propose(
            adapter,
            count=allocation["surrogate"],
            parents=elite_pool,
            rng=rng,
            reader=ModelTrainingReader(store),
        )
        surrogate_iter = iter(proposed)
        while sum(item[1] == "surrogate" for item in selected) < allocation["surrogate"]:
            try:
                strategy = next(surrogate_iter)
            except StopIteration:
                strategy = adapter.mutate(adapter.decode(rng.choice(elite_pool)), rng)
            add_strategy(strategy, "surrogate", None, None)
        while sum(item[1] == "random" for item in selected) < allocation["random"]:
            add_strategy(adapter.sample(rng), "random", None, None)

        next_index = current_index + 1
        generation_id = f"{session_id}:g{next_index:04d}"
        new_revisions = [item[0] for item in selected if item[1] != "elite"]
        store.add_agent_revisions(new_revisions)
        store.add_strategies(
            [adapter.decode(item) for item in new_revisions], "league-next-generation", session_id
        )
        store.connection.execute(
            "INSERT INTO generations (generation_id, session_id, generation_index, status) VALUES (?, ?, ?, 'OPEN')",
            [generation_id, session_id, next_index],
        )
        member_rows = []
        lineage_rows = []
        for revision, method, parent_a, parent_b in selected:
            inherited = ratings.get(revision.revision_id, GlickoRating())
            member_rows.append(
                [generation_id, revision.revision_id, method, inherited.rating, inherited.deviation, inherited.volatility]
            )
            lineage_rows.append([generation_id, revision.revision_id, parent_a, parent_b, method])
        store.connection.executemany(
            "INSERT INTO generation_members VALUES (?, ?, ?, ?, ?, ?)", member_rows
        )
        store.connection.executemany(
            "INSERT INTO lineage VALUES (?, ?, ?, ?, ?)", lineage_rows
        )
        return generation_id


def generation_status(store: ExperimentStore, session_id: str) -> dict[str, Any]:
    generation_id, generation_index, status = _current_generation(store, session_id)
    counts = dict(
        store.connection.execute(
            """
            SELECT status, count(*) FROM league_pairings
            WHERE generation_id = ? GROUP BY status
            """,
            [generation_id],
        ).fetchall()
    )
    hooks = dict(
        store.connection.execute(
            """
            SELECT ee.status, count(*) FROM episode_evaluations ee
            JOIN pairing_episodes pe USING (match_id)
            JOIN league_pairings lp USING (pairing_id)
            WHERE lp.generation_id = ? GROUP BY ee.status
            """,
            [generation_id],
        ).fetchall()
    )
    reevaluation_requests = int(
        store.connection.execute(
            """
            SELECT count(*) FROM episode_evaluations ee
            JOIN pairing_episodes pe USING (match_id)
            JOIN league_pairings lp USING (pairing_id)
            WHERE lp.generation_id = ? AND ee.status = 'COMPLETE'
              AND coalesce(
                    CAST(json_extract(ee.result_json, '$.should_reevaluate') AS BOOLEAN),
                    false
                  )
            """,
            [generation_id],
        ).fetchone()[0]
    )
    return {
        "session_id": session_id,
        "generation_id": generation_id,
        "generation_index": generation_index,
        "status": status,
        "pairings": counts,
        "hook_evaluations": hooks,
        "reevaluation_requests": reevaluation_requests,
    }


def session_status(session_id: str, database: str | Path) -> dict[str, Any]:
    with ExperimentStore(database) as store:
        return generation_status(store, session_id)


def leaderboard(session_id: str, database: str | Path, limit: int = 20) -> list[dict[str, Any]]:
    with ExperimentStore(database) as store:
        generation_id, _, _ = _current_generation(store, session_id)
        rows = store.connection.execute(
            """
            SELECT rs.revision_id, ar.strategy_id, gm.role, rs.rating, rs.deviation,
                   rs.games, gs.mean_margin, gs.wins, gs.draws, gs.losses
            FROM rating_snapshots rs
            JOIN agent_revisions ar USING (revision_id)
            JOIN generation_members gm USING (generation_id, revision_id)
            LEFT JOIN generation_stats gs USING (generation_id, revision_id)
            WHERE rs.generation_id = ?
            ORDER BY rs.rating - 2 * rs.deviation DESC,
                     gs.mean_margin DESC NULLS LAST, rs.revision_id
            LIMIT ?
            """,
            [generation_id, limit],
        ).fetchall()
    keys = ["revision_id", "strategy_id", "role", "rating", "deviation", "games", "mean_margin", "wins", "draws", "losses"]
    return [dict(zip(keys, row)) for row in rows]
