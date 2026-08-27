"""Persistent collector and incremental updater for the unscaffolded market.

Structurally a fork of `agents/dqn_market_scorer/training/updater.py`, and the
parts that are not about the action space are imported from it rather than
copied -- lineage resume, opponent mixing, the promotion rule, the degenerate
result guards. None of that was ever action-specific, and two divergent copies
of a promotion gate is how one of them quietly stops matching the other.

What is genuinely different here is only the action space: the network has an
order head rather than a modifier head, so the replay assembly and the
Double-DQN target index into `ACTIONS`. The buffer can also be seeded with
demonstrations read out of stored league replays, because uniform exploration
measurably cannot find a solvent trajectory in this action space on its own.

Both the training resume and the rollout policy follow `lineage_checkpoint`.
**Release is gated; the behaviour distribution must not be.** An earlier version
gated rollouts on a viability check of their own, so collection stayed on the cold
start until something passed -- and nothing did: three hundred batches collected
by an untrained near-silent policy, and every candidate trained on them rejected.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import torch
from torch import nn

from agents.dqn_market_free_v3c.diagnostics import (
    BehaviorCounts,
    OrderDiagnostics,
    behavior_from_replay,
)
from agents.dqn_market_free_v3c.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free_v3c.order_space import (
    ACTION_COUNT,
    MASKED_SCORE,
    NOOP_INDEX,
)
from agents.dqn_market_free_v3c.policy import OrderQNetwork
from agents.dqn_market_free_v3c.training import network as net_module
from agents.dqn_market_free_v3c.training import rare
from agents.dqn_market_free_v3c.training import replay as replay_module
from agents.dqn_market_free_v3c.training.clone import load_demonstrations
from agents.dqn_market_free_v3c.training.collect import (
    HIDDEN_STRIDE,
    LEARNER_ENTRY_POINT,
    SELF_SPEC,
    EpisodeRecord,
    _single_threaded_workers,
    actor_sigma_scales,
    collect_batch,
    load_episode_batch,
    save_episode_batch,
)
from agents.dqn_market_free_v3c.training.ladder import load_ladder_demonstrations
from agents.dqn_market_free_v3c.training.members import (
    MemberEntry,
    admit_foreign,
    find_member,
    foreign_candidates,
    member_count,
    member_for_cutoff,
    member_for_sequence,
    member_paths,
    new_lineage_id,
    rank_and_prune,
    read_index,
    roster_candidates,
    write_index,
)
from agents.dqn_market_free_v3c.training.members import (
    member_batches as _member_batches,
)
from agents.dqn_market_free_v3c.training.replay import (
    DEFAULT_CAPACITY,
    DEFAULT_GENERATION_WEIGHT_FLOOR,
    DEFAULT_GENERATION_WEIGHT_POWER,
    DEFAULT_N_STEP,
    DEFAULT_PRIORITY_ALPHA,
    DEFAULT_PRIORITY_BETA,
    DEFAULT_RESERVOIR_FRACTION,
    ReplayBuffer,
)
from agents.dqn_market_free_v3c.training.reward import relabel
from agents.dqn_market_scorer.training.device import move_optimizer, resolve_device
from agents.dqn_market_scorer.training.progress import _opponent_policy, _policy_identity

# Action-agnostic machinery, shared on purpose. See the module docstring.
from agents.dqn_market_scorer.training.updater import (
    LineageCheckpoint,
    _benchmark_kind,
    _file_digest,
    _reject_degenerate,
    compose_opponents,
    configured_benchmarks,
    league_opponents,
    rejections_since_promotion,
    rotate_opponents,
)
from agents.dqn_market_scorer.training.updater import (
    lineage_checkpoint as _newest_lineage_checkpoint,
)
from kaggriculture_lab.model_trainers import EvaluationPolicy
from kaggriculture_lab.model_updates import (
    ModelEvaluation,
    ModelTrainingReader,
    ModelUpdateContext,
    ModelUpdateResult,
)

#: Warnings that must reach the run log without failing an update -- an
#: all-zero GRU is the only one so far. See `_reject_inert_network`.
LOGGER = logging.getLogger(__name__)

# `SELF_SPEC` is imported above rather than defined here. It moved to
# `training.collect` so the reward's mirror gate could read it without importing
# this module back, and it stays reachable as `updater.SELF_SPEC` for everything
# that already referred to it there -- the same re-export
# `_single_threaded_workers` got when it moved to `sim.workers`.
LEARNER_MANIFEST = "agents/dqn_market_free_v3c/match_agent.toml"
SOURCE_KIND = "free_market_rollout"
_PROMOTION_RULES = {"activity_gate"}
_PROMOTION_SCORE_SIGNALS = {"benchmark", "rollout"}
# Which checkpoint collection and training continue from under the activity gate.
# "promoted": the slot's best promoted checkpoint (the manifest control).
# "latest": the newest checkpoint, promoted or not -- the shared scorer rule.
_LINEAGE_SOURCES = {"promoted", "latest"}
#: Which measurement the learner slot's plateau detector reads. See `_slot_score`.
_SCORE_SIGNALS = {"rollout", "benchmark"}

# Active-run-safe anti-drag weights. These are deliberately code constants rather
# than manifest config keys: changing the training-agent config would invalidate
# the running session manifest, while this weighting is derived at load time and
# does not alter stored rollout artifacts.
TERMINAL_QUALITY_MIN_MONEY = 3_000.0
TERMINAL_QUALITY_FULL_MONEY = 20_000.0
TERMINAL_QUALITY_MIN_WEIGHT = 0.10

# What the promotion gate actually reads. Everything else that used to live here
# was consumed by `_arena_viability`, which this family's simplified gate deleted;
# the keys outlived the function and sat here defining floors nothing applied.
# Removed rather than restored, because the simplification was deliberate --
# `_order_viability` documents it as "survival plus activity, with benchmark money
# carrying the quality signal" -- but a config naming a `benchmark_money_ratchet`
# no code consults is worse than one admitting the gate is two checks wide.
#
# Gone, all unread: `viability_min_mean_money_ratio`, `viability_min_median_money_ratio`,
# `viability_cold_start_reference_money`, `viability_grace_after_rejections`,
# `viability_grace_cold_start_reference_money`, `benchmark_release_min_money`,
# `benchmark_cold_start_min_money`, `benchmark_grace_min_money`,
# `benchmark_money_ratchet`, `benchmark_reference_window`. The window the gate does
# read is `promotion_reference_window`, a different key, set in the manifest.
VIABILITY_DEFAULTS = {
    # A policy that never trades keeps its money perfectly, so a money-only gate
    # waves the NOOP attractor straight through. This is the only structural bar
    # against it. Applied on every update including a cold start -- the call site
    # passes `require_activity=True` unconditionally. An earlier comment here
    # claimed the floor waited for a released incumbent; it never did, and
    # `market-free-members-v2` rejected 97 of 105 updates on it with no incumbent
    # anywhere, which is the floor working rather than misfiring.
    "viability_min_orders_per_decision": 0.02,
    "generation_weight_power": DEFAULT_GENERATION_WEIGHT_POWER,
    "generation_weight_floor": DEFAULT_GENERATION_WEIGHT_FLOOR,
}

#: Stable semantic labels for the roles a policy plays here, mirroring the
#: scorer's. The learner is a raw order policy rather than a residual layer, so
#: it gets its own kind: a query that groups by kind must not silently merge the
#: scaffolded and unscaffolded learners into one series.
POLICY_KINDS: dict[str, str] = {
    "market_learner": "raw_order_q",
    "training_opponent": "training_opponent",
}


def _json_column(value: Any) -> dict[str, Any] | None:
    """A JSON column as a mapping, whatever the driver handed back.

    psycopg decodes `jsonb` for us but not always -- a `text` column, a driver
    without the adapter registered, or a value that round-tripped through a dump all
    arrive as `str`/`bytes`. Every reader below wants the same thing, so they share
    one decode rather than eight copies of the same three lines.

    `None` for anything that is not a mapping, including a decode failure: an
    unreadable payload is *not recorded*, which is the reading every caller needs.
    """

    if isinstance(value, (str, bytes, bytearray)):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def _released_gate(reader, active) -> dict[str, Any] | None:
    """The `gate_json` the active checkpoint was promoted on, or None.

    A pure read of a measurement already paid for: `_arena_evaluate` wrote it when
    the incumbent was itself a candidate, so everything downstream costs one row and
    no arena time. One helper because the three readers that want it -- the ratchet's
    money, the cull's trading rate, the archive's fingerprint -- were each fetching
    *the same row for the same checkpoint*, three round-trips per update.
    """

    if active is None:
        return None
    row = reader.fetchone(
        "SELECT gate_json FROM model_training_checkpoints WHERE checkpoint_id = %s",
        [active.checkpoint_id],
    )
    return None if row is None else _json_column(row[0])


def _benchmark_entry(gate: dict[str, Any] | None, opponent: str) -> dict[str, Any] | None:
    """One gate payload's result for `opponent`, or None if it holds none.

    Matched on the opponent spec rather than index 0, so reconfiguring `benchmarks`
    drops the term instead of silently comparing against a different opponent's
    number -- the one way any of these could be wrong without looking wrong.
    """

    if not isinstance(gate, dict):
        return None
    for result in gate.get("challenger_benchmarks") or []:
        if isinstance(result, dict) and str(result.get("opponent", "")) == opponent:
            return result
    return None


def _benchmark_money(gate: dict[str, Any] | None, opponent: str) -> float | None:
    """`a_money_mean` from one gate payload's benchmark entry, if it is usable."""

    result = _benchmark_entry(gate, opponent)
    if result is None or result.get("errors") or result.get("a_money_mean") is None:
        return None
    return float(result["a_money_mean"])




def _promotion_score_signal(config: dict[str, Any]) -> str:
    signal = str(config.get("promotion_score_signal", "benchmark"))
    if signal not in _PROMOTION_SCORE_SIGNALS:
        raise ValueError(
            f"promotion_score_signal must be one of {sorted(_PROMOTION_SCORE_SIGNALS)}: {signal}"
        )
    return signal


def _checkpoint_score_sql(score_signal: str) -> str:
    if score_signal == "benchmark":
        return """
        COALESCE(
            CAST(gate_json -> 'benchmark_score' ->> 'candidate_money'
                 AS DOUBLE PRECISION),
            CAST(gate_json -> 'benchmark_ratchet' ->> 'candidate_money'
                 AS DOUBLE PRECISION)
        )
        """
    if score_signal == "rollout":
        return """
        COALESCE(
            CAST(metrics_json ->> 'recent_mean_final_money' AS DOUBLE PRECISION),
            CAST(metrics_json ->> 'own_mean_final_money' AS DOUBLE PRECISION)
        )
        """
    raise ValueError(
        f"promotion_score_signal must be one of {sorted(_PROMOTION_SCORE_SIGNALS)}: {score_signal}"
    )


def _best_promoted_checkpoint_score(
    reader, model_id: str, score_signal: str, window: int = 0
) -> tuple[float | None, str | None]:
    """Best promoted checkpoint under the configured activation score.

    `window > 0` reads the best of the last `window` promoted checkpoints rather
    than the all-time best. An all-time high-water mark latches on the luckiest
    arena draw a slot ever took -- the warm-start's 6,112 on `terminalselfladder_01`
    was never approached again by that slot's own children -- and a 100-seed arena
    is a ~1,000-coin measurement. The window lets the bar drift back down with the
    lineage instead of pinning it at one draw for the session.
    """

    score_sql = _checkpoint_score_sql(score_signal)
    if int(window) > 0:
        row = reader.fetchone(
            f"""
            SELECT score, checkpoint_id FROM (
                SELECT {score_sql} AS score,
                       checkpoint_id,
                       created_at
                FROM model_training_checkpoints
                WHERE model_id = %s
                  AND promoted
                  AND {score_sql} IS NOT NULL
                ORDER BY created_at DESC, checkpoint_id DESC
                LIMIT %s
            ) recent
            ORDER BY score DESC, created_at DESC, checkpoint_id DESC
            LIMIT 1
            """,
            [model_id, int(window)],
        )
    else:
        row = reader.fetchone(
            f"""
            SELECT {score_sql} AS score,
                   checkpoint_id
            FROM model_training_checkpoints
            WHERE model_id = %s
              AND promoted
              AND {score_sql} IS NOT NULL
            ORDER BY score DESC, created_at DESC, checkpoint_id DESC
            LIMIT 1
            """,
            [model_id],
        )
    if row is None:
        return None, None
    return float(row[0]), str(row[1])


def _promotion_tolerance(
    config: dict[str, Any],
    score_signal: str,
    benchmark_result: dict[str, Any] | None,
    best_money: float | None = None,
) -> float:
    """Coins a candidate may land under the reference score and still promote.

    `promotion_tolerance_money` plus, under `promotion_tolerance_ci`, the arena's
    own confidence half-width -- then **capped at a fraction of the bar it is
    relaxing**, and that cap is the whole correctness argument.

    `margin_ci` is on the *paired margin*, so its width is the candidate's money
    noise plus the opponent's. Against this benchmark the opponent banks ~140,000
    and the candidate three or four figures, so the half-width is 3,000-5,000
    coins essentially independent of what the candidate did. Uncapped it swamps
    the bar: `market-free-rewardtogo-v2` promoted checkpoints scoring **0.0** in
    six of ten slots at cutoff 140-160, against standing bars of 1,500-13,400,
    because `0.0 >= 2864.01 - 2999.71`. The gate stopped gating, and the released
    duds then entered the self-play mirror pool through
    `self_play_latest_backfill`. Sizing a tolerance on candidate money from a
    statistic dominated by the opponent's money was the error.

    The cap makes it scale-correct: at `promotion_tolerance_fraction = 0.10` a bar
    of 13,407 forgives 1,341 coins and a bar of 2,864 forgives 286. Recovering
    from a bad bar is strict, which is what it should be. `best_money` of None or
    zero means there is no bar to relax and `_activation_reasons` compares nothing.

    The CI half applies to the benchmark signal only: under "rollout" the
    candidate score is collection money from a different distribution, and an
    arena CI says nothing about it.
    """

    tolerance = max(0.0, float(config.get("promotion_tolerance_money", 0.0)))
    if (
        score_signal == "benchmark"
        and bool(config.get("promotion_tolerance_ci", False))
        and benchmark_result is not None
    ):
        ci = benchmark_result.get("margin_ci")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            try:
                tolerance += abs(float(ci[1]) - float(ci[0])) / 2.0
            except (TypeError, ValueError):
                pass
    fraction = max(0.0, float(config.get("promotion_tolerance_fraction", 0.10)))
    if best_money is not None:
        try:
            bar = float(best_money)
        except (TypeError, ValueError):
            bar = 0.0
        if bar > 0.0:
            tolerance = min(tolerance, fraction * bar)
    return tolerance


def _activation_reasons(
    *,
    activity_passed: bool,
    candidate_money: float | None,
    best_money: float | None,
    score_signal: str = "benchmark",
    tolerance: float = 0.0,
) -> list[str]:
    """Why a candidate should not replace the active checkpoint.

    `tolerance` is how far under `best_money` a candidate may land and still
    promote, in coins. Zero is the strict ratchet. The arena's own confidence
    interval is the honest size for it: a candidate inside the noise of the best
    is not a regression, it is the same policy measured twice.
    """

    reasons: list[str] = []
    if not activity_passed:
        reasons.append("activity_gate_failed")
    if candidate_money is None:
        reasons.append(f"missing_{score_signal}_score")
    elif best_money is not None and float(candidate_money) < best_money - max(0.0, float(tolerance)):
        reasons.append(f"below_best_promoted_{score_signal}_money")
    return reasons


def lineage_checkpoint(context, reader) -> LineageCheckpoint | None:
    """Checkpoint that collection and training should continue from.

    The shared scorer lineage follows the newest checkpoint, promoted or not. That
    is correct when the release gate can reject and hold an incumbent, but it is
    the wrong rule for this family's simplified activity gate: rejected idle
    checkpoints are exactly the policies we do not want to carry forward. In that
    mode, use the best promoted benchmark score for the slot and leave release
    selection to the same metric.

    Other promotion rules keep the shared behavior, including the reset-to-release
    hatch after consecutive rejections.
    """

    if str(getattr(context, "config", {}).get("promotion_rule", "")) != "activity_gate":
        return _newest_lineage_checkpoint(context, reader)

    # `lineage_source = "latest"` hands the activity gate back the shared rule,
    # and it exists because of what "promoted" cost when nothing promotes.
    # `terminal-selfplay-ladder-v1` warm-started ten slots, released two more
    # checkpoints by under 200 coins, and then rejected every one of the next 24
    # updates -- so every update trained 960 steps from the same warm-start
    # weights on a fresh buffer and discarded the result. Three generations, ten
    # slots, zero net parameter movement. Promotion should decide what is
    # *released* (the mirror pool, the export); it must not decide what is
    # *trained*, or a gate that says no freezes the learner it is grading.
    # `resume_reset_after_rejections` is still the snapback for a lineage that
    # drifts.
    lineage_source = str(context.config.get("lineage_source", "promoted"))
    if lineage_source not in _LINEAGE_SOURCES:
        raise ValueError(
            f"lineage_source must be one of {sorted(_LINEAGE_SOURCES)}: {lineage_source}"
        )
    if lineage_source == "latest":
        return _newest_lineage_checkpoint(context, reader)

    score_signal = _promotion_score_signal(context.config)
    score_sql = _checkpoint_score_sql(score_signal)
    rows = reader.fetchall(
        f"""
        SELECT checkpoint_id, trainer_artifact_path, inference_artifact_path,
               {score_sql} AS score
        FROM model_training_checkpoints
        WHERE model_id = %s
          AND promoted
          AND {score_sql} IS NOT NULL
        ORDER BY score DESC, created_at DESC, checkpoint_id DESC
        LIMIT 20
        """,
        [context.model_id],
    )
    active = context.active_checkpoint
    for row in rows or []:
        trainer_path = Path(str(row[1]))
        inference_path = Path(str(row[2]))
        if not trainer_path.is_file() or not inference_path.is_file():
            continue
        checkpoint_id = str(row[0])
        return LineageCheckpoint(
            checkpoint_id,
            trainer_path,
            inference_path,
            released=active is not None and checkpoint_id == active.checkpoint_id,
        )
    return None


def _batch_lineage(batch) -> str:
    """The member lineage that collected `batch`, or `""` for a pre-member row.

    Read off batch metadata rather than joined from anywhere: the batch row is the
    only place that records which member was rolling when the episodes were played,
    and `collect` stamps it there for exactly this filter.
    """

    metadata = getattr(batch, "metadata", None)
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("member_lineage_id") or "")


def _member_checkpoint(member) -> LineageCheckpoint | None:
    """A member's stored checkpoint as a resume source, or None to cold-start.

    Both artifacts have to be on disk: the `.pt` carries the optimizer moments and
    the step counters, and resuming weights without them would silently restart
    Adam mid-lineage. A missing pair is a cold start, which is the honest reading
    for a member minted this round and for an artifact root cleaned by hand alike.

    `released` is False and the streak fields keep their defaults. Both belong to
    the reject-streak snapback, which pruning supersedes -- a member that stops
    earning its place is dropped by the prune, not walked back by a counter.
    """

    if member is None or not member.trainer_path or not member.inference_path:
        return None
    trainer, inference = Path(member.trainer_path), Path(member.inference_path)
    if not trainer.is_file() or not inference.is_file():
        return None
    return LineageCheckpoint(
        str(member.checkpoint_id or ""), trainer, inference, released=False
    )


def _weights_spec(weights_path: Path | str) -> str:
    """Address this agent's inference entry point at a specific checkpoint.

    Encoded rather than interpolated. Artifact roots are configurable, so a
    directory containing a space or an `&` would otherwise truncate the path at
    the separator and silently resolve to the packaged weights -- which makes
    challenger and incumbent play as one policy and every paired margin come out
    exactly 0.0.
    """

    return f"{SELF_SPEC}?{urlencode({'weights': str(weights_path)})}"


def _self_play_pool(
    context,
    reader,
    limit: int,
    after_checkpoint_id: str | None = None,
) -> list[str]:
    """Promoted checkpoints of this model, addressed as agent specs.

    Self-play is part of the reward distribution. Under the activity gate, a
    rejected checkpoint is exactly the kind of behaviour the next update should not
    imitate, even if it is recent. Rank by the same benchmark money used for
    active lineage so the mirror pool teaches from the best trusted checkpoints,
    not from the newest child.

    `after_checkpoint_id` bounds the pool below at a reseed: a restarted slot whose
    mirror is still the graduated policy trains against the trajectory it was
    restarted to escape. Returning nothing meanwhile is intended -- the caller falls
    back to the mature group.
    """

    if limit < 1:
        return []
    score_signal = _promotion_score_signal(context.config)
    score_sql = _checkpoint_score_sql(score_signal)
    if after_checkpoint_id:
        rows = reader.fetchall(
            f"""
            SELECT inference_artifact_path FROM model_training_checkpoints
            WHERE model_id = %s AND created_at > (
                SELECT created_at FROM model_training_checkpoints
                WHERE checkpoint_id = %s
            )
              AND promoted
              AND {score_sql} IS NOT NULL
            ORDER BY {score_sql} DESC,
                     created_at DESC,
                     checkpoint_id DESC
            LIMIT %s
            """,
            [context.model_id, after_checkpoint_id, limit],
        )
    else:
        rows = reader.fetchall(
            f"""
            SELECT inference_artifact_path FROM model_training_checkpoints
            WHERE model_id = %s
              AND promoted
              AND {score_sql} IS NOT NULL
            ORDER BY {score_sql} DESC,
                     created_at DESC,
                     checkpoint_id DESC
            LIMIT %s
            """,
            [context.model_id, limit],
        )
    return [_weights_spec(str(row[0])) for row in rows if Path(str(row[0])).is_file()]


def _latest_checkpoint_backfill(
    context,
    reader,
    pool: list[str],
    limit: int,
    after_checkpoint_id: str | None = None,
) -> list[str]:
    """Top `pool` up with the slot's newest checkpoints, promoted or not.

    Pairs with `lineage_source = "latest"`. Once the trained lineage advances past
    the release, a mirror pool drawn from promoted checkpoints alone is a frozen
    copy of a policy the learner has already left -- and it was the *only*
    opponent `terminal-selfplay-ladder-v1` ever rolled out against. Newest first,
    so the mirror tracks the learner; `after_checkpoint_id` keeps a reseeded slot
    from sparring with the trajectory it was restarted to escape.
    """

    if len(pool) >= limit:
        return pool
    if after_checkpoint_id:
        rows = reader.fetchall(
            """
            SELECT inference_artifact_path FROM model_training_checkpoints
            WHERE model_id = %s AND created_at > (
                SELECT created_at FROM model_training_checkpoints
                WHERE checkpoint_id = %s
            )
            ORDER BY created_at DESC, checkpoint_id DESC
            LIMIT %s
            """,
            [context.model_id, after_checkpoint_id, limit],
        )
    else:
        rows = reader.fetchall(
            """
            SELECT inference_artifact_path FROM model_training_checkpoints
            WHERE model_id = %s
            ORDER BY created_at DESC, checkpoint_id DESC
            LIMIT %s
            """,
            [context.model_id, limit],
        )
    seen = set(pool)
    out = list(pool)
    for row in rows or []:
        path = Path(str(row[0]))
        if not path.is_file():
            continue
        spec = _weights_spec(str(path))
        if spec in seen:
            continue
        out.append(spec)
        seen.add(spec)
        if len(out) >= limit:
            break
    return out


def _mixed_self_play_pool(
    context,
    reader,
    limit: int,
    after_checkpoint_id: str | None = None,
) -> list[str]:
    """A quality-gated but not purely in-lineage self-play pool."""

    if limit < 1:
        return []
    latest_backfill = bool(context.config.get("self_play_latest_backfill", False))
    if not bool(context.config.get("self_play_mature_backfill", True)):
        pool = _self_play_pool(context, reader, limit, after_checkpoint_id)[:limit]
        if latest_backfill and len(pool) < limit:
            pool = _latest_checkpoint_backfill(
                context, reader, pool, limit, after_checkpoint_id
            )
        return pool
    lineage_share = float(context.config.get("self_play_lineage_share", 0.6))
    lineage_share = max(0.0, min(1.0, lineage_share))
    lineage_limit = min(limit, max(1, int(math.ceil(limit * lineage_share))))
    pool = _self_play_pool(context, reader, lineage_limit, after_checkpoint_id)
    if len(pool) >= limit:
        return pool[:limit]
    if latest_backfill:
        pool = _latest_checkpoint_backfill(context, reader, pool, limit, after_checkpoint_id)
        if len(pool) >= limit:
            return pool[:limit]
    return pool[:limit]


def _learner_policy(
    weights_path: Path,
    gamma: float,
    sigma_scale: float = 1.0,
) -> EvaluationPolicy:
    """Identity of the exploring behaviour policy for one batch.

    The spec actually played also carries a per-episode trace path and exploration
    seed; folding those in would mint one revision per rollout episode. What defines
    the policy is its weights and how much it explores.

    Only the *base* `sigma_scale` is carried, not the per-actor spread around it:
    folding sixteen of those in would mint eight revisions a batch, none with enough
    games to rate.
    """

    query: dict[str, Any] = {
        "weights": str(weights_path.resolve()),
        "gamma": gamma,
    }
    if sigma_scale != 1.0:
        query["sigma_scale"] = round(sigma_scale, 4)
    spec = f"{LEARNER_ENTRY_POINT}?" + urlencode(query)
    return EvaluationPolicy(
        _policy_identity("free-market-policy", spec),
        spec,
        LEARNER_MANIFEST,
        POLICY_KINDS["market_learner"],
    )


def initial_network(config: dict[str, Any]) -> OrderQNetwork:
    """Build the first behavior policy for a new training instance.

    A random hidden trunk over a zeroed output layer: every action scores 0.0, the
    argmax ties to index 0, and index 0 is NOOP, so the cold-start policy trades
    nothing. That is a survivable starting point and, unlike the scorer's cold
    start, a deliberately weak one -- there is no donor market layer underneath to
    make it strong. `config` is read only for the seed.

    It must not be an all-zero MLP. Zero everywhere makes both hidden
    activations identically zero for any input, which zeroes the gradient of
    every parameter but the output bias, and `relu'(0) == 0` closes the path back
    to the biases. The symmetry never breaks, so a run that starts there emits
    NOOP forever while reporting a healthy falling loss.
    """

    network = OrderQNetwork.quiet_start(seed=int(config.get("seed", 0)))
    noisy = _noisy(config)
    if not noisy.enabled:
        return network

    last = len(network.layers) - 1
    sigma = [
        (
            (
                np.full(weight.shape, noisy.sigma0 / np.sqrt(weight.shape[0]), dtype=np.float32),
                np.full(bias.shape, noisy.sigma0 / np.sqrt(weight.shape[0]), dtype=np.float32),
            )
            if noisy.build_kind == "all" or index == last
            else None
        )
        for index, (weight, bias) in enumerate(network.layers)
    ]
    return OrderQNetwork(network.layers, sigma, gru=network.gru)


def _quiet_opponent(context, config: dict[str, Any]) -> str:
    """The cold-start network as a rollout opponent, materialised on first use.

    Reached in exactly one place: batch 0 of a fresh `model_id` under
    `opponents = []`, where `self_play_after` is read as `batch_index >= 1` so the
    mirror pool is never consulted and the mix is the empty base. That used to
    raise, which made a new `SLOT_PREFIX` illegal on a pure-mirror run -- and since
    the prefix is what severs a stale reward regime, a rule that could not be
    followed was protecting nothing.

    One batch against an opponent that emits nothing is a small price for that, and
    it is not a novel object: the same file is what the promotion gate measures a
    cold-start candidate against, on the argument that the quiet network is a real
    incumbent rather than an absence of one. Written once and reused, so the
    opponent a run faces at batch 0 does not depend on when the file was made.
    """

    path = Path(context.artifacts) / "viability_baselines" / "quiet.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        initial_network(config).save(path)
    return _weights_spec(path)


#: `sim.arena` resolves this by name in each worker, so it must stay importable
#: without torch -- which is why it lives in `diagnostics` and not here.
ARENA_REDUCER = "agents.dqn_market_free_v3c.diagnostics:arena_reduction"


def _arena_evaluate(
    weights_path: Path, opponent: str, seeds: int, jobs: int, seed_start: int = 0
) -> dict[str, Any]:
    """One paired series against a fixed opponent.

    Wrapped in the same thread pin the collector uses. `sim.arena` builds its own
    spawn pool and deliberately knows nothing about any agent, so it cannot pin
    threads on this agent's behalf. Pinned here rather than in `sim/` because
    that module is shared by every family.
    """

    from sim.arena import play_series
    from sim.metrics import summarize

    spec = _weights_spec(weights_path)
    with _single_threaded_workers():
        results = play_series(
            spec,
            opponent,
            range(seed_start, seed_start + seeds),
            {},
            jobs,
            reduce=ARENA_REDUCER,
        )
    summary = summarize(spec, opponent, results)
    diagnostics, behavior = _arena_summary(results)
    return {
        "opponent": opponent,
        "pairs": summary.pairs,
        "win_rate": summary.win_rate,
        "margin": summary.margin_mean,
        "margin_ci": list(summary.margin_ci),
        "significant": summary.significant,
        "errors": summary.errors,
        "a_money_mean": summary.a_money_mean,
        "a_money_median": summary.a_money_median,
        "b_money_mean": summary.b_money_mean,
        "b_money_median": summary.b_money_median,
        "order_diagnostics": diagnostics.as_dict(),
        # The strategy fingerprint the mature group's reserved cells are keyed
        # on. Measured here rather than from the rollout ring so it describes the
        # same episodes the score does -- the arena is greedy against a pinned
        # benchmark, the ring is noisy against the training mix, and a policy's
        # crop spread is not the same in both.
        "behavior": behavior.as_dict(),
    }


def _arena_summary(
    results: dict[int, tuple[Any, Any]],
) -> tuple[OrderDiagnostics, BehaviorCounts]:
    """The learner's seat over complete paired arena results, in one pass.

    Both aggregates walk the same replays under the same completeness rule, so they
    are read together rather than by two functions with an identical loop body
    reading every replay dict twice.

    The walk itself now happens in the worker -- `diagnostics.arena_reduction`,
    passed to `play_series` as `ARENA_REDUCER` -- so what arrives here is already
    the pair of aggregates and `replay` is None. This loop only folds them, which
    is what makes it cheap enough to run five of concurrently. The fold order is
    unchanged, so the totals are identical to walking the replays here.

    Only complete seat-swapped pairs contribute, matching `summarize()`.
    """

    diagnostics = OrderDiagnostics()
    behavior = BehaviorCounts()
    for forward, reverse in results.values():
        if forward.failed or reverse.failed:
            continue
        for result in (forward, reverse):
            if result.reduction is None:
                continue
            match_diagnostics, match_behavior = result.reduction
            diagnostics += match_diagnostics
            behavior += match_behavior
    return diagnostics, behavior


def _rollout_behavior(rollouts: list[Any]) -> BehaviorCounts:
    """The learner's fingerprint over the batch it just collected.

    `_arena_summary` above reads the promotion gate's paired matches, which are
    played greedily against a fixed benchmark. This reads the noisy training
    rollouts instead, so the land-use columns track the behaviour the buffer is
    actually being filled with rather than the one being graded.
    """

    behavior = BehaviorCounts()
    for rollout in rollouts:
        match = getattr(rollout, "match", None)
        replay = getattr(match, "replay", None)
        if isinstance(replay, dict):
            behavior += behavior_from_replay(replay, int(rollout.learner_seat))
    return behavior


def _records_diagnostics(records: list[EpisodeRecord]) -> OrderDiagnostics:
    diagnostics = OrderDiagnostics()
    for record in records:
        diagnostics += record.order_diagnostics
    return diagnostics


#: Q-units of clearance the large-margin clone loss asks for between a demonstrated
#: action and its best alternative, before the loss goes to exactly zero.
#:
#: Sized against the reward, not picked. The value is discounted cash-to-come over
#: `reward.MONEY_SCALE`: this policy's own seasons sit at 0.1-0.5 at turn 0 and
#: decay toward zero, the ladder pack's at 2-6. 0.3 is a clear preference on the
#: own range and well under the demo range, so satisfying it cannot by itself move
#: the Q scale the way cross-entropy did -- `market-free-splitladder-v1` ran
#: `max_abs_q` at 20-27 for a whole session on the unbounded loss.
#:
#: Overridable per model as `behavior_clone_margin`. Raise it if a clone fits the
#: demonstrations and the policy still does not follow them; lower it if
#: `max_abs_q` climbs on a demonstration-seeded slot while `mean_loss` stays flat,
#: which is the margin outrunning the reward rather than ranking against it.
BEHAVIOR_CLONE_MARGIN = 0.3

#: The lot-accounting columns lifted out of batch metadata and onto the update.
#:
#: The collector computes them and writes them to `training_batches.metadata_json`
#: (`writer.record(..., metadata=...)` below), and `model_updates.metrics_json`
#: carried nothing about lots at all -- so the one instrument that says whether the
#: herd and the seed purchases pay for themselves sat in a different table from
#: `own_mean_final_money`, the gate, and every other number an update is judged on.
#: Descriptive only: the reward reads the money trace and none of these.
#:
#: The share columns come along because the coins are a volume as much as a rate
#: -- `animal_waste_cost` doubling because the policy bought twice as many animals
#: and `animal_waste_cost` doubling because it recovered half as often are the same
#: column moving for opposite reasons, and `unrecovered_animal_share` is what
#: separates them.
_BATCH_LOT_KEYS = (
    "seed_waste_cost",
    "seed_gain_credit",
    "unrecovered_seed_share",
    "animal_waste_cost",
    "animal_gain_credit",
    "unrecovered_animal_share",
    "animal_lots",
    "animal_lots_matched_share",
    "animal_spend",
    "animal_proceeds",
    "stock_waste_cost",
    "stock_gain_credit",
    "stock_resale_share",
    "stock_resale_margin",
    "stock_spend",
    "lagged_wash_turns",
)


def _batch_lot_metrics(batches: Sequence[Any]) -> dict[str, float]:
    """Mean of each lot column over the batches this update consumed.

    Not a re-derivation. Every value here was computed once by the collector from
    the rollout replays and written to `training_batches.metadata_json`; this reads
    those rows back off `context.new_batches` and averages them, so the number on
    the update and the number on its batches cannot disagree.

    The mean is unweighted across batches rather than pooled across episodes. Each
    batch's value is already a per-episode mean over a fixed `episodes_per_batch`,
    so the two agree except where a batch came up short, and an unweighted mean is
    the one that stays stable when it does.

    Batches with no lot columns are skipped rather than counted as zero -- an
    evaluation-sourced batch has no rollouts behind it, and folding it in as a zero
    would report the herd term as decaying every time one landed. A key absent from
    every batch stays absent from the metrics rather than reading 0.0, which is the
    difference between "not measured" and "measured as nothing".
    """

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in batches:
        metadata = getattr(batch, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        for key in _BATCH_LOT_KEYS:
            value = metadata.get(key)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue
            totals[key] = totals.get(key, 0.0) + numeric
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}


def _limit(config: dict[str, Any], name: str) -> float:
    return float(config.get(name, VIABILITY_DEFAULTS[name]))


def _bootstrap_clamp(config: dict[str, Any]) -> tuple[float, float] | None:
    """`(low, high)` bounds on the bootstrapped value, in reward units, or None.

    Sized against `reward.MONEY_SCALE`. The value is the discounted cash still to
    be banked from a position, so its floor is the cash a position can *lose*, not
    the stake: a turn that holds 50k and ends the season at 3k is worth -1.9, and
    a stock-resale policy grossing ~200k a season can sit well below that. The
    pack's best ladder season (155,637) is about 6.1 on the high side. The
    defaults are symmetric at 8.0 so the clamp catches the max-over-noise runaway
    and nothing a real trajectory produces. `bootstrap_clamp_low` /
    `bootstrap_clamp_high` override; `bootstrap_clamp = false` disables.
    """

    if config.get("bootstrap_clamp", True) is False:
        return None
    low = float(config.get("bootstrap_clamp_low", -8.0))
    high = float(config.get("bootstrap_clamp_high", 8.0))
    if not low < high:
        raise ValueError(f"bootstrap_clamp_low must be below bootstrap_clamp_high: {low} >= {high}")
    return low, high


def _terminal_quality_weight(final_money: float | None) -> float:
    """Gradient multiplier from terminal money alone.

    Low-ending episodes are still sampled and still teach low value, but they no
    longer carry the same gradient mass as the trajectories the policy should
    imitate. The ramp is in dollars rather than reward units so diagnostics line
    up with every rollout table.
    """

    if final_money is None:
        return 1.0
    try:
        value = float(final_money)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value):
        return 1.0
    if value <= TERMINAL_QUALITY_MIN_MONEY:
        return TERMINAL_QUALITY_MIN_WEIGHT
    if value >= TERMINAL_QUALITY_FULL_MONEY:
        return 1.0
    span = TERMINAL_QUALITY_FULL_MONEY - TERMINAL_QUALITY_MIN_MONEY
    ratio = (value - TERMINAL_QUALITY_MIN_MONEY) / span
    return TERMINAL_QUALITY_MIN_WEIGHT + ratio * (1.0 - TERMINAL_QUALITY_MIN_WEIGHT)


def _order_viability(
    result: dict[str, Any], config: dict[str, Any], *, require_activity: bool = False
) -> dict[str, Any]:
    """Did the candidate trade at all, in one arena.

    Read off `trading_rate`, the share of turns the simulator acted on, not the wire
    order count: `HIRE n` is one decision whose wire form is n rows, so a hire-only
    policy reports 13 orders per decision while trading on one turn in fifty.

    Take `OrderDiagnostics.as_dict`'s published share as given rather than
    recomputing it from the counts -- a recomputation naming a key the payload does
    not carry reads 0.0 for every real arena and fails the floor unconditionally.

    Lot quality and refused-order rates are recorded by arena diagnostics, but
    promotion follows the older gate shape: survival plus activity, with benchmark
    money carrying the quality signal.
    """

    diagnostics = dict(result.get("order_diagnostics") or {})
    decisions = max(1, int(diagnostics.get("decisions", 0)))
    trading_rate = float(diagnostics.get("trading_rate", 0.0))
    checks: dict[str, bool] = {}
    minimum_activity = _limit(config, "viability_min_orders_per_decision")
    if require_activity and minimum_activity > 0.0:
        checks["min_orders_per_decision"] = trading_rate >= minimum_activity
    return {
        "passed": not result.get("errors")
        and int(result.get("pairs", 0)) > 0
        and all(checks.values()),
        "checks": checks,
        "trading_rate": trading_rate,
        "rejected_order_rate": float(diagnostics.get("rejected_order_rate", 0.0)),
        "wire_orders_per_decision": int(diagnostics.get("orders", 0)) / decisions,
        "min_orders_per_decision_required": bool(require_activity and minimum_activity > 0.0),
    }






















def _learning_rates(config: dict[str, Any]) -> tuple[float, float]:
    """Configured (TD, behaviour-clone) learning rates."""

    learning_rate = float(config.get("learning_rate", 1e-4))
    clone_rate = float(config.get("behavior_clone_learning_rate", learning_rate))
    return learning_rate, clone_rate


def _reset_learning_rates(
    optimizer: torch.optim.Optimizer,
    clone_optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    learning_rate: float | None = None,
) -> None:
    """Re-apply the learning rates over whatever a resume restored.

    `learning_rate` overrides the configured peak with what the slot schedule
    resolved for this update; the clone rate keeps its configured ratio to the
    peak so a decaying slot decays both phases together.
    """

    peak, clone_rate = _learning_rates(config)
    if learning_rate is not None and peak > 0.0:
        clone_rate *= learning_rate / peak
        peak = learning_rate
    for group in optimizer.param_groups:
        group["lr"] = peak
    for group in clone_optimizer.param_groups:
        group["lr"] = clone_rate


# --- Population training -----------------------------------------------------
#
# The single ever-resuming lineage this replaced measured its own stall: hundreds
# of batches with no trend in either direction, loss and `mean_abs_q` both flat.
# A trajectory that has stopped moving cannot be pushed out of its basin by the
# gradient that put it there, and every exploration lever aimed at the behaviour
# distribution rather than at the trajectory.
#
# So named learner slots train in parallel (one `training_models` row each,
# namespaced under one `declared_model_id`), and each slot carries several member
# lineages rather than one. A member that stops earning its place is *dropped* by
# the generation prune rather than nursed -- see `training/members.py`.


def _roll_chunk(
    module,
    batch,
    device: torch.device,
    burn_in: int,
    train: int,
    *,
    grad: bool,
):
    """Burn in, then roll the training window and the n-step tail.

    Returns `(train_hidden, post_hidden)`: the `[B, train, H]` states the training
    decisions score against, and the `[B * (train + n_step), H]` flattened grid
    `next_turn` indexes into -- the training latents for the intra-turn bootstrap,
    the tail's for the turn boundary. A bootstrap that carries gradient is not a
    target, hence the detach.

    The burn-in is always under `no_grad`, whatever `grad` says. Letting the
    gradient into it would make BPTT run the whole chunk rather than the training
    window, so "truncated" would stop being true and the cost would grow with
    `burn_in` -- which is the one dial meant to be cheap.

    The tail is also always under `no_grad` and starts from a detached state: it
    exists only to produce bootstrap latents.
    """

    turn_input = torch.from_numpy(batch.turn_input).to(device)
    valid = torch.from_numpy(batch.turn_valid).to(device)
    hidden = torch.from_numpy(batch.hidden0).to(device)

    with torch.no_grad():
        if burn_in:
            hidden = module.recur(
                turn_input[:, :burn_in], hidden, valid[:, :burn_in]
            )[:, -1]

    window = turn_input[:, burn_in : burn_in + train]
    window_valid = valid[:, burn_in : burn_in + train]
    if grad:
        train_hidden = module.recur(window, hidden, window_valid)
    else:
        with torch.no_grad():
            train_hidden = module.recur(window, hidden, window_valid)

    with torch.no_grad():
        tail = module.recur(
            turn_input[:, burn_in + train :],
            train_hidden[:, -1].detach(),
            valid[:, burn_in + train :],
        )
    post = torch.cat((train_hidden.detach(), tail), dim=1)
    return train_hidden, post.reshape(-1, module.hidden_size)


def _probe_features(
    module,
    batch,
    device: torch.device,
    burn_in: int,
    train: int,
):
    """`(concatenated features, mask)` for the sigma probe, latents and all.

    The geometry is passed rather than recovered from the batch's shapes: a chunk
    is `burn_in + train` turns and only `train` is recoverable from `reward`, so
    inferring the split here would be guessing at the very number whose
    misalignment the probe could not detect.
    """

    with torch.no_grad():
        train_hidden, _post = _roll_chunk(
            module, batch, device, burn_in, train, grad=False
        )
        rows = torch.from_numpy(batch.decision_turn).to(device)
        states = torch.from_numpy(batch.state).to(device)
        hidden = train_hidden.reshape(-1, module.hidden_size)[rows]
        features = torch.cat((states, hidden), dim=1)
    return features, torch.from_numpy(batch.mask).to(device)


def _large_margin_loss(
    logits: torch.Tensor, chosen: torch.Tensor, margin: float
) -> torch.Tensor:
    """DQfD's large-margin classification loss, in Q units.

        relu( max_a[ Q(s, a) + margin * (a != a_E) ] - Q(s, a_E) )

    **Bounded, which is the whole reason it replaced cross-entropy here.** These
    logits are Q-values, not class scores: `max_abs_q` is a divergence detector and
    the label is a reward-to-go bounded by the best season in the buffer, so a
    correct Q sits in single digits. Cross-entropy's optimum is the chosen logit at `+inf` -- it has no fixed
    point in Q units at all -- and `market-free-splitladder-v1` measured what that
    costs. On the *first* update carrying demonstrations, `max_abs_q` went 0.49 ->
    20.82 and `mean_loss` 0.0013 -> 1.194, against control slots flat at ~0.5 and
    0.0005 for the whole session; `_sync_target` then copied the inflated head into
    the target net. `grad_clip_norm` never had a say: it bounds the step, not the
    destination.

    This one is exactly zero, with exactly zero gradient, the moment the
    demonstrated action leads every alternative by `margin`. So it composes with the
    TD loss instead of fighting it -- it states a *preference ordering* and stops,
    where cross-entropy states a target distribution it can never reach.

    Infeasible actions arrive already at `MASKED_SCORE`, so the max cannot select
    one and the margin cannot push the expert above one that the live mask would
    refuse anyway.
    """

    offset = torch.full_like(logits, float(margin))
    offset.scatter_(1, chosen.unsqueeze(1), 0.0)
    expert = logits.gather(1, chosen.unsqueeze(1)).squeeze(1)
    return torch.relu((logits + offset).max(dim=1).values - expert).mean()


def _behavior_clone_step(
    online,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    chunks: int,
    device: torch.device,
    burn_in: int,
    train: int,
    clip: float,
    margin: float = BEHAVIOR_CLONE_MARGIN,
) -> float:
    """Fit the action head to a stored expert action for one chunk batch.

    A large-margin hinge over `ACTION_COUNT` actions -- see `_large_margin_loss`
    for why it is not the cross-entropy it used to be. The previous ten-slot
    action made this a cross-entropy over ten heads whose targets were 88% NOOP
    -- stored donor replays average 1.2 orders a turn -- so the clone's
    minimum-loss answer was silence, which is the failure demonstrations exist to
    prevent.

    Sequence-shaped in v3, and graded through the recurrence rather than only
    through the head. The clone is what shapes the *representation* a cold run
    starts from, and a latent trained only by the TD loss while the head is being
    fitted by cross-entropy would have the two disagreeing about what the hidden
    state means.

    Demonstrations carry the *least* staleness of anything in the buffer, which is
    not the same as none: `load_demonstrations(..., gru=...)` re-derives their
    stored states from the current GRU once at the top of each update, and
    `clone.py` writes them through `float16`. So their burn-in still refreshes
    something -- one update's drift plus a quantisation -- rather than starting
    from a state this exact network produced. Bounded by one update instead of by
    buffer residency, which is the whole of the improvement.
    """

    batch = buffer.sample_sequences(
        chunks, burn_in=burn_in, train=train, demo_fraction=1.0
    )
    if batch.state.shape[0] == 0:
        return 0.0
    optimizer.zero_grad(set_to_none=True)
    train_hidden, _post = _roll_chunk(online, batch, device, burn_in, train, grad=True)

    rows = torch.from_numpy(batch.decision_turn).to(device)
    states = torch.from_numpy(batch.state).to(device)
    hidden = train_hidden.reshape(-1, online.hidden_size)[rows]
    logits = online(torch.cat((states, hidden), dim=1))
    # Infeasible actions carry no probability, so the clone never spends its
    # capacity learning to rank orders the game would refuse.
    logits = logits.masked_fill(~torch.from_numpy(batch.mask).to(device), MASKED_SCORE)
    chosen = torch.from_numpy(batch.chosen).to(device)
    loss = _large_margin_loss(logits, chosen, margin)
    loss.backward()
    # The optimizer's own parameters, not `online.parameters()`, and both halves of
    # that matter. `clone_optimizer` excludes every sigma, so `zero_grad` never
    # clears a sigma gradient -- while a clip over the whole module would keep
    # counting them, and they accumulate across the phase. The total norm would
    # then grow monotonically and the clip factor would shrink the head's and the
    # GRU's real gradients a little further on every step, decaying the clone's
    # effective learning rate for a reason nothing reports.
    #
    # `clip` rather than a hardcoded 10.0: this backpropagates through `train`
    # GRU steps exactly as the TD step does, so it wants the same bound
    # `grad_clip_norm` sets there -- and the clone phase runs first, so it is the
    # path that meets an exploding recurrence earliest.
    clipped = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.grad is not None
    ]
    nn.utils.clip_grad_norm_(clipped, clip)
    optimizer.step()
    return float(loss.detach())


def _behavior_clone_steps_per_update(config: dict[str, Any]) -> int:
    """Default demo-seeded runs to imitation without changing run identity."""

    default = 100 if int(config.get("demonstration_episodes", 0)) > 0 else 0
    return int(config.get("behavior_clone_steps_per_update", default))


@dataclass(frozen=True)
class NoisySpec:
    """How this run explores in weight space; empty when NoisyNet is off.

    Read in one place because the collector and the updater must agree. They build the same network from different sides --
    the updater from `net_module.build`, the collector from a `weights.npz` the
    updater exported -- and a disagreement about `layers` surfaces as
    `from_numpy` raising in the middle of an update rather than as a config
    error at the start of one.
    """

    enabled: bool
    layers: str  # "none" | "head" | "all"
    sigma0: float
    cadence: str
    actor_alpha: float
    scale_floor: float
    clone_noise: bool
    order_rate_target: float
    scale_limit: float

    @property
    def build_kind(self) -> str:
        return self.layers if self.enabled else "none"

    @property
    def collect_cadence(self) -> str:
        return self.cadence if self.enabled else "none"

    @property
    def controlled(self) -> bool:
        """Whether collection sigma is chosen from the measured curve at all."""

        return self.enabled and self.order_rate_target > 0.0


_NOISE_CADENCES = ("none", "decision", "day", "episode")

#: Sigma multipliers the flip-rate probe measures, geometric around 1.0.
#:
#: Geometric rather than linear because the quantity being read is a rate against
#: a perturbation whose effect is roughly log-linear in its size: half a decade
#: below the learned sigma and half a decade above covers the whole usable range
#: in seven points, and each point costs one no-grad forward on one minibatch.
#: `flip_rate_curve` clips it to the same bounds `resolve_sigma_scale` may return
#: -- `noisy_sigma_scale_limit` above, and `noisy_sigma_scale_floor` (falling back
#: to `1 / limit`) below -- so widening the ladder cannot widen what a run is
#: allowed to play at, and narrowing the floor cannot hide a rung it may return.
SIGMA_LADDER: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

#: Independent draws averaged per ladder point.
#:
#: One draw of the 186-wide head over a 256-state minibatch is a very noisy
#: estimate of a rate near 0.05 -- of order 13 flipped states, so a single draw
#: swings it by tens of percent, and the resolver would chase that swing every
#: update. Eight draws is 56 no-grad forwards on one minibatch for the whole
#: ladder, against `gradient_steps_per_batch` full forward/backward passes.
SIGMA_PROBE_DRAWS = 8


def _noisy(config: dict[str, Any]) -> NoisySpec:
    enabled = bool(config.get("noisy", False))
    layers = str(config.get("noisy_layers", "head"))
    cadence = str(config.get("noisy_cadence", "decision"))
    if layers not in net_module.NOISY_LAYERS:
        raise ValueError(
            f"unknown noisy_layers {layers!r}, expected one of {net_module.NOISY_LAYERS}"
        )
    if cadence not in _NOISE_CADENCES:
        raise ValueError(f"unknown noisy_cadence {cadence!r}, expected one of {_NOISE_CADENCES}")
    # Retired keys, refused rather than ignored. A config carrying them was asking
    # for rungs above the setpoint, and silently dropping that request would leave
    # the run reading as if it still had them.
    retired = sorted(
        {"noisy_actor_hot", "noisy_hot_order_rate", "noisy_hot_scale_limit"} & set(config)
    )
    if retired:
        raise ValueError(
            f"{', '.join(retired)} was retired with the hot actor ladder; the sigma "
            "spread is downward only and rare-action widening is `rare_sigma_gain`"
        )
    return NoisySpec(
        enabled=enabled,
        layers=layers,
        sigma0=float(config.get("noisy_sigma0", 0.5)),
        cadence=cadence,
        actor_alpha=float(config.get("noisy_actor_alpha", 32.0)),
        # 0.0 keeps the implicit `1 / noisy_sigma_scale_limit` clamp, so a config
        # that never names it is unchanged.
        scale_floor=float(config.get("noisy_sigma_scale_floor", 0.0)),
        clone_noise=bool(config.get("behavior_clone_noise", False)),
        order_rate_target=float(config.get("noisy_order_rate_target", 0.0)),
        scale_limit=float(config.get("noisy_sigma_scale_limit", 8.0)),
    )


def flip_rate_curve(
    module: nn.Module,
    features: torch.Tensor,
    mask: torch.Tensor,
    scale_limit: float = 8.0,
    ladder: tuple[float, ...] = SIGMA_LADDER,
    draws: int = SIGMA_PROBE_DRAWS,
    scale_floor: float = 0.0,
) -> tuple[dict[float, float], dict[float, float]]:
    """Flip rate and flip-to-order rate per sigma multiplier, on one minibatch.

    `features` is `[N, STATE_DIM + HIDDEN_SIZE]` -- the decision state with its
    turn's latent already concatenated, which is what `OrderQModule.forward` takes.
    The probe measures the *head's* sensitivity to its own sigma, and the GRU
    carries no sigma, so rolling the recurrence per rung would redo identical work
    `draws * len(ladder)` times to reach the same latents. The caller rolls it once
    and hands the result in.

    Returns `(flip_curve, order_curve)`, both keyed by the same rungs. `flip_curve`
    maps a multiplier to the share of probe states where the perturbation changed
    the argmax -- the quantity epsilon named directly. `order_curve` maps it to the
    share of states the greedy policy would sit out that the perturbation talks
    into a real order, and is empty when the greedy policy never chooses NOOP on
    the probe, since there is then nothing to talk out of it.

    The second exists because the first is not enough on its own: on a collapsed
    policy every action is near-tied around NOOP, so the flip rate reads high while
    the agent trades nothing, and a controller reading it throttles exploration at
    the one moment it needs raising. That is why the controller reads `order_curve`.

    Swept rather than measured once, because one reading says whether exploration is
    still there while the curve says what it would take to put it back.

    The ladder is clipped to the range `resolve_sigma_scale` may return, so a run is
    never measured at a scale it is not allowed to play at -- and, just as important,
    is never denied a measurement at one it is. `scale_floor` is that resolver's
    `floor` argument and must be passed through: it *replaces* the implicit
    `1 / scale_limit` lower clamp rather than tightening it, so with a limit of 2 and
    a floor of 0.25 the resolver may return 0.25 while an unclipped-below ladder
    stops at 0.5. The resolver only ever returns a scale the curve mentions, so the
    floor is dead config unless the ladder reaches it.

    Leaves the module's noise in whatever state the last draw left it: every caller
    either re-draws before its next forward or is done with the module.
    """

    with torch.no_grad():
        net_module.set_noise(module, False)
        greedy = module(features).masked_fill(~mask, MASKED_SCORE).argmax(dim=1)
        greedy_noop = greedy == NOOP_INDEX
        has_noop = bool(greedy_noop.any())
        low = float(scale_floor) if float(scale_floor) > 0.0 else 1.0 / scale_limit
        rungs = [scale for scale in ladder if low <= scale <= scale_limit] or [1.0]

        curve: dict[float, float] = {}
        order_curve: dict[float, float] = {}
        for scale in rungs:
            flips: list[float] = []
            to_order: list[float] = []
            for _ in range(draws):
                net_module.reset_noise(module, scale=scale)
                perturbed = module(features).masked_fill(~mask, MASKED_SCORE).argmax(dim=1)
                flips.append(float((greedy != perturbed).float().mean()))
                # The NOOP attractor read directly: states the greedy policy
                # would sit out, that the perturbation talks into a real order.
                # Recorded at every rung rather than only at the learned sigma,
                # because that is what makes it usable as a control signal and
                # not merely as a diagnostic. NoisyNet still does not *fix* the
                # NOOP attractor -- it only stops the controller from mistaking
                # a policy sitting on the flat plateau for one that is exploring.
                if has_noop:
                    to_order.append(
                        float((perturbed[greedy_noop] != NOOP_INDEX).float().mean())
                    )
            curve[scale] = float(np.mean(flips))
            if to_order:
                order_curve[scale] = float(np.mean(to_order))
    return curve, order_curve


def resolve_sigma_scale(
    curve: dict[float, float],
    target: float,
    limit: float = 8.0,
    floor: float = 0.0,
) -> float:
    """The sigma multiplier whose measured rate is `target`.

    `curve` maps a sigma scale to a rate measured at that scale -- either the
    argmax flip rate or the flip-to-order rate, whichever the run controls on.
    Both rise with scale, so the answer is a lookup with interpolation rather
    than a controller: no gain to tune, no integral state to persist, and no
    oscillation from chasing a stale measurement. Interpolation is in log-scale
    because the ladder is geometric and the rate is closer to linear in
    `log(scale)` than in `scale`.

    Returns 1.0 -- the learned sigma, untouched -- whenever the curve cannot
    answer: the controller is off (`target <= 0`), the probe never ran, or the
    curve is flat, which is what a collapsed sigma looks like. Saturating at
    `limit` instead would be worse in exactly that case: it would read as a
    controller doing its job while exploration is in fact zero at every scale.
    Clamped to `[1 / limit, limit]` at both ends, so a target the ladder cannot
    reach costs a bounded departure from the learned sigma rather than an
    unbounded one.
    """

    if target <= 0.0 or not curve:
        return 1.0
    points = sorted(curve.items())
    rates = [rate for _scale, rate in points]
    if max(rates) == min(rates):
        return 1.0
    # `floor` overrides the implicit `1 / limit` lower clamp, and exists because
    # that clamp is not a floor on *exploration*, only a bound on how far the
    # resolver may depart from the learned sigma. On a cold start the two come
    # apart: an untrained network's flip-to-order rate is high because the network
    # is random, not because sigma is large, so the controller reads
    # over-exploration and turns sigma down at the one moment there is nothing yet
    # to exploit. Measured on `market-free-landgate-v10`, update 20 resolved
    # exactly 0.125 -- the `1 / 8` clamp, hit dead on -- with `mean_abs_q` at
    # 0.046, and the run went quiet.
    #
    # 0.0 means "use `1 / limit`", so an unset floor is byte-for-byte what this
    # returned before the parameter existed.
    low = float(floor) if float(floor) > 0.0 else 1.0 / float(limit)
    high = float(limit)
    # Below the whole ladder: even the smallest scale over-explores, so take the
    # bottom. Strictly above it: even the largest under-explores, so take the top.
    #
    # Strictly, because a target the ladder exactly *reaches* is reachable and
    # belongs to the loop below, which returns the smallest scale that gets there.
    # This is dead weight on the argmax flip rate, which is strictly increasing
    # and never ties -- but the flip-to-order rate saturates at 1.0 once every
    # NOOP state is talked into an order, so `>=` here would answer a reachable
    # target with the largest scale sharing it rather than the least. Sigma above
    # what the setpoint asks for is not free: under episode cadence one
    # over-perturbed draw costs a whole 720-turn episode.
    if target <= points[0][1]:
        return max(low, min(high, points[0][0]))
    if target > points[-1][1]:
        return max(low, min(high, points[-1][0]))
    for (scale_a, rate_a), (scale_b, rate_b) in zip(points, points[1:]):
        if not rate_a <= target <= rate_b:
            continue
        if rate_b == rate_a:
            return max(low, min(high, scale_a))
        weight = (target - rate_a) / (rate_b - rate_a)
        resolved = math.exp(
            math.log(scale_a) + weight * (math.log(scale_b) - math.log(scale_a))
        )
        return max(low, min(high, resolved))
    # The curve is not monotone -- possible when two ladder points sit inside the
    # probe's own noise. Fall back to the point closest to the target rather than
    # inventing a scale between two readings that disagree.
    return max(low, min(high, min(points, key=lambda item: abs(item[1] - target))[0]))


def collection_sigma_scale(
    reader,
    model_id: str,
    key: str = "noisy_sigma_scale_target",
) -> float:
    """The scale the last update's probe resolved, or 1.0.

    `collect` and `train` are different methods on different rounds, so the number
    crosses through the database rather than memory -- the same route
    `lineage_checkpoint` takes. That also survives a restart and makes what an actor
    played at answerable in SQL.

    1.0 covers every way there is nothing to read. Deliberately not the newest row
    unconditionally: a rejected update still measured a real curve, so its scale is
    usable, but a row that never ran the probe must not read as a scale of nothing.

    `->> IS NOT NULL` rather than the `?` containment operator: `?` is a placeholder
    in several drivers and survives here only by psycopg's parameter style, while
    `->>` returns NULL for both an absent key and a JSON null.
    """

    # Interpolated rather than parameterised: `metrics_json ->> %s` would make the
    # key a value, and the same placeholder then appears in a `SELECT` cast and a
    # `WHERE` predicate where PostgreSQL infers two different types for it. The
    # input is not user data -- it is a literal chosen in this module -- so it is
    # rejected outright rather than escaped.
    if key != "noisy_sigma_scale_target":
        raise ValueError(f"unknown collection sigma key {key!r}")
    row = reader.fetchone(
        f"""
        SELECT (metrics_json ->> '{key}')::float
        FROM model_updates
        WHERE model_id = %s
          AND metrics_json ->> '{key}' IS NOT NULL
        ORDER BY started_at DESC
        LIMIT 1
        """,
        [model_id],
    )
    return 1.0 if row is None or row[0] is None else float(row[0])


def _load_online_state(module: nn.Module, state: dict[str, Any]) -> bool:
    """Load a trainer state dict, remapping one written before NoisyNet.

    Returns True when it had to migrate.

    A pre-NoisyNet checkpoint stores `4.weight`; a noisy module wants
    `4.weight_mu`. Remapped rather than rejected, because the lineage a live run
    follows is dozens of checkpoints deep and refusing it would throw all of that
    away -- and starting a fresh lineage instead is not the easy alternative it
    looks like: a mirror-only run has no self-play pool on batch 0 and collection
    raises for want of an opponent.

    Sigma is deliberately *not* invented here. It stays at the module's own
    `sigma0` initialization, the same rule `from_numpy` follows.
    """

    try:
        module.load_state_dict(state)
        return False
    except RuntimeError:
        pass

    wanted = module.state_dict()
    remapped = dict(state)
    for key in list(remapped):
        for suffix in ("weight", "bias"):
            if key.endswith(f".{suffix}") and key not in wanted:
                candidate = f"{key}_mu"
                if candidate in wanted:
                    remapped[candidate] = remapped.pop(key)
    missing, unexpected = module.load_state_dict(remapped, strict=False)
    unresolved = [name for name in missing if "sigma" not in name]
    if unresolved or unexpected:
        raise RuntimeError(
            f"cannot load trainer state: missing {unresolved}, unexpected {list(unexpected)}"
        )
    return True


def _recurrence_metrics(
    online,
    buffer: ReplayBuffer,
    device: torch.device,
    burn_in: int,
    train: int,
) -> dict[str, float | None]:
    """What the stored hidden states and the burn-in are actually worth.

    Three questions, none of which any existing metric answers:

    `hidden_recovery_*` asks how much of the stored state the burn-in did *not*
    reconstruct, by rolling the same window twice -- once from the stored `h0`,
    once from zeros -- and comparing where the two land. Converging to zero is the
    **good** reading: it means the burn-in has already rebuilt everything the
    stored state was carrying, so staleness cannot be biasing the update. It also
    means the stored state is redundant and `hidden_stride` could be raised.

    `q_discrepancy` is R2D2's own diagnostic and the only one of the three that
    reads on the quantity being learned: how far the Q values move when the
    burn-in is cut to a quarter, scaled by the spread of Q on the same rows. A
    chunk carries only `burn_in` turns of history, so the comparison has to run
    downward -- see the note at the call. Near zero means the configured burn-in is
    longer than it needs to be and `hidden_stride` has room; large means it is
    carrying the update and should not be cut.

    `hidden_norm` and `hidden_saturation` watch the recurrence itself. A GRU that
    saturates its gates or collapses to a constant is invisible in `mean_loss` --
    it looks exactly like a network that has learned a very good state-independent
    value.

    Computed on one probe batch per update, not per gradient step, and entirely
    under `no_grad`.
    """

    metrics: dict[str, float | None] = {}
    if not buffer.episodes:
        return metrics
    # Silence the noise before measuring anything. This runs after
    # `flip_rate_curve`, whose documented contract is that it leaves the module at
    # whatever draw it last made -- and the last rung of `SIGMA_LADDER` is the
    # widest one the controller is allowed. Every number below would otherwise be
    # read through a head perturbed at up to that multiple of its learned sigma,
    # and `q_discrepancy` in particular is a difference of two forwards, so the
    # perturbation does not cancel.
    net_module.set_noise(online, False)
    try:
        batch = buffer.sample_sequences(
            max(1, min(8, len(buffer.episodes))),
            burn_in=burn_in,
            train=train,
            demo_fraction=0.0,
            prioritized=False,
        )
    except ValueError:
        # No episode long enough to hold a training window. Not an error here --
        # the update itself will have raised first if it mattered.
        return metrics

    with torch.no_grad():
        turn_input = torch.from_numpy(batch.turn_input).to(device)
        valid = torch.from_numpy(batch.turn_valid).to(device)
        stored = torch.from_numpy(batch.hidden0).to(device)
        zeros = torch.zeros_like(stored)

        def burn(start, steps):
            if not steps:
                return start
            return online.recur(turn_input[:, :steps], start, valid[:, :steps])[:, -1]

        from_stored = burn(stored, burn_in)
        from_zero = burn(zeros, burn_in)
        gap = (from_stored - from_zero).norm(dim=1)
        scale = from_stored.norm(dim=1).clamp_min(1e-6)
        metrics["hidden_recovery_l2"] = float((gap / scale).mean())
        metrics["hidden_recovery_cosine"] = float(
            1.0
            - torch.nn.functional.cosine_similarity(from_stored, from_zero, dim=1).mean()
        )

        # A *quarter* of the burn-in against the configured one, and the direction
        # matters. A chunk physically holds `burn_in` turns of history and nothing
        # more, so a longer refresh cannot be constructed from it -- an earlier
        # version tried, clamping `4 * burn_in` to `burn_in + n_step`, which at the
        # shipped geometry rolled 24 turns *into* the graded window and reported
        # how far Q moves across 24 turns of play rather than how far it moves
        # under a longer refresh. That number is large for a healthy
        # time-sensitive GRU and would read as "burn-in too short" forever.
        #
        # Read it the other way instead: if a quarter of the burn-in already
        # reproduces the configured one's Q, the burn-in is longer than it needs to
        # be. If it does not, the burn-in is doing real work and the configured
        # length is the floor rather than a comfortable margin.
        short_burn = burn_in // 4
        from_short = burn(stored, short_burn)
        states = torch.from_numpy(batch.state).to(device)
        if states.shape[0]:
            rows = torch.from_numpy(batch.decision_turn).to(device)
            # Per-turn latents, indexed exactly as the TD step indexes
            # `train_hidden` -- not one chunk-level latent applied to every turn in
            # the window, which is not a hidden state the model ever scores against.
            window = turn_input[:, burn_in : burn_in + train]
            window_valid = valid[:, burn_in : burn_in + train]
            full_rolled = online.recur(window, from_stored, window_valid)
            short_rolled = online.recur(window, from_short, window_valid)
            full_hidden = full_rolled.reshape(-1, online.hidden_size)[rows]
            short_hidden = short_rolled.reshape(-1, online.hidden_size)[rows]
            full_q = online(torch.cat((states, full_hidden), dim=1))
            short_q = online(torch.cat((states, short_hidden), dim=1))
            spread = (full_q.max(dim=1).values - full_q.min(dim=1).values).clamp_min(1e-6)
            metrics["q_discrepancy"] = float(
                ((short_q - full_q).abs().mean(dim=1) / spread).mean()
            )

        rolled = online.recur(
            turn_input[:, burn_in : burn_in + train],
            from_stored,
            valid[:, burn_in : burn_in + train],
        )
        metrics["hidden_norm"] = float(rolled.norm(dim=-1).mean())
        metrics["hidden_saturation"] = float((rolled.abs() > 0.99).float().mean())
    return metrics


def _sigma_metrics(module: nn.Module) -> dict[str, float]:
    """Mean |sigma| and the sigma/mu ratio per noisy layer.

    Sigma shrinks under a regression loss unless noise genuinely helps, and the
    failure is silent -- the run quietly becomes greedy-only and no existing
    metric moves. That is the same shape as the NOOP collapse that took a whole
    run to notice, so it is instrumented before it is needed rather than after.
    """

    metrics: dict[str, float] = {}
    index = 0
    for layer in module.modules():
        if not isinstance(layer, net_module.NoisyLinear):
            continue
        sigma = float(layer.weight_sigma.detach().abs().mean())
        mu = float(layer.weight_mu.detach().abs().mean())
        metrics[f"sigma_mean_{index}"] = sigma
        metrics[f"sigma_ratio_{index}"] = sigma / mu if mu > 0 else 0.0
        index += 1
    return metrics


def _reject_inert_network(network: OrderQNetwork) -> None:
    """Fail loudly when the exported network can only ever emit NOOP.

    The recurrence is *reported*, not rejected. An all-zero head trunk is
    unrecoverable -- `relu'(0) == 0` closes every gradient path -- but an all-zero
    GRU is not: at zero weights `r = z = 0.5` and `tanh'(0) = 1`, so `dh/dn = 0.5`
    and the recurrence can still learn its way out. Raising on it would also take
    down `OrderQNetwork.zeros`, the inference fallback, which is legitimately
    all-zero by design.
    """

    if net_module.has_inert_hidden_trunk(network):
        raise RuntimeError(
            "exported free-market network has all-zero hidden weights: every action "
            "ties to NOOP and the agent never trades. An all-zero network also has "
            "no gradient outside its output bias, so the run cannot recover -- the "
            "hidden trunk collapsed after the cold start, which starts it random"
        )
    if net_module.has_inert_recurrence(network):
        # Not fatal, and not silent either. A v3 checkpoint whose GRU is dead plays
        # v2: the head still reads the raw decision state, so the policy trades and
        # every gate metric looks ordinary, while 128 of the head's 408 input
        # columns carry nothing and the whole premise of the fork is inert.
        LOGGER.warning(
            "exported free-market network has an all-zero GRU: the recurrence "
            "contributes nothing and this checkpoint is effectively pre-v3. Check "
            "`hidden_norm` and `mean_grad_norm` on the last few updates."
        )


def fill_buffer(
    buffer: ReplayBuffer, batches, *, terminal_quality_weighting: bool = False
) -> int:
    """Load a run's rollout history into the buffer, opening as little as possible.

    `all_batches` is the whole history and only grows, while the buffer holds
    `buffer_episodes` of it -- so reading every artifact meant digesting and
    decompressing the entire run to throw almost all of it away, once per update,
    forever.

    Survival is decided first, from `sample_count` on the batch row: the FIFO cut
    is positional and the reservoir draw is content-blind, so neither needs the
    episode it is ruling on. Only batches that still own a surviving episode are
    opened. Returns the number of batches skipped that way.

    A batch skipped this way is also not digest-checked. That is the intended
    trade: it contributes nothing to this update, so a corrupt artifact there
    cannot affect the gradient, and it will be checked again on any update where
    it does contribute.
    """

    rollouts = [
        batch for batch in batches if batch.source_kind == SOURCE_KIND and batch.artifact_path
    ]
    counts = [max(0, int(batch.sample_count)) for batch in rollouts]
    generation_money: dict[str, tuple[float, int]] = {}
    if buffer.generation_weight_power > 0.0:
        for batch, count in zip(rollouts, counts):
            metadata = getattr(batch, "metadata", None)
            if not isinstance(metadata, dict) or count <= 0:
                continue
            generation = metadata.get("generation_id")
            mean = metadata.get("mean_final_money")
            if generation is None or mean is None:
                continue
            try:
                value = float(mean)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            key = str(generation)
            total, samples = generation_money.get(key, (0.0, 0))
            generation_money[key] = (total + value * count, samples + count)
    generation_scores = {
        key: total / samples
        for key, (total, samples) in generation_money.items()
        if samples > 0
    }
    best_generation_score = max(
        (max(0.0, value) for value in generation_scores.values()), default=0.0
    )
    recent_ids, reservoir_ids, evicted_seen = buffer.plan_intake(counts)
    wanted = set(recent_ids) | set(reservoir_ids)

    skipped_evicted = 0
    records_by_id: dict[int, EpisodeRecord] = {}
    offset = 0
    for batch, count in zip(rollouts, counts):
        first, offset = offset, offset + count
        owned = [key for key in range(first, offset) if key in wanted]
        if not owned:
            skipped_evicted += 1
            continue
        path = Path(batch.artifact_path)
        if batch.artifact_sha256 and _file_digest(path) != batch.artifact_sha256:
            raise ValueError(f"rollout artifact digest mismatch: {path}")
        records = load_episode_batch(path)
        metadata = getattr(batch, "metadata", None)
        generation = (
            str(metadata.get("generation_id"))
            if isinstance(metadata, dict) and metadata.get("generation_id") is not None
            else None
        )
        generation_weight = 1.0
        if (
            buffer.generation_weight_power > 0.0
            and generation is not None
            and best_generation_score > 0.0
            and generation in generation_scores
        ):
            ratio = max(0.0, generation_scores[generation]) / best_generation_score
            generation_weight = max(
                buffer.generation_weight_floor,
                min(1.0, ratio ** buffer.generation_weight_power),
            )
        for key in owned:
            index = key - first
            if index < len(records):
                # Relabelled from the money trace, as the ladder pack is: an
                # artifact stores whatever `reward.py` said when it was collected,
                # and the buffer outlives that. Without this a reward edit leaves
                # the reservoir training on the old objective row-for-row beside
                # the new one -- the `scale-v1` rebase, silently, every time.
                record = relabel(records[index])
                # Off by default, and off for a reason measured rather than argued.
                # On `terminal-selfplay-ladder-v1` 81% of the learner's own
                # episodes ended insolvent and the ramp weighted each at 0.10 --
                # the buffer that should have taught "buying land and sheep on
                # day 10 bankrupts you" was down-weighted to noise, and the gate
                # arena then recorded 0.98 land purchases an episode at day 10.
                # Insolvent seasons are the signal here, not the nuisance.
                terminal_weight = (
                    _terminal_quality_weight(float(record.money[-1]))
                    if terminal_quality_weighting
                    else 1.0
                )
                record.generation_weight = generation_weight
                record.terminal_quality_weight = terminal_weight
                record.episode_weight = generation_weight * terminal_weight
                records_by_id[key] = record
    buffer.load_planned(recent_ids, reservoir_ids, evicted_seen, records_by_id)
    return skipped_evicted


class FreeMarketCollector:
    """Roll out exploration episodes for one closed league round."""

    def collect(self, context, reader, writer) -> list[str]:
        config = context.config
        count = int(config.get("batches_per_round", 1))
        if count < 1:
            raise ValueError("batches_per_round must be positive")
        episodes = int(config.get("episodes_per_batch", 16))
        if episodes < 2 or episodes % 2:
            raise ValueError("episodes_per_batch must be a positive even number")

        # Which of the slot's members owns this round. The schedule is contiguous
        # and derived from `round_index` alone, so member `k` draws the same rounds
        # in every generation and the mapping needs no state and no query.
        members = member_count(config)
        per_member = _member_batches(config)
        position = member_for_sequence(int(context.round_index), per_member, members)
        entries = read_index(context.artifacts)
        entry = find_member(entries, position)

        # Cold start: no index yet, so mint one. Every member begins from the same
        # `initial_network` -- the silent policy -- and they separate immediately,
        # because from this round on each collects and trains on its own batches
        # under its own noise draws. Seeding them differently would be a second
        # variable on top of the one this change is testing.
        if entry is None or not entry.inference_path or not Path(entry.inference_path).is_file():
            weights, _trainer = member_paths(context.artifacts, position)
            weights.parent.mkdir(parents=True, exist_ok=True)
            if not weights.is_file():
                initial_network(config).save(weights)
            entry = MemberEntry(
                position=position,
                lineage_id=new_lineage_id(context.model_id, position, 0),
                inference_path=str(weights),
                batch_floor=0,
                generation=int(getattr(context, "generation_index", 0) or 0),
            )
            others = [item for item in entries if item.position != position]
            write_index(context.artifacts, [*others, entry])

        checkpoint = None
        behavior_path = Path(entry.inference_path)

        total_transitions = int(
            reader.fetchone(
                """
                SELECT coalesce(sum(CAST(metadata_json ->> 'transitions' AS BIGINT)), 0)
                FROM training_batches WHERE model_id = %s
                """,
                [context.model_id],
            )[0]
        )
        trace_root = (
            Path(context.artifacts)
            / "episode_traces"
            / context.generation_id
            / f"round-{context.round_index:04d}"
        )

        seed = int(config.get("seed", 0))
        rng = np.random.default_rng(seed + total_transitions)
        gamma = float(config.get("gamma", 0.999))
        noisy = _noisy(config)
        jobs = int(config.get("jobs", 1))

        # Directed exploration. The counts live beside the behaviour policy and
        # under the *model*, not the checkpoint: they describe what this trainer
        # has sampled, which is a fact about the run rather than about any one set
        # of weights, and a checkpoint-keyed file would reset the novelty of every
        # action on each promotion.
        rare_gain = float(config.get("rare_sigma_gain", 0.0))
        rare_sigma_cap = float(config.get("rare_sigma_cap", rare_gain))
        rare_decay = float(config.get("rare_sigma_decay", 0.9))
        # Which novelty statistic the widening reads. `"count"` is what every run
        # through v28 played and stays the default, so leaving this unset keeps a
        # run byte-identical to those. See `rare.sigma_scale_vector`.
        rare_novelty = str(config.get("rare_sigma_novelty", "count"))
        if rare_novelty not in rare.NOVELTY_KINDS:
            raise ValueError(
                f"rare_sigma_novelty must be one of {list(rare.NOVELTY_KINDS)}: {rare_novelty}"
            )
        # Held to a minority of the pairs by default, counted from the widest end
        # of the ladder. See `collect_batch`.
        rare_sigma_pairs = int(config.get("rare_sigma_pairs", max(1, episodes // 4)))
        rare_path = Path(context.artifacts) / "rare_counts" / "visits.npz"
        rare_counts = rare.load(rare_path) if rare_gain > 0.0 else None

        base = [str(item) for item in config.get("opponents", [])]
        if config.get("league_opponents", True):
            base = list(dict.fromkeys([*base, *league_opponents(context, reader)]))

        self_play_after = int(config.get("self_play_after", 0))
        episode_config = {"episodeSteps": int(config.get("episode_steps", 720))}
        # Resolved once per round rather than per batch: it comes from the last
        # update's probe, and no update runs between the batches of one round, so
        # re-reading it inside the loop would return the same row every time.
        sigma_base = (
            collection_sigma_scale(reader, context.model_id) if noisy.controlled else 1.0
        )
        written: list[str] = []
        for _ in range(count):
            batch_index = int(
                reader.fetchone(
                    "SELECT count(*) FROM training_batches WHERE model_id = %s",
                    [context.model_id],
                )[0]
            )

            # Self-play against this model's own recent checkpoints, once the run
            # has enough batches behind it to have trained anything worth
            # playing. Held to a ratio when there is a base to hold it against:
            # the promotion gate always measures the benchmarks, so a higher
            # share trains on a distribution the objective never sees.
            #
            # Gated on `batch_index`, which is what `self_play_after` counts.
            # The batch count is only known inside this loop, which is why the
            # whole opponent mix is assembled here rather than once above.
            opponents = list(base)
            if self_play_after and batch_index >= self_play_after:
                # Unbounded below, unlike the reseed era it replaces. That bound
                # existed so a slot restarted from an ancestor would not spar with
                # the trajectory it had just been restarted to escape. A member
                # pool has no such moment: every checkpoint the slot holds
                # survived a prune, so all of them are legitimate mirrors, and the
                # pool is ranked by score rather than by recency anyway.
                pool = _mixed_self_play_pool(
                    context,
                    reader,
                    int(config.get("self_play_pool", 5)),
                )
                if pool:
                    opponents = compose_opponents(
                        base,
                        pool,
                        float(config.get("self_play_ratio", 0.3)),
                        pool_offset=batch_index,
                    )

            # Resolved here rather than beside `base`, because an empty
            # `opponents` list is a legitimate configuration until the mirror
            # pool has been consulted: `compose_opponents` returns the pool
            # unmixed when there is no base, which is how pure self-play is
            # expressed.
            if not opponents:
                opponents = [_quiet_opponent(context, config)]

            # `collect_batch` draws one opponent per seed *pair*, so a batch
            # advances the list by `episodes // 2`. Without the rotation a fixed
            # start would replay the head of the list every batch and never
            # reach the tail -- where the self-play slots above land.
            opponents = rotate_opponents(opponents, batch_index * (episodes // 2))

            # One seed per pair, played twice, so the learner takes each seat.
            seeds = [
                int(value)
                for value in rng.integers(0, 2**31 - 1, size=episodes // 2)
                for _ in range(2)
            ]
            # One scale per seed pair, so a batch carries both exploratory and
            # near-greedy data. `sigma_base` is what the last probe resolved, so
            # actor 0 plays at the configured rate and the spread runs down from
            # there; the identity policy carries the base only, since folding
            # sixteen per-actor scales in would mint eight revisions a batch.
            #
            # One ladder, downward only: widening the whole policy bankrupts the
            # wallet before rare behaviour becomes affordable. `rare_sigma_gain`
            # widens single rows, and escaping a stuck trajectory is the
            # population's job.
            if not noisy.enabled:
                sigma_scales = None
            else:
                sigma_scales = [
                    sigma_base * scale
                    for scale in actor_sigma_scales(episodes // 2, noisy.actor_alpha)
                ]
            learner = _learner_policy(behavior_path, gamma, sigma_base)
            rollouts = collect_batch(
                behavior_path,
                opponents,
                seeds,
                gamma,
                trace_root,
                jobs,
                exploration_seed=seed + total_transitions,
                policy_revision_id=learner.revision_id,
                config=episode_config,
                noise_cadence=noisy.collect_cadence,
                sigma_scales=sigma_scales,
                # Only once the counts exist. The first batch of a run has nothing
                # to be rare *against* -- every row is equally unvisited, so the
                # widening is uniform and buys no direction at all while still
                # loosening 97 spending rows. It arrives on batch two.
                rare_sigma_gain=rare_gain if rare_counts is not None else 0.0,
                rare_sigma_cap=rare_sigma_cap,
                rare_sigma_novelty=rare_novelty,
                rare_sigma_path=rare_path if rare_counts is not None else None,
                rare_sigma_pairs=rare_sigma_pairs,
            )
            if not rollouts:
                raise RuntimeError("every free-market rollout episode failed")
            records = [rollout.record for rollout in rollouts]

            if rare_gain > 0.0:
                rare_counts = rare.fold(rare_counts, rare.visit_counts(records), rare_decay)
                rare.save(rare_path, *rare_counts)

            # Record each rollout as an ordinary league pairing. Without this
            # the episodes exist only inside a batch `.npz`: nothing lands in
            # `episodes`/`episode_players`, the agent's `evaluate_episode` hook
            # never fires, and none of it is answerable in SQL.
            artifact_ids: list[str] = []
            for rollout in rollouts:
                artifact_ids.extend(
                    writer.record_policy_episode(
                        context=context,
                        pairing_kind="free_market_rollout",
                        pairing_index=batch_index * (episodes // 2) + rollout.pair_index,
                        left=learner,
                        right=_opponent_policy(
                            rollout.opponent_spec,
                            role=POLICY_KINDS["training_opponent"],
                            prefix="free-market-opponent",
                        ),
                        # Seat order as played: orientation 0 is (learner,
                        # opponent), which is the learner holding seat 0.
                        orientation=rollout.learner_seat,
                        result=rollout.match,
                    )
                )

            source_id = f"rollout-{batch_index:08d}"
            path = Path(context.artifacts) / "training_batches" / f"{source_id}.npz"
            save_episode_batch(records, path)
            transitions = sum(record.decisions for record in records)
            orders_placed = sum(record.orders_placed for record in records)
            order_diagnostics = _records_diagnostics(records)
            final_money = [float(record.money[-1]) for record in records]
            played = [rollout.opponent_spec for rollout in rollouts]
            # Recorded whether or not the widening is on: the point of the columns is
            # to compare a run that has it against the runs that did not.
            batch_chosen, batch_opportunities = rare.visit_counts(records)
            # `buy_land_rate` and `land_gate_open_rate` were both read off the
            # `BUY_LAND` column and are gone with it (v44). Neither is a loss.
            # They were already the same event measured twice -- the mask was the
            # decision, so a row that opened was a row that was chosen -- and the
            # replay-derived block below answers the question they were standing in
            # for, off the quadrant count rather than off the request. Read
            # `land_buying_episodes` where `buy_land_rate` was read.
            #
            # This says what the purchase then bought: when it was made, what it cost, how
            # many of the 25 new tiles were ever planted, and whether the seat's
            # hands went idle afterwards. Without it a falling money curve cannot
            # be attributed to expansion being negative rather than to the field
            # scheduler never working the tiles.
            land_use = _rollout_behavior(rollouts).as_dict()
            batch = writer.record(
                source_kind=SOURCE_KIND,
                source_id=source_id,
                sample_count=len(records),
                artifact_path=path,
                artifact_sha256=_file_digest(path),
                metadata={
                    "episodes": len(records),
                    "transitions": transitions,
                    "turns": sum(record.steps for record in records),
                    "noise_cadence": noisy.collect_cadence,
                    # The setpoint the ladder hangs from, and the spread around it
                    # that the actors actually played. Recorded apart so a batch
                    # says both what exploration was aimed at and what each pair
                    # got. Every rung is at or below `sigma_base`.
                    "sigma_base": sigma_base if noisy.enabled else None,
                    "actor_sigma_scales": sigma_scales,
                    "seeds": seeds,
                    # The two numbers that separate this agent's failure modes
                    # from each other. All-zero orders is the NOOP collapse;
                    # orders on nearly every decision with money at the floor is
                    # the bankruptcy collapse. Recorded per batch so either is
                    # visible in SQL before it has eaten a whole run.
                    "orders_placed": orders_placed,
                    "orders_per_decision": orders_placed / max(1, transitions),
                    "order_diagnostics": order_diagnostics.as_dict(),
                    # v2's own axis, and the one neither number above can see: a
                    # policy that stops before ordering and one that fills all ten
                    # slots both report a plausible `orders_per_decision`. 1.0 is the
                    # STOP collapse, `MAX_DECISIONS_PER_TURN` is slot stuffing, and a
                    # healthy run sits between them. Weighted by turns rather than
                    # averaged over episodes, so a short episode cannot swing it.
                    "decisions_per_turn": transitions
                    / max(1, sum(record.steps for record in records)),
                    # And the shape behind that mean, which is the number that
                    # would have caught v2's collapse. v2 finished reporting 5.71
                    # above -- squarely "healthy, between them" -- while its real
                    # distribution was 40% of turns on 1 and 47% on the cap. A mean
                    # over a bimodal split reports the gap between the modes, so
                    # the average above cannot see the failure it was added for.
                    #
                    # Index i is the count of turns holding i + 1 sub-decisions.
                    "decisions_per_turn_histogram": [
                        sum(counts)
                        for counts in zip(
                            *(record.decisions_per_turn_histogram for record in records)
                        )
                    ]
                    if records
                    else [],
                    # Turns that queued a buy and a sale of the same item -- the
                    # zero-P&L round trip v2 spent 47% of its turns on.
                    # `order_space.feasible_actions` now shuts it structurally, so
                    # this is the check that the mask holds rather than a dial:
                    # anything but zero means a wash is reachable again.
                    "wash_turns": sum(record.wash_turns for record in records),
                    # The same round trip with its two legs on different turns, which
                    # the mask does not shut and `wash_turns` cannot see. This is the
                    # dial: it read the whole of `market-free-splitladder-v1`'s
                    # behaviour while `wash_turns` sat at zero every batch.
                    "lagged_wash_turns": (
                        sum(record.lagged_wash_turns for record in records)
                        / max(1, len(records))
                    ),
                    # What the widening was worth this batch and whether it moved
                    # anything. Entropy is the gain-free reading -- a policy
                    # pouring every turn into three rows reads low here whatever
                    # its money curve says. Land was the row that motivated the
                    # mechanism and is no longer a row at all, so the question that
                    # started this is now answered by `land_buying_episodes`.
                    "rare_sigma_gain": rare_gain if rare_gain > 0.0 else None,
                    # Which statistic produced that gain's multipliers. The two are
                    # not calibrated alike, so a money series is only comparable
                    # across batches that agree on this column.
                    "rare_sigma_novelty": rare_novelty if rare_gain > 0.0 else None,
                    # Counted from the widest end of the ladder, so the clean arm
                    # is the greedy tail. A money series filters on both columns
                    # together.
                    "rare_sigma_pairs": rare_sigma_pairs if rare_gain > 0.0 else None,
                    "selection_entropy": rare.selection_entropy(batch_chosen),
                    "land_accepted": land_use["land_accepted"],
                    "land_buying_episodes": land_use["land_episodes"],
                    "land_day": land_use["land_day"],
                    "land_money": land_use["land_money"],
                    # Against 25 tiles per quadrant. Near zero means bought land
                    # is never worked, which is a scheduler problem, not a Q one.
                    "land_tiles_planted_per_purchase": land_use["land_tiles_planted_per_purchase"],
                    "land_weed_digs_per_episode": land_use["land_weed_digs_per_episode"],
                    "post_land_idle_delta": land_use["post_land_idle_delta"],
                    # The lot ledger's side of the same question. The land columns
                    # above say what expansion did; these say what
                    # a *seed* did, which is what `LOT_POTENTIAL_WEIGHT` pays
                    # against. Holding turns falling while the margin holds is the
                    # term working; spend rising with `unrecovered_seed_share` is
                    # it buying more rather than earlier, which the potential
                    # cannot cause -- look at `LAND_PURCHASE_BONUS` for that.
                    "seed_lots": land_use["seed_lots"],
                    "seed_lots_matched": land_use["seed_lots_matched"],
                    "seed_lots_matched_share": land_use["seed_lots_matched_share"],
                    "seed_spend": land_use["seed_spend"],
                    "seed_proceeds": land_use["seed_proceeds"],
                    "seed_margin": land_use["seed_margin"],
                    "seed_holding_turns": land_use["seed_holding_turns"],
                    "unrecovered_seed_share": land_use["unrecovered_seed_share"],
                    # The coins the two non-invariant lot legs moved on buy turns.
                    # `unrecovered_seed_share` is the same fact as a ratio; the coins
                    # say how much money it was. Read as a pair: the credit rising
                    # while the charge falls is the policy trading better, both
                    # rising together is only volume.
                    "seed_waste_cost": land_use["seed_waste_cost_per_episode"],
                    "seed_gain_credit": land_use["seed_gain_credit_per_episode"],
                    # The same pair for the herd. `animal_lots` sitting at zero while
                    # the seat is buying animals means the trace never reached the
                    # ledger; `animal_waste_cost` climbing with `animal_gain_credit`
                    # flat means the animals are being bought and not placed, or
                    # bought past the day they could still yield.
                    "animal_lots": land_use["animal_lots"],
                    "animal_lots_matched": land_use["animal_lots_matched"],
                    "animal_lots_matched_share": land_use["animal_lots_matched_share"],
                    "animal_spend": land_use["animal_spend"],
                    "animal_proceeds": land_use["animal_proceeds"],
                    "animal_margin": land_use["animal_margin"],
                    "animal_holding_turns": land_use["animal_holding_turns"],
                    "unrecovered_animal_share": land_use["unrecovered_animal_share"],
                    "animal_waste_cost": land_use["animal_waste_cost_per_episode"],
                    "animal_gain_credit": land_use["animal_gain_credit_per_episode"],
                    # The third pair, for stock bought off the market to resell. These
                    # lots were 89% of this agent's turnover and appeared in no column
                    # here until the bought-stock reward pair landed beside them (the
                    # pair is gone; the columns stay), which is how a policy that
                    # grossed 213,280 coins an episode and kept 1,843 of them looked
                    # healthy on every other number.
                    #
                    # `stock_resale_share` is the one to read: bought units resold
                    # rather than fed to the herd or spread on the tiles, 0.84 on
                    # `market-free-splitladder-v1`. Falling while
                    # `order_diagnostics.trading_rate` holds is the intended shape;
                    # both falling together is the NOOP collapse.
                    "stock_lots": land_use["stock_lots"],
                    "stock_spend": land_use["stock_spend_per_episode"],
                    "stock_margin": land_use["stock_margin"],
                    "stock_resale_margin": land_use["stock_resale_margin_per_episode"],
                    "stock_resale_share": land_use["stock_resale_share"],
                    "stock_units": land_use["stock_units_per_episode"],
                    "stock_waste_cost": land_use["stock_waste_cost_per_episode"],
                    "stock_gain_credit": land_use["stock_gain_credit_per_episode"],
                    # The same three, split by species. Splatted rather than named,
                    # because the roster is `donor.ANIMALS` and this file must not
                    # restate it -- a fourth animal appears here on its own.
                    #
                    # This is the column set v1 did not have, and the reason a herd
                    # regression ran for 54 generations unnoticed: `animal_spend`
                    # climbed from 160 to 6,100 coins an episode while goose purchases
                    # fell from 10.3 an episode to 0.05 and EGG sales to zero. In
                    # aggregate that reads as a growing herd. It was a monoculture.
                    **{
                        key: value
                        for key, value in land_use.items()
                        if key.startswith(("animals_", "animal_lots_", "animal_margin_"))
                        and key
                        not in (
                            "animal_lots_matched",
                            "animal_lots_matched_share",
                        )
                    },
                    "mean_final_money": float(np.mean(final_money)),
                    "max_final_money": float(np.max(final_money)),
                    "solvent_episodes": int(sum(1 for value in final_money if value > 10_000)),
                    "behavior_checkpoint_id": (
                        checkpoint.checkpoint_id if checkpoint is not None else None
                    ),
                    # Which member collected this batch. `member_lineage_id` is the
                    # one the replay filter reads -- a member that survives a prune
                    # into a different position keeps its history because the
                    # identity travels with the lineage, not with the slot in the
                    # round-robin. `member_position` is the schedule, and a
                    # diagnostic.
                    "member_position": position,
                    "member_lineage_id": entry.lineage_id,
                    "behavior_policy_revision_id": learner.revision_id,
                    "behavior_policy_kind": POLICY_KINDS["market_learner"],
                    "policy_kinds": POLICY_KINDS,
                    "session_id": context.session_id,
                    "generation_id": context.generation_id,
                    "round_index": context.round_index,
                    # Which opponents actually played, so a collapse can be
                    # traced to the mix rather than guessed at.
                    "opponents": played,
                    "self_play_episodes": sum(1 for spec in played if "?weights=" in spec),
                },
            )
            writer.link_artifacts(batch.batch_id, artifact_ids)
            total_transitions += transitions
            written.append(source_id)
        return written


class FreeMarketSelector:
    """Ranks a slot's members at the generation boundary and prunes to the best N.

    Runs from `on_round(scope="generation")`, which `league.py` fires from
    `_close_generation_training` -- after every update in the generation has
    completed and before the generation is marked COMPLETE. So all `N` of this
    generation's checkpoints and their scores exist by the time this reads them.

    That hook is **not** a barrier: `run_round_hooks` fans it out over every model
    on a thread pool, so ten copies of this run at once with no shared lock. It is
    safe only because a prune is slot-local -- it reads this slot's checkpoints and
    writes this slot's `members/`, and touches nothing another slot can reach.
    """

    def select(self, context, reader) -> dict[str, Any]:
        config = context.config
        members = member_count(config)
        candidates = roster_candidates(reader, context.model_id, members)
        generation = int(getattr(context, "generation_index", 0) or 0)
        if not candidates:
            # Nothing scored yet -- a generation that collected but never updated.
            # Leaving the index untouched is what lets the next generation resume
            # the members `update` already wrote.
            LOGGER.info("%s generation %s: no prune candidates", context.model_id, generation)
            return {"action": "skipped", "candidates": 0, "members": members}

        cap = config.get("member_lineage_cap")
        cutoff = max(int(item.get("cutoff_batch_seq") or 0) for item in candidates)

        # Cross-slot rescue. Off by default, because it trades the roster's
        # independent-replicate property -- which is what carried
        # `market-free-rewardtogo-v2`, where two of ten slots finished above 11,000 and
        # eight below 7,000 -- for the ability to restart a slot that is going nowhere.
        # `admit_foreign` holds both gates; this only decides how many may land.
        adopted: list[dict[str, Any]] = []
        if bool(config.get("cross_slot_reseed", False)):
            # One below `members` by default, so a slot always keeps a lineage of its
            # own and the roster cannot converge on this generation's leader in a
            # single prune.
            # `""` is the manifest's unset value, as with `member_lineage_cap`;
            # `int("")` raises, so it is resolved to the default rather than passed on.
            configured_adopt = config.get("cross_slot_reseed_max_adopt", "")
            max_adopt = (
                max(1, members - 1)
                if configured_adopt in (None, "")
                else int(configured_adopt)
            )
            adopted = admit_foreign(
                candidates,
                foreign_candidates(reader, context.model_id, members, max_adopt * 4),
                min_score=float(config.get("cross_slot_reseed_min_score", 0.0)),
                margin=float(config.get("cross_slot_reseed_margin", 0.25)),
                max_adopt=max_adopt,
            )
            if adopted:
                LOGGER.info(
                    "%s generation %s: adopting %d foreign checkpoint(s) %s",
                    context.model_id,
                    generation,
                    len(adopted),
                    [
                        (item.get("source_model_id"), item.get("score"))
                        for item in adopted
                    ],
                )

        survivors = rank_and_prune(
            candidates + adopted,
            members,
            model_id=context.model_id,
            generation=generation,
            cutoff=cutoff,
            lineage_cap=None if cap in (None, "", 0) else int(cap),
        )

        # Materialize each survivor into its position's slot before the index names
        # it. `collect` reads `members/<k>.npz` by path and never resolves a
        # checkpoint, so the copy is what makes the next generation roll out the
        # policy the prune actually chose.
        for entry in survivors:
            weights, trainer = member_paths(context.artifacts, entry.position)
            weights.parent.mkdir(parents=True, exist_ok=True)
            if entry.inference_path and Path(entry.inference_path) != weights:
                shutil.copyfile(entry.inference_path, weights)
            if entry.trainer_path and Path(entry.trainer_path) != trainer:
                shutil.copyfile(entry.trainer_path, trainer)

        write_index(
            context.artifacts,
            [
                replace(entry, inference_path=str(member_paths(context.artifacts, entry.position)[0]),
                        trainer_path=str(member_paths(context.artifacts, entry.position)[1]))
                for entry in survivors
            ],
        )
        summary = {
            "action": "pruned",
            "candidates": len(candidates),
            "foreign_adopted": len(adopted),
            "foreign_sources": [
                {"model_id": item.get("source_model_id"), "score": item.get("score")}
                for item in adopted
            ],
            "members": members,
            "generation": generation,
            "cutoff": cutoff,
            "kept": [
                {
                    "position": entry.position,
                    "lineage_id": entry.lineage_id,
                    "checkpoint_id": entry.checkpoint_id,
                    "score": entry.score,
                    "forked_from": entry.forked_from,
                }
                for entry in survivors
            ],
            "distinct_lineages": len({entry.lineage_id for entry in survivors}),
        }
        LOGGER.info(
            "%s generation %s: kept %s of %s, %s distinct lineages, best %s",
            context.model_id,
            generation,
            len(survivors),
            len(candidates),
            summary["distinct_lineages"],
            survivors[0].score if survivors else None,
        )
        return summary


class FreeMarketUpdater:
    def update(
        self, context: ModelUpdateContext, reader: ModelTrainingReader
    ) -> ModelUpdateResult:
        config = context.config
        device = resolve_device(str(config.get("device", "auto")))
        torch.manual_seed(int(config.get("seed", 0)))
        # The discount replay applies over the `n_step` horizon; also part of the
        # policy identity string and of the demonstration loaders' signatures.
        gamma = float(config.get("gamma", 0.999))
        # Fallbacks come from `replay.py` rather than being restated here.
        # Restated, they went stale.
        buffer = ReplayBuffer(
            capacity=int(config.get("buffer_episodes", DEFAULT_CAPACITY)),
            gamma=gamma,
            n_step=int(config.get("n_step", DEFAULT_N_STEP)),
            seed=int(config.get("seed", 0)),
            reservoir_fraction=float(
                config.get("reservoir_fraction", DEFAULT_RESERVOIR_FRACTION)
            ),
            priority_alpha=float(config.get("priority_alpha", DEFAULT_PRIORITY_ALPHA)),
            priority_beta=float(config.get("priority_beta", DEFAULT_PRIORITY_BETA)),
            generation_weight_power=float(
                config.get("generation_weight_power", DEFAULT_GENERATION_WEIGHT_POWER)
            ),
            generation_weight_floor=float(
                config.get("generation_weight_floor", DEFAULT_GENERATION_WEIGHT_FLOOR)
            ),
        )

        # Which member this update is training, and the lineage it belongs to.
        # Read before anything else because the lineage and the batch floor it
        # carries decide what the buffer is even allowed to see.
        members = member_count(config)
        per_member = _member_batches(config)
        position = member_for_cutoff(int(context.cutoff_batch_seq), per_member, members)
        member_entries = read_index(context.artifacts)
        member = find_member(member_entries, position)
        member_lineage = (
            member.lineage_id
            if member is not None
            else new_lineage_id(context.model_id, position, 0)
        )
        member_floor = member.batch_floor if member is not None else 0
        member_weights, member_trainer = member_paths(context.artifacts, position)

        # This member's own history and nothing else. Two filters, and they answer
        # different questions: the lineage keeps a member off its siblings' data,
        # so the five searches stay independent and the prune ranks learning rather
        # than luck; the floor keeps a *fork* off its parent's data, so a position
        # refilled from a strong lineage starts empty instead of inheriting a
        # trajectory it never played.
        #
        # `landgate-v10` is the record of skipping the floor: a restored 18k
        # checkpoint trained immediately on ~400 episodes generated by the 357-coin
        # policy that had just been thrown away.
        usable = [
            batch
            for batch in context.all_batches
            if batch.sequence > member_floor and _batch_lineage(batch) == member_lineage
        ]
        skipped_evicted_batches = fill_buffer(
            buffer,
            usable,
            terminal_quality_weighting=bool(config.get("terminal_quality_weighting", False)),
        )

        # Demonstrations, pinned so the ring cannot flush them. Read from stored
        # league replays and converted into this action space; no donor code
        # runs at inference, and the agent's own policy is unaffected.
        # Demonstrations are loaded *after* the network is built and resumed, not
        # here. A cloned episode's stored hidden states have to be rolled with a
        # GRU, and the only correct one is the policy this update is about to train
        # -- which does not exist yet at this point in the function. See the
        # `load_demonstrations(..., gru=...)` call below. v2 loaded them here
        # because a flat replay row needed nothing but the replay.
        demo_stats = None
        demo_limit = int(config.get("demonstration_episodes", 0))

        # Demonstrations count toward this, which is why it is not tested until
        # after they are pinned -- see the check below `buffer.add(demos, ...)`.
        # v2 loaded them before this point and so could take the test here; v3
        # cannot, because rolling their hidden states needs a network that does not
        # exist yet, and leaving the test here would refuse a demo-seeded cold
        # start that v2 trained.
        warmup = int(config.get("warmup_episodes", 16))

        noisy = _noisy(config)
        # `gru_hidden` is read here rather than defaulted silently: it was declared
        # in the manifest with a sizing rationale while nothing consumed it, so a
        # sweep of the recurrence's width would have changed the documentation and
        # not the network. It still defaults to the constant the numpy mirror
        # ships, and a checkpoint at a different width is refused by name on load.
        gru_hidden = int(config.get("gru_hidden", net_module.HIDDEN_SIZE))
        online = net_module.build(
            hidden_size=gru_hidden, noisy=noisy.build_kind, sigma0=noisy.sigma0
        ).to(device)
        target = net_module.build(
            hidden_size=gru_hidden, noisy=noisy.build_kind, sigma0=noisy.sigma0
        ).to(device)
        optimizer = torch.optim.Adam(
            online.parameters(),
            lr=float(config.get("learning_rate", 1e-4)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        # Its own optimizer, not the TD one. Cross-entropy over 186 classes and a
        # Huber TD residual have very different gradient scales, and Adam's
        # second moment is a running average: sharing one optimizer lets the
        # clone phase set the step sizes the fine-tuning that follows starts from.
        #
        # Sigma is excluded from it. Noise strictly hurts a classification fit, so
        # the clone's cheapest answer is to drive sigma to zero -- and the clone
        # phase runs *before* the TD loop, so it could zero out exploration before
        # a single TD step has seen it.
        clone_optimizer = torch.optim.Adam(
            [
                parameter
                for name, parameter in online.named_parameters()
                if "sigma" not in name
            ],
            lr=float(config.get("behavior_clone_learning_rate", config.get("learning_rate", 1e-4))),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        gradient_steps = 0
        behavior_clone_steps = 0
        # A reseeded slot resumes from the perturbed artifact rather than from
        # its own lineage, and takes no optimizer state with it. Adam's moments
        # are a running average over the gradients of the trajectory that just
        # ended; carrying them into a slot whose weights were deliberately
        # redrawn would damp the very first steps of the new search with the
        # history of the old one. The step counters restart for the same reason.
        resumed = _member_checkpoint(member)
        resumed_from = resumed.checkpoint_id if resumed is not None else None
        # `resumed is None` covers two different events -- a genuine cold start,
        # and a snapback on a run with nothing released to snap back to. Only the
        # second is a reset, and `rejections_since_promotion` is what separates
        # them. Recording it in `gate_json` is not just for reading later: the
        # streak query stops at a recorded reset, so without this the limit trips
        # again on every subsequent update.
        reset_limit = int(config.get("resume_reset_after_rejections", 0))
        rejection_streak = (
            resumed.rejection_streak
            if resumed is not None
            else rejections_since_promotion(reader, context.model_id)
        )
        lineage_reset = (
            resumed.reset
            if resumed is not None
            else reset_limit > 0 and rejection_streak >= reset_limit
        )
        if resumed is not None:
            state = torch.load(
                resumed.trainer_artifact_path, map_location="cpu", weights_only=False
            )
            noisy_migration = _load_online_state(online, state["online"])
            _load_online_state(target, state["target"])
            if noisy_migration:
                # Deliberately not loaded. Sigma adds tensors to the parameter
                # group, so `optimizer.load_state_dict` raises on the count
                # regardless of how the module state dict was remapped. This is
                # the same event `reset_optimizer_moments` exists to express --
                # the run is leaving a regime behind -- and it happens once per
                # lineage.
                move_optimizer(optimizer, device)
                move_optimizer(clone_optimizer, device)
            else:
                optimizer.load_state_dict(state["optimizer"])
                move_optimizer(optimizer, device)
                clone_optimizer.load_state_dict(state["clone_optimizer"])
                move_optimizer(clone_optimizer, device)
            # `load_state_dict` restores the lr the checkpoint was written with,
            # so without this a config change to `learning_rate` is silently
            # ignored for the entire life of a resumed lineage. The config is the
            # authority; the checkpoint only carries the moments.
            _reset_learning_rates(optimizer, clone_optimizer, config, None)
            if bool(config.get("reset_optimizer_moments", False)):
                # Adam's second moment is a running average over the gradients of
                # the policy being abandoned. Dropping it makes the first steps
                # after a config change large again instead of damped by the
                # history of a regime that no longer applies.
                optimizer.state.clear()
                clone_optimizer.state.clear()
            gradient_steps = int(state["gradient_steps"])
            behavior_clone_steps = int(state["behavior_clone_steps"])
            torch.set_rng_state(state["torch_rng_state"])
        else:
            noisy_migration = False
            # `collect` mints the member's weights on its first round and rolls them
            # out, so the file is normally already here and this update starts from
            # the same bytes the episodes were played with. `initial_network` is the
            # fallback for a member whose file went missing under it.
            seed_network = (
                OrderQNetwork.load(member_weights)
                if member_weights.is_file()
                else initial_network(config)
            )
            online = net_module.from_numpy(seed_network, online)
            target.load_state_dict(online.state_dict())
            _reset_learning_rates(optimizer, clone_optimizer, config, None)

        if demo_limit > 0:
            # Rolled with the policy that is about to train on them, so the pinned
            # pool carries no staleness at all -- unlike the self-play segments,
            # whose stored states were written by whatever the policy was when the
            # episode was collected. Strictly better than storing a demo's hidden
            # state once, and it costs one rollout an update. Both sources below
            # get the same treatment: the ladder pack's stored column is a
            # placeholder that `load_ladder_demonstrations` re-rolls.
            demo_gru = net_module.to_numpy(online).gru
            source = str(config.get("demonstration_source", "league"))
            if source == "ladder":
                # Off disk, not out of PostgreSQL. These are Kaggle ladder replays
                # mirrored by `scripts/pull_kaggle_episodes.py`; they are not league
                # episodes and deliberately have no rows in `episode_players`, so
                # the SQL path cannot see them and nothing that counts league
                # episodes counts them.
                pack = str(config.get("demonstration_pack", "")).strip()
                if not pack:
                    raise ValueError(
                        "demonstration_source = 'ladder' needs demonstration_pack; "
                        "build one with agents.dqn_market_free_v3c.training.ladder"
                    )
                demos, demo_stats = load_ladder_demonstrations(
                    pack, gru=demo_gru, limit=demo_limit
                )
            elif source == "league":
                demos, demo_stats = load_demonstrations(
                    reader,
                    limit=demo_limit,
                    min_final_money=float(config.get("demonstration_min_money", 10_000.0)),
                    family_id=config.get("demonstration_family"),
                    gamma=gamma,
                    gru=demo_gru,
                )
            else:
                raise ValueError(
                    f"unknown demonstration_source {source!r}; expected 'league' or 'ladder'"
                )
            buffer.add(demos, pinned=True)

        # Own episodes only. `buffer.episodes` includes the pinned demonstrations,
        # which satisfy any warmup on their own, and a member whose lineage filter
        # left it nothing would then run its whole update on ladder data and write
        # a checkpoint nothing distinguishes from a trained one.
        own_episodes = len(buffer.recent) + len(buffer.reservoir)
        if own_episodes < warmup:
            raise ValueError(
                f"free-market training requires at least {warmup} own replay episodes; "
                f"have {own_episodes} (pinned demonstrations do not count)"
            )

        # `reduction="none"`: every row carries its own importance-sampling
        # weight from the final-money-weighted draw, and a mean taken inside the loss
        # would average them away before they could apply.
        loss_fn = nn.SmoothL1Loss(reduction="none")
        # Chunks now, not rows. 32 chunks x 40 turns x ~3 decisions is ~3,840
        # gradient rows -- fifteen times v2's 256 -- but from 32 independent
        # episode draws rather than 256, so the *independent* content of a step is
        # far smaller than the row count suggests.
        batch_size = int(config.get("batch_size", 32))
        burn_in = int(config.get("burn_in_turns", replay_module.DEFAULT_BURN_IN))
        train_turns = int(config.get("train_turns", replay_module.DEFAULT_TRAIN))
        # An update is ~480 gradient steps now rather than ~4,000, so v2's 2000
        # would sync the target once every four updates. This keeps the two syncs
        # an update v2 had, and is the single easiest value to leave stale across
        # the fork.
        target_sync = int(config.get("target_sync", 250))
        grad_clip_norm = float(config.get("grad_clip_norm", 5.0))
        demo_fraction = float(config.get("demo_fraction", 0.25)) if buffer.pinned else 0.0
        # Scale the optimizer budget by the data the update actually consumes,
        # as the scorer does. A fixed count meant one update spent 100 steps on
        # `min_new_batches` worth of rollouts -- 20 batches, ~230k decisions --
        # and left `target_sync` needing twenty updates to fire once.
        batches_per_update = max(
            1, int(config.get("batches_per_update", len(context.new_batches)))
        )
        steps_to_run = int(config.get("gradient_steps_per_batch", 48)) * batches_per_update
        losses: list[float] = []
        # Pre-clip gradient norms. New in v3 and not a nicety: BPTT over
        # `train_turns` GRU steps is the most likely new failure and v2 measured
        # the gradient nowhere, so a run held together entirely by the clip would
        # look identical to a healthy one.
        grad_norms: list[float] = []
        # Rows per chunk, which makes the decision-rate bias the two-level mean
        # cancels visible rather than merely handled.
        rows_per_chunk: list[float] = []
        unranked_batches = 0
        # Gradient steps that actually ran, as opposed to `steps_to_run`: a draw
        # that produced no rows is skipped, and the metric has to say so.
        steps_taken = 0
        # The same per-chunk residual without the importance weights -- the
        # two-level mean up to but not including `batch.weight`. `mean_loss` is
        # the quantity actually optimised and has to stay that, but it changes
        # scale the moment weighting turns on, so a loss curve spanning the change
        # would read as a step that never happened. This one is comparable to
        # every generation recorded before it. Not a row mean: that would differ
        # from `mean_loss` by the whole decision-rate normalisation, not by the
        # weights.
        unweighted_losses: list[float] = []
        # Effective sample size of the weight vector, as a fraction of the batch.
        # 1.0 is an unweighted batch; 1/batch_size is a batch whose gradient is one
        # chunk.
        #
        # The denominator is the *chunk* count in v3, not the row count, so this is
        # not comparable with the same metric under v2: 8 out of 32 chunks is a far
        # worse state than 30 out of 256 rows was.
        weight_ess: list[float] = []
        clone_losses: list[float] = []
        # Divergence detector. The label is reward-to-go in units of
        # `reward.MONEY_SCALE`, so a correct Q is bounded by the best season in
        # the buffer -- under 2 for this policy, ~6 for the ladder pack -- and
        # anything in double digits is the bootstrap chain, not a scale. `mean_loss` alone
        # cannot say whether a rising number is a harder target or an exploding
        # one, and by the time the gate rejects the candidate the update that
        # started it is several checkpoints back on the lineage.
        q_magnitudes: list[float] = []
        q_peak = 0.0
        # Rows whose bootstrap value fell outside `bootstrap_clamp`. A few per
        # step is the bias being caught; a large share means the head has
        # already diverged and the clamp is holding it up. Accumulated on the
        # device and read once after the loop: an `int()` per step is a
        # host sync per step.
        bootstrap_clamp = _bootstrap_clamp(config)
        value_clamped_rows = torch.zeros((), dtype=torch.int64, device=device)
        horizon_clamped_rows = 0
        started = time.perf_counter()
        # Per update, as the key says. This used to multiply by
        # `batches_per_update`, so the driver's 12 ran 120 hinge steps against
        # 480 TD steps -- a fifth of the gradient budget on imitation, followed
        # by a target sync from the cloned head, on every update.
        clone_steps_to_run = _behavior_clone_steps_per_update(config) if buffer.pinned else 0
        # Silenced for the clone phase by default: see `clone_optimizer` above.
        # Fitting a 186-way cross-entropy through a stochastic representation
        # teaches the trunk to cancel its own noise, which is not what any of
        # this is for.
        clone_margin = float(
            config.get("behavior_clone_margin", BEHAVIOR_CLONE_MARGIN)
        )
        if clone_steps_to_run:
            net_module.set_noise(online, noisy.enabled and noisy.clone_noise)
        for _ in range(clone_steps_to_run):
            clone_losses.append(
                _behavior_clone_step(
                    online,
                    clone_optimizer,
                    buffer,
                    batch_size,
                    device,
                    burn_in,
                    train_turns,
                    grad_clip_norm,
                    clone_margin,
                )
            )
            behavior_clone_steps += 1
        if clone_steps_to_run:
            # Bootstrap from the cloned policy rather than a stale pre-clone
            # target network during the DQN fine-tuning that follows.
            target.load_state_dict(online.state_dict())

        for _ in range(steps_to_run):
            batch = buffer.sample_sequences(
                batch_size,
                burn_in=burn_in,
                train=train_turns,
                demo_fraction=demo_fraction,
            )
            unranked_batches += 0 if batch.ranked else 1
            if batch.state.shape[0] == 0:
                continue
            # Training turns whose bootstrap did not land on the turn `n_step`
            # ahead. Must stay zero; see `replay.sample_sequences`.
            horizon_clamped_rows += batch.clamped

            # The online rollout, graded on the training window only. The burn-in
            # inside `_roll_chunk` is `no_grad`, which is what keeps BPTT the
            # length of the training window rather than of the whole chunk.
            train_hidden, post_online = _roll_chunk(
                online, batch, device, burn_in, train_turns, grad=True
            )

            # The bootstrap is computed *before* the graded head forward, which is
            # a reordering NoisyNet forces rather than a preference. Rainbow draws
            # independent noise for the action-selection pass, the evaluation pass
            # and the graded pass, and `reset_noise` rebinds the epsilon buffers --
            # so a resample landing between `predicted`'s forward and its
            # `backward` would rebind a tensor autograd saved for the gradient.
            #
            # v3 appears to break that rule, because the graded *recurrent* rollout
            # above happens before the two `reset_noise` calls below. It is legal
            # only because `OrderQModule` builds the GRU with no `NoisyLinear` in
            # it, so nothing in the recurrent graph references an epsilon buffer.
            # Put noise in the cell and this ordering silently becomes wrong;
            # `test_the_recurrence_is_never_noisy` is what stands between the two.
            with torch.no_grad():
                # The target burns in from the *same stored state* with its *own*
                # parameters, rather than being handed the online net's latent.
                # Copying it across would evaluate the target's head on the online
                # net's representation, which is not Double DQN -- it is one
                # network wearing two heads, and the decorrelation the target
                # network exists to provide is gone. The failure is silent: the
                # target simply tracks the online net.
                _train_t, post_target = _roll_chunk(
                    target, batch, device, burn_in, train_turns, grad=False
                )
                boot = torch.from_numpy(batch.next_turn).to(device)
                boot_state = torch.from_numpy(batch.next_state).to(device)
                # The bootstrap maximises over what the *next* state actually
                # allows. Leaving it unmasked would value the successor by an
                # order the simulator would refuse, which is the same fiction the
                # acting policy is no longer permitted to believe.
                boot_mask = torch.from_numpy(batch.next_mask).to(device)
                # Double DQN: online picks the argmax, target scores it.
                if noisy.enabled:
                    net_module.reset_noise(online)
                best = online(torch.cat((boot_state, post_online[boot]), dim=1))
                best = best.masked_fill(~boot_mask, MASKED_SCORE).argmax(dim=1, keepdim=True)
                # The target scores the online net's pick with its noise *off*.
                # Rainbow resamples here, but Rainbow's chain is one step per
                # reward; this one runs up to eleven intra-turn rows at discount
                # 1.0 with no reward between them, so a noisy `max` compounds
                # its own optimism eleven times before a label touches it.
                # `terminal-selfplay-ladder-v1` read `max_abs_q` 22-51 on three
                # slots against labels that never reached 0.1 at the old scale
                # (under 2 at this one). Double DQN's decorrelation is the
                # *online* argmax above, which stays noisy. This is the target's
                # only forward -- `_roll_chunk` touches the GRU alone, which
                # carries no `NoisyLinear` -- so the target is deterministic
                # everywhere it is read, and nothing re-arms it after a sync.
                if noisy.enabled:
                    net_module.set_noise(target, False)
                bootstrap = (
                    target(torch.cat((boot_state, post_target[boot]), dim=1))
                    .gather(1, best)
                    .squeeze(1)
                )
                # And then bounded. A correct value cannot exceed the best season
                # in the buffer nor fall below the most cash a position held;
                # anything past the bound is the max-over-noise bias, not
                # information.
                discount = torch.from_numpy(batch.discount).to(device)
                if bootstrap_clamp is not None:
                    low, high = bootstrap_clamp
                    # Counted only where the bootstrap reaches the target. A row
                    # whose episode ended inside the horizon carries discount 0
                    # and a zero successor state, and its arbitrary Q would
                    # otherwise dominate the count.
                    outside = ((bootstrap < low) | (bootstrap > high)) & (discount > 0)
                    value_clamped_rows += outside.sum()
                    bootstrap = bootstrap.clamp(low, high)
                # Per decision, not per turn. An intra-turn row carries reward 0
                # and discount 1, so its target is purely `max_a Q(s_next, a)`
                # against the same latent -- the "having queued that, what is the
                # next order worth?" link. The row that closes the turn carries
                # the turn's n-step return and `gamma ** n_step` against the
                # latent of the turn `n_step` ahead.
                goal = torch.from_numpy(batch.reward).to(device) + discount * bootstrap

            if noisy.enabled:
                net_module.reset_noise(online)
            rows = torch.from_numpy(batch.decision_turn).to(device)
            state_tensor = torch.from_numpy(batch.state).to(device)
            chosen = torch.from_numpy(batch.chosen).to(device)
            hidden_rows = train_hidden.reshape(-1, online.hidden_size)[rows]
            predicted = (
                online(torch.cat((state_tensor, hidden_rows), dim=1))
                .gather(1, chosen.unsqueeze(1))
                .squeeze(1)
            )
            per_row = loss_fn(predicted, goal)

            # A two-level mean, and the level that matters is the inner one.
            #
            # v2 could take a plain row mean because it drew a decision uniformly
            # *within* an episode, so a chatty episode's extra sub-decisions bought
            # it no extra gradient mass -- which is exactly what `_own_priorities`
            # means by "episode weighting is transition weighting here". A chunk
            # draw is uniform over *turns*, and a turn holds one to eleven rows, so
            # under a row mean an episode trading eleven orders a turn would
            # contribute eleven times the mass of one that had collapsed to STOP.
            # That tilts the objective toward precisely the behaviour
            # `decisions_per_turn` exists to watch for.
            #
            # `turn_share` is 1/rows-on-that-turn, so `index_add_` averages within
            # a turn; the mean over the window then averages across turns, and the
            # objective is invariant to the decision rate.
            share = torch.from_numpy(batch.turn_share).to(device)
            per_turn = torch.zeros(
                batch_size * train_turns, device=device, dtype=per_row.dtype
            ).index_add_(0, rows, per_row * share)
            per_chunk = per_turn.view(batch_size, train_turns).mean(dim=1)
            # `.mean()`, not `/ weight.sum()`. The weights are normalised to a
            # maximum of 1, so a plain mean is bounded above by the unweighted
            # loss and the effective step can only ever shrink. Renormalising by
            # the weight sum would put the scale back and hand the correction a
            # way to enlarge the step on a batch that happened to draw badly.
            loss = (per_chunk * torch.from_numpy(batch.weight).to(device)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Backpropagating through `train_turns` GRU steps is a new way to
            # explode that v2 had no exposure to, and v2 observed the gradient norm
            # nowhere at all. Recorded pre-clip so a run that is being held together
            # by the clip says so.
            grad_norm = nn.utils.clip_grad_norm_(online.parameters(), grad_clip_norm)
            optimizer.step()
            grad_norms.append(float(grad_norm))
            losses.append(float(loss.detach()))
            unweighted_losses.append(float(per_chunk.detach().mean()))
            rows_per_chunk.append(batch.state.shape[0] / float(batch_size))
            # Numpy, so this costs no GPU sync. The denominator is the *chunk*
            # count now, not the row count: an effective sample size of 8 out of 32
            # chunks is a far worse state than 30 out of 256 rows was, so the two
            # numbers are not comparable across the fork.
            row_weight = batch.weight.astype(np.float64)
            weight_ess.append(
                float(row_weight.sum() ** 2 / (np.square(row_weight).sum() * batch_size))
            )
            with torch.no_grad():
                magnitude = predicted.detach().abs()
                q_magnitudes.append(float(magnitude.mean()))
                q_peak = max(q_peak, float(magnitude.max()))
            gradient_steps += 1
            steps_taken += 1
            if gradient_steps % target_sync == 0:
                # Copies mu and sigma. The epsilon buffers are non-persistent, so
                # the online net's current draw does not come across; and the
                # target is silenced before its only forward anyway.
                target.load_state_dict(online.state_dict())
        train_seconds = time.perf_counter() - started
        bootstrap_clamped_rows = int(value_clamped_rows.item())

        # How often the perturbation actually changed the decision, measured on
        # one held-out minibatch of the states the policy really visits. This is
        # the quantity epsilon named directly, and the only one that says whether
        # sigma is still buying exploration or has quietly annealed to nothing.
        #
        # Swept across a ladder of sigma multipliers rather than measured once at
        # the learned sigma. One reading says whether exploration is still there;
        # the curve says what it would take to put it back, which is what
        # `collect` needs to choose a scale. `SIGMA_LADDER[3]` is 1.0, so the
        # learned-sigma reading is still in here and still reported under its own
        # key.
        flip_rate: float | None = None
        flip_curve: dict[float, float] | None = None
        order_curve: dict[float, float] | None = None
        flip_to_order: float | None = None
        if noisy.enabled and len(buffer.episodes):
            # Unprioritised on purpose. This probe resolves sigma against the
            # states the actor actually visits, and a final-money-weighted draw would
            # resolve the controller against a distribution no actor ever sees --
            # invisibly, since the flip rate comes back plausible either way.
            probe = buffer.sample_sequences(
                max(1, batch_size // 4),
                burn_in=burn_in,
                train=train_turns,
                demo_fraction=0.0,
                prioritized=False,
            )
            # The latent each probe row was actually scored against, produced by
            # rolling the recurrence over the chunk exactly as the TD step does --
            # burn-in included. Scoring the head against a zero latent instead
            # would resolve sigma against states no actor visits, which is the same
            # objection the unprioritised draw above exists to answer.
            probe_features, probe_mask = _probe_features(
                online, probe, device, burn_in, train_turns
            )
            flip_curve, order_curve = flip_rate_curve(
                online,
                probe_features,
                probe_mask,
                noisy.scale_limit,
                # Same floor `resolve_sigma_scale` is handed below. Passing one and
                # not the other is what made `noisy_sigma_scale_floor` inert: the
                # resolver was allowed down to the floor and the probe never
                # measured there, so it could only ever answer with the lowest rung
                # the ladder happened to carry.
                scale_floor=noisy.scale_floor,
            )
            flip_rate = flip_curve.get(1.0)
            flip_to_order = order_curve.get(1.0)

        # Chosen here because this is where the network is; `collect` reads it back
        # out of `metrics_json` next round. Recorded whether or not the controller
        # is on, and both curves are recorded either way, so a run that switches
        # targets can see what the other would have said.
        sigma_scale_target = (
            resolve_sigma_scale(
                order_curve or {},
                noisy.order_rate_target,
                noisy.scale_limit,
                noisy.scale_floor,
            )
            if noisy.controlled
            else 1.0
        )

        checkpoint_id = hashlib.sha256(
            f"{context.model_id}\0{context.update_id}\0{context.dataset_sha256}".encode()
        ).hexdigest()[:24]
        destination = context.artifacts / "checkpoints"
        destination.mkdir(parents=True, exist_ok=True)
        trainer_path = destination / f"{checkpoint_id}.pt"
        inference_path = destination / f"{checkpoint_id}.npz"
        torch.save(
            {
                "online": online.state_dict(),
                "target": target.state_dict(),
                "optimizer": optimizer.state_dict(),
                "clone_optimizer": clone_optimizer.state_dict(),
                "gradient_steps": gradient_steps,
                "behavior_clone_steps": behavior_clone_steps,
                "torch_rng_state": torch.get_rng_state(),
                "dataset_sha256": context.dataset_sha256,
                "device": str(device),
                # v2 checked widths on the numpy export and never on the `.pt`.
                # Under a recurrence a coincidental width match is more dangerous,
                # because `HIDDEN_SIZE + STATE_DIM` is one integer away from
                # colliding with another configuration's head width -- so the
                # schema is written down and `_load_online_state` refuses a
                # mismatch rather than reporting a missing-key list.
                "schema": {
                    "state_dim": STATE_DIM,
                    "gru_hidden": online.hidden_size,
                    "actions": ACTION_COUNT,
                },
                # The chunk geometry these weights were fitted under. Not read on
                # resume -- the config is the authority there, as it is for the
                # learning rate -- but recorded, because changing `burn_in_turns`
                # or `train_turns` mid-run changes what the recurrence was trained
                # to do and nothing else on disk would say so.
                "chunk": {
                    "burn_in": burn_in,
                    "train": train_turns,
                    "hidden_stride": HIDDEN_STRIDE,
                },
            },
            trainer_path,
        )
        exported = net_module.to_numpy(online)
        _reject_inert_network(exported)
        exported.save(inference_path)

        eval_seeds = int(config.get("eval_seeds", 40))
        seed_base = int(config.get("promotion_seed_base", 1_000_000))
        benchmark_seed_stride = int(config.get("benchmark_seed_stride", 0))
        benchmark_seed_start = seed_base + context.cutoff_batch_seq * benchmark_seed_stride
        jobs = int(config.get("jobs", 1))

        # One arena answers both questions this gate still asks: did the candidate
        # complete a benchmark, and did it trade at all? Money is recorded here for
        # offline selection, not for online rejection.
        benchmarks = configured_benchmarks(config)
        benchmark = benchmarks[0] if benchmarks else None
        # Two tiers, because arenas are the slow part and there are now `members`
        # updates a generation per slot rather than one. Every member pays a cheap
        # arena to earn its rank; only a member that beats the slot's standing best
        # on that cheap read pays the full one, because only such a member can
        # possibly be a release candidate.
        #
        # At 5 members that is 5 x 20 + 200 = 300 seed-pairs a generation against
        # the 1,000 a full arena per member would cost, for a ranking whose only
        # job is to order five siblings.
        rank_seeds = max(1, int(config.get("member_rank_eval_seeds", eval_seeds)))
        rank_result = (
            _arena_evaluate(inference_path, benchmark, rank_seeds, jobs, benchmark_seed_start)
            if benchmark is not None
            else None
        )
        if rank_result is not None:
            _reject_degenerate(rank_result, "member rank")
        rank_money = _benchmark_money(
            {"challenger_benchmarks": [rank_result]} if rank_result else None, benchmark
        )

        # The bar is the best *rank* score the slot's current members hold, so the
        # two sides of the comparison are the same measurement on the same number
        # of seeds. Comparing a 20-seed candidate against a 100-seed incumbent
        # would reject on the seed count rather than on the policy.
        contender_bar = max(
            (
                float(item.score)
                for item in member_entries
                if item.position != position and item.score is not None
            ),
            default=None,
        )
        contends = (
            rank_money is not None
            and (contender_bar is None or rank_money >= contender_bar)
        )

        if rank_seeds >= eval_seeds or rank_result is None:
            # The cheap tier is the full tier; do not pay twice.
            challenger_benchmarks = [rank_result] if rank_result is not None else []
        elif contends:
            challenger_benchmarks = [
                _arena_evaluate(inference_path, benchmark, eval_seeds, jobs, benchmark_seed_start)
            ]
        else:
            # Not a release candidate this generation. The rank arena still stands
            # as its score, and the gate below rejects it on the missing benchmark
            # rather than on a measurement nobody paid for.
            challenger_benchmarks = []
        for result in challenger_benchmarks:
            _reject_degenerate(result, "benchmark")

        promotion_rule = str(config.get("promotion_rule", "activity_gate"))
        if promotion_rule not in _PROMOTION_RULES:
            raise ValueError(
                f"promotion_rule must be one of {sorted(_PROMOTION_RULES)}: {promotion_rule}"
            )
        promotion_score_signal = _promotion_score_signal(config)

        own = [record for record in buffer.recent + buffer.reservoir if record.decisions]
        own_diagnostics = _records_diagnostics(own)
        # The FIFO segment alone, which is a different question from `own` and has
        # to be asked separately. `own` is what the gradient saw and includes the
        # reservoir -- a uniform sample of everything evicted since the reseed --
        # so `own_mean_final_money` is a third historical by construction. That is
        # correct for "what was trained on" and wrong for "how is the policy
        # doing", which is what `lr_score_signal = "rollout"` reads: measured on
        # this session, `own_mean_final_money` ran 4,493 where the ten batches just
        # collected ran 8,570, and `_rollout_score` then medians that over eleven
        # updates. The plateau detector was grading a policy several updates old.
        recent = [record for record in buffer.recent if record.decisions]
        recent_mean_final_money = (
            float(np.mean([record.money[-1] for record in recent])) if recent else None
        )

        active = context.active_checkpoint
        benchmark_result = challenger_benchmarks[0] if challenger_benchmarks else None
        activity = (
            _order_viability(benchmark_result, config, require_activity=True)
            if benchmark_result is not None
            else {
                "passed": False,
                "checks": {},
                "trading_rate": 0.0,
                "rejected_order_rate": 0.0,
                "wire_orders_per_decision": 0.0,
                "min_orders_per_decision_required": True,
            }
        )
        benchmark_score = {
            "opponent": benchmark,
            "candidate_money": _benchmark_money(
                {"challenger_benchmarks": challenger_benchmarks}, benchmark
            )
            if benchmark is not None
            else None,
            "seed_start": benchmark_seed_start,
            "seeds": eval_seeds,
        }
        rejection_reasons: list[str] = []
        if benchmark_result is None:
            rejection_reasons.append("missing_benchmark")
        elif benchmark_result.get("errors"):
            rejection_reasons.append("benchmark_errors")
        elif int(benchmark_result.get("pairs", 0)) <= 0:
            rejection_reasons.append("no_benchmark_pairs")
        if not activity["passed"]:
            rejection_reasons.append("idle_or_invalid_activity")
        activity_passed = not rejection_reasons
        best_money, best_checkpoint_id = _best_promoted_checkpoint_score(
            reader,
            context.model_id,
            promotion_score_signal,
            window=int(config.get("promotion_reference_window", 0)),
        )
        candidate_money = (
            benchmark_score.get("candidate_money")
            if promotion_score_signal == "benchmark"
            else recent_mean_final_money
        )
        promotion_tolerance = _promotion_tolerance(
            config, promotion_score_signal, benchmark_result, best_money
        )
        activation_reasons = _activation_reasons(
            activity_passed=activity_passed,
            candidate_money=candidate_money,
            best_money=best_money,
            score_signal=promotion_score_signal,
            tolerance=promotion_tolerance,
        )
        activate = not activation_reasons
        gate: dict[str, Any] = {
            "kind": "activity_best_gate",
            "policy_kinds": POLICY_KINDS,
            "promotion_rule": promotion_rule,
            "promotion_score_signal": promotion_score_signal,
            "benchmark_seed_start": benchmark_seed_start,
            "benchmark_score": benchmark_score,
            "activation": {
                "passed": activate,
                "score_signal": promotion_score_signal,
                "candidate_score": candidate_money,
                "best_promoted_score": best_money,
                "candidate_money": candidate_money,
                "best_promoted_money": best_money,
                "best_promoted_checkpoint_id": best_checkpoint_id,
                "tolerance_money": promotion_tolerance,
                "reference_window": int(config.get("promotion_reference_window", 0)),
                "reasons": activation_reasons,
            },
            "challenger_benchmarks": challenger_benchmarks,
            "resumed_from_checkpoint_id": resumed_from,
            "rejection_streak": rejection_streak,
            "lineage_reset": lineage_reset,
            "incumbent_checkpoint_id": active.checkpoint_id if active is not None else None,
            "activity": {
                **activity,
                "min_trading_rate": _limit(config, "viability_min_orders_per_decision"),
                "reasons": rejection_reasons,
            },
            "passed": activate,
        }
        evaluations: list[ModelEvaluation] = []
        if benchmark_result is not None:
            evaluations.append(ModelEvaluation(_benchmark_kind(0), benchmark_result, activate))
        metrics = {
            "policy_kinds": POLICY_KINDS,
            "promotion_score_signal": promotion_score_signal,
            "resumed_from_checkpoint_id": resumed_from,
            "resumed_from_released": resumed is not None and resumed.released,
            "rejection_streak": rejection_streak,
            "lineage_reset": lineage_reset,
            "episodes_in_buffer": len(buffer.episodes),
            "pinned_demonstrations": len(buffer.pinned),
            "demo_fraction": demo_fraction,
            "n_step": buffer.n_step,
            "bootstrap_horizon_clamped_rows": horizon_clamped_rows,
            # Read off the buffer rather than the config, so these say what the
            # draw was actually built with and not what was asked for.
            "priority_alpha": buffer.priority_alpha,
            "priority_beta": buffer.priority_beta,
            "generation_weight_power": buffer.generation_weight_power,
            "generation_weight_floor": buffer.generation_weight_floor,
            # The health check is one line: `sampled_mean_priority_money` strictly
            # above `own_mean_priority_money`, with `mean_weight_effective_fraction` above
            # ~0.8. Equal means alpha is inert, which is exactly how
            # `demo_fraction` failed -- it moved no number, so nobody noticed.
            # A collapsing ESS means alpha is too high and the batch has become a
            # handful of episodes.
            **buffer.priority_stats(),
            "mean_weight_effective_fraction": (
                float(np.mean(weight_ess)) if weight_ess else None
            ),
            # Batches whose every episode had already aged out of both buffer
            # segments, so their artifact was never opened.
            "skipped_evicted_batches": skipped_evicted_batches,
            "transitions_in_buffer": buffer.transitions,
            "buffer_megabytes": buffer.nbytes() / 1e6,
            "gradient_steps": gradient_steps,
            "gradient_steps_this_update": steps_taken,
            "gradient_steps_skipped": steps_to_run - steps_taken,
            "behavior_clone_steps": behavior_clone_steps,
            "behavior_clone_steps_this_update": clone_steps_to_run,
            "mean_behavior_clone_loss": (
                float(np.mean(clone_losses)) if clone_losses else None
            ),
            "mean_loss": float(np.mean(losses)) if losses else None,
            # `mean_loss` is the weighted objective actually optimised. This is
            # the same per-chunk residual without the importance weights, so a
            # loss curve spanning the arrival of return weighting stays one
            # readable series. `mean_loss <= mean_unweighted_loss` always, the
            # weights being normalised to a maximum of 1.
            "mean_unweighted_loss": (
                float(np.mean(unweighted_losses)) if unweighted_losses else None
            ),
            # Read beside `mean_loss`. Both rising together is divergence; loss
            # rising while these hold is a target that genuinely got harder. See
            # the bound argued where they are collected.
            "mean_abs_q": float(np.mean(q_magnitudes)) if q_magnitudes else None,
            "max_abs_q": q_peak if q_magnitudes else None,
            "bootstrap_value_clamped_rows": bootstrap_clamped_rows,
            "bootstrap_clamp": list(bootstrap_clamp) if bootstrap_clamp is not None else None,
            # --- the recurrence ------------------------------------------------
            #
            # Pre-clip gradient norms. BPTT over `train_turns` GRU steps is v3's
            # most likely new failure and v2 measured the gradient nowhere, so a
            # run held together entirely by the clip would look healthy.
            "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else None,
            "max_grad_norm": float(np.max(grad_norms)) if grad_norms else None,
            # Gradient rows per chunk. The two-level mean cancels the decision-rate
            # bias in the loss; this is what makes the bias itself visible, and it
            # doubles as a read on `decisions_per_turn` from the training side.
            "mean_rows_per_chunk": (
                float(np.mean(rows_per_chunk)) if rows_per_chunk else None
            ),
            # Gradient steps whose own-episode draw fell back to uniform because
            # some episode in the buffer was shorter than a training window. Not
            # visible anywhere else: `priority_*` is computed over the unfiltered
            # pool and keeps reporting the law this draw did not use.
            "unranked_sequence_batches": unranked_batches,
            # How much of the stored hidden state the burn-in did not reconstruct,
            # split by pool because the two staleness regimes are entirely
            # different: the FIFO segment turns over inside one generation, while
            # a reservoir episode's stored state can be tens of updates old and in
            # a coordinate system the current GRU no longer uses. An average over
            # both hides whichever one is failing.
            **_recurrence_metrics(online, buffer, device, burn_in, train_turns),
            # Sigma collapse is silent: exploration stops, the run goes
            # greedy-only, and nothing else in this dict moves. `noisy_argmax_
            # flip_rate` is the number directly comparable to epsilon -- the
            # share of states where the perturbation actually changed the
            # decision -- and `sigma_ratio_*` says whether sigma is still there
            # to do it.
            **({"noisy_migration": True} if noisy_migration else {}),
            **(_sigma_metrics(online) if noisy.enabled else {}),
            "noisy_argmax_flip_rate": flip_rate,
            # The same measurement across the sigma ladder, and the scale
            # `collect` will play at next round. Keys are stringified because
            # this lands in JSON, where an object key cannot be a float.
            "noisy_flip_rate_curve": (
                {f"{scale:g}": rate for scale, rate in flip_curve.items()}
                if flip_curve
                else None
            ),
            "noisy_sigma_scale_target": sigma_scale_target,
            # Reads on the NOOP attractor, which the flip rate does not: a
            # collapsed policy sits on a flat plateau where flips are cheap and
            # plentiful but almost none of them buy anything.
            "noisy_flip_to_order_rate": flip_to_order,
            "noisy_flip_to_order_curve": (
                {f"{scale:g}": rate for scale, rate in order_curve.items()}
                if order_curve
                else None
            ),
            "train_seconds": train_seconds,
            "action_count": ACTION_COUNT,
            # Collapse detectors. This is what the policy *emitted*; read it
            # beside `own_order_diagnostics.trading_rate`, which is what the
            # simulator accepted. A wide gap between the two is the disguised
            # NOOP collapse -- ordering every turn and being refused every turn.
            "own_orders_per_decision": (
                sum(record.orders_placed for record in own)
                / max(1, sum(record.decisions for record in own))
            ),
            "own_mean_final_money": (
                float(np.mean([record.money[-1] for record in own])) if own else None
            ),
            # The same money over the FIFO segment only. This is what the slot's
            # plateau detector reads; `own_mean_final_money` stays as it was so a
            # series spanning the change is still one readable curve and the two
            # can be compared in SQL. The gap between them is the reservoir's
            # share, so it also says how much history the buffer is carrying.
            "recent_episodes": len(recent),
            "recent_mean_final_money": recent_mean_final_money,
            "own_solvent_fraction": (
                float(np.mean([record.money[-1] > 10_000 for record in own])) if own else None
            ),
            "own_order_diagnostics": own_diagnostics.as_dict(),
        }
        # Lot accounting for the batches this update consumed, lifted off their
        # metadata rather than recomputed. Same names as the batch columns
        # deliberately: a second spelling for the same quantity would be one more
        # thing to know.
        #
        # Merged with a collision check rather than splatted into the literal above.
        # A `**` at the end of a dict literal wins, so the day someone adds an
        # `animal_spend` of their own up there it would be replaced by this one
        # silently and the column would quietly start meaning something else. There
        # is no overlap today; this is what keeps it that way.
        lot_metrics = _batch_lot_metrics(context.new_batches)
        collisions = sorted(lot_metrics.keys() & metrics.keys())
        if collisions:
            raise RuntimeError(
                "batch lot metrics collide with update metrics: " + ", ".join(collisions)
            )
        metrics.update(lot_metrics)
        if demo_stats is not None:
            metrics["demonstrations"] = {
                # Which pool these came from. "ladder" is the Kaggle mirror off
                # disk and "league" the stored league replays; the two are
                # different opponents entirely, so a metrics row that did not say
                # which would not be comparable across runs -- and this roster
                # runs both halves at once.
                "source": str(config.get("demonstration_source", "league")),
                # Pinned episodes are never evicted, so they are a permanent
                # addition to the buffer `buffer_megabytes` sizes. A ladder seat
                # trades hard and its record is large; without this the pinned
                # share of a 4 GB buffer is invisible.
                "megabytes": sum(record.nbytes() for record in buffer.pinned) / 1e6,
                "episodes": demo_stats.episodes,
                "decisions": demo_stats.decisions,
                "orders_seen": demo_stats.orders_seen,
                "orders_mapped": demo_stats.orders_mapped,
                "mapped_fraction": demo_stats.mapped_fraction,
                "orders_unrepresentable": demo_stats.orders_unrepresentable,
                # Neither mapped nor lost: `BUY_LAND` has no column in the action
                # space because `should_buy_land` owns that decision off the head.
                # Reported so the three buckets add up to `orders_seen`.
                "orders_land": demo_stats.orders_land,
                # The feasibility mask being narrower than real play. Against
                # league demonstrations this should stay near zero -- a solvent
                # demonstrator's orders were accepted by the simulator. Against
                # the ladder it does not, and that is a measurement rather than a
                # fault: ladder agents buy quantity rungs past the room
                # `pending.seed_room` allows, and `clone` forces the bit so the
                # label survives. The cost is that this share of the imitation
                # gradient points at rungs the argmax cannot reach in play. Watch
                # it per source: a rise on the league half means the mask moved.
                "orders_masked_infeasible": demo_stats.orders_masked_infeasible,
                "order_diagnostics": _records_diagnostics(buffer.pinned).as_dict(),
            }
        # --- This member's bookkeeping -------------------------------------
        #
        # Deliberately last, and it writes only files inside this slot's own
        # artifacts root. The generation hook that prunes runs concurrently across
        # every slot with no shared lock, so nothing here may touch state another
        # slot can also reach.
        # The cheap arena, always — so every member has a comparable score even
        # when it never contended for release and ran no full arena.
        # Ranked on money, but *tiered* on activity: an inert member sorts below
        # every trading sibling regardless of what it banked.
        #
        # Money alone is the wrong sole key here and it is the pressure that
        # compounded `market-free-members-v2` into silence. A policy that never
        # trades keeps its stake, so on money it beats a sibling that traded and
        # lost -- generation 0's rosters scored 3000/2993/2778 (3000 is the stake)
        # and generation 2's were 2806/850/516/140/13. The release gate already
        # refuses these, on this same floor, 97 times in 105. The prune did not,
        # so the pool bred exactly what the gate rejects and the fork-from-survivor
        # step propagated it.
        #
        # `rank_active` is None when no rank arena ran, which is distinct from
        # False: `_rank_key` keeps unscored members available to fill a position
        # rather than dropping a lineage over one failed arena, and an unmeasured
        # member must not be demoted below a measured-inert one on missing
        # evidence.
        member_rank_score = rank_money
        member_rank_active = (
            bool(_order_viability(rank_result, config, require_activity=True)["passed"])
            if rank_result is not None
            else None
        )
        net_module.to_numpy(online).save(member_weights)
        member_trainer.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(trainer_path, member_trainer)

        member_record = {
            "position": position,
            "members": members,
            "lineage_id": member_lineage,
            "forked_from": member.forked_from if member is not None else None,
            "batch_floor": member_floor,
            "rank_score": member_rank_score,
            "rank_active": member_rank_active,
            "rank_eval_seeds": rank_seeds,
            "rank_contends": contends,
            "rank_bar": contender_bar,
            "full_arena": bool(challenger_benchmarks),
            "resumed": resumed is not None,
            # What the lab actually handed this update against what the round-robin
            # was told to expect. They disagree when `member_batches` and the
            # `[[models]]` key `min_new_batches` have drifted apart, which the
            # trainer cannot check any other way -- `min_new_batches` is not in
            # `context.config`. A mismatch means members are training on each
            # other's batches, and this is the only place it is visible.
            "batches_expected": per_member,
            "batches_received": len(context.new_batches),
            "batches_trained": len(usable),
        }

        # Carried on the checkpoint, because the prune reads exactly these three
        # off `metrics_json` and nothing else: which lineage the row belongs to,
        # where its history starts, and what it scored.
        metrics["member_lineage_id"] = member_lineage
        metrics["member_batch_floor"] = member_floor
        if member_rank_score is not None:
            metrics["member_rank_score"] = float(member_rank_score)
        if member_rank_active is not None:
            metrics["member_rank_active"] = bool(member_rank_active)
        metrics["member"] = member_record
        gate["member"] = member_record

        # The index entry for this position, so a prune that never runs -- a
        # generation cut short, a single-member roster -- still leaves the member
        # resumable from what it just trained.
        write_index(
            context.artifacts,
            [
                *[item for item in member_entries if item.position != position],
                MemberEntry(
                    position=position,
                    lineage_id=member_lineage,
                    checkpoint_id=checkpoint_id,
                    score=member_rank_score,
                    batch_floor=member_floor,
                    generation=member.generation if member is not None else 0,
                    inference_path=str(member_weights),
                    trainer_path=str(member_trainer),
                    forked_from=member.forked_from if member is not None else None,
                ),
            ],
        )

        combined_digest = hashlib.sha256(
            trainer_path.read_bytes() + inference_path.read_bytes()
        ).hexdigest()
        return ModelUpdateResult(
            checkpoint_id,
            str(trainer_path),
            str(inference_path),
            combined_digest,
            metrics,
            gate,
            activate,
            tuple(evaluations),
        )
