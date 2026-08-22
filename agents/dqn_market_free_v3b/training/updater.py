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
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import torch
from torch import nn

from agents.dqn_market_free_v3b.diagnostics import (
    BehaviorCounts,
    OrderDiagnostics,
    behavior_from_replay,
)
from agents.dqn_market_free_v3b.lots import median
from agents.dqn_market_free_v3b.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free_v3b.order_space import (
    ACTION_COUNT,
    MASKED_SCORE,
    NOOP_INDEX,
)
from agents.dqn_market_free_v3b.policy import OrderQNetwork
from agents.dqn_market_free_v3b.training import network as net_module
from agents.dqn_market_free_v3b.training import rare, reseed
from agents.dqn_market_free_v3b.training import replay as replay_module
from agents.dqn_market_free_v3b.training.clone import load_demonstrations
from agents.dqn_market_free_v3b.training.ladder import load_ladder_demonstrations
from agents.dqn_market_free_v3b.training.collect import (
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
from agents.dqn_market_free_v3b.training.replay import (
    DEFAULT_CAPACITY,
    DEFAULT_N_STEP,
    DEFAULT_PRIORITY_ALPHA,
    DEFAULT_PRIORITY_BETA,
    DEFAULT_RESERVOIR_FRACTION,
    ReplayBuffer,
)
from agents.dqn_market_scorer.training.device import move_optimizer, resolve_device
from agents.dqn_market_scorer.training.progress import _opponent_policy, _policy_identity

# Action-agnostic machinery, shared on purpose. See the module docstring.
from agents.dqn_market_scorer.training.updater import (
    _benchmark_kind,
    _file_digest,
    _reject_degenerate,
    compose_opponents,
    configured_benchmarks,
    league_opponents,
    lineage_checkpoint,
    policy_improvement,
    rejections_since_promotion,
    rotate_opponents,
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
LEARNER_MANIFEST = "agents/dqn_market_free_v3b/match_agent.toml"
SOURCE_KIND = "free_market_rollout"
_PROMOTION_RULES = {"significant", "point_estimate", "always", "benchmark_ratchet"}
#: Which measurement the learner slot's plateau detector reads. See `_slot_score`.
_SCORE_SIGNALS = {"rollout", "benchmark"}

VIABILITY_DEFAULTS = {
    "viability_min_mean_money_ratio": 0.80,
    "viability_min_median_money_ratio": 0.75,
    # The gate's other half: a policy that never trades keeps its money perfectly,
    # so the floors above wave the NOOP attractor through. Applied only once there
    # is a released incumbent -- a cold start is legitimately near-silent.
    "viability_min_orders_per_decision": 0.02,
    # What the money floors measure against before any release. Not the quiet
    # baseline's seat: it keeps the full stake, so `0.80 *` it asks a candidate to
    # trade for a season and lose nothing.
    "viability_cold_start_reference_money": 1500.0,
    # The cold-start floors' own escape hatch, separate from the ratchet's because
    # the two gates fail independently -- standing one down does nothing for a slot
    # the other is holding. 0 leaves it off.
    "viability_grace_after_rejections": 0,
    # The cold-start reference under that grace; unset means
    # `benchmark_cold_start_min_money`. Only the *reference* moves -- both ratios
    # still bind and the activity floor is untouched, or this becomes
    # `promotion_rule = "always"` for any slot that has rejected enough times.
    "viability_grace_cold_start_reference_money": None,
    # The fixed-reference release bar. Absolute, because everything else in this
    # gate is relative to the previous update and therefore drifts.
    "benchmark_release_min_money": 1000.0,
    # The same bar for the first release only: a foothold, not a good policy. Below
    # it there is no incumbent, the state both anti-stall mechanisms are inert in.
    "benchmark_cold_start_min_money": 750.0,
    # The bar once ratchet grace is in force; unset means the cold-start floor.
    # Grace drops the *absolute* term rather than the relative one, because the
    # absolute is what a stalled run cannot clear -- a lineage can sit far below a
    # fixed bar while never having regressed against its own recorded reference.
    "benchmark_grace_min_money": None,
    # Non-regression against the incumbent's *recorded* benchmark money. Not 1.0: a
    # gate admitting no noise eventually admits nothing.
    "benchmark_money_ratchet": 0.9,
    # How many measurements the ratchet reference is a median over. 1 is the single
    # stored draw. Above 1 because ratcheting one draw against another is not just
    # weak but biased: the stored reference is whatever was lucky enough to promote,
    # so each release re-aims the bar at a higher quantile of unchanged noise.
    "benchmark_reference_window": 11,
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
    """The most recent checkpoints of this model, addressed as agent specs.

    Deliberately does **not** filter on `promoted`, unlike the scorer's shared
    helper: release is gated, the behaviour distribution is not. Filtering makes
    self-play unreachable exactly when it would help most, since a run that
    promotes nothing has an empty promoted pool on every batch. The scorer's helper
    is left alone, because a shared helper silently changing which checkpoints two
    agents mirror is how one stops matching its own gate.

    `after_checkpoint_id` bounds the pool below at a reseed: a restarted slot whose
    mirror is still the graduated policy trains against the trajectory it was
    restarted to escape. Returning nothing meanwhile is intended -- the caller falls
    back to the mature group.
    """

    if limit < 1:
        return []
    if after_checkpoint_id:
        rows = reader.fetchall(
            """
            SELECT inference_artifact_path FROM model_training_checkpoints
            WHERE model_id = %s AND created_at > (
                SELECT created_at FROM model_training_checkpoints
                WHERE checkpoint_id = %s
            )
            ORDER BY created_at DESC, checkpoint_id DESC LIMIT %s
            """,
            [context.model_id, after_checkpoint_id, limit],
        )
    else:
        rows = reader.fetchall(
            """
            SELECT inference_artifact_path FROM model_training_checkpoints
            WHERE model_id = %s
            ORDER BY created_at DESC, checkpoint_id DESC LIMIT %s
            """,
            [context.model_id, limit],
        )
    return [_weights_spec(str(row[0])) for row in rows if Path(str(row[0])).is_file()]


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

    return OrderQNetwork.quiet_start(seed=int(config.get("seed", 0)))


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
ARENA_REDUCER = "agents.dqn_market_free_v3b.diagnostics:arena_reduction"


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


#: The lot-accounting columns lifted out of batch metadata and onto the update.
#:
#: `reward.py` says of the animal pair: "Watch `animal_waste_cost` against
#: `animal_gain_credit` per batch." That was not something this row could answer.
#: The collector computes both and writes them to `training_batches.metadata_json`
#: (`writer.record(..., metadata=...)` below), and `model_updates.metrics_json`
#: carried nothing about lots at all -- so the one instrument the two shaping pairs
#: are documented to be read through sat in a different table from
#: `own_mean_final_money`, the gate, and every other number an update is judged on.
#: Answering "is the herd term biting" meant joining two tables by hand and knowing
#: to.
#:
#: These are the columns that answer it. The share columns come along because the
#: coins are a volume as much as a rate -- `animal_waste_cost` doubling because the
#: policy bought twice as many animals and `animal_waste_cost` doubling because it
#: recovered half as often are the same column moving for opposite reasons, and
#: `unrecovered_animal_share` is what separates them.
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

    No upper limit: over-trading can only show up as lost money, which the floors in
    `_arena_viability` already measure.
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
        "wire_orders_per_decision": int(diagnostics.get("orders", 0)) / decisions,
        "min_orders_per_decision_required": bool(require_activity and minimum_activity > 0.0),
    }


def _collapsed_against_baseline(result: dict[str, Any], config: dict[str, Any]) -> bool:
    """Did the candidate itself emit nothing in a zero-margin baseline match.

    Only meaningful once the paired margin is already known to be zero-width. The
    question is *which* of two things it is: the NOOP collapse, or a candidate that
    traded all season against a counterparty offering nothing. The margin alone
    cannot say -- round-trip trades are cash-neutral against an unchanged market, so
    a seat can place thousands of orders and still finish on exactly the stake.

    Read off `trading_rate` rather than the order count, because `HIRE n` is one
    decision whose wire form is n rows. Shares the activity floor, so setting
    `viability_min_orders_per_decision = 0` turns this off with it.
    """

    diagnostics = dict(result.get("order_diagnostics") or {})
    trading_rate = float(diagnostics.get("trading_rate", 0.0))
    return trading_rate < _limit(config, "viability_min_orders_per_decision")


def _arena_viability(
    result: dict[str, Any],
    reference: dict[str, Any],
    config: dict[str, Any],
    *,
    require_activity: bool = False,
    reference_kind: str = "incumbent_seat",
    grace: bool = False,
) -> dict[str, Any]:
    """Absolute money preservation plus order activity for one arena.

    Only applied where `reference` is the *same opponent on the same seeds*, so
    the floor compares like with like. That is the head-to-head, where the
    reference is the incumbent's own seat.

    `reference_kind` and `grace` are recorded rather than used: a grace floor and an
    ordinary one are both just numbers once written, so without them `gate_json`
    cannot say which of the two hatches moved.
    """

    order = _order_viability(result, config, require_activity=require_activity)
    mean_ratio = _limit(config, "viability_min_mean_money_ratio")
    median_ratio = _limit(config, "viability_min_median_money_ratio")
    reference_mean = max(0.0, float(reference.get("a_money_mean", 0.0)))
    reference_median = max(0.0, float(reference.get("a_money_median", 0.0)))
    mean_floor = reference_mean * mean_ratio
    median_floor = reference_median * median_ratio
    money_checks = {
        "mean_money": float(result.get("a_money_mean", 0.0)) >= mean_floor,
        "median_money": float(result.get("a_money_median", 0.0)) >= median_floor,
    }
    return {
        **order,
        "passed": order["passed"] and all(money_checks.values()),
        "checks": {**order["checks"], **money_checks},
        "a_money_mean": float(result.get("a_money_mean", 0.0)),
        "a_money_median": float(result.get("a_money_median", 0.0)),
        "reference_money_mean": reference_mean,
        "reference_money_median": reference_median,
        "mean_money_floor": mean_floor,
        "median_money_floor": median_floor,
        "reference_kind": reference_kind,
        "grace": grace,
    }


def _released_trading_rate(reader, active, opponent: str) -> float | None:
    """The released incumbent's own recorded trading rate on this benchmark.

    The activity half of `_released_benchmark_money`, read the same way: from the
    incumbent's own `gate_json`, so it costs one row and no arena time. Matched on
    the opponent spec, so reconfiguring `benchmarks` drops the term rather than
    comparing against a different opponent.

    `None` means *not recorded*, deliberately not zero: a missing measurement is not
    evidence of idleness, and treating it as such would retire every pre-existing
    member the first time the pool was culled.

    `hard_viability.benchmarks` is a **mapping keyed on the opponent spec** --
    `update` writes `dict(zip(benchmarks, benchmark_viability))` -- so it is read by
    key. Iterating it instead yields the spec strings, every `isinstance(item, dict)`
    test fails, and the whole branch falls through to the head-to-head: a different
    arena against a different opponent, silently, with a plausible number. That
    happened, and it turned the pool's activity discount into a reading of the
    incumbent match rather than the benchmark it claims to describe.
    """

    gate = _released_gate(reader, active)
    if gate is None:
        return None
    viability = gate.get("hard_viability")
    if isinstance(viability, dict):
        benchmarks = viability.get("benchmarks")
        if isinstance(benchmarks, dict):
            entry = benchmarks.get(opponent)
            if isinstance(entry, dict) and entry.get("trading_rate") is not None:
                return float(entry["trading_rate"])
        head = viability.get("head_to_head")
        if isinstance(head, dict) and head.get("trading_rate") is not None:
            return float(head["trading_rate"])
    result = _benchmark_entry(gate, opponent)
    diagnostics = (result or {}).get("order_diagnostics")
    if isinstance(diagnostics, dict) and diagnostics.get("trading_rate") is not None:
        return float(diagnostics["trading_rate"])
    return None


def _released_behavior(reader, active, opponent: str) -> dict[str, Any] | None:
    """The released incumbent's recorded strategy fingerprint on this benchmark.

    Read exactly the way `_released_trading_rate` above reads its number and for
    the same reason: `_arena_evaluate` wrote it into that update's `gate_json`
    when the incumbent was a candidate, so this costs one row and no arena time.

    `None` means *not recorded*, and the distinction from an all-zero descriptor
    is load-bearing here in a way it is not for the trading rate. A member with
    no fingerprint is never reserve-eligible, so returning zeros for one would
    file every pre-existing checkpoint into the no-land, no-crop cell and let a
    row nobody has measured occupy a reserved berth.
    """

    result = _benchmark_entry(_released_gate(reader, active), opponent)
    behavior = (result or {}).get("behavior")
    if isinstance(behavior, dict) and behavior.get("episodes"):
        return dict(behavior)
    return None


#: Upper edges of the archive's two axes. Coarse on purpose: a cell nothing can
#: reach is a reserved berth that stays empty, and the roster is ten slots, so
#: nine cells is already more structure than the population can fill.
CROP_BINS = (1, 2, 3)
LAND_BINS = (0, 1, 2)


def _behavior_cell(behavior: dict[str, Any] | None) -> tuple[int, int] | None:
    """Which archive cell a fingerprint falls in, or `None` if unmeasured.

    Two axes, both binned to their top edge so the highest bin is open-ended:
    `crop_bin` is distinct crops planted capped at 3, `land_bin` is quadrants
    bought per episode capped at 2.

    The land axis is rounded rather than truncated because it is a per-episode
    mean over the whole arena series -- a policy buying one quadrant in most of
    its episodes is running the expansion strategy, and truncation would file it
    with the policies that never buy any.
    """

    if not isinstance(behavior, dict) or not behavior.get("episodes"):
        return None
    crops = int(behavior.get("distinct_crops", 0) or 0)
    land = behavior.get("land_per_episode")
    if land is None:
        land = float(behavior.get("land", 0) or 0) / max(1, int(behavior["episodes"]))
    crop_bin = min(CROP_BINS[-1], max(CROP_BINS[0], crops))
    land_bin = min(LAND_BINS[-1], max(LAND_BINS[0], int(round(float(land)))))
    return crop_bin, land_bin


def _reserved_cell(behavior: dict[str, Any] | None) -> tuple[int, int] | None:
    """The cell a member may hold a *reserved* berth in, if any.

    `(1, 0)` -- monoculture, no expansion -- is excluded, and it is the only
    exclusion. That cell is the attractor the archive exists to escape:
    `logmoney-v26` collapsed to 1.29 distinct crops per seat by generation 15 and
    held there, on rising money, so the score bucket is already full of it.
    Reserving a berth for it would spend one of six on the thing six others
    already hold.
    """

    cell = _behavior_cell(behavior)
    if cell is None or cell == (CROP_BINS[0], LAND_BINS[0]):
        return None
    return cell


def _population_score(
    money: float | None, trading_rate: float | None, config: dict[str, Any]
) -> dict[str, Any]:
    """What a graduating checkpoint carries into the mature pool.

    Benchmark money discounted by how much the policy actually traded, because
    the pool is ranked on this number and ranked on money alone it selects for
    doing nothing. The stake is 3,000 and a policy that never trades keeps it, so
    a hoarder scores near 3,000 while a policy that trades for a season scores in
    the hundreds. Rank by raw money and the hoarder enters at the top, is never
    culled -- the cull is `ORDER BY score DESC` -- and becomes both the reseed
    source and the mirror opponent for every slot in the roster. The population
    would then propagate the collapse it exists to escape.

    On this family money and activity are anti-correlated across the roster, so the
    hoarder is not a hypothetical.

    A linear ramp to `population_activity_reference`, saturating above it, and
    deliberately not a threshold: hard viability already owns the binary "did it act
    at all" question, and repeating that shape here would only move the same cliff.
    What this adds is that *barely* clearing that floor no longer buys a hoarder the
    top of the pool. No upper penalty -- over-trading shows up as lost money, which
    the money term already carries.

    The reference is a judgement about how much activity counts as trading for real,
    set from the observed bimodality rather than from a result. 0 restores raw-money
    ranking.
    """

    base = 0.0 if money is None else float(money)
    reference = float(config.get("population_activity_reference", 0.10))
    if reference <= 0.0 or trading_rate is None:
        # Not recorded, or the discount disabled. Rank on money, and say which,
        # so a member's score can be read against the rule that produced it.
        return {
            "score": base,
            "money": base,
            "trading_rate": trading_rate,
            "activity_factor": 1.0,
            "activity_reference": reference,
            "scored_by": "money" if reference <= 0.0 else "money_unmeasured_activity",
        }
    factor = min(1.0, max(0.0, float(trading_rate) / reference))
    return {
        "score": base * factor,
        "money": base,
        "trading_rate": float(trading_rate),
        "activity_factor": factor,
        "activity_reference": reference,
        "scored_by": "money_discounted_by_activity",
    }


def _released_benchmark_money(reader, active, opponent: str) -> tuple[float | None, str | None]:
    """The released incumbent's own recorded score on this benchmark.

    A pure read of a measurement already paid for: the number was written into that
    update's `gate_json` when the incumbent was itself a candidate, so the ratchet
    costs one row and no arena time.

    Matched on the opponent spec rather than index 0, so reconfiguring `benchmarks`
    resets the ratchet instead of silently comparing against a different opponent's
    score -- the one way this could be wrong without looking wrong.

    `None` means *no ratchet term*, never "the ratchet passes". The absolute floor
    still binds, at the cold-start value.
    """

    if active is None:
        return None, None
    money = _benchmark_money(_released_gate(reader, active), opponent)
    return (None, None) if money is None else (money, active.checkpoint_id)


def _benchmark_reference_window(
    reader, model_id: str, active, opponent: str, window: int
) -> list[float]:
    """Every benchmark measurement taken while the current incumbent has stood.

    One arena series is a draw, not a number, and a wide one here: the policy is
    bankrupt on most seeds, so the mean rides on a handful of survivors.

    Ratcheting one such draw against another is a *biased* coin flip -- the stored
    reference is whatever was lucky enough to promote, so each release raises the
    bar to a higher quantile of unchanged noise and the gate tightens on itself. A
    run can then fail `no_regression` dozens of times without having regressed.

    So the reference is a median over the window. Candidates measured while an
    incumbent stands are near-neighbours of it -- under a held snapback, literally
    the same weights plus one update's gradient steps -- so they are repeated
    samples of one quantity.

    What this gives up, stated rather than hidden: a lineage that decays slowly
    and uniformly drags its own window down with it, so the ratchet follows the
    decay instead of catching it. That is the same self-reference the head-to-head
    has and the reason this gate reads the benchmark at all. It is bounded on the
    other side by the absolute floor, which is only stood down under grace, and by
    hard viability, which is never stood down.

    Ordered newest first and capped at `window`, so a long stall re-estimates on
    recent evidence rather than on a checkpoint's whole history.
    """

    if window <= 1 or active is None:
        return []
    rows = reader.fetchall(
        """
        SELECT gate_json FROM model_updates
        WHERE model_id = %s AND gate_json IS NOT NULL AND started_at >= coalesce(
            (SELECT max(started_at) FROM model_updates
             WHERE model_id = %s AND status = 'PROMOTED'),
            '-infinity'::timestamp
        )
        ORDER BY started_at DESC LIMIT %s
        """,
        [model_id, model_id, window],
    )
    measurements: list[float] = []
    for row in rows or []:
        money = _benchmark_money(_json_column(row[0]), opponent)
        if money is not None:
            measurements.append(money)
    return measurements


def _benchmark_ratchet(
    result: dict[str, Any] | None,
    reference: float | None,
    reference_checkpoint_id: str | None,
    config: dict[str, Any],
    *,
    grace: bool,
    reference_samples: list[float] | None = None,
) -> dict[str, Any]:
    """Does the candidate hold its ground against the fixed benchmark.

    Two terms, and they fail differently on purpose. The absolute floor says a
    release must be worth releasing at all. The ratchet says a release must not be
    materially worse than the one it replaces, and it is relative to the
    incumbent's recorded benchmark money.

    `reference_samples`, when the window is wider than one, replaces that single
    recorded number with the median of every measurement taken while the incumbent
    has stood. One arena series on this family is a draw, not a number -- 61 of
    them on an unmoving policy ran 0.93 to 427 around a median of 36.7 -- so
    ratcheting one draw against another is a coin flip biased toward whatever was
    lucky enough to promote. See `_benchmark_reference_window`.

    Read against the *benchmark*, not the head-to-head. The head-to-head
    incumbent is the previous update, so it moves with the candidate and a run
    can walk itself down indefinitely while every single comparison reads as an
    improvement.

    `grace` drops the **floor** and keeps the ratchet, which is the opposite of the
    obvious reading. The floor is the term a stalled run cannot clear -- a lineage
    can sit far below a fixed bar while never having regressed against its own
    reference, and the incumbent protecting it may not clear that bar either.

    Keeping the ratchet under grace is what stops this becoming
    `promotion_rule = "always"` in disguise: the decay this gate exists to catch is
    relative, so the relative term is the one worth holding on to.
    """

    floor_name = (
        "benchmark_cold_start_min_money" if reference is None else "benchmark_release_min_money"
    )
    if grace:
        # `None` means "the cold-start foothold", so a config that never sets this
        # behaves exactly as it did before the key existed.
        configured = config.get(
            "benchmark_grace_min_money", VIABILITY_DEFAULTS["benchmark_grace_min_money"]
        )
        floor_name = (
            "benchmark_cold_start_min_money" if configured is None else "benchmark_grace_min_money"
        )
        floor = (
            _limit(config, "benchmark_cold_start_min_money")
            if configured is None
            else float(configured)
        )
    else:
        floor = _limit(config, floor_name)
    ratchet = _limit(config, "benchmark_money_ratchet")
    candidate = None if result is None else float(result.get("a_money_mean", 0.0))
    # The incumbent's own draw is one of the samples, not a separate authority --
    # it is the same measurement on the same benchmark, and singling it out is
    # exactly the selection bias the window exists to remove.
    samples = list(reference_samples or [])
    if reference is not None:
        samples.append(reference)
    reference_used = reference
    reference_kind = "released_measurement"
    if len(samples) > 1:
        reference_used = median(samples)
        reference_kind = "window_median"
    checks: dict[str, bool] = {
        "measured": result is not None and not result.get("errors"),
        "min_money": candidate is not None and candidate >= floor,
    }
    if reference_used is not None:
        checks["no_regression"] = candidate is not None and candidate >= ratchet * reference_used
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "opponent": None if result is None else result.get("opponent"),
        "candidate_money": candidate,
        "reference_money": reference_used,
        # The single stored draw, kept beside the statistic actually used so a
        # rejection can be read against both without a second query.
        "released_money": reference,
        "reference_kind": reference_kind,
        "reference_samples": len(samples),
        "reference_checkpoint_id": reference_checkpoint_id,
        "floor": floor,
        "floor_name": floor_name,
        "ratchet": ratchet,
        "grace": grace,
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
# namespaced under one `declared_model_id`) and a slot that stops improving is
# *ended* rather than nursed: its released checkpoint graduates into
# `population_members`, the group drops its lowest scorer to stay at capacity, and
# the slot restarts from a mature member with its learning rate at zero and a
# fraction of its weights redrawn.


@dataclass(frozen=True)
class SlotSpec:
    """One learner slot's warmup-plateau-graduate schedule.

    Parsed in one place because `collect` and `update` both consult the slot --
    the collector to decide which weights to roll out, the updater to decide what
    to train and whether the slot is finished -- and they run on different rounds
    from different processes.
    """

    peak: float
    warmup_updates: int
    patience: int
    min_delta: float
    best_decay: float
    score_signal: str
    decay_factor: float
    graduate_cut: float
    min_updates: int
    mature_capacity: int
    reserved_capacity: int
    reserved_min_money: float
    entry_updates: int
    warm_start: bool
    warm_start_min_updates: int
    dropout: float
    shrink: float
    sigma_reset: bool

    @property
    def enabled(self) -> bool:
        """A slot with no cut never graduates, which is the pre-population run."""

        return self.graduate_cut > 0.0


def _slot(config: dict[str, Any]) -> SlotSpec:
    peak = float(config.get("learning_rate", 1e-4))
    spec = SlotSpec(
        peak=peak,
        # Updates spent climbing from zero to the peak after a reseed. The first
        # update of a slot therefore trains at *exactly* zero, which is the
        # point: it lets the ring refill with the reseeded policy's own episodes
        # before the weights move at all. Training on the graduated policy's data
        # is the "reset was half a reset" failure recorded at `landgate-v10`, and
        # `reseed_batch_floor` alone only covers the batches, not the first step.
        warmup_updates=int(config.get("lr_warmup_updates", 3)),
        # Consecutive updates without a real improvement before the rate halves.
        patience=int(config.get("lr_plateau_patience", 4)),
        # Relative, not absolute. The score is money, which spans three orders of
        # magnitude across a run, so a fixed delta is either noise at the top of
        # that range or unreachable at the bottom.
        min_delta=float(config.get("lr_plateau_min_delta", 0.05)),
        # How fast the plateau detector's high-water mark leaks. See
        # `_settle_slot`: a plain maximum latches onto the luckiest draw the slot
        # ever took and then demands `min_delta` on top of it, which ends slots
        # that are still improving. 0.0 restores that latch exactly.
        best_decay=float(config.get("lr_best_decay", 0.08)),
        # Which measurement the plateau detector reads. See `_slot_score`.
        score_signal=str(config.get("lr_score_signal", "rollout")).lower(),
        decay_factor=float(config.get("lr_decay_factor", 0.5)),
        # Below this the slot is finished. 0.0 switches the whole mechanism off
        # and reproduces the fixed-rate single lineage exactly.
        graduate_cut=float(config.get("lr_graduate_cut", 0.0)),
        # A floor on slot lifetime, so warmup noise cannot end a slot that has
        # not had a chance to train. Counted from the reseed, not from the run.
        min_updates=int(config.get("lr_min_updates_before_graduation", 10)),
        mature_capacity=int(config.get("mature_capacity", 30)),
        # How many of those berths the behaviour archive holds instead of the
        # score ranking. Carved out of `mature_capacity`, not added to it, so the
        # group's size and its arena cost are unchanged. 0 is the pre-archive cull.
        reserved_capacity=int(config.get("mature_reserved_capacity", 0)),
        # The money floor a cell occupant clears. Defaulted to the number this
        # file already uses for "traded a season and did not die" rather than to a
        # new one, so the archive and the cold-start viability check agree.
        reserved_min_money=float(
            config.get(
                "mature_reserved_min_money",
                VIABILITY_DEFAULTS["viability_cold_start_reference_money"],
            )
        ),
        # How long a release must stand unbeaten by its own slot before it may
        # enter the mature group. Parsed here rather than at the call site so the
        # default lives in one place instead of being duplicated between this
        # module and the manifest.
        entry_updates=int(config.get("population_entry_updates", 2)),
        # Adopt the mature group on a slot that has never been reseeded, rather
        # than leaving it on the cold start. See `_warm_starts`.
        warm_start=bool(config.get("population_warm_start", True)),
        warm_start_min_updates=int(config.get("population_warm_start_min_updates", 0)),
        dropout=float(config.get("reseed_dropout", 0.10)),
        shrink=float(config.get("reseed_shrink", 1.0)),
        sigma_reset=bool(config.get("reseed_sigma_reset", True)),
    )
    if spec.warmup_updates < 0:
        raise ValueError(f"lr_warmup_updates must not be negative: {spec.warmup_updates}")
    if not 0.0 < spec.decay_factor < 1.0:
        raise ValueError(f"lr_decay_factor must be in (0, 1): {spec.decay_factor}")
    if spec.patience < 1:
        raise ValueError(f"lr_plateau_patience must be positive: {spec.patience}")
    if not 0.0 <= spec.best_decay < 1.0:
        raise ValueError(f"lr_best_decay must be in [0, 1): {spec.best_decay}")
    if spec.score_signal not in _SCORE_SIGNALS:
        raise ValueError(
            f"lr_score_signal must be one of {sorted(_SCORE_SIGNALS)}: {spec.score_signal}"
        )
    if spec.mature_capacity < 1:
        raise ValueError(f"mature_capacity must be positive: {spec.mature_capacity}")
    if not 0 <= spec.reserved_capacity < spec.mature_capacity:
        # Equal is refused as well as greater: an archive that takes every berth
        # leaves the score ranking none, so nothing enters on merit and the pool
        # stops selecting for anything at all.
        raise ValueError(
            f"mature_reserved_capacity {spec.reserved_capacity} must be in "
            f"[0, mature_capacity) with mature_capacity {spec.mature_capacity}"
        )
    if spec.entry_updates < 1:
        # 0 is entry-on-promotion, which `_release_age` and the manifest both
        # record as strictly worse than the single lineage the population
        # replaced: whichever slot released last becomes the ancestor for
        # whichever slot reseeds next, and the roster converges on clones.
        raise ValueError(
            f"population_entry_updates must be positive: {spec.entry_updates}"
        )
    if spec.warm_start_min_updates < 0:
        raise ValueError(
            "population_warm_start_min_updates must not be negative: "
            f"{spec.warm_start_min_updates}"
        )
    if spec.enabled and spec.graduate_cut >= spec.peak:
        # Otherwise the very first steady update is already below the cut and the
        # slot graduates having trained nothing, forever, on every learner.
        raise ValueError(
            f"lr_graduate_cut {spec.graduate_cut} is not below learning_rate "
            f"{spec.peak}, so a slot would graduate before it trains"
        )
    return spec


@dataclass(frozen=True)
class SlotState:
    """Where a slot is in its cycle, as carried through `metrics_json`.

    Through the database rather than through memory, the same route
    `collection_sigma_scale` and `lineage_checkpoint` already take: `collect` and
    `update` are different hooks on different rounds, the state has to survive a
    restart, and what a slot was doing has to be answerable in SQL beside the
    measurement that decided it.
    """

    learning_rate: float
    stage: str  # "warmup" | "steady" | "decayed"
    updates_since_reseed: int
    decay_steps: int
    best_score: float | None
    stall_updates: int
    reseed_index: int
    # The two watermarks that make a reseed a *whole* reset, carried for the
    # slot's whole life rather than for the one update that set them. Holding
    # them only across the reseed is the `landgate-v10` bug: the buffer refills
    # from the discarded policy's batches and the mirror plays its checkpoints,
    # so restored weights train on abandoned data and the reset undoes itself
    # over the following generation.
    batch_floor: int
    after_checkpoint_id: str | None

    def as_metrics(self) -> dict[str, Any]:
        return {
            "lr": self.learning_rate,
            "lr_stage": self.stage,
            "lr_updates_since_reseed": self.updates_since_reseed,
            "lr_decay_steps": self.decay_steps,
            "lr_best_score": self.best_score,
            "lr_stall_updates": self.stall_updates,
            "lr_reseed_index": self.reseed_index,
            "lr_batch_floor": self.batch_floor,
            "lr_after_checkpoint_id": self.after_checkpoint_id,
        }


def _slot_metrics(reader, model_id: str) -> dict[str, Any]:
    """The newest update's slot state for one learner, or an empty dict.

    Selected on the state key rather than on `started_at` alone, so updates that
    predate the population -- or that failed before recording anything -- read as
    "no state" instead of as a slot at update zero.
    """

    row = reader.fetchone(
        """
        SELECT metrics_json FROM model_updates
        WHERE model_id = %s
          AND metrics_json ->> 'lr_updates_since_reseed' IS NOT NULL
        ORDER BY started_at DESC
        LIMIT 1
        """,
        [model_id],
    )
    if row is None:
        return {}
    return _json_column(row[0]) or {}


def _slot_state(previous: dict[str, Any], spec: SlotSpec) -> SlotState:
    """The state this update trains at, derived from the last one recorded.

    A reseed marker on the previous row restarts the counters here rather than
    where it was written: the graduating update still had to finish its own
    training and its own gate, and rewinding it would discard a candidate that
    was already paid for.
    """

    if previous.get("reseed_pending") or not previous:
        since = 0
        decay_steps = 0
        best: float | None = None
        stall = 0
        reseed_index = int(previous.get("lr_reseed_index", -1)) + 1
        batch_floor = int(previous.get("reseed_batch_floor", 0) or 0)
        after = previous.get("reseed_after_checkpoint_id")
    else:
        since = int(previous.get("lr_updates_since_reseed", 0)) + 1
        decay_steps = int(previous.get("lr_decay_steps", 0))
        best = previous.get("lr_best_score")
        best = None if best is None else float(best)
        stall = int(previous.get("lr_stall_updates", 0))
        reseed_index = int(previous.get("lr_reseed_index", 0))
        batch_floor = int(previous.get("lr_batch_floor", 0) or 0)
        after = previous.get("lr_after_checkpoint_id")

    # Inert without a graduation cut. `lr_warmup_updates` and `lr_decay_factor`
    # both carry non-trivial defaults, and letting them apply to a config that
    # never asked for a population would silently warm every existing run up from
    # zero and decay it afterwards. One key turns the whole mechanism on.
    if not spec.enabled:
        rate, stage = spec.peak, "steady"
    elif spec.warmup_updates > 0 and since < spec.warmup_updates:
        rate = spec.peak * (since / spec.warmup_updates)
        stage = "warmup"
    else:
        rate = spec.peak * spec.decay_factor**decay_steps
        stage = "decayed" if decay_steps else "steady"
    return SlotState(
        learning_rate=rate,
        stage=stage,
        updates_since_reseed=since,
        decay_steps=decay_steps,
        best_score=best,
        stall_updates=stall,
        reseed_index=reseed_index,
        batch_floor=batch_floor,
        after_checkpoint_id=None if after is None else str(after),
    )


def _settle_slot(state: SlotState, score: float | None, spec: SlotSpec) -> SlotState:
    """Fold this update's score into the slot, decaying on a plateau.

    `score` is a windowed money median, never `mean_loss`: loss is not a progress
    signal here and is not even the right *sign*, since an improving policy reaches
    harder states and its TD residual rises.

    **`best` leaks, and that is `spec.best_decay`.** A plain maximum latches on the
    single luckiest draw the slot ever took, which every later update then has to
    clear by `min_delta` on top -- unreachable on a signal moving tens of percent
    between updates, so the slot graduates while it is still improving.

    Three knobs that look interchangeable and are not: `patience` says how long a
    genuine plateau must last, `min_delta` how big a move counts, and this how long
    an unrepeatable measurement stands as the thing to beat. The decay is sized
    against the patience budget -- the largest excess cleared before the
    `patience`-th stall is `1 / ((1 + min_delta) * (1 - decay)**(patience - 1))` --
    so raising patience widens this rather than duplicating it.

    `max(score, ...)` floors the leak at the current observation, which is what makes
    a large decay safe: a slot genuinely flat at `L` keeps `best == L` whatever the
    decay, so it still stalls and still graduates. Only excess leaks.

    Two things that are **not** stalls:

    * `None` -- the score did not measure. A broken arena would otherwise graduate
      every slot in the roster.
    * Warmup. A slot climbing from zero is deliberately under-trained, so counting
      it spends the patience budget before a single step at the resolved rate and
      sets `best_score` from a measurement the slot is guaranteed to beat.
    """

    if score is None or not spec.enabled or state.stage == "warmup":
        return state
    best, stall, decay_steps = state.best_score, state.stall_updates, state.decay_steps
    if best is None or score > best * (1.0 + spec.min_delta):
        best, stall = score, 0
    else:
        stall += 1
        if stall >= spec.patience:
            decay_steps, stall = decay_steps + 1, 0
        # Floored at this update's own score, so the leak only ever gives back
        # ground an outlier took and never turns a decline into an improvement.
        if spec.best_decay:
            best = max(score, best * (1.0 - spec.best_decay))
    return SlotState(
        learning_rate=state.learning_rate,
        stage=state.stage,
        updates_since_reseed=state.updates_since_reseed,
        decay_steps=decay_steps,
        best_score=best,
        stall_updates=stall,
        reseed_index=state.reseed_index,
        batch_floor=state.batch_floor,
        after_checkpoint_id=state.after_checkpoint_id,
    )


def _rollout_score(
    reader,
    model_id: str,
    candidate: float | None,
    window: int,
    batch_floor: int,
) -> float | None:
    """`_slot_score`'s rollout arm: a window median of `recent_mean_final_money`.

    The same shape as the benchmark arm below -- this update's own measurement
    plus the previous `window - 1`, bounded below by the slot's reseed, median
    taken -- so switching `lr_score_signal` changes only which number is being
    smoothed and not how. That matters because `lr_plateau_min_delta` is a
    *relative* threshold and both signals are money on the same stake, so the
    patience budget means the same thing either way.

    A median even though the input is already a many-episode mean, and the extra
    smoothing is close to free here: the ring turns over about 3.75 generations,
    so adjacent updates are correlated rather than independent draws and the
    median is guarding against the reseed discontinuity more than against noise.

    The FIFO segment, not the whole buffer. `own_mean_final_money` includes the
    reservoir -- a uniform sample of everything evicted since the reseed -- so it
    lags the policy by a third of the buffer's history, and this median on top of
    that lag was grading a slot on a policy several updates old.

    Rows written before `recent_mean_final_money` existed are **skipped rather than
    substituted**: the old key is the diluted quantity, and folding it into the same
    median would put a step in the series that no policy change caused. The window
    is short for one cycle after the change and exact throughout.
    """

    if candidate is None:
        return None
    if window <= 1:
        return float(candidate)
    rows = reader.fetchall(
        """
        SELECT metrics_json FROM model_updates
        WHERE model_id = %s AND metrics_json IS NOT NULL AND cutoff_batch_seq > %s
        ORDER BY started_at DESC LIMIT %s
        """,
        [model_id, batch_floor, window - 1],
    )
    samples = [float(candidate)]
    for row in rows or []:
        metrics = _json_column(row[0])
        money = None if metrics is None else metrics.get("recent_mean_final_money")
        if money is not None:
            samples.append(float(money))
    return median(samples)


def _slot_score(
    reader,
    model_id: str,
    opponent: str | None,
    candidate: float | None,
    window: int,
    batch_floor: int,
    signal: str = "rollout",
    rollout: float | None = None,
) -> float | None:
    """The slot's progress signal: a median of its own recent money.

    `signal` picks which money. The rollout mean changes ~12% between updates and
    the benchmark arena ~48%, so the arena is four times noisier -- and that is not
    a precision problem the window fixes, because adjacent windows share ten of
    eleven samples and the draw's heavy tail survives the smoothing. Slots have been
    graduated as plateaus on the arena number while their rollout money climbed
    steadily.

    `"benchmark"` keeps the old signal, and the two are not the same *quantity*: the
    rollout mean is measured against the training mix at collection sigma, the arena
    against the pinned benchmark greedily. This changes which one ends a slot, never
    which one promotes.

    A median rather than the latest draw either way -- a detector reading single
    draws would find an "improvement" by luck every few updates and the rate would
    never decay, reproducing the stagnation it exists to end.

    Bounded below by the slot's own reseed: earlier measurements describe a policy
    this slot no longer is, and an inherited high score would pin a restarted slot's
    plateau counter. `None` when nothing measured, which `_settle_slot` reads as no
    information rather than as a stall.
    """

    if signal == "rollout":
        return _rollout_score(reader, model_id, rollout, window, batch_floor)
    if candidate is None or opponent is None:
        return candidate
    if window <= 1:
        return float(candidate)
    rows = reader.fetchall(
        """
        SELECT gate_json FROM model_updates
        WHERE model_id = %s AND gate_json IS NOT NULL AND cutoff_batch_seq > %s
        ORDER BY started_at DESC LIMIT %s
        """,
        [model_id, batch_floor, window - 1],
    )
    samples = [float(candidate)]
    for row in rows or []:
        money = _benchmark_money(_json_column(row[0]), opponent)
        if money is not None:
            samples.append(money)
    return median(samples)


def _graduates(state: SlotState, spec: SlotSpec) -> bool:
    """Whether the settled slot has decayed past its cut.

    Read off the *next* update's rate rather than off `state.learning_rate`,
    because the decay that ends a slot is recorded by `_settle_slot` after the
    rate for this update was already resolved -- otherwise the cut is always
    noticed one update late.

    Warmup is excluded explicitly. During warmup the rate is legitimately below
    the cut, and at `updates_since_reseed = 0` it is exactly zero, so a slot would
    graduate on the first update after being reseeded and the roster would do
    nothing but churn.
    """

    if not spec.enabled:
        return False
    if state.updates_since_reseed < max(spec.warmup_updates, spec.min_updates):
        return False
    return spec.peak * spec.decay_factor**state.decay_steps < spec.graduate_cut


def _warm_starts(
    state: SlotState, spec: SlotSpec, members: list[dict[str, Any]]
) -> bool:
    """Whether a slot that has never been reseeded should adopt the group instead.

    A cold start is the right initialisation only when there is nothing to start
    from. `declared_model_id` carries no session suffix, so the mature group
    outlives the session that filled it and a fresh roster begins beside
    checkpoints that already work. Starting from `initial_network` there is not
    exploring a different basin -- it is re-deriving from noise what the group
    already holds, for a whole decay cycle before `_graduates` supplies an ancestor.

    Not a graduation, and deliberately not routed through `_graduates`' floors.
    Those floors exist so warmup noise cannot end a slot that has not had a chance
    to train: they gate a claim about evidence. This reads no score and makes no
    such claim. It corrects the slot's *initialisation*, and every update spent
    waiting the floor out is another update spent training the initialisation
    being corrected. `population_warm_start_min_updates` is the escape hatch for
    anyone who disagrees.

    Fires at most once per reseed lineage, and needs no flag of its own to say so:
    the marker this update writes advances `lr_reseed_index` in `_slot_state`, and
    the predicate is `reseed_index == 0`. So it is self-cancelling, and it
    survives a process restart because the state is the update row rather than
    memory.
    """

    if not spec.enabled or not spec.warm_start:
        return False
    if state.reseed_index > 0:
        return False
    if state.updates_since_reseed < spec.warm_start_min_updates:
        return False
    # The filtered list rather than a count. `mature_members` drops members whose
    # `.npz` has been cleaned off disk, so a group of thirty rows with no
    # artifacts is an empty group and the slot must cold-start rather than fail.
    return bool(members)


def _declared_model_id(reader, model_id: str) -> str:
    """The population every learner slot of this model shares.

    `declared_model_id` rather than `model_id`, because the roster is N named
    instances of one declared model and the whole point is that a checkpoint
    graduating out of one slot is what another slot reseeds from. Falls back to
    the model's own id, so a single unnamed model is a population of one and
    nothing here has to special-case it.
    """

    row = reader.fetchone(
        "SELECT declared_model_id FROM training_models WHERE model_id = %s",
        [model_id],
    )
    if row is None or row[0] is None:
        return model_id
    return str(row[0])


def mature_members(
    reader, declared_model_id: str, reserved_min_money: float = 0.0
) -> list[dict[str, Any]]:
    """The live mature group, best first.

    Missing artifacts are dropped rather than raising: a member whose `.npz` has
    been cleaned off disk is unusable as a reseed source or a mirror opponent,
    and the run has to keep going. It stays a live row so the count still reflects
    what graduated.

    `reserved_min_money` is the archive's own floor and has to be passed by anything
    whose result reaches `_reseed_source`'s reserve bucket -- see `_mark_reserved`.
    0.0 is the unfloored marking, correct for a caller that only wants the ranked
    list (the mirror pool) and never reads `reserved`.
    """

    rows = reader.fetchall(
        """
        SELECT p.member_id, p.model_id, p.checkpoint_id, p.score,
               c.inference_artifact_path, p.behavior_json
        FROM population_members p
        JOIN model_training_checkpoints c USING (checkpoint_id)
        WHERE p.declared_model_id = %s AND p.retired_at IS NULL
        ORDER BY p.score DESC, p.member_id
        """,
        [declared_model_id],
    )
    members = [
        {
            "member_id": str(row[0]),
            "model_id": str(row[1]),
            "checkpoint_id": str(row[2]),
            "score": float(row[3]),
            "inference_artifact_path": str(row[4]),
            "behavior": _decode_behavior(row[5]),
        }
        for row in rows or []
        if Path(str(row[4])).is_file()
    ]
    return _mark_reserved(members, reserved_min_money)


def _decode_behavior(value: Any) -> dict[str, Any] | None:
    """`behavior_json` as a mapping, whatever the driver handed back.

    The column is nullable and predates nothing -- every member graduated before
    it existed carries NULL -- so an unreadable or absent value is `None` rather
    than an error, which `_json_column` already gets right. `_reserved_cell` then
    declines to file it anywhere.
    """

    decoded = _json_column(value)
    return None if decoded is None else dict(decoded)


def _mark_reserved(
    members: list[dict[str, Any]], reserved_min_money: float = 0.0
) -> list[dict[str, Any]]:
    """Tag the best-scoring live member of each reserved cell.

    Computed here, once, off the same score-ordered list the cull reads, so
    `_reseed_source` never re-derives which rows the archive is holding and the
    two cannot disagree about it. Ties break on the incoming order, which is
    `score DESC, member_id` -- deterministic, and the same tiebreak the cull's
    `DISTINCT ON` uses.

    **The money floor is applied here too, and has to be.** An earlier version
    reasoned that a live row is a row the cull already accepted -- true of the
    `kept_by_cell` half and false of the group as a whole, because
    `kept_by_score` keeps a member on rank alone and never consults the floor.
    So a bankrupt land-buyer held by score was still tagged `reserved` and drawn
    by `reseed_source_explorer_share`, which is exactly the reseed the floor is
    the safety argument against. `_graduate`'s `money >= %s` predicate reads the
    same undiscounted `behavior_json ->> 'money'` this does.

    A member with no fingerprint, or one carrying no `money`, stays unreserved:
    unmeasured is not the same as below the floor, and neither may hold a berth.
    """

    seen: set[tuple[int, int]] = set()
    marked: list[dict[str, Any]] = []
    for member in members:
        cell = _reserved_cell(member.get("behavior"))
        reserved = (
            cell is not None
            and cell not in seen
            and _behavior_money(member.get("behavior")) >= float(reserved_min_money)
        )
        if reserved:
            seen.add(cell)
        marked.append({**member, "cell": cell, "reserved": reserved})
    return marked


def _behavior_money(behavior: dict[str, Any] | None) -> float:
    """The undiscounted benchmark money a fingerprint carries, or `-inf`.

    `-inf` rather than 0.0 for an absent or unreadable value, so an unmeasured
    member fails any floor including `0.0` -- the same "not recorded is not
    evidence" rule `_released_trading_rate` follows, and the reading `_graduate`'s
    SQL already has for free since `money >= %s` is NULL-false.
    """

    if not isinstance(behavior, dict) or behavior.get("money") is None:
        return float("-inf")
    try:
        return float(behavior["money"])
    except (TypeError, ValueError):
        return float("-inf")


def _release_age(reader, model_id: str, checkpoint_id: str | None) -> int | None:
    """How many of this slot's updates a release has stood without being replaced.

    The lag on pool entry, and load-bearing rather than cautious: a release copyable
    the moment it is promoted closes a loop where the newest release becomes the
    next reseed's ancestor and the roster converges on clones of the current best --
    a greedy hill-climb running ten copies. Gating entry on *graduation* is also a
    lag, but an enormous one: a whole decay cycle, and only if the slot dies.

    What the count tests is survival: a policy has to stand through a stretch of its
    own slot's training without a newer candidate beating it. A slot promoting on
    every update has a moving target for a best and contributes nothing until it
    settles.

    `None` when the release cannot be dated -- an unknown checkpoint, or one whose
    update row is gone. Callers treat that as "not old enough" rather than as
    zero, so an unreadable history never opens the gate it exists to hold shut.
    """

    if checkpoint_id is None:
        return None
    row = reader.fetchone(
        """
        SELECT count(*)
        FROM model_updates u
        WHERE u.model_id = %s
          AND u.started_at > (
              SELECT p.started_at FROM model_updates p
              JOIN model_training_checkpoints c ON c.update_id = p.update_id
              WHERE c.checkpoint_id = %s
          )
        """,
        [model_id, checkpoint_id],
    )
    if row is None or row[0] is None:
        return None
    return int(row[0])


_SLOT_ORDINAL = re.compile(r"(\d+)$")


def _materialize_warm_start(context, reader, slot_metrics: dict[str, Any]) -> Path | None:
    """Adopt the mature group *before* generation 0 is collected, or return None.

    `_warm_starts` already decides a fresh roster should start from the group, but
    it runs inside `update`, which fires only after `rounds_per_generation` rounds --
    so a session would spend its whole first generation collecting from a NOOP-only
    cold start and then train its first update on those episodes. The adoption has
    to happen here or it lands a generation too late to be collected from.

    So the materialisation moves here, and only here. The update side loads these
    bytes rather than drawing its own mask, which is the invariant
    `reseed_artifact_path` exists for: `collect` and `update` must roll out and
    train the same network, not two draws from one seed.

    Returns None -- and leaves the cold-start path untouched -- whenever the
    adoption does not apply: the population is off, `population_warm_start` is
    off, the slot has already trained, or the mature group is empty. That last
    one is the first session of a family, where a cold start is correct because
    there is genuinely nothing to start from.
    """

    config = context.config
    if slot_metrics:
        # Not virgin. Either the slot has state and `update` owns the decision,
        # or it has already been through here and `lineage_checkpoint` applies.
        return None
    spec = _slot(config)
    if not spec.enabled or not spec.warm_start:
        return None

    network_path, sidecar = warm_start_paths(context.artifacts)
    if network_path.is_file() and sidecar.is_file():
        # Rounds two through ten of generation 0 land here. One draw, one mask,
        # one set of bytes for the whole generation.
        return network_path

    declared = _declared_model_id(reader, context.model_id)
    members = mature_members(reader, declared, spec.reserved_min_money)
    if not members:
        return None

    chosen = _reseed_source(
        members,
        config,
        seed=int(config.get("seed", 0)),
        # A virgin slot is at reseed zero by definition, and `spread` is exactly
        # the whole-roster-restarts-at-once case this is -- ten slots drawing
        # independently from one group collide most of the time.
        reseed_index=0,
        ordinal=_slot_ordinal(config, context.model_id),
        roster=int(config.get("population_slots", 0)),
        spread=True,
    )
    if chosen is None:
        return None

    seeded = reseed.perturb(
        OrderQNetwork.load(chosen["inference_artifact_path"]),
        spec.dropout,
        spec.shrink,
        # `+ 1` to match the update path's `reseed_index + 1` at reseed zero, so
        # moving the materialisation did not quietly move the mask as well.
        seed=int(config.get("seed", 0)) + 1,
        sigma_reset=_noisy(config).sigma0 if spec.sigma_reset else None,
    )
    # The same guard the update path applies. A perturbation that zeroed the
    # trunk leaves a network with no gradient outside its output bias, and a slot
    # that starts there can never leave.
    _reject_inert_network(seeded)

    network_path.parent.mkdir(parents=True, exist_ok=True)
    seeded.save(network_path)
    sidecar.write_text(
        json.dumps(
            {
                "checkpoint_id": chosen["checkpoint_id"],
                "model_id": chosen["model_id"],
                "score": chosen["score"],
            },
            sort_keys=True,
        )
    )
    return network_path


def warm_start_paths(artifacts) -> tuple[Path, Path]:
    """Where a virgin slot's warm start lives, and its sidecar.

    One place, named by neither an `update_id` nor a checkpoint id, because the
    thing that materialises it runs before either exists. `collect` writes both
    on the first round of generation 0 and `update` reads both on the first
    update, so this function is the only thing the two sides have to agree on.

    The sidecar carries the ancestor's identity rather than the filename doing
    it. `update` re-derives its own draw from `_reseed_source` to fill
    `reseed_source_rank` and the archive columns, and cross-checks it against
    this -- a disagreement means the mature group moved between the collection
    and the update, which cannot happen today and would silently pair one
    ancestor's bytes with another's provenance if it ever did.
    """

    root = Path(artifacts) / "reseeds"
    return root / "warm-start.npz", root / "warm-start.json"


def _slot_ordinal(config: dict[str, Any], model_id: str) -> int | None:
    """This slot's index in the roster, or None if it is not one of many.

    The roster is `effective_model_id`'d instances of one declared model, so the
    ordinal is already recorded in two places and needs deriving from neither:
    `register_training_agent` stamps `name` into `config_json`, and the same
    string is the `@` suffix of `model_id`. Preferring `name` avoids parsing an
    identifier at all; the suffix is the fallback for a config that predates the
    stamp.

    None for `default`, or any name without a trailing number. That is a
    population of one, and it draws the way it always did.
    """

    name = str(config.get("name") or "")
    if not name and "@" in model_id:
        name = model_id.rsplit("@", 1)[-1]
    match = _SLOT_ORDINAL.search(name)
    return None if match is None else int(match.group(1))


def _capped_members(
    members: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """The ranked group with each originating slot held to a fixed share.

    The group is not `mature_capacity` policies -- it is that many *checkpoints of
    however few*, since a slot that is genuinely ahead promotes repeatedly and
    occupies the top of the ranking several times over. One lineage holding two of
    the top three hands itself most of every restart, so a population of ten reads
    as a population of about four: the failure `reseed_source_pool` exists to
    prevent, reappearing one level down.

    The best entry of each lineage rather than an arbitrary one, so the cap costs
    rank and never costs quality: the member dropped is always one its own lineage
    already beat.

    What this gives up, stated rather than hidden: a lineage better than every
    other by more than one rung cannot supply two slots' ancestry at once. That is
    the intended trade. A second perturbation of an ancestor the roster already
    holds is the cheapest thing the roster can hold, and diversity is the entire
    reason there is a population.
    """

    cap = max(1, int(config.get("reseed_source_lineage_cap", 1)))
    taken: dict[str, int] = {}
    capped: list[dict[str, Any]] = []
    for member in members:
        lineage = str(member["model_id"])
        if taken.get(lineage, 0) >= cap:
            continue
        taken[lineage] = taken.get(lineage, 0) + 1
        capped.append(member)
    return capped


def _reseed_source(
    members: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    seed: int,
    reseed_index: int,
    ordinal: int | None = None,
    roster: int = 0,
    spread: bool = False,
) -> dict[str, Any] | None:
    """Which mature member a restarting slot is rebuilt from.

    Drawn uniformly from the top `reseed_source_pool` by score rather than from the
    front of the list, for two reasons:

    * A member's score is a windowed median of a benchmark spanning three orders of
      magnitude, so it *estimates* that policy's worth. Argmax over stored noisy
      estimates selects the luckiest one, every time, because the estimate is reused.
    * Diversity. Every restart pulling index 0 makes ten slots ten perturbations of
      one ancestor in one basin -- a worse search than the single lineage it replaced.

    Ranked rather than score-weighted: a softmax needs a temperature tied to the
    scale, and the scale is neither stable nor fixed, since `_population_score`
    discounts it by activity. Top-k is scale-free.

    Seeded from the slot and its reseed count so a restart is reproducible, with
    entropy distinct from the perturbation's own seed so the ancestor and the mask
    do not move together.

    Three things layered on top:

    * **The cap, always** -- see `_capped_members`. The top-k argument assumes k
      *distinct policies*; without it, k checkpoints of however few.
    * **The spread, only on a whole-roster restart.** Ten slots choosing
      independently collide, and each is a separate process with no coordination
      point. The ordinal is the coordination: every slot indexes the same ranked
      list by its own position. Deliberately *not* used for ordinary reseeds --
      there is nothing to spread against, and a permanent ordinal-to-rank map would
      hand slot 0 the single best on every restart forever.
    * **The reserve bucket**, whenever `reseed_source_explorer_share` is positive.
      A list ranked on one number can only return variations on what that number
      rewards; a draw from the archive's behaviour cells restarts a slot on a policy
      the ranking would have culled. `share = 0.0` reproduces the pre-archive draw.

    `roster` is how many slots exist, needed only by the spread path. 0 disables the
    reserve there rather than guessing.
    """

    capped = _capped_members(members, config)
    if not capped:
        return None
    # The lineage cap runs on each bucket separately. It is about ancestry, and a
    # reserve holding three checkpoints of one lineage is the same failure inside
    # the archive as outside it.
    reserve = _capped_members([member for member in members if member.get("reserved")], config)
    share = max(0.0, min(1.0, float(config.get("reseed_source_explorer_share", 0.0))))

    if spread and ordinal is not None:
        # Deterministic, because the whole roster is choosing at once and there is
        # no coordination point but the arithmetic. The *highest* ordinals take
        # the reserve: slot 0 is the one carrying the league's surrogate-propose
        # entry, and the explorer slots are the roster's tail by convention in
        # `train_market_free.py`, so the two agree without either knowing it.
        explorers = int(round(share * roster)) if reserve and roster > 0 else 0
        if explorers and ordinal >= roster - explorers:
            return reserve[(ordinal - (roster - explorers)) % len(reserve)]
        # Wraps when the group holds fewer lineages than the roster has slots.
        # Collisions there are unavoidable -- there is nothing distinct left to
        # hand out -- and wrapping keeps the shortfall at the *bottom* of the
        # list, so the slots that double up double up on the best entries.
        return capped[ordinal % len(capped)]

    # Drawn with its own entropy, so which bucket a slot reads and which member
    # it takes out of that bucket do not move together -- the same separation the
    # member draw below already keeps from the perturbation's mask.
    bucket = capped
    if reserve and share > 0.0:
        roll = float(np.random.default_rng([seed, reseed_index, 0xA5C]).random())
        if roll < share:
            bucket = reserve
    pool = max(1, int(config.get("reseed_source_pool", 3)))
    top = bucket[:pool]
    if len(top) == 1:
        return top[0]
    index = int(np.random.default_rng([seed, reseed_index, 0x5EED]).integers(len(top)))
    return top[index]


def _graduate(
    reader,
    declared_model_id: str,
    model_id: str,
    checkpoint_id: str,
    score: float,
    capacity: int,
    behavior: dict[str, Any] | None = None,
    reserved: int = 0,
    reserved_min_money: float = 0.0,
) -> dict[str, Any]:
    """Enter one checkpoint into the mature group and cull back to capacity.

    `RETURNING` on both statements because the trainer's only database handle is
    the read view, whose `fetchall` expects a result set. The connection is
    autocommit, so this lands the moment it runs -- which is why the caller does
    it last, after everything that can still fail.

    Culling reads the live set *after* the insert, so a graduate that is itself
    the worst member is the one retired and the group cannot be made worse by
    admitting to it. Retirement sets a timestamp; the row and its artifacts stay,
    because a run has to be answerable afterwards about what it dropped.

    **The conflict clause must clear `retired_at`, not just refresh the score.**
    Entry is idempotent per checkpoint and a standing release re-enters on every one
    of its slot's updates, so a member culled while its slot still holds it as the
    active release would have every later update rewrite the score on a row that
    stayed retired. The cull counts only live rows, so it then retires nothing and
    the group sits below capacity permanently.

    `reserved` splits the capacity rather than adding to it: `capacity - reserved`
    berths go to the score ranking, the rest to the best-scoring member of each
    behaviour cell that is not the monoculture attractor. At `reserved = 0` the two
    `EXCEPT` clauses collapse to a plain `OFFSET capacity`, which is what makes the
    whole feature retirable through one config key. Cells are ranked against each
    other before the limit, so an oversubscribed archive spends berths on the best
    occupied cells rather than on whichever sorts first.

    `reserved_min_money` is the floor a cell occupant clears, and the entire safety
    argument: unfloored, a cell would hold day-0 land bankrupts permanently and
    reseed the roster from them. It reads the *undiscounted* money rather than
    `score`, since applying the activity discount here would bend the bar against a
    diversified policy twice.
    """

    member_id = hashlib.sha256(
        f"{declared_model_id}\0{checkpoint_id}".encode()
    ).hexdigest()[:24]
    reader.fetchall(
        """
        INSERT INTO population_members
            (member_id, declared_model_id, model_id, checkpoint_id, score, behavior_json)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (member_id) DO UPDATE SET score = excluded.score,
                                              behavior_json = coalesce(
                                                  excluded.behavior_json,
                                                  population_members.behavior_json
                                              ),
                                              retired_at = NULL,
                                              retired_reason = NULL
        RETURNING member_id
        """,
        [
            member_id,
            declared_model_id,
            model_id,
            checkpoint_id,
            float(score),
            json.dumps(behavior) if behavior is not None else None,
        ],
    )
    retired = reader.fetchall(
        """
        WITH live AS (
            SELECT member_id, score, behavior_json,
                   (behavior_json ->> 'crop_bin')::int AS crop_bin,
                   (behavior_json ->> 'land_bin')::int AS land_bin,
                   (behavior_json ->> 'money')::float AS money
            FROM population_members
            WHERE declared_model_id = %s AND retired_at IS NULL
        ),
        kept_by_score AS (
            SELECT member_id FROM live ORDER BY score DESC, member_id LIMIT %s
        ),
        kept_by_cell AS (
            SELECT member_id FROM (
                SELECT DISTINCT ON (crop_bin, land_bin) member_id, score
                FROM live
                WHERE crop_bin IS NOT NULL AND land_bin IS NOT NULL
                  AND NOT (crop_bin = %s AND land_bin = %s)
                  AND money >= %s
                ORDER BY crop_bin, land_bin, score DESC, member_id
            ) cells
            ORDER BY score DESC, member_id
            LIMIT %s
        )
        UPDATE population_members SET retired_at = current_timestamp,
                                      retired_reason = 'capacity'
        WHERE member_id IN (
            SELECT member_id FROM live
            EXCEPT SELECT member_id FROM kept_by_score
            EXCEPT SELECT member_id FROM kept_by_cell
        )
        RETURNING member_id, score, behavior_json
        """,
        [
            declared_model_id,
            max(0, capacity - max(0, reserved)),
            CROP_BINS[0],
            LAND_BINS[0],
            float(reserved_min_money),
            max(0, reserved),
        ],
    )
    cell = _reserved_cell(behavior)
    return {
        "member_id": member_id,
        "checkpoint_id": checkpoint_id,
        "score": float(score),
        "capacity": capacity,
        "reserved_capacity": max(0, reserved),
        "cell": list(cell) if cell is not None else None,
        "retired": [
            {
                "member_id": str(row[0]),
                "score": float(row[1]),
                "cell": (
                    list(retired_cell)
                    if (retired_cell := _reserved_cell(_decode_behavior(row[2]))) is not None
                    else None
                ),
            }
            for row in retired or []
        ],
    }


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
    decisions score against, and the `[B, (train + n_step) * B_flat, H]` flattened
    grid `boot_turn` indexes into.

    The burn-in is always under `no_grad`, whatever `grad` says. Letting the
    gradient into it would make BPTT run the whole chunk rather than the training
    window, so "truncated" would stop being true and the cost would grow with
    `burn_in` -- which is the one dial meant to be cheap.

    The tail is also always under `no_grad` and starts from a detached state: it
    exists only to produce bootstrap latents, and a bootstrap that carries gradient
    is not a target.
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
    is `burn_in + train + n_step` turns and only `train` is recoverable from
    `reward`, so inferring the split here would be guessing at the very number
    whose misalignment the probe could not detect.
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


def _behavior_clone_step(
    online,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    chunks: int,
    device: torch.device,
    burn_in: int,
    train: int,
    clip: float,
) -> float:
    """Fit the action head to a stored expert action for one chunk batch.

    Ordinary classification over `ACTION_COUNT` classes. The previous ten-slot
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
    loss = nn.functional.cross_entropy(
        logits, torch.from_numpy(batch.chosen).to(device)
    )
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
#: `_flip_rate_curve` clips it to `noisy_sigma_scale_limit`, so widening the
#: ladder cannot widen what a run is allowed to play at.
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

    The ladder is clipped to `scale_limit` so a run is never measured at a scale
    `resolve_sigma_scale` is not allowed to return.

    Leaves the module's noise in whatever state the last draw left it: every caller
    either re-draws before its next forward or is done with the module.
    """

    with torch.no_grad():
        net_module.set_noise(module, False)
        greedy = module(features).masked_fill(~mask, MASKED_SCORE).argmax(dim=1)
        greedy_noop = greedy == NOOP_INDEX
        has_noop = bool(greedy_noop.any())
        rungs = [
            scale for scale in ladder if 1.0 / scale_limit <= scale <= scale_limit
        ] or [1.0]

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
        # every gate metric looks ordinary, while 128 of the head's 357 input
        # columns carry nothing and the whole premise of the fork is inert.
        LOGGER.warning(
            "exported free-market network has an all-zero GRU: the recurrence "
            "contributes nothing and this checkpoint is effectively pre-v3. Check "
            "`hidden_norm` and `mean_grad_norm` on the last few updates."
        )


def fill_buffer(buffer: ReplayBuffer, batches) -> int:
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
        for key in owned:
            index = key - first
            if index < len(records):
                records_by_id[key] = records[index]
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

        # Collection follows the lineage, not the release. See the module
        # docstring for what gating this on the promotion gate cost.
        #
        # Unless the slot has been reseeded and not yet trained, in which case it
        # follows the reseed artifact instead. The rounds between a graduation and
        # the update that consumes it are `rounds_per_generation` batches long --
        # twenty here -- and collecting them from the graduated policy is the
        # `landgate-v10` failure exactly: a restart that trains its first update
        # on a full generation of the trajectory it was restarted to leave.
        slot_metrics = _slot_metrics(reader, context.model_id)
        reseed_path = slot_metrics.get("reseed_artifact_path")
        if slot_metrics.get("reseed_pending") and reseed_path and Path(reseed_path).is_file():
            # Already on disk and already content-addressed by `update_id`, so
            # there is nothing to write and nothing to load here -- the workers
            # read the weights themselves.
            checkpoint = None
            behavior_path = Path(reseed_path)
        else:
            checkpoint = lineage_checkpoint(context, reader)
            warm_path = (
                _materialize_warm_start(context, reader, slot_metrics)
                if checkpoint is None
                else None
            )
            if warm_path is not None:
                # Same shape as the `reseed_pending` branch above: the bytes are
                # already on disk and the workers read them themselves.
                checkpoint = None
                behavior_path = warm_path
            else:
                network = (
                    OrderQNetwork.load(checkpoint.inference_artifact_path)
                    if checkpoint is not None
                    else initial_network(config)
                )
                behavior_path = (
                    Path(context.artifacts)
                    / "rollout_policies"
                    / f"{checkpoint.checkpoint_id if checkpoint is not None else 'initial'}.npz"
                )
                behavior_path.parent.mkdir(parents=True, exist_ok=True)
                # Checkpoint paths are content-addressed and immutable; `initial`
                # is not, so it is rewritten in case an older run left a stale one
                # behind.
                if checkpoint is None or not behavior_path.is_file():
                    network.save(behavior_path)

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
                pool = _self_play_pool(
                    context,
                    reader,
                    int(config.get("self_play_pool", 5)),
                    slot_metrics.get("reseed_after_checkpoint_id")
                    or slot_metrics.get("lr_after_checkpoint_id"),
                )
                # A slot reseeded moments ago has no post-reseed checkpoints of
                # its own yet. The mature group is the mirror in the meantime --
                # a better one than the ordinary pool would have been, since it
                # is the population's best rather than the slot's most recent.
                if not pool:
                    pool = [
                        _weights_spec(member["inference_artifact_path"])
                        for member in mature_members(
                            reader, _declared_model_id(reader, context.model_id)
                        )[: int(config.get("self_play_pool", 5))]
                    ]
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
                    # `unrecovered_seed_share` is the same fact as a ratio, but the
                    # reward acts on the coins and a constant with no column showing
                    # what it actually billed cannot be told apart from an inert one.
                    # Read as a pair: the credit rising while the charge falls is the
                    # policy trading better, both rising together is only volume.
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
                    # Null on an ordinary batch, so the rounds a slot spent
                    # collecting from a perturbed restart can be told apart from
                    # the rest of its life. They are meant to look different on
                    # money, and without this column there is nothing in the row
                    # that says why.
                    "reseed_artifact_path": (
                        str(behavior_path) if checkpoint is None and reseed_path else None
                    ),
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


class FreeMarketUpdater:
    def update(
        self, context: ModelUpdateContext, reader: ModelTrainingReader
    ) -> ModelUpdateResult:
        config = context.config
        device = resolve_device(str(config.get("device", "auto")))
        torch.manual_seed(int(config.get("seed", 0)))
        gamma = float(config.get("gamma", 0.999))
        # Fallbacks come from `replay.py` rather than being restated here.
        # Restated, they went stale: this read `n_step=8` long after the manifest
        # moved to 24, so a config that dropped the key silently trained on a
        # third of a day of credit -- the exact regime the move abandoned.
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
        )

        # Where this learner slot is in its cycle, and whether the last update
        # ended it. Read before anything else because the batch floor it carries
        # decides what the buffer is even allowed to see.
        slot_spec = _slot(config)
        previous_slot = _slot_metrics(reader, context.model_id)
        slot = _slot_state(previous_slot, slot_spec)
        reseeded = bool(previous_slot.get("reseed_pending"))
        reseed_source = previous_slot.get("reseed_artifact_path") if reseeded else None

        # `collect` adopts the mature group *before* generation 0 and rolls it out
        # -- see `_materialize_warm_start`. This update has to start from those same
        # bytes, and the condition is deliberately the same one that wrote them: a
        # virgin slot with the warm start on and both files on disk. Reading it off
        # the artifacts rather than off a query is what makes the two sides unable
        # to disagree about which network generation 0 actually played.
        #
        # Falling through to `initial_network` here is what the first version did,
        # and it cost a whole update per slot three ways over: the cold start was
        # trained on episodes collected from the adopted policy, the silent network
        # was exported and paid a full `eval_seeds + promotion_seeds` arena, and the
        # adoption was then deferred to a *second* update that `_slot_state` also
        # ran at a learning rate of exactly zero.
        warm_network_path, warm_sidecar_path = warm_start_paths(context.artifacts)
        adopting_warm_start = (
            not previous_slot
            and slot_spec.enabled
            and slot_spec.warm_start
            and warm_network_path.is_file()
            and warm_sidecar_path.is_file()
        )

        # Batches collected before the reseed belong to a policy this slot has
        # abandoned. `landgate-v10` is the record of skipping this: a restored
        # 18k checkpoint trained immediately on ~400 episodes generated by the
        # 357-coin policy that had just been thrown away, and the next twenty
        # batches were collected from the result.
        usable = [batch for batch in context.all_batches if batch.sequence > slot.batch_floor]
        skipped_evicted_batches = fill_buffer(buffer, usable)

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
        resumed = None if reseeded else lineage_checkpoint(context, reader)
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
            _reset_learning_rates(optimizer, clone_optimizer, config, slot.learning_rate)
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
            if reseed_source and Path(reseed_source).is_file():
                seed_network = OrderQNetwork.load(reseed_source)
            elif adopting_warm_start:
                seed_network = OrderQNetwork.load(warm_network_path)
            else:
                seed_network = initial_network(config)
            online = net_module.from_numpy(seed_network, online)
            target.load_state_dict(online.state_dict())
            _reset_learning_rates(optimizer, clone_optimizer, config, slot.learning_rate)

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
                        "build one with agents.dqn_market_free_v3b.training.ladder"
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

        if len(buffer.episodes) < warmup:
            raise ValueError(f"free-market training requires at least {warmup} replay episodes")

        # `reduction="none"`: every row carries its own importance-sampling
        # weight from the return-weighted draw, and a mean taken inside the loss
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
        # Training turns whose bootstrap did not land on the turn `n_step` ahead.
        # Must stay zero: every turn records at least a NOOP terminator, so the
        # search cannot skip one. Nonzero means a turn recorded no decision at all
        # and the credit horizon quietly shortened.
        clamped_rows = 0
        unranked_batches = 0
        # The same residual without the importance weights. `mean_loss` is the
        # quantity actually optimised and has to stay that, but it changes scale
        # the moment weighting turns on, so a loss curve spanning the change
        # would read as a step that never happened. This one is comparable to
        # every generation recorded before it.
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
        # Divergence detector. Every reward term is bounded -- the outcome is a
        # tanh in (-1, 1) paid twice for a flat contribution of about that, and
        # the two per-turn charges sum to about -1.4 over a whole episode -- so a
        # correct Q sits inside single digits whatever the mode. `mean_loss` alone
        # cannot say whether a rising number is a harder target or an exploding
        # one, and by the time the gate rejects the candidate the update that
        # started it is several checkpoints back on the lineage.
        q_magnitudes: list[float] = []
        q_peak = 0.0
        started = time.perf_counter()
        clone_steps_to_run = (
            _behavior_clone_steps_per_update(config) * batches_per_update
            if buffer.pinned
            else 0
        )
        # Silenced for the clone phase by default: see `clone_optimizer` above.
        # Fitting a 186-way cross-entropy through a stochastic representation
        # teaches the trunk to cancel its own noise, which is not what any of
        # this is for.
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
            clamped_rows += batch.clamped
            unranked_batches += 0 if batch.ranked else 1
            if batch.state.shape[0] == 0:
                continue

            # The online rollout, graded on the training window only. The burn-in
            # and the n-step tail inside `_roll_chunk` are both `no_grad`, which is
            # what keeps BPTT the length of the training window rather than of the
            # whole chunk.
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
                if noisy.enabled:
                    net_module.reset_noise(target)
                bootstrap = (
                    target(torch.cat((boot_state, post_target[boot]), dim=1))
                    .gather(1, best)
                    .squeeze(1)
                )
                # Per decision now, not per turn. An intra-turn row carries reward
                # 0 and discount 1, so its target is purely `max_a Q(s_next, a)`
                # against the same latent -- the "having queued that, what is the
                # next order worth?" link. A boundary row carries the turn's n-step
                # return and `gamma ** n_step`.
                goal = torch.from_numpy(batch.reward).to(device) + (
                    torch.from_numpy(batch.discount).to(device) * bootstrap
                )

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
            unweighted_losses.append(float(per_row.detach().mean()))
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
            if gradient_steps % target_sync == 0:
                target.load_state_dict(online.state_dict())
                # The sync copies mu and sigma, which is the point, but the
                # epsilon buffers are non-persistent precisely so it cannot also
                # copy the online net's current draw and correlate the two.
                if noisy.enabled:
                    net_module.reset_noise(target)
        train_seconds = time.perf_counter() - started

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
            # states the actor actually visits, and a return-weighted draw would
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
                    "n_step": buffer.n_step,
                    "hidden_stride": HIDDEN_STRIDE,
                },
            },
            trainer_path,
        )
        exported = net_module.to_numpy(online)
        _reject_inert_network(exported)
        exported.save(inference_path)

        eval_seeds = int(config.get("eval_seeds", 40))
        promotion_seeds = int(config.get("promotion_seeds", 60))
        seed_base = int(config.get("promotion_seed_base", 1_000_000))
        seed_stride = int(config.get("promotion_seed_stride", 10_000))
        seed_start = seed_base + context.cutoff_batch_seq * seed_stride
        # The benchmark window is pinned separately, because the head-to-head and
        # the benchmark want opposite things from it. The head-to-head compares
        # two policies inside one update, so rotating its seeds each update stops
        # the gate being read off one fixed hundred seeds forever. The benchmark
        # compares *this* update against one recorded twenty batches ago, and a
        # rotating window makes those two numbers unpaired -- a stride of 0 makes
        # `benchmark_money_ratchet` a like-for-like comparison.
        #
        # Nothing is lost to overfitting: the benchmark opponent is fixed, never
        # appears in `opponents`, and arena evaluation draws no noise
        # (`_weights_spec` passes only `weights`, and `noise_cadence` defaults to
        # "none" precisely so the gate path is deterministic).
        benchmark_seed_stride = int(config.get("benchmark_seed_stride", seed_stride))
        benchmark_seed_start = seed_base + context.cutoff_batch_seq * benchmark_seed_stride
        jobs = int(config.get("jobs", 1))

        # Three arena series an update, not five. The incumbent is not replayed
        # against the benchmarks: `_released_benchmark_money` reads the score it
        # already recorded when it was itself a candidate, which is what makes
        # the fixed-reference ratchet below cost one row rather than most of the
        # evaluation wall clock.
        benchmarks = configured_benchmarks(config)
        challenger_benchmarks = [
            _arena_evaluate(inference_path, benchmark, eval_seeds, jobs, benchmark_seed_start)
            for benchmark in benchmarks
        ]
        for result in challenger_benchmarks:
            _reject_degenerate(result, "benchmark")

        promotion_rule = str(config.get("promotion_rule", "significant"))
        if promotion_rule not in _PROMOTION_RULES:
            raise ValueError(
                f"promotion_rule must be one of {sorted(_PROMOTION_RULES)}: {promotion_rule}"
            )

        # The quiet network is a real incumbent, not an absence of one. Without
        # this baseline the first trained checkpoint would promote merely for
        # completing evaluation, which is how an over-ordering bankruptcy policy
        # once displaced the much safer cold start.
        active = context.active_checkpoint
        baseline_path = context.artifacts / "viability_baselines" / "quiet.npz"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        initial_network(config).save(baseline_path)
        incumbent_path = (
            Path(active.inference_artifact_path) if active is not None else baseline_path
        )
        incumbent_spec = _weights_spec(incumbent_path)
        head_to_head = _arena_evaluate(
            inference_path, incumbent_spec, promotion_seeds, jobs, seed_start
        )
        # A zero-width paired margin against a *released* incumbent is a harness
        # fault -- a dropped `?weights=`, a missing checkpoint -- and must fail
        # loudly. Against the quiet baseline it is not: the baseline emits nothing,
        # so a candidate that collapsed to NOOP genuinely plays every paired seed
        # identically, and that belongs in the gate as a rejection rather than as an
        # ERROR discarding the diagnostics that show the collapse.
        #
        # But a zero margin does not *imply* the collapse. Round-trip trades are
        # cash-neutral against an unchanged market, so a candidate can trade all
        # season against a counterparty offering nothing and end where it started.
        # So read the collapse off the candidate's own activity in that match. A
        # genuine NOOP policy reads 0.0 and is still rejected, which is the case
        # this exists for.
        degenerate_head_to_head = False
        try:
            _reject_degenerate(head_to_head, "head-to-head")
        except RuntimeError:
            if active is not None:
                raise
            degenerate_head_to_head = _collapsed_against_baseline(head_to_head, config)

        # The money floors measure against the incumbent's own seat in this same
        # paired series -- except on a cold start, where that seat is the quiet
        # baseline, which never trades and so keeps its full stake on every seed.
        # `0.80 *` a perfect stake is a bar no policy that trades for a season can
        # clear, so a fixed cold-start reference is what makes the *first* release
        # reachable; from the second onward the real seat takes over.
        #
        # That fixed reference is itself escapable, and only on a cold start. Once
        # `active is not None` the reference is a real opponent's seat in this same
        # paired series, which is the comparison worth keeping strict -- standing it
        # down would let a run walk itself down through exactly the drift
        # `_benchmark_ratchet` refuses. The synthetic bar has no such standing: it is
        # a number this config picked, and a slot converging on it from below across
        # every update it has ever run is being held by the number rather than by its
        # own decay. See `viability_grace_after_rejections`.
        head_viability_grace = False
        if active is not None:
            head_reference_kind = "incumbent_seat"
            head_reference = {
                "a_money_mean": head_to_head["b_money_mean"],
                "a_money_median": head_to_head["b_money_median"],
            }
        else:
            viability_grace_limit = int(config.get("viability_grace_after_rejections", 0))
            head_viability_grace = (
                viability_grace_limit > 0 and rejection_streak >= viability_grace_limit
            )
            if head_viability_grace:
                # `None` means "the ratchet's own first-release foothold", so a
                # config that sets the streak limit and nothing else lands on a
                # number it already declares rather than on a second new one.
                configured = config.get(
                    "viability_grace_cold_start_reference_money",
                    VIABILITY_DEFAULTS["viability_grace_cold_start_reference_money"],
                )
                cold_start_reference = (
                    _limit(config, "benchmark_cold_start_min_money")
                    if configured is None
                    else float(configured)
                )
                head_reference_kind = "cold_start_grace"
            else:
                cold_start_reference = _limit(config, "viability_cold_start_reference_money")
                head_reference_kind = "cold_start_fixed"
            head_reference = {
                "a_money_mean": cold_start_reference,
                "a_money_median": cold_start_reference,
            }
        # The activity floor binds unconditionally on the *benchmark* arena, and on
        # the head-to-head only once there is a released incumbent to play. Both
        # halves are load-bearing:
        #
        #   * The benchmark half stops a candidate that never trades becoming the
        #     incumbent everything afterwards is measured against. With no incumbent
        #     the head-to-head is against a policy that reliably bankrupts, so a
        #     pure-NOOP candidate keeps its stake, wins outright and passes hard
        #     viability. It reads 0.0 against the benchmark, where this catches it.
        #   * On the cold-start head-to-head the floor measures nothing: the quiet
        #     baseline offers no counterparty, so an order has nothing to fill
        #     against. Binding it there rejected slots that were trading fine
        #     against the benchmark and left the mature group empty.
        #
        # With no benchmark configured there is no other arena to carry it, so the
        # head-to-head keeps the floor rather than dropping it silently.
        require_head_activity = active is not None or not challenger_benchmarks
        head_viability = _arena_viability(
            head_to_head,
            head_reference,
            config,
            require_activity=require_head_activity,
            reference_kind=head_reference_kind,
            grace=head_viability_grace,
        )
        benchmark_viability = [
            _order_viability(result, config, require_activity=True)
            for result in challenger_benchmarks
        ]
        hard_viability_passed = head_viability["passed"] and all(
            item["passed"] for item in benchmark_viability
        )

        # `policy_improvement` falls through to the `significant` branch for any
        # name it does not know, so `benchmark_ratchet` would silently mean
        # `significant` if it were passed straight in. Mapped explicitly. Under
        # the ratchet this number is recorded and never consulted by `promote`:
        # the incumbent it measures against is the previous update, so it drifts
        # with the candidate by construction -- `falloff_v1` scored 0.29, 0.00,
        # 0.16, 1.00, 0.00, 0.38, 0.59, 0.94 on it while its benchmark money fell
        # from 2,979 to 107.
        head_reference_rule = (
            "point_estimate" if promotion_rule == "benchmark_ratchet" else promotion_rule
        )
        head_to_head_ok = policy_improvement(head_to_head, head_reference_rule)

        benchmark_reference, benchmark_reference_id = (
            _released_benchmark_money(reader, active, benchmarks[0]) if benchmarks else (None, None)
        )
        # A ratchet that can never be released is how a gated run freezes:
        # `simple_v2` ran 33 updates promoting none, `falloff_v1` 57. After this
        # many consecutive rejections the absolute floor drops to
        # `benchmark_grace_min_money` and the non-regression check alone decides;
        # see `_benchmark_ratchet` for why it is that term and not the other.
        #
        # This is only reachable because `rejections_since_promotion` counts
        # through a *held* lineage reset. It did not, and the streak read 0 on all
        # 55 held updates of `market-free-rare-v20`, so the hatch below could never
        # open on the one run that needed it.
        grace_limit = int(config.get("promotion_ratchet_grace_after_rejections", 0))
        ratchet_grace = grace_limit > 0 and rejection_streak >= grace_limit
        benchmark_ratchet = _benchmark_ratchet(
            challenger_benchmarks[0] if challenger_benchmarks else None,
            benchmark_reference,
            benchmark_reference_id,
            config,
            grace=ratchet_grace,
            reference_samples=_benchmark_reference_window(
                reader,
                context.model_id,
                active,
                benchmarks[0],
                int(_limit(config, "benchmark_reference_window")),
            )
            if benchmarks
            else None,
        )
        # Three rules, and the middle arm is the live one for this family.
        #
        # Nothing here gates *training*: `lineage_checkpoint` takes the newest
        # checkpoint promoted or not, `_self_play_pool` deliberately does not
        # filter on `promoted`, and every update writes a checkpoint row
        # regardless -- so a rejection withholds the release and nothing else.
        # What it withholds is `export-active` and the head-to-head incumbent,
        # and withholding the second is most of the point: an unreleased
        # regression stops becoming the thing the next update is measured
        # against.
        #
        # Two guards above are *not* part of this and must keep raising:
        # `_reject_inert_network` on an all-zero exported trunk, and
        # `_reject_degenerate` on a zero-width interval. Both signal a harness or
        # export fault rather than a weak candidate, and both fail silently in the
        # other direction -- an inert export would release as a healthy policy.
        promote = (
            promotion_rule == "always"
            or (
                promotion_rule == "benchmark_ratchet"
                and hard_viability_passed
                and not degenerate_head_to_head
                and benchmark_ratchet["passed"]
            )
            or (
                promotion_rule in {"significant", "point_estimate"}
                and hard_viability_passed
                and head_to_head_ok
                and not degenerate_head_to_head
            )
        )
        gate: dict[str, Any] = {
            "kind": (
                "unconditional_release"
                if promotion_rule == "always"
                else "benchmark_ratchet_and_hard_viability"
                if promotion_rule == "benchmark_ratchet"
                else f"{promotion_rule}_incumbent_improvement_and_hard_viability"
            ),
            "policy_kinds": POLICY_KINDS,
            "promotion_rule": promotion_rule,
            "seed_start": seed_start,
            "benchmark_seed_start": benchmark_seed_start,
            "benchmark_ratchet": benchmark_ratchet,
            "challenger_benchmarks": challenger_benchmarks,
            "resumed_from_checkpoint_id": resumed_from,
            "rejection_streak": rejection_streak,
            "lineage_reset": lineage_reset,
            "initial_baseline": str(baseline_path),
            "incumbent_checkpoint_id": active.checkpoint_id if active is not None else None,
            "head_to_head": head_to_head,
            # True when the candidate played the quiet baseline to an identically
            # zero margin, i.e. it emits nothing either. A rejection, not a fault.
            "degenerate_head_to_head": degenerate_head_to_head,
            "head_to_head_significantly_better": head_to_head_ok,
            "hard_viability": {
                "passed": hard_viability_passed,
                "head_to_head": head_viability,
                "benchmarks": dict(zip(benchmarks, benchmark_viability)),
                # Skip the keys whose default is `None`, which means "fall back to
                # another key" rather than a number -- `_limit` would `float(None)`
                # and take the whole update down with a TypeError. Both such keys
                # resolve through their own branch and both record what they
                # resolved to where they are used, so nothing is lost here.
                #
                # `benchmark_grace_min_money` survived this for as long as it did
                # only because `FREE_CONFIG` happens to set it; a config that left
                # it unset would have hit exactly the same crash.
                "limits": {
                    name: _limit(config, name)
                    for name in VIABILITY_DEFAULTS
                    if config.get(name, VIABILITY_DEFAULTS[name]) is not None
                },
            },
            "passed": promote,
        }
        # A recorded evaluation says whether that arena would let the candidate
        # through, so it has to follow the same rule the promotion did. Under the
        # ratchet the head-to-head no longer carries an improvement term -- it
        # contributes viability and the degeneracy check -- and the index-0
        # benchmark carries the release decision, so it reports the ratchet
        # rather than its activity floor alone.
        ratchet_rule = promotion_rule == "benchmark_ratchet"
        evaluations: list[ModelEvaluation] = [
            ModelEvaluation(
                "incumbent_head_to_head",
                head_to_head,
                (ratchet_rule or head_to_head_ok)
                and not degenerate_head_to_head
                and head_viability["passed"],
                active.checkpoint_id if active is not None else None,
            )
        ]
        evaluations.extend(
            ModelEvaluation(
                _benchmark_kind(index),
                challenger,
                benchmark_ratchet["passed"]
                if ratchet_rule and index == 0
                else viability["passed"],
            )
            for index, (challenger, viability) in enumerate(
                zip(challenger_benchmarks, benchmark_viability)
            )
        )

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
        metrics = {
            "policy_kinds": POLICY_KINDS,
            "resumed_from_checkpoint_id": resumed_from,
            "resumed_from_released": resumed is not None and resumed.released,
            "rejection_streak": rejection_streak,
            "lineage_reset": lineage_reset,
            "episodes_in_buffer": len(buffer.episodes),
            "pinned_demonstrations": len(buffer.pinned),
            "demo_fraction": demo_fraction,
            # Read off the buffer rather than the config, so these say what the
            # draw was actually built with and not what was asked for.
            "priority_alpha": buffer.priority_alpha,
            "priority_beta": buffer.priority_beta,
            # The health check is one line: `sampled_mean_return` strictly above
            # `own_mean_return`, with `mean_weight_effective_fraction` above
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
            "gradient_steps_this_update": steps_to_run,
            "behavior_clone_steps": behavior_clone_steps,
            "behavior_clone_steps_this_update": clone_steps_to_run,
            "mean_behavior_clone_loss": (
                float(np.mean(clone_losses)) if clone_losses else None
            ),
            "mean_loss": float(np.mean(losses)) if losses else None,
            # `mean_loss` is the weighted objective actually optimised. This is
            # the same residual without the importance weights, so a loss curve
            # spanning the arrival of return weighting stays one readable series.
            "mean_unweighted_loss": (
                float(np.mean(unweighted_losses)) if unweighted_losses else None
            ),
            # Read beside `mean_loss`. Both rising together is divergence; loss
            # rising while these hold is a target that genuinely got harder. See
            # the bound argued where they are collected.
            "mean_abs_q": float(np.mean(q_magnitudes)) if q_magnitudes else None,
            "max_abs_q": q_peak if q_magnitudes else None,
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
            # Must be zero. Nonzero means some turn recorded no decision at all,
            # so `searchsorted` skipped past the horizon and that row's credit
            # window quietly shortened -- the exact silent-bias failure the
            # turns-not-decisions `n_step` comment was written about.
            "bootstrap_clamped_rows": clamped_rows,
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
        # deliberately: `reward.py` names these two pairs as the instrument its
        # weights are read through, and a second spelling for the same quantity
        # would be one more thing to know.
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
                # which would not be comparable across runs.
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
                # fault: 27% of mapped orders on the first mirrored episodes,
                # concentrated in `BUY_SEED` (84%), `BUY_ANIMAL` (63%) and `SELL`
                # (45%), because ladder agents buy quantity rungs past the room
                # `pending.seed_room` and the housing gate allow. `clone` forces
                # the bit so the label survives; the cost is that this share of
                # the imitation gradient points at rungs the argmax cannot reach
                # in play. Watch it: a rise means the mask moved, not the meta.
                "orders_masked_infeasible": demo_stats.orders_masked_infeasible,
                "order_diagnostics": _records_diagnostics(buffer.pinned).as_dict(),
            }
        # --- The learner slot's own bookkeeping -----------------------------
        #
        # Deliberately last. The connection is autocommit, so a `population_members`
        # row lands the instant it is written; doing it after everything that can
        # still raise is what keeps a graduation from surviving an update that
        # failed on its way out.
        slot_score = _slot_score(
            reader,
            context.model_id,
            benchmarks[0] if benchmarks else None,
            benchmark_ratchet.get("candidate_money"),
            int(_limit(config, "benchmark_reference_window")),
            slot.batch_floor,
            signal=slot_spec.score_signal,
            rollout=recent_mean_final_money,
        )
        slot = _settle_slot(slot, slot_score, slot_spec)
        metrics.update(slot.as_metrics())
        metrics["lr_score"] = slot_score
        # Which measurement ended up in `lr_score`, recorded beside it. The two
        # signals are different quantities on the same stake, so a series that
        # spans a config change is only readable if each point says which it is.
        metrics["lr_score_signal"] = slot_spec.score_signal
        population: dict[str, Any] = {"action": "train", "reseed_index": slot.reseed_index}
        declared = _declared_model_id(reader, context.model_id)
        graduating = _graduates(slot, slot_spec)
        # Hoisted out of the reseed branch below, because the warm-start predicate
        # is a question about the group and the group has to be read before the
        # answer exists. Read only when a reseed is actually possible, so an
        # ordinary training update still costs no extra query.
        members: list[dict[str, Any]] = []
        if graduating or (slot_spec.warm_start and slot.reseed_index == 0):
            members = mature_members(reader, declared, slot_spec.reserved_min_money)
        warming = not graduating and _warm_starts(slot, slot_spec, members)
        reseeding = graduating or warming

        # Contribution and restart are separate events with separate triggers. When
        # they shared a branch a checkpoint could only enter the pool as its slot
        # was destroyed, so a slot could reseed from `initial_network` with a
        # perfectly good ancestor sitting unreachable.
        #
        # What enters is the *released* checkpoint, never the candidate this update
        # just trained: a mature member is a policy the gate has already accepted.
        #
        # **A warm-starting slot contributes nothing, whatever the age.** Every slot
        # warm-starts inside one generation from a separate process and `_graduate`
        # is autocommit, so a contribution landing mid-generation changes the ranked
        # list the slots still choosing are reading and the ordinal mapping silently
        # collides. Suppressing it leaves the group with no writer for the duration.
        # The `graduating` rule below does not transfer: that slot is finishing and
        # would lose its release entirely, while this one keeps its id and can
        # contribute again in `entry_updates`.
        if active is not None and not warming:
            release_age = _release_age(reader, context.model_id, active.checkpoint_id)
            entry_after = slot_spec.entry_updates
            # A graduating slot contributes whatever the age -- it is finishing
            # anyway, and holding its release back would lose it entirely.
            eligible = graduating or (release_age is not None and release_age >= entry_after)
            population["release_age"] = release_age
            population["entry_after"] = entry_after
            if eligible:
                # The windowed median the ratchet already computed for this
                # incumbent, so the score a member carries is on the same footing
                # as the number the gate refused to let it regress from -- and it
                # costs no arena time, which is the whole reason the ratchet reads
                # a recorded measurement rather than replaying the incumbent.
                score = benchmark_ratchet.get("reference_money")
                if score is None:
                    score = benchmark_ratchet.get("released_money")
                if score is None:
                    score = slot.best_score
                # Discounted by the incumbent's own recorded activity before it
                # becomes a pool rank. See `_population_score`: the cull is
                # `ORDER BY score DESC`, so ranking on money alone lets a policy
                # that never trades enter at the top and be reseeded from.
                #
                # Only the pool rank is discounted. `benchmark_ratchet` itself is
                # untouched -- it is the gate's own comparison and the ratchet
                # reference the next update is measured against, and bending that
                # here would change what the gate means as a side effect of
                # changing what the population prefers.
                scored = _population_score(
                    score,
                    _released_trading_rate(
                        reader, active, benchmarks[0] if benchmarks else ""
                    ),
                    config,
                )
                population["score_detail"] = scored
                population["entered"] = True
                # The strategy fingerprint, read back off the same gate payload
                # the score came from so the two describe one arena series.
                # `money` rides along undiscounted because the archive's floor is
                # a viability question rather than a ranking one -- see
                # `_graduate`. A member with no fingerprint keeps a NULL column
                # and is never reserve-eligible.
                behavior = _released_behavior(
                    reader, active, benchmarks[0] if benchmarks else ""
                )
                if behavior is not None:
                    cell = _behavior_cell(behavior)
                    behavior = {
                        **behavior,
                        "money": float(scored["money"]),
                        "crop_bin": None if cell is None else cell[0],
                        "land_bin": None if cell is None else cell[1],
                    }
                population["behavior"] = behavior
                population.update(
                    _graduate(
                        reader,
                        declared,
                        context.model_id,
                        active.checkpoint_id,
                        float(scored["score"]),
                        slot_spec.mature_capacity,
                        behavior=behavior,
                        reserved=slot_spec.reserved_capacity,
                        reserved_min_money=slot_spec.reserved_min_money,
                    )
                )

        if reseeding:
            # `graduate` when there was something to contribute, `restart` when
            # the slot reached its cut having never released, `warm_start` when it
            # never reached its cut at all and is adopting the group instead. All
            # three reseed, and the three are distinguishable in SQL because which
            # one happened is the first question asked of a roster afterwards.
            population["action"] = (
                "warm_start"
                if warming
                else "graduate"
                if active is not None
                else "restart"
            )

            # Seeded from the top of the mature group, which after a graduation
            # may include the member that just entered. That is correct rather
            # than circular: the slot restarts from the population's best, and if
            # its own release is among them then what the restart buys is the
            # perturbation and the cleared optimizer, not a different ancestor.
            #
            # From the *top*, not from the front -- see `_reseed_source`. The
            # scores are noisy estimates and the best of them is the luckiest one
            # as often as it is the best policy.
            #
            # Re-read after a graduation and not otherwise, which is the whole
            # reason the hoisted read is not simply reused: `_graduate` ran above
            # and is autocommit, so the list this slot should draw from is the one
            # that includes its own release. A warm start contributes nothing, and
            # neither does any other slot warm-starting in the same generation, so
            # the group has no writer and the hoisted list is still current --
            # which is exactly what makes the ordinal spread distinct rather than
            # merely likely to be.
            if graduating:
                members = mature_members(reader, declared, slot_spec.reserved_min_money)
            chosen = _reseed_source(
                members,
                config,
                seed=int(config.get("seed", 0)),
                reseed_index=slot.reseed_index,
                ordinal=_slot_ordinal(config, context.model_id),
                roster=int(config.get("population_slots", 0)),
                spread=warming,
            )
            if chosen is not None:
                source_path = chosen["inference_artifact_path"]
                source_id = chosen["checkpoint_id"]
                population["reseed_source_rank"] = next(
                    index
                    for index, member in enumerate(members)
                    if member["checkpoint_id"] == chosen["checkpoint_id"]
                )
                population["reseed_source_pool"] = min(
                    len(members), max(1, int(config.get("reseed_source_pool", 3)))
                )
                # `reseed_source_rank` above is the index into the *uncapped*
                # list, so the row still reads "rank 4 of thirty checkpoints".
                # These three say how many policies those checkpoints were, which
                # is the number the diversity argument is actually about.
                population["reseed_source_model_id"] = chosen["model_id"]
                population["reseed_source_lineages"] = len(
                    {member["model_id"] for member in members}
                )
                population["reseed_source_candidates"] = len(
                    _capped_members(members, config)
                )
                # Which half of the group the restart came from, and where in the
                # archive. Without these two columns "did reserve draws ever pay"
                # is unanswerable, and that is the only question that retires the
                # archive -- so they are written whether or not it is switched on.
                population["reseed_source_bucket"] = (
                    "reserve" if chosen.get("reserved") else "score"
                )
                population["reseed_source_cell"] = (
                    list(chosen["cell"]) if chosen.get("cell") is not None else None
                )
                population["reseed_reserve_candidates"] = len(
                    _capped_members(
                        [member for member in members if member.get("reserved")], config
                    )
                )
            elif active is not None:
                source_path, source_id = active.inference_artifact_path, active.checkpoint_id
            else:
                source_path, source_id = None, None

            # An adopted warm start is a reseed for *provenance* and not for
            # evidence, and the two used to be conflated at the cost of a whole
            # update per slot. `_materialize_warm_start` drew the ancestor, applied
            # the mask and wrote the bytes before generation 0; `collect` rolled
            # that generation out of them; and the resume above trained from those
            # same bytes. So there is nothing left to materialize, nothing for a
            # warmup to wait for, and no abandoned policy for a batch floor to
            # discard -- `reseed_batch_floor` would have thrown away the generation
            # collected from the very network being adopted.
            #
            # What the marker did carry and still has to happen is the bookkeeping:
            # `lr_reseed_index` advances so `_warm_starts` self-cancels on the next
            # update, and `lr_after_checkpoint_id` bounds the mirror pool below at
            # the adoption. Both survive a restart because they are the update row.
            #
            # `warming` without `adopting_warm_start` is the
            # `population_warm_start_min_updates > 0` case: the slot has trained,
            # nothing was materialized before its collection, and it takes the
            # ordinary reseed path below.
            if warming and adopting_warm_start:
                recorded = json.loads(warm_sidecar_path.read_text())
                if source_id is not None and str(recorded["checkpoint_id"]) != str(source_id):
                    # The group moved between the collection and this update,
                    # which nothing writes to it during generation 0 can cause.
                    # Loud rather than silent: the alternative pairs one
                    # ancestor's weights with another's provenance columns.
                    raise ValueError(
                        "warm start drifted: generation 0 collected from "
                        f"{recorded['checkpoint_id']} but this update drew "
                        f"{source_id}"
                    )
                source_id = str(recorded["checkpoint_id"])
                slot = replace(
                    slot,
                    reseed_index=slot.reseed_index + 1,
                    after_checkpoint_id=checkpoint_id,
                )
                # Rewritten rather than merged: `as_metrics` ran before `warming`
                # was known, so the row would otherwise record the pre-adoption
                # index and `_warm_starts` would fire again next update.
                metrics.update(slot.as_metrics())
                metrics["warm_start_adopted"] = True
                metrics["warm_start_artifact_path"] = str(warm_network_path)
            else:
                source = (
                    OrderQNetwork.load(source_path)
                    if source_path is not None
                    else initial_network(config)
                )
                seeded = reseed.perturb(
                    source,
                    slot_spec.dropout,
                    slot_spec.shrink,
                    seed=int(config.get("seed", 0)) + slot.reseed_index + 1,
                    sigma_reset=noisy.sigma0 if slot_spec.sigma_reset else None,
                )
                # The same guard the exported candidate goes through. A
                # perturbation that somehow zeroed the trunk produces a network
                # with no gradient outside its output bias, and a slot restarted
                # onto one can never leave -- which would read as ten learners
                # quietly going silent.
                _reject_inert_network(seeded)
                reseed_path = context.artifacts / "reseeds" / f"{context.update_id}.npz"
                reseed_path.parent.mkdir(parents=True, exist_ok=True)
                seeded.save(reseed_path)

                # Materialized rather than perturbed in place on the next update,
                # so `collect` and `update` load identical bytes. The alternative
                # -- each side drawing its own mask from a shared seed -- makes the
                # behaviour policy and the trained policy agree only as long as two
                # separate code paths keep agreeing, which is the class of drift the
                # `to_numpy` exporter bug in this file already cost a debugging
                # session to.
                metrics.update(
                    {
                        "reseed_pending": True,
                        "reseed_artifact_path": str(reseed_path),
                        "reseed_source_checkpoint_id": source_id,
                        "reseed_batch_floor": context.cutoff_batch_seq,
                        "reseed_after_checkpoint_id": checkpoint_id,
                        "reseed_dropout": slot_spec.dropout,
                        "reseed_shrink": slot_spec.shrink,
                    }
                )
            population["reseed_source_checkpoint_id"] = source_id
            population["mature_members"] = len(members)

        metrics["population"] = population
        gate["population"] = population

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
            promote,
            tuple(evaluations),
        )
