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

Both the training resume and the rollout policy follow `lineage_checkpoint`, the
same as the scorer. An earlier version gated rollouts on a hard viability check
of their own, so collection stayed on the cold start until something passed the
gate. Nothing did: three hundred batches were collected by an untrained,
near-silent policy, its measured orders per decision decayed from 0.19 to 0.03 as
epsilon annealed over a policy that never moved, and every candidate trained on
that data was rejected. Release is gated; the behaviour distribution must not be.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import torch
from torch import nn

from agents.dqn_market_free.diagnostics import OrderDiagnostics, diagnose_replay
from agents.dqn_market_free.order_space import (
    ACTION_COUNT,
    ACTION_INDEX,
    MASKED_SCORE,
    NOOP_INDEX,
)
from agents.dqn_market_free.policy import OrderQNetwork
from agents.dqn_market_free.training import network as net_module
from agents.dqn_market_free.training import rare, reseed
from agents.dqn_market_free.training.clone import load_demonstrations
from agents.dqn_market_free.training.collect import (
    LEARNER_ENTRY_POINT,
    EpisodeRecord,
    _single_threaded_workers,
    actor_sigma_scales,
    collect_batch,
    load_episode_batch,
    save_episode_batch,
)
from agents.dqn_market_free.training.replay import (
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

SELF_SPEC = "agents/dqn_market_free/main.py"
LEARNER_MANIFEST = "agents/dqn_market_free/match_agent.toml"
SOURCE_KIND = "free_market_rollout"
_PROMOTION_RULES = {"significant", "point_estimate", "always", "benchmark_ratchet"}

#: The row `rare.py` exists for, kept as a named batch column. Not special to the
#: mechanism -- the widening knows nothing about land -- but it is the action whose
#: absence was measurable across four sessions, so it is the one a run is read on.
LAND_INDEX = ACTION_INDEX[("BUY_LAND", None, 0)]

VIABILITY_DEFAULTS = {
    "viability_min_mean_money_ratio": 0.80,
    "viability_min_median_money_ratio": 0.75,
    # The gate's other half. A policy that never trades preserves its money
    # perfectly, so the money floors above pass -- they catch bankruptcy and wave
    # the NOOP attractor straight through.
    #
    # Only applied once the model has a released incumbent to fall back to. A
    # cold start is legitimately near-silent, and failing it there would leave
    # the run with no promotable first checkpoint.
    "viability_min_orders_per_decision": 0.02,
    # What the money floors above measure against before anything is released.
    #
    # It used to be the quiet baseline's own seat, and that is not a bar a
    # trading policy can clear: the baseline emits nothing, so it keeps its full
    # 3,000 stake on every seed and `0.80 * 3000 = 2400` asks the candidate to
    # trade for a season and lose nothing. `coldstart_v1` failed exactly that,
    # on `mean_money`/`median_money` only, at seq 40/60/80/120/140 -- seven
    # updates, zero promotions, and with nothing released the snapback machinery
    # has nothing to snap back to, so the run cannot escape on its own either.
    "viability_cold_start_reference_money": 1500.0,
    # The fixed-reference release bar, read off the candidate's own money in the
    # index-0 benchmark series. Absolute, because everything else in this gate is
    # relative to the previous update and therefore drifts: `falloff_v1` released
    # nine consecutive updates whose benchmark money ran 2979, 54, 20, 1146, 18,
    # 359, 1, 11, 107 while every head-to-head compared it only against whatever
    # it had itself produced twenty batches earlier.
    "benchmark_release_min_money": 1000.0,
    # The same bar for the first release only. Lower on purpose: a foothold, not
    # a good policy. Below it a run has no incumbent at all, and that is the
    # state the two anti-stall mechanisms are both inert in.
    "benchmark_cold_start_min_money": 750.0,
    # The bar once `promotion_ratchet_grace_after_rejections` has stood the ratchet
    # down. Left unset it is `benchmark_cold_start_min_money`, which keeps every
    # existing config reading exactly as it did.
    #
    # It exists because the absolute floor turned out to be the term a stalled run
    # cannot clear, not the relative one, which is the opposite of what grace
    # originally assumed. Measured on `market-free-rare-v20`: 57 consecutive
    # rejections against a floor of 1,000 with the candidate's benchmark money
    # between 0.93 and 427, a median near 35, and a win rate of exactly 0.000 on
    # every one of them -- while the *reference* those rejections were ratcheted
    # against was 145.8, so the incumbent being protected could not have passed its
    # own floor either. Dropping the floor and keeping the ratchet is what makes
    # the hatch escapable: eight of those 57 updates already measured above
    # `0.9 * 145.8`, so the relative term admits real movement while still refusing
    # the walk-itself-down decay the ratchet exists to catch.
    "benchmark_grace_min_money": None,
    # Non-regression against the released incumbent's *recorded* benchmark money.
    # Deliberately not 1.0 -- a gate that admits no noise at all is a gate that
    # eventually admits nothing.
    "benchmark_money_ratchet": 0.9,
    # How many benchmark measurements the ratchet reference is a median over.
    # 1 is the single stored draw, byte for byte the behaviour before the key
    # existed.
    #
    # Above 1 because one 100-seed arena series is a draw and not a number here.
    # `falloff_v1` measured 61 of them without the policy meaningfully moving --
    # the snapback held it on `9071d2ad` for 55 -- and they ran 0.93 to 427 around
    # a median of 36.7. `a_money_median` is 2 against a mean of 32.6, so the policy
    # is bankrupt on most seeds and the mean is a lottery on the survivors.
    #
    # Ratcheting one such draw against another is not a weak gate, it is a biased
    # one: the stored reference is whatever was lucky enough to promote, so each
    # release re-aims the bar at a higher quantile of unchanged noise. `9071d2ad`
    # recorded 145.8, its own p75, and the run then failed `no_regression` 57 times
    # without having regressed.
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

    The scorer's shared `self_play_pool` filters on `promoted`, which is right
    for a model whose gate passes. It is not right here, for the same reason the
    module docstring gives for rollouts following `lineage_checkpoint` rather
    than the release: *release is gated; the behaviour distribution must not be*.
    Filtering on promotion makes self-play unreachable exactly when it would help
    most -- `market-free-simple-v2` ran 33 updates and promoted none of them, so
    the promoted pool was empty on every batch and the mirror never played.

    The scorer's helper is left alone. Its promotion record is healthy, and a
    shared helper that silently changed which checkpoints two agents self-play
    against is how one of them stops matching its own gate.

    `after_checkpoint_id` bounds the pool below at a reseed. A restarted slot
    whose mirror is still the graduated policy is training against the trajectory
    it was restarted to escape, and the pool refills with its own checkpoints
    within a few updates anyway -- so this only bites for exactly as long as it
    has to. Returning nothing while it does is the intended answer: the caller
    falls back to the mature group, which is a better mirror for a fresh slot
    than the wreckage of the previous one.
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

    The spec actually played also carries a per-episode trace path and
    exploration seed; folding those into the identity would mint one revision
    per rollout episode. What defines the policy is its weights and how much it
    explores, so the identity spec carries only those -- and stays resolvable.

    `sigma_scale` is bucketed to four decimals and carried because it is how much
    the policy explores. Only the *base* is: the per-actor spread around it is
    deliberately left out, since folding sixteen of them in would mint eight
    revisions a batch, none with enough games to rate.
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
        results = play_series(spec, opponent, range(seed_start, seed_start + seeds), {}, jobs)
    summary = summarize(spec, opponent, results)
    diagnostics = _arena_order_diagnostics(results)
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
    }


def _arena_order_diagnostics(results: dict[int, tuple[Any, Any]]) -> OrderDiagnostics:
    """Aggregate the learner's seat from complete paired arena results."""

    diagnostics = OrderDiagnostics()
    for forward, reverse in results.values():
        # Match summarize(): only complete seat-swapped pairs contribute.
        if forward.failed or reverse.failed:
            continue
        if isinstance(forward.replay, dict):
            diagnostics += diagnose_replay(forward.replay, 0)
        if isinstance(reverse.replay, dict):
            diagnostics += diagnose_replay(reverse.replay, 1)
    return diagnostics


def _records_diagnostics(records: list[EpisodeRecord]) -> OrderDiagnostics:
    diagnostics = OrderDiagnostics()
    for record in records:
        diagnostics += record.order_diagnostics
    return diagnostics


def _limit(config: dict[str, Any], name: str) -> float:
    return float(config.get(name, VIABILITY_DEFAULTS[name]))


def _order_viability(
    result: dict[str, Any], config: dict[str, Any], *, require_activity: bool = False
) -> dict[str, Any]:
    """Did the candidate trade at all, in one arena.

    Read off `trading_rate`, the share of turns the simulator acted on, not off
    the wire order count. Those differ: `HIRE n` is one decision whose wire form
    is n rows, so a policy that only ever hires reports 13 wire orders per
    decision while trading on one turn in fifty. The floor is about how often the
    policy acts, so it is the decision share that belongs here.

    `OrderDiagnostics.as_dict` publishes that share already, over its own
    `max(1, decisions)` guard. Take it as given rather than recomputing from the
    counts: the recomputation named a key the payload does not carry, so the rate
    read 0.0 for every real arena and the floor failed unconditionally wherever
    `require_activity` was set.

    There is still no upper limit. Over-trading can only show up as lost money,
    which the money floors in `_arena_viability` already measure. The three
    cross-slot limits that used to live here (exact duplicates, repeated targets,
    buy/sell conflicts within one decision) were counting a failure mode that a
    single-order decision cannot have.
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

    Only meaningful once the paired margin has already been found to be
    zero-width, which against the quiet baseline is a rejection rather than a
    harness fault. The question this answers is *which* of the two it is: the
    NOOP collapse this family falls into, or a candidate that traded all season
    against a counterparty offering nothing.

    A zero margin does not settle it. Round-trip trades are cash-neutral against
    an unchanged market, so `market-free-population-v25@learner_05` placed 16,686
    orders over 35,950 decisions at cutoff 143 and still finished every paired
    seed on exactly 3,000.0 against a baseline that did the same.

    Read off `trading_rate` for the same reason `_order_viability` does: `HIRE n`
    is one decision whose wire form is n rows, so the order count overstates a
    hire-only policy by an order of magnitude. Shares the activity floor, so
    setting `viability_min_orders_per_decision = 0` turns this off with it.
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
) -> dict[str, Any]:
    """Absolute money preservation plus order activity for one arena.

    Only applied where `reference` is the *same opponent on the same seeds*, so
    the floor compares like with like. That is the head-to-head, where the
    reference is the incumbent's own seat.
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
    }


def _released_trading_rate(reader, active, opponent: str) -> float | None:
    """The released incumbent's own recorded trading rate on this benchmark.

    The activity half of `_released_benchmark_money`, read the same way and for
    the same reason: the number was written into that update's `gate_json` when
    the incumbent was itself a candidate, so this costs one row and no arena time.

    Matched on the opponent spec, like the money, so reconfiguring `benchmarks`
    drops the term instead of comparing against a different opponent's activity.

    `None` means *not recorded* -- an older checkpoint, or a gate payload from
    before this was written -- and is deliberately not the same as zero. A missing
    measurement is not evidence of idleness, and treating it as such would retire
    every pre-existing member the first time the pool was culled.
    """

    if active is None:
        return None
    row = reader.fetchone(
        "SELECT gate_json FROM model_training_checkpoints WHERE checkpoint_id = %s",
        [active.checkpoint_id],
    )
    if row is None:
        return None
    gate = row[0]
    if isinstance(gate, (str, bytes, bytearray)):
        gate = json.loads(gate)
    if not isinstance(gate, dict):
        return None
    viability = gate.get("hard_viability")
    if isinstance(viability, dict):
        for item in viability.get("benchmarks") or []:
            if isinstance(item, dict) and item.get("trading_rate") is not None:
                return float(item["trading_rate"])
        head = viability.get("head_to_head")
        if isinstance(head, dict) and head.get("trading_rate") is not None:
            return float(head["trading_rate"])
    for result in gate.get("challenger_benchmarks") or []:
        if not isinstance(result, dict) or str(result.get("opponent", "")) != opponent:
            continue
        diagnostics = result.get("order_diagnostics")
        if isinstance(diagnostics, dict) and diagnostics.get("trading_rate") is not None:
            return float(diagnostics["trading_rate"])
    return None


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

    Measured on this run at the time this was written: eight of ten slots had
    collapsed to a trading rate of 0.0000 carrying 468 to 924 coins, while the two
    still trading (0.32 and 0.88) carried 8 and 231. Money and activity were
    anti-correlated across the whole roster.

    The discount is a linear ramp to `population_activity_reference`, saturating
    at 1.0 above it. Deliberately not a threshold: hard viability already owns the
    binary "did it act at all" question at
    `viability_min_orders_per_decision`, and repeating that shape here would only
    move the same cliff. What this adds is that *barely* clearing that floor no
    longer buys a hoarder the top of the pool.

    No upper penalty, for the reason `_order_viability` gives: over-trading can
    only show up as lost money, which the money term already carries.

    Stated plainly because it is a judgement rather than a measurement: the
    reference is a policy choice about how much activity counts as "trading for
    real", and it is set from the observed bimodality (0.000 against 0.32-0.88)
    rather than from any result. Setting it to 0 restores raw-money ranking.
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

    A pure read of a measurement that has already been paid for. Re-simulating
    the incumbent against every benchmark is what the comment above
    `challenger_benchmarks` removed for wall clock, and this does not reinstate
    it: the number was written into that update's `gate_json` when the incumbent
    was itself a candidate, so the ratchet costs one row.

    Matched on the opponent spec rather than on index 0. Reconfiguring
    `benchmarks` then resets the ratchet instead of silently comparing the
    candidate's score against a different opponent's, which is the one way this
    could be wrong without any of it looking wrong.

    `None` means *no ratchet term* -- there is nothing comparable to ratchet
    against -- and never "the ratchet passes". The absolute floor still binds in
    that case, and binds at the cold-start value.
    """

    if active is None:
        return None, None
    row = reader.fetchone(
        "SELECT gate_json FROM model_training_checkpoints WHERE checkpoint_id = %s",
        [active.checkpoint_id],
    )
    if row is None:
        return None, None
    gate = row[0]
    if isinstance(gate, (str, bytes, bytearray)):
        gate = json.loads(gate)
    if not isinstance(gate, dict):
        return None, None
    for result in gate.get("challenger_benchmarks") or []:
        if not isinstance(result, dict) or str(result.get("opponent", "")) != opponent:
            continue
        if result.get("errors") or result.get("a_money_mean") is None:
            return None, None
        return float(result["a_money_mean"]), active.checkpoint_id
    return None, None


def _benchmark_reference_window(
    reader, model_id: str, active, opponent: str, window: int
) -> list[float]:
    """Every benchmark measurement taken while the current incumbent has stood.

    One arena series is not a number, it is a draw, and on this family it is a
    very wide one. `falloff_v1` measured its benchmark money 61 times without the
    policy meaningfully moving -- the snapback held it on one checkpoint for 55 of
    them -- and got a median of 36.7, a p90 of 186 and a maximum of 427. The
    reason is visible in the same payload: `a_money_median` is 2 against a mean of
    32.6, so the policy is bankrupt on most of the 100 seeds and the mean is
    carried by a handful of survivors.

    Ratcheting a draw from that against a single earlier draw from it is a
    coin-flip wearing a gate's clothes, and it is a *biased* coin-flip: the stored
    reference is whatever was lucky enough to promote, so each release raises the
    bar to a higher quantile of unchanged noise and the gate tightens on itself.
    `9071d2ad` recorded 145.8, which is its p75; the run then failed
    `no_regression` 57 times against a policy it had not actually regressed from.

    So the reference is a median over the window rather than one draw. Candidates
    measured while an incumbent stands are near-neighbours of it -- under a held
    snapback, literally the same weights plus one update's gradient steps -- so
    they are repeated samples of the same underlying quantity and the median of
    them estimates it far better than any single one.

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
        gate = row[0]
        if isinstance(gate, (str, bytes, bytearray)):
            gate = json.loads(gate)
        if not isinstance(gate, dict):
            continue
        for result in gate.get("challenger_benchmarks") or []:
            if not isinstance(result, dict) or str(result.get("opponent", "")) != opponent:
                continue
            if result.get("errors") or result.get("a_money_mean") is None:
                continue
            measurements.append(float(result["a_money_mean"]))
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

    `grace` drops the **floor**, to `benchmark_grace_min_money`, and keeps the
    ratchet. That is the reverse of how it started, and the reversal is measured
    rather than argued. The original reading -- floor absolute and permanent,
    ratchet relative and droppable -- assumed the floor is the term a genuinely
    improving run clears on its way past a stall. `market-free-rare-v20` is the
    counterexample: 57 consecutive rejections, every one of them failing
    `min_money` against a floor of 1,000 with the candidate between 0.93 and 427,
    and the reference protecting them sitting at 145.8 -- below its own floor. The
    floor was not waiting for the run to improve, it was unreachable, and grace as
    written could only stand down the one term the run was already clearing on
    eight of those 57 updates.

    Keeping the ratchet under grace is what stops this from becoming
    `promotion_rule = "always"` in disguise. The decay it exists to catch is
    relative, so the relative term is the one worth holding on to when the run has
    proven it cannot clear an absolute bar.
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
    samples = sorted(reference_samples or [])
    if reference is not None:
        samples = sorted([*samples, reference])
    reference_used = reference
    reference_kind = "released_measurement"
    if len(samples) > 1:
        middle, odd = divmod(len(samples), 2)
        reference_used = (
            samples[middle] if odd else 0.5 * (samples[middle - 1] + samples[middle])
        )
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
# The single ever-resuming lineage this replaced measured its own failure over
# `market-free-falloff-v18`'s batches 1100-1688: rollout money oscillating around
# 5,000 with no trend, `mean_loss` pinned at 0.0015, `mean_abs_q` at 0.2, and
# benchmark money against pilkwang between 0.9 and 1,208 from a 3,000 stake. A
# trajectory that has stopped moving cannot be pushed out of its basin by the
# thing that put it there, and every exploration lever tried through v19 was
# aimed at the behaviour distribution rather than at the trajectory.
#
# So a roster of named learner slots trains in parallel (one `training_models`
# row each, `effective_model_id` namespacing them under one `declared_model_id`)
# and a slot that stops improving is *ended* rather than nursed: its released
# checkpoint graduates into `population_members`, the mature group drops its
# lowest scorer to stay at capacity, and the slot restarts from the best mature
# member with its learning rate back at zero and a fraction of its weights
# redrawn.


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
    decay_factor: float
    graduate_cut: float
    min_updates: int
    mature_capacity: int
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
        # Relative, not absolute. The score is benchmark money, which spans three
        # orders of magnitude across a run, so a fixed delta is either noise at
        # the top of that range or unreachable at the bottom.
        min_delta=float(config.get("lr_plateau_min_delta", 0.05)),
        decay_factor=float(config.get("lr_decay_factor", 0.5)),
        # Below this the slot is finished. 0.0 switches the whole mechanism off
        # and reproduces the fixed-rate single lineage exactly.
        graduate_cut=float(config.get("lr_graduate_cut", 0.0)),
        # A floor on slot lifetime, so warmup noise cannot end a slot that has
        # not had a chance to train. Counted from the reseed, not from the run.
        min_updates=int(config.get("lr_min_updates_before_graduation", 10)),
        mature_capacity=int(config.get("mature_capacity", 30)),
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
    if spec.mature_capacity < 1:
        raise ValueError(f"mature_capacity must be positive: {spec.mature_capacity}")
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
    metrics = row[0]
    if isinstance(metrics, (str, bytes, bytearray)):
        metrics = json.loads(metrics)
    return metrics if isinstance(metrics, dict) else {}


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

    `score` is the windowed benchmark median rather than `mean_loss`. Loss is not
    a progress signal on this family -- it sat at 0.0015 across 500 batches of a
    policy that was going nowhere -- and it is not even the right *sign*: a
    genuinely improving policy reaches harder states and its TD residual rises.

    `None` means the benchmark did not measure, which is not a stall. An update
    that failed to produce a comparison says nothing about progress, and counting
    it would let a broken arena graduate every slot in the roster.

    Warmup is not a stall either, and for the same reason. A slot climbing from
    zero is deliberately under-trained, so its scores are evidence about the ramp
    rather than about the policy -- counting them spends the patience budget
    before the slot has taken a single step at the resolved rate, and worse, sets
    `best_score` from a measurement the slot is guaranteed to beat.
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


def _slot_score(
    reader,
    model_id: str,
    opponent: str | None,
    candidate: float | None,
    window: int,
    batch_floor: int,
) -> float | None:
    """The slot's progress signal: a median of its own recent benchmark money.

    A median rather than the latest draw, for the reason `_benchmark_reference_window`
    argues at length: one 100-seed arena series on this family swings between 0.93
    and 427 on an unchanged policy, because `a_money_median` is 2 against a mean of
    32.6 and the mean rides on a handful of survivors. A plateau detector reading
    single draws would find an "improvement" by luck every few updates and the rate
    would never decay -- which is the stagnation this exists to end, reproduced by
    the thing meant to end it.

    Bounded below by the slot's own reseed. Measurements from before it describe a
    policy this slot no longer is, and including them would let an inherited high
    score keep a restarted slot's plateau counter pinned.

    `None` when nothing measured, which `_settle_slot` treats as "no information"
    rather than as a stall.
    """

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
        gate = row[0]
        if isinstance(gate, (str, bytes, bytearray)):
            gate = json.loads(gate)
        if not isinstance(gate, dict):
            continue
        for result in gate.get("challenger_benchmarks") or []:
            if not isinstance(result, dict) or str(result.get("opponent", "")) != opponent:
                continue
            if result.get("errors") or result.get("a_money_mean") is None:
                continue
            samples.append(float(result["a_money_mean"]))
    samples.sort()
    middle, odd = divmod(len(samples), 2)
    return samples[middle] if odd else 0.5 * (samples[middle - 1] + samples[middle])


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


def mature_members(reader, declared_model_id: str) -> list[dict[str, Any]]:
    """The live mature group, best first.

    Missing artifacts are dropped rather than raising: a member whose `.npz` has
    been cleaned off disk is unusable as a reseed source or a mirror opponent,
    and the run has to keep going. It stays a live row so the count still reflects
    what graduated.
    """

    rows = reader.fetchall(
        """
        SELECT p.member_id, p.model_id, p.checkpoint_id, p.score,
               c.inference_artifact_path
        FROM population_members p
        JOIN model_training_checkpoints c USING (checkpoint_id)
        WHERE p.declared_model_id = %s AND p.retired_at IS NULL
        ORDER BY p.score DESC, p.member_id
        """,
        [declared_model_id],
    )
    return [
        {
            "member_id": str(row[0]),
            "model_id": str(row[1]),
            "checkpoint_id": str(row[2]),
            "score": float(row[3]),
            "inference_artifact_path": str(row[4]),
        }
        for row in rows or []
        if Path(str(row[4])).is_file()
    ]


def _release_age(reader, model_id: str, checkpoint_id: str | None) -> int | None:
    """How many of this slot's updates a release has stood without being replaced.

    The lag on pool entry, and it is load-bearing rather than cautious. A release
    that can be copied the moment it is promoted closes a loop: whichever slot
    released most recently becomes the ancestor for whichever slot reseeds next,
    and the roster converges on clones of the current best. That is a greedy
    hill-climb running ten copies, which is strictly worse than the single lineage
    the population replaced.

    Gating entry on graduation, as this did before, *is* a lag -- an enormous one,
    a whole decay cycle and only if the slot dies. This is the same idea sized to
    something a healthy slot can reach.

    What the count actually tests: a policy has to survive a stretch of its own
    slot's continued training without a newer candidate beating it. A slot
    promoting on every update is improving fast, its current best is a moving
    target, and it contributes nothing until it settles.

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


def _reseed_source(
    members: list[dict[str, Any]], config: dict[str, Any], *, seed: int, reseed_index: int
) -> dict[str, Any] | None:
    """Which mature member a restarting slot is rebuilt from.

    Drawn uniformly from the top `reseed_source_pool` by score rather than taken
    from the front of the list, for two reasons.

    The first is the argument this file already makes about single draws, applied
    one level up. A member's score is a windowed median of a benchmark that swings
    three orders of magnitude, so it is an *estimate* of that policy's worth.
    Taking the argmax of noisy estimates does not select the best policy, it
    selects the luckiest estimate -- and it does so every time, because the
    estimate is stored and reused. `_benchmark_reference_window` records what the
    same mistake cost the ratchet.

    The second is diversity, which is the whole reason a population exists. If
    every restart pulls from index 0 then ten slots become ten perturbations of
    one ancestor exploring one basin, which is a worse search than the single
    lineage it replaced. Sampling from the top keeps the pressure toward good
    policies while letting the roster spread.

    Ranked rather than score-weighted on purpose. A softmax over these scores
    would need a temperature tied to their scale, and the scale is neither stable
    -- money spans orders of magnitude -- nor fixed, since `_population_score`
    discounts it by activity. Top-k is scale-free and survives both.

    Seeded from the slot and its reseed count so a restart is reproducible, and
    with entropy distinct from the perturbation's own seed so the choice of
    ancestor and the mask drawn from it do not move together.
    """

    if not members:
        return None
    pool = max(1, int(config.get("reseed_source_pool", 3)))
    top = members[:pool]
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
    """

    member_id = hashlib.sha256(
        f"{declared_model_id}\0{checkpoint_id}".encode()
    ).hexdigest()[:24]
    reader.fetchall(
        """
        INSERT INTO population_members
            (member_id, declared_model_id, model_id, checkpoint_id, score)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (member_id) DO UPDATE SET score = excluded.score
        RETURNING member_id
        """,
        [member_id, declared_model_id, model_id, checkpoint_id, float(score)],
    )
    retired = reader.fetchall(
        """
        UPDATE population_members SET retired_at = current_timestamp,
                                      retired_reason = 'capacity'
        WHERE member_id IN (
            SELECT member_id FROM population_members
            WHERE declared_model_id = %s AND retired_at IS NULL
            ORDER BY score DESC, member_id
            OFFSET %s
        )
        RETURNING member_id, score
        """,
        [declared_model_id, capacity],
    )
    return {
        "member_id": member_id,
        "checkpoint_id": checkpoint_id,
        "score": float(score),
        "capacity": capacity,
        "retired": [
            {"member_id": str(row[0]), "score": float(row[1])} for row in retired or []
        ],
    }


def _behavior_clone_step(
    online: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
    device: torch.device,
) -> float:
    """Fit the action head to a stored expert action for one minibatch.

    Ordinary classification over `ACTION_COUNT` classes. The previous ten-slot
    action made this a cross-entropy over ten heads whose targets were 88% NOOP
    -- stored donor replays average 1.2 orders a turn -- so the clone's
    minimum-loss answer was silence, which is the failure demonstrations exist to
    prevent.
    """

    state, chosen, mask = buffer.sample_demonstrations(batch_size)
    optimizer.zero_grad(set_to_none=True)
    logits = online(torch.from_numpy(state).to(device))
    # Infeasible actions carry no probability, so the clone never spends its
    # capacity learning to rank orders the game would refuse.
    logits = logits.masked_fill(~torch.from_numpy(mask).to(device), MASKED_SCORE)
    loss = nn.functional.cross_entropy(logits, torch.from_numpy(chosen).to(device))
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
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

    Returns `(flip_curve, order_curve)`, both keyed by the same rungs. `flip_curve`
    maps a multiplier to the share of probe states where the perturbation changed
    the argmax -- the quantity epsilon named directly. `order_curve` maps it to the
    share of states the greedy policy would sit out that the perturbation talks
    into a real order, and is empty when the greedy policy never chooses NOOP on
    the probe, since there is then nothing to talk out of it.

    Both are swept, and the second one is swept *because the first is not enough
    on its own*. On a collapsed policy every action is near-tied around NOOP, so
    the flip rate reads high while the agent trades nothing: `market-free-noisy-v6`
    measured 0.28-0.39 against a 0.05 target and had sigma divided by up to six for
    it, throttling all eight actors to `[0.33 .. 0.010]`. A controller cannot tell
    those apart from one curve, which is why the controller reads this one.

    Swept rather than measured once because one reading says whether exploration
    is still there while the curve says what it would take to put it back, which
    is what `collect` needs in order to choose a scale.

    The ladder is clipped to `scale_limit` so a run is never measured at a scale
    `resolve_sigma_scale` is not allowed to return, which would put a point on
    the curve that the resolver silently clamps away from.

    Leaves the module's noise in whatever state the last draw left it. Every
    caller either re-draws before its next forward or is finished with the
    module, and silencing it here would be the more surprising behaviour.
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

    `collect` and `train` are different methods on different rounds, so the
    number crosses through the database rather than through memory -- the same
    route `lineage_checkpoint` and `_self_play_pool` already take. That also
    makes it survive a restart, and makes what an actor played at answerable in
    SQL beside the measurement that chose it.

    1.0 covers every way there is nothing to read: a cold start, a run with the
    controller off, and a lineage whose updates predate the key. Deliberately not
    the newest row unconditionally -- a rejected update still measured a real
    curve on a real network, so its scale is usable, but a row that never ran the
    probe must not be read as a scale of nothing.

    `->> IS NOT NULL` rather than the `?` containment operator, which does say
    "has this key" more directly. `?` is a placeholder in several drivers and
    survives here only by the grace of psycopg's parameter style; `->>` returns
    NULL both for an absent key and for a JSON null, which is the same set of
    rows and cannot break on a driver change.
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
    """Fail loudly when the exported network can only ever emit NOOP."""

    if net_module.has_inert_hidden_trunk(network):
        raise RuntimeError(
            "exported free-market network has all-zero hidden weights: every action "
            "ties to NOOP and the agent never trades. An all-zero network also has "
            "no gradient outside its output bias, so the run cannot recover -- the "
            "hidden trunk collapsed after the cold start, which starts it random"
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
            # Checkpoint paths are content-addressed and immutable; `initial` is
            # not, so it is rewritten in case an older run left a stale one behind.
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

            # Checked here rather than beside `base`, because an empty
            # `opponents` list is a legitimate configuration until the mirror
            # pool has been consulted: `compose_opponents` returns the pool
            # unmixed when there is no base, which is how pure self-play is
            # expressed. Only having neither is an error.
            if not opponents:
                raise ValueError("free-market collection needs at least one opponent")

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
            # One scale per seed pair, so a single batch carries both
            # exploratory and near-greedy data. The *identity* policy keeps the
            # base: folding sixteen per-actor scales into it would mint eight
            # revisions a batch, none of them with enough games to rate.
            #
            # `sigma_base` is what the last probe resolved, so actor 0 plays at
            # the configured rate and the spread runs down from there. With the
            # controller off it is 1.0 and this list is byte for byte what it was
            # before the controller existed.
            #
            # The identity policy carries `sigma_base` only. The per-actor spread
            # around it is deliberately left out: folding sixteen scales in would
            # mint eight revisions a batch, none with enough games to rate.
            #
            # One ladder, downward only. Both mechanisms that put rungs above the
            # setpoint are retired -- the idle burst at v15, `noisy_actor_hot`
            # with the population -- and for one measured reason: widening the
            # whole policy bankrupts the wallet before rare behaviour becomes
            # affordable. `rare_sigma_gain` widens single rows instead, and
            # escaping a stuck trajectory is now the population's job rather than
            # the ladder's.
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
            batch_chosen, _ = rare.visit_counts(records)
            land_episodes = sum(
                1 for record in records if int(np.count_nonzero(record.order_chosen == LAND_INDEX))
            )
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
                    # What the widening was worth this batch and whether it moved
                    # anything. Entropy is the gain-free reading -- a policy
                    # pouring every turn into three rows reads low here whatever
                    # its money curve says -- and `buy_land_rate` is the specific
                    # row that motivated the mechanism, kept as a named column so
                    # the question that started this is answerable in one query.
                    "rare_sigma_gain": rare_gain if rare_gain > 0.0 else None,
                    # Counted from the widest end of the ladder, so the clean arm
                    # is the greedy tail. A money series filters on both columns
                    # together.
                    "rare_sigma_pairs": rare_sigma_pairs if rare_gain > 0.0 else None,
                    "selection_entropy": rare.selection_entropy(batch_chosen),
                    "buy_land_rate": land_episodes / max(1, len(records)),
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
        demo_stats = None
        demo_limit = int(config.get("demonstration_episodes", 0))
        if demo_limit > 0:
            demos, demo_stats = load_demonstrations(
                reader,
                limit=demo_limit,
                min_final_money=float(config.get("demonstration_min_money", 10_000.0)),
                family_id=config.get("demonstration_family"),
                gamma=gamma,
            )
            buffer.add(demos, pinned=True)

        warmup = int(config.get("warmup_episodes", 16))
        if len(buffer.episodes) < warmup:
            raise ValueError(f"free-market training requires at least {warmup} replay episodes")

        noisy = _noisy(config)
        online = net_module.build(noisy=noisy.build_kind, sigma0=noisy.sigma0).to(device)
        target = net_module.build(noisy=noisy.build_kind, sigma0=noisy.sigma0).to(device)
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
            seed_network = (
                OrderQNetwork.load(reseed_source)
                if reseed_source and Path(reseed_source).is_file()
                else initial_network(config)
            )
            online = net_module.from_numpy(seed_network, online)
            target.load_state_dict(online.state_dict())
            _reset_learning_rates(optimizer, clone_optimizer, config, slot.learning_rate)

        # `reduction="none"`: every row carries its own importance-sampling
        # weight from the return-weighted draw, and a mean taken inside the loss
        # would average them away before they could apply.
        loss_fn = nn.SmoothL1Loss(reduction="none")
        batch_size = int(config.get("batch_size", 256))
        target_sync = int(config.get("target_sync", 2000))
        demo_fraction = float(config.get("demo_fraction", 0.25)) if buffer.pinned else 0.0
        # Scale the optimizer budget by the data the update actually consumes,
        # as the scorer does. A fixed count meant one update spent 100 steps on
        # `min_new_batches` worth of rollouts -- 20 batches, ~230k decisions --
        # and left `target_sync` needing twenty updates to fire once.
        batches_per_update = max(
            1, int(config.get("batches_per_update", len(context.new_batches)))
        )
        steps_to_run = int(config.get("gradient_steps_per_batch", 100)) * batches_per_update
        losses: list[float] = []
        # The same residual without the importance weights. `mean_loss` is the
        # quantity actually optimised and has to stay that, but it changes scale
        # the moment weighting turns on, so a loss curve spanning the change
        # would read as a step that never happened. This one is comparable to
        # every generation recorded before it.
        unweighted_losses: list[float] = []
        # Effective sample size of the weight vector, as a fraction of the batch.
        # 1.0 is an unweighted batch; 1/batch_size is a batch whose gradient is
        # one row. This is the number that says whether the correction has
        # quietly turned 256 rows into 30.
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
                _behavior_clone_step(online, clone_optimizer, buffer, batch_size, device)
            )
            behavior_clone_steps += 1
        if clone_steps_to_run:
            # Bootstrap from the cloned policy rather than a stale pre-clone
            # target network during the DQN fine-tuning that follows.
            target.load_state_dict(online.state_dict())

        for _ in range(steps_to_run):
            batch = buffer.sample(batch_size, demo_fraction=demo_fraction)
            state_tensor = torch.from_numpy(batch.state).to(device)
            chosen = torch.from_numpy(batch.chosen).to(device)

            # The bootstrap is computed *before* the graded forward, which is a
            # reordering NoisyNet forces rather than a preference. Rainbow draws
            # independent noise for the action-selection pass, the evaluation
            # pass and the graded pass, and `reset_noise` rebinds the epsilon
            # buffers -- so a resample landing between `predicted`'s forward and
            # its `backward` would rebind a tensor autograd saved for the
            # gradient. Doing the no-grad block first leaves the online net's
            # last draw untouched across the forward/backward pair.
            with torch.no_grad():
                next_state = torch.from_numpy(batch.next_state).to(device)
                # The bootstrap maximises over what the *next* state actually
                # allows. Leaving it unmasked would value the successor by an
                # order the simulator would refuse, which is the same fiction
                # the acting policy is no longer permitted to believe.
                next_mask = torch.from_numpy(batch.next_mask).to(device)
                # Double DQN: online picks the argmax, target scores it.
                if noisy.enabled:
                    net_module.reset_noise(online)
                best = online(next_state).masked_fill(~next_mask, MASKED_SCORE)
                best = best.argmax(dim=1, keepdim=True)
                if noisy.enabled:
                    net_module.reset_noise(target)
                bootstrap = target(next_state).gather(1, best).squeeze(1)
                goal = torch.from_numpy(batch.reward).to(device) + torch.from_numpy(
                    batch.discount
                ).to(device) * bootstrap

            if noisy.enabled:
                net_module.reset_noise(online)
            predicted = online(state_tensor).gather(1, chosen.unsqueeze(1)).squeeze(1)
            per_row = loss_fn(predicted, goal)
            # `.mean()`, not `/ weight.sum()`. The weights are normalised to a
            # maximum of 1, so a plain mean is bounded above by the unweighted
            # loss and the effective step can only ever shrink. Renormalising by
            # the weight sum would put the scale back and hand the correction a
            # way to enlarge the step on a batch that happened to draw badly.
            loss = (per_row * torch.from_numpy(batch.weight).to(device)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            unweighted_losses.append(float(per_row.detach().mean()))
            # Numpy, so this costs no GPU sync.
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
            probe = buffer.sample(batch_size, demo_fraction=0.0, prioritized=False)
            flip_curve, order_curve = flip_rate_curve(
                online,
                torch.from_numpy(probe.state).to(device),
                torch.from_numpy(probe.mask).to(device),
                noisy.scale_limit,
            )
            flip_rate = flip_curve.get(1.0)
            flip_to_order = order_curve.get(1.0)

        # Chosen here because this is where the network is; `collect` reads it
        # back out of `metrics_json` on the next round. Recorded whether or not
        # the controller is on, so a run that turns it on can see what it would
        # have been playing at beforehand.
        #
        # Which curve drives it is the whole of the v6 fix. Controlling on the
        # argmax flip rate reads a NOOP-collapsed policy as exploring -- every
        # action near-tied, so flips are cheap -- and divides sigma down at the
        # one moment it needs raising. Both curves are recorded either way, so a
        # run that switches targets can see what the other would have said.
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
        # loudly. Against the quiet baseline it is not: the baseline emits
        # nothing, so a candidate that collapsed to NOOP genuinely plays every
        # paired seed identically. That is this agent's headline failure mode and
        # it belongs in the gate as a rejection, not as an ERROR that discards
        # the gate JSON and the diagnostics showing the collapse.
        #
        # But a zero margin does not *imply* the collapse, which is what this
        # read as before, and the difference cost the population run its
        # releases. Measured on `market-free-population-v25@learner_05`, cutoff
        # 143: 16,686 orders over 35,950 decisions -- a trading rate of 0.464 --
        # into a head-to-head that still scored an identically zero margin, with
        # both seats on exactly 3,000.0. Round-trip trades are cash-neutral
        # against an unchanged market by construction (README), so a candidate
        # can trade all season against a counterparty that offers nothing and
        # end where it started. That update passed the ratchet at 987 benchmark
        # money and passed hard viability, and this check alone rejected it.
        #
        # So read the collapse off the candidate's own activity in that match
        # rather than inferring it from the margin. A genuine NOOP policy reads
        # a trading rate of 0.0 and is still rejected here, which is the case
        # the check exists for. `viability_min_orders_per_decision = 0` turns it
        # off along with the other activity floors, deliberately.
        degenerate_head_to_head = False
        try:
            _reject_degenerate(head_to_head, "head-to-head")
        except RuntimeError:
            if active is not None:
                raise
            degenerate_head_to_head = _collapsed_against_baseline(head_to_head, config)

        # The money floors measure against the incumbent's own seat in this same
        # paired series -- except on a cold start, where that seat is the quiet
        # baseline and the baseline never trades. It therefore keeps its full
        # stake on every seed, and `0.80 *` a perfect stake is a bar no policy
        # that trades for a season can clear. `coldstart_v1` failed exactly and
        # only those two checks on five of seven updates, promoted nothing, and
        # with nothing released could not snap back either. A fixed cold-start
        # reference is what makes the *first* release reachable; from the second
        # onward the incumbent's real seat takes over again.
        if active is not None:
            head_reference = {
                "a_money_mean": head_to_head["b_money_mean"],
                "a_money_median": head_to_head["b_money_median"],
            }
        else:
            cold_start_reference = _limit(config, "viability_cold_start_reference_money")
            head_reference = {
                "a_money_mean": cold_start_reference,
                "a_money_median": cold_start_reference,
            }
        # The activity floor binds unconditionally on the *benchmark* arena, and
        # on the head-to-head only once there is a released incumbent to play.
        #
        # Both halves are load-bearing and the benchmark half is what preserves
        # the guard this used to be. It exists because a candidate that never
        # trades must never become the incumbent everything afterwards is
        # measured against: with `opponents = [seeded_random]` and no incumbent
        # the head-to-head is against a policy that reliably bankrupts --
        # `evaluation_opponent` has never recorded a non-zero `final_money` -- so
        # a pure-NOOP candidate keeps its 3,000 stake, wins 3,000-to-0
        # significantly, and passes hard viability. `market-free-noisy-v6`
        # recorded exactly that shape: `trading_rate: 0.0` beside
        # `hard_viability.passed: true`. That candidate reads 0.0 against
        # pilkwang as well, so `benchmark_viability` below still rejects it and
        # nothing silent can release.
        #
        # What binding it on the cold-start head-to-head as well cost, measured
        # on `market-free-population-v25` at cutoff 143: learner_01, learner_02
        # and learner_09 traded at 0.633, 0.332 and 0.398 against the benchmark
        # and emitted 0, 210 and 0 orders against the quiet baseline. Five of ten
        # slots had never released after 143 batches and `population_members`
        # held one row for the whole run, so every graduation took the `restart`
        # branch and the mature group the population exists to build stayed
        # empty.
        #
        # The floor measures nothing in that match. The baseline is a quiet
        # network: it offers no counterparty, so an order the candidate would
        # place against a live opponent has nothing to fill against, and the
        # arena the release is actually decided on -- the index-0 benchmark
        # under `benchmark_ratchet` -- already measures the same quantity
        # against an opponent that trades.
        #
        # With no benchmark configured there is no other arena to carry it, so
        # the head-to-head keeps the floor rather than dropping it silently.
        require_head_activity = active is not None or not challenger_benchmarks
        head_viability = _arena_viability(
            head_to_head, head_reference, config, require_activity=require_head_activity
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
                "limits": {name: _limit(config, name) for name in VIABILITY_DEFAULTS},
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
            "own_solvent_fraction": (
                float(np.mean([record.money[-1] > 10_000 for record in own])) if own else None
            ),
            "own_order_diagnostics": own_diagnostics.as_dict(),
        }
        if demo_stats is not None:
            metrics["demonstrations"] = {
                "episodes": demo_stats.episodes,
                "decisions": demo_stats.decisions,
                "orders_seen": demo_stats.orders_seen,
                "orders_mapped": demo_stats.orders_mapped,
                "mapped_fraction": demo_stats.mapped_fraction,
                "orders_unrepresentable": demo_stats.orders_unrepresentable,
                "orders_over_budget": demo_stats.orders_over_budget,
                # Should stay at zero. A solvent demonstrator's orders were
                # accepted by the simulator, so anything here is the feasibility
                # mask being over-restrictive.
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
        )
        slot = _settle_slot(slot, slot_score, slot_spec)
        metrics.update(slot.as_metrics())
        metrics["lr_score"] = slot_score
        population: dict[str, Any] = {"action": "train", "reseed_index": slot.reseed_index}
        declared = _declared_model_id(reader, context.model_id)
        graduating = _graduates(slot, slot_spec)

        # Contribution and restart are separate events with separate triggers.
        # They used to share this branch, which meant a checkpoint could only
        # enter the pool at the moment its slot was destroyed -- so three slots
        # could hold released policies no other slot was able to use, and the
        # first slot to reach its cut would reseed from `initial_network` with a
        # perfectly good ancestor sitting unreachable.
        #
        # What enters is still the *released* checkpoint, never the candidate this
        # update just trained: a mature member is meant to be a policy the gate
        # has already accepted, and the candidate has by definition not been.
        if active is not None:
            release_age = _release_age(reader, context.model_id, active.checkpoint_id)
            entry_after = int(config.get("population_entry_updates", 4))
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
                population.update(
                    _graduate(
                        reader,
                        declared,
                        context.model_id,
                        active.checkpoint_id,
                        float(scored["score"]),
                        slot_spec.mature_capacity,
                    )
                )

        if graduating:
            # `graduate` when there was something to contribute, `restart` when
            # the slot reached its cut having never released. Both reseed.
            population["action"] = "graduate" if active is not None else "restart"

            # Seeded from the top of the mature group, which after a graduation
            # may include the member that just entered. That is correct rather
            # than circular: the slot restarts from the population's best, and if
            # its own release is among them then what the restart buys is the
            # perturbation and the cleared optimizer, not a different ancestor.
            #
            # From the *top*, not from the front -- see `_reseed_source`. The
            # scores are noisy estimates and the best of them is the luckiest one
            # as often as it is the best policy.
            members = mature_members(reader, declared)
            chosen = _reseed_source(
                members,
                config,
                seed=int(config.get("seed", 0)),
                reseed_index=slot.reseed_index,
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
            elif active is not None:
                source_path, source_id = active.inference_artifact_path, active.checkpoint_id
            else:
                source_path, source_id = None, None

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
            # The same guard the exported candidate goes through. A perturbation
            # that somehow zeroed the trunk produces a network with no gradient
            # outside its output bias, and a slot restarted onto one can never
            # leave -- which would read as ten learners quietly going silent.
            _reject_inert_network(seeded)
            reseed_path = context.artifacts / "reseeds" / f"{context.update_id}.npz"
            reseed_path.parent.mkdir(parents=True, exist_ok=True)
            seeded.save(reseed_path)

            # Materialized rather than perturbed in place on the next update, so
            # `collect` and `update` load identical bytes. The alternative -- each
            # side drawing its own mask from a shared seed -- makes the behaviour
            # policy and the trained policy agree only as long as two separate
            # code paths keep agreeing, which is the class of drift the `to_numpy`
            # exporter bug in this file already cost a debugging session to.
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
