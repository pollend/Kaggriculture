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
from agents.dqn_market_free.training import rare
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
_PROMOTION_RULES = {"significant", "point_estimate", "always"}

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


def _self_play_pool(context, reader, limit: int) -> list[str]:
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
    """

    if limit < 1:
        return []
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


def _learning_rates(config: dict[str, Any]) -> tuple[float, float]:
    """Configured (TD, behaviour-clone) learning rates."""

    learning_rate = float(config.get("learning_rate", 1e-4))
    clone_rate = float(config.get("behavior_clone_learning_rate", learning_rate))
    return learning_rate, clone_rate


def _reset_learning_rates(
    optimizer: torch.optim.Optimizer,
    clone_optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> None:
    """Re-apply the configured learning rates over whatever a resume restored."""

    learning_rate, clone_rate = _learning_rates(config)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    for group in clone_optimizer.param_groups:
        group["lr"] = clone_rate


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
    actor_hot: int
    hot_order_rate: float
    hot_scale_limit: float
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
    hot_order_rate = float(config.get("noisy_hot_order_rate", 0.0))
    order_target = float(config.get("noisy_order_rate_target", 0.0))
    # The hot rungs are sized by the rate they collect at, not by a multiplier,
    # and that rate has to be above the ordinary setpoint -- otherwise the "hot"
    # pairs explore *less* than the ladder they sit on top of, and the two are
    # indistinguishable in `metrics_json` afterwards.
    if hot_order_rate > 0.0:
        if order_target > 0.0 and hot_order_rate <= order_target:
            raise ValueError(
                f"noisy_hot_order_rate {hot_order_rate} is not above "
                f"noisy_order_rate_target {order_target}, so the hot rungs would explore "
                "no harder than the ordinary ones"
            )
    return NoisySpec(
        enabled=enabled,
        layers=layers,
        sigma0=float(config.get("noisy_sigma0", 0.5)),
        cadence=cadence,
        actor_alpha=float(config.get("noisy_actor_alpha", 32.0)),
        # How many of the batch's seed pairs collect *above* the setpoint. 0 --
        # the default -- is the one-sided ladder every run through v13 played,
        # where no episode anywhere explores harder than the controller resolved.
        # The hot rungs are sized off `noisy_hot_order_rate`, so a config that
        # asks for them without a hot rate to resolve gets the one-sided ladder
        # rather than a silent multiplier of its own.
        actor_hot=int(config.get("noisy_actor_hot", 0)),
        hot_order_rate=hot_order_rate,
        # Defaults to the ordinary limit, so a config that never names it behaves
        # exactly as it did. It exists because the ordinary limit is far too loose
        # to bound the hot end: at 8.0 it let `noisy_hot_order_rate = 0.35` resolve
        # to 3.159 on v9's curve, which measured 49% of episodes bankrupt across
        # 212 seats and bought no land at all.
        hot_scale_limit=float(
            config.get(
                "noisy_hot_scale_limit",
                config.get("noisy_sigma_scale_limit", 8.0),
            )
        ),
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

    `key` selects which resolved scale to read: the ordinary setpoint, or
    `noisy_hot_sigma_scale_target` for the hot rungs. Both are written by the
    same update off the same measured curve, so reading them separately keeps a
    hot rungs' scale on the same footing as the setpoint -- including the
    "a rejected update still measured a real curve" rule below, which applies to
    both or to neither.

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
    # input is not user data -- it is one of two literals chosen in this module --
    # so it is rejected outright rather than escaped.
    if key not in {"noisy_sigma_scale_target", "noisy_hot_sigma_scale_target"}:
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
        # Checkpoint paths are content-addressed and immutable; `initial` is not,
        # so it is rewritten in case an older run left a stale one behind.
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
        # Held to a minority of the pairs by default, counted from the hot end --
        # the same end `noisy_actor_hot` counts from. See `collect_batch`.
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
        # The hot rungs' scale, resolved by the same update off the same curve
        # against `noisy_hot_order_rate`. Read here rather than derived from
        # `sigma_base` because the hot end is a target rate, not a multiple of the
        # setpoint: the setpoint moves every update as the controller tracks it,
        # and the scale a given rate costs moves with the network.
        hot_base = (
            collection_sigma_scale(
                reader, context.model_id, "noisy_hot_sigma_scale_target"
            )
            if noisy.actor_hot > 0
            else 1.0
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
                pool = _self_play_pool(context, reader, int(config.get("self_play_pool", 5)))
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
            # around it, hot rungs included, is deliberately left out: folding
            # sixteen scales in would mint eight revisions a batch, none with
            # enough games to rate.
            #
            # One ladder, always. The idle burst that used to rebase this flat on
            # a silent policy was removed at v15 -- it and `noisy_actor_hot` were
            # two mechanisms for one job, fighting over the same variable: the
            # controller holds flip-to-order at `noisy_order_rate_target` while a
            # burst jumped to `noisy_hot_order_rate`, and the hot rungs already
            # put pairs above the setpoint on *every* batch rather than
            # occasionally. The reward now says idling is bad directly
            # (`IDLE_PENALTY`), so forcing it through the behaviour distribution as
            # well was second-guessing the objective.
            if not noisy.enabled:
                sigma_scales = None
            else:
                hot_top = max(1.0, hot_base / sigma_base) if sigma_base > 0.0 else 1.0
                sigma_scales = [
                    sigma_base * scale
                    for scale in actor_sigma_scales(
                        episodes // 2,
                        noisy.actor_alpha,
                        noisy.actor_hot,
                        hot_top,
                    )
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
            # The same diagnostics over the pairs that played at or below the
            # setpoint. `noisy_actor_hot` pairs collect *above* it, and they trade
            # more by construction, so the whole-batch `trading_rate` reads higher
            # than the policy actually is -- which is the number an idle-escape
            # triggers on. Without this the hot rungs would quietly disable the
            # escape hatch that answers a collapsed policy.
            #
            # Selected on `pair_index` rather than by slicing the first `hot * 2`
            # records off the front: a failed episode is dropped from `rollouts`,
            # so position and pair stop agreeing exactly when something went wrong.
            #
            # `None` when there are no hot pairs, so an ordinary run records
            # nothing new and `last_batch_trading_rate` falls through to the
            # whole-batch rate.
            cold_records = [
                rollout.record for rollout in rollouts if rollout.pair_index >= noisy.actor_hot
            ]
            cold_diagnostics = (
                _records_diagnostics(cold_records)
                if noisy.actor_hot > 0 and cold_records
                else None
            )
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
                    # The base the last probe resolved, and the spread around it
                    # that the actors actually played. Recorded apart so a batch
                    # says both what exploration was aimed at and what each actor
                    # got.
                    # The setpoint the ladder hangs from. `actor_sigma_scales`
                    # beside it is what each pair actually played, hot rungs
                    # included.
                    "sigma_base": sigma_base if noisy.enabled else None,
                    "actor_sigma_scales": sigma_scales,
                    # How many of those rungs were above the setpoint. The scales
                    # say it too, but only against a `sigma_base` in the same row;
                    # this is the column a money series can be filtered on, and a
                    # batch with hot pairs in it is bimodal on money by design.
                    "noisy_actor_hot": noisy.actor_hot if noisy.enabled else None,
                    "seeds": seeds,
                    # The two numbers that separate this agent's failure modes
                    # from each other. All-zero orders is the NOOP collapse;
                    # orders on nearly every decision with money at the floor is
                    # the bankruptcy collapse. Recorded per batch so either is
                    # visible in SQL before it has eaten a whole run.
                    "orders_placed": orders_placed,
                    "orders_per_decision": orders_placed / max(1, transitions),
                    "order_diagnostics": order_diagnostics.as_dict(),
                    "cold_order_diagnostics": (
                        cold_diagnostics.as_dict() if cold_diagnostics is not None else None
                    ),
                    # What the widening was worth this batch and whether it moved
                    # anything. Entropy is the gain-free reading -- a policy
                    # pouring every turn into three rows reads low here whatever
                    # its money curve says -- and `buy_land_rate` is the specific
                    # row that motivated the mechanism, kept as a named column so
                    # the question that started this is answerable in one query.
                    "rare_sigma_gain": rare_gain if rare_gain > 0.0 else None,
                    # Counted from the hot end, so these pairs overlap
                    # `noisy_actor_hot` rather than sitting opposite it and the
                    # clean arm is the greedy tail. A money series filters on both
                    # columns together.
                    "rare_sigma_pairs": rare_sigma_pairs if rare_gain > 0.0 else None,
                    "selection_entropy": rare.selection_entropy(batch_chosen),
                    "buy_land_rate": land_episodes / max(1, len(records)),
                    "mean_final_money": float(np.mean(final_money)),
                    "max_final_money": float(np.max(final_money)),
                    "solvent_episodes": int(sum(1 for value in final_money if value > 10_000)),
                    "behavior_checkpoint_id": (
                        checkpoint.checkpoint_id if checkpoint is not None else None
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

        skipped_evicted_batches = fill_buffer(buffer, context.all_batches)

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
        resumed = lineage_checkpoint(context, reader)
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
            _reset_learning_rates(optimizer, clone_optimizer, config)
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
            online = net_module.from_numpy(initial_network(config), online)
            target.load_state_dict(online.state_dict())

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

        # The hot rungs' own scale, off the same curve. Resolved here for the
        # same reason the setpoint is -- this is where the network and the probe
        # are -- and stored separately so `collect` can pick per round without
        # re-deriving anything. 1.0 when there is no schedule, no curve, or no
        # order target, which makes the hot rungs identical to the setpoint
        # rather than silently enormous.
        hot_scale_target = (
            resolve_sigma_scale(
                order_curve or {},
                noisy.hot_order_rate,
                noisy.hot_scale_limit,
                noisy.scale_floor,
            )
            if noisy.actor_hot > 0 and noisy.hot_order_rate > 0.0
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
        jobs = int(config.get("jobs", 1))

        # Three arena series an update, not five. The benchmarks are recorded as
        # diagnostics and read for order activity, but they no longer gate:
        # non-regression against the incumbent *on a benchmark* asks a strictly
        # weaker version of the question the paired head-to-head below already
        # answers directly, and replaying both the incumbent and the quiet
        # baseline against every benchmark to ask it was most of the evaluation
        # wall clock.
        benchmarks = configured_benchmarks(config)
        challenger_benchmarks = [
            _arena_evaluate(inference_path, benchmark, eval_seeds, jobs, seed_start)
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
        degenerate_head_to_head = False
        try:
            _reject_degenerate(head_to_head, "head-to-head")
        except RuntimeError:
            if active is not None:
                raise
            degenerate_head_to_head = True

        head_reference = {
            "a_money_mean": head_to_head["b_money_mean"],
            "a_money_median": head_to_head["b_money_median"],
        }
        # The activity floor binds always, including on a cold start.
        #
        # It used to be `active is not None`, on the reasoning that a cold start
        # is legitimately near-silent. True of the *policy* and false of the
        # *release*: a candidate that never trades must never become the incumbent
        # everything afterwards is measured against, and a cold start is precisely
        # where that can happen unopposed. With `opponents = [seeded_random]` and
        # no incumbent the head-to-head is against a policy that reliably
        # bankrupts -- `evaluation_opponent` has never recorded a non-zero
        # `final_money` -- so a pure-NOOP candidate keeps its 3,000 stake, wins
        # 3,000-to-0 significantly, and passed hard viability because this was
        # off. `market-free-noisy-v6` recorded exactly that shape:
        # `trading_rate: 0.0` beside `hard_viability.passed: true`.
        #
        # Same failure as two other guards in this family: written to fire "once
        # released", silently inert on a run that has never released, which is
        # where it was needed. Being unpromotable is not a cost here -- rejection
        # does not stop training, and 0.02 is cleared by any policy that trades at
        # all (a live trading run measures 0.29). Set
        # `viability_min_orders_per_decision = 0` to turn it off deliberately.
        require_activity = True
        head_viability = _arena_viability(
            head_to_head, head_reference, config, require_activity=require_activity
        )
        benchmark_viability = [
            _order_viability(result, config, require_activity=require_activity)
            for result in challenger_benchmarks
        ]
        hard_viability_passed = head_viability["passed"] and all(
            item["passed"] for item in benchmark_viability
        )

        head_to_head_ok = policy_improvement(head_to_head, promotion_rule)
        # `always` releases every update, and the whole arena above still runs.
        #
        # Nothing here ever gated *training*: `lineage_checkpoint` takes the
        # newest checkpoint promoted or not, `_self_play_pool` deliberately does
        # not filter on `promoted`, and every update writes a checkpoint row
        # regardless -- so a rejection only ever withheld the release. What it
        # gated was `export-active` and the head-to-head incumbent.
        #
        # `hard_viability`, the money floors, the activity floor, the head-to-head
        # and the pilkwang benchmark are all still computed and recorded in
        # `gate_json`; only their effect on this boolean is gone. Losing the
        # measurement was never the point.
        #
        # Two guards above are *not* part of this and must keep raising:
        # `_reject_inert_network` on an all-zero exported trunk, and
        # `_reject_degenerate` on a zero-width interval. Both signal a harness or
        # export fault rather than a weak candidate, and both fail silently in the
        # other direction -- an inert export would release as a healthy policy.
        promote = promotion_rule == "always" or (
            hard_viability_passed and head_to_head_ok and not degenerate_head_to_head
        )
        gate: dict[str, Any] = {
            "kind": (
                "unconditional_release"
                if promotion_rule == "always"
                else f"{promotion_rule}_incumbent_improvement_and_hard_viability"
            ),
            "policy_kinds": POLICY_KINDS,
            "promotion_rule": promotion_rule,
            "seed_start": seed_start,
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
        # through, so it has to follow the same rule the promotion did.
        evaluations: list[ModelEvaluation] = [
            ModelEvaluation(
                "incumbent_head_to_head",
                head_to_head,
                head_to_head_ok and not degenerate_head_to_head and head_viability["passed"],
                active.checkpoint_id if active is not None else None,
            )
        ]
        evaluations.extend(
            ModelEvaluation(_benchmark_kind(index), challenger, viability["passed"])
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
            # The scale the hot rungs will collect at, resolved off the same
            # curve. Recorded whether or not a schedule is running, so a run that
            # turns them on can see beforehand what they would have played at --
            # and, more usefully, so a run with them on can be checked against
            # `noisy_flip_to_order_curve` for whether the target is still landing
            # where it was meant to as the network moves.
            "noisy_hot_sigma_scale_target": hot_scale_target,
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
