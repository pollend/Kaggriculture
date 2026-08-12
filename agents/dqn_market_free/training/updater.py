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
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import torch
from torch import nn

from agents.dqn_market_free.diagnostics import OrderDiagnostics, diagnose_replay
from agents.dqn_market_free.order_space import ACTION_COUNT
from agents.dqn_market_free.policy import OrderQNetwork
from agents.dqn_market_free.training import network as net_module
from agents.dqn_market_free.training.clone import load_demonstrations
from agents.dqn_market_free.training.collect import (
    LEARNER_ENTRY_POINT,
    EpisodeRecord,
    _single_threaded_workers,
    actor_epsilons,
    collect_batch,
    load_episode_batch,
    save_episode_batch,
)
from agents.dqn_market_free.training.replay import (
    DEFAULT_CAPACITY,
    DEFAULT_N_STEP,
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
    rotate_opponents,
    self_play_pool,
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
_PROMOTION_RULES = {"significant", "point_estimate"}

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


def _learner_policy(weights_path: Path, epsilon: float, gamma: float) -> EvaluationPolicy:
    """Identity of the exploring behaviour policy for one batch.

    The spec actually played also carries a per-episode trace path and
    exploration seed; folding those into the identity would mint one revision
    per rollout episode. What defines the policy is its weights and how much it
    explores, so the identity spec carries only those -- and stays resolvable.

    Epsilon is bucketed to four decimals. It decays continuously, and it is now
    a small absolute number rather than a rate scaled by a separate act rate, so
    two decimals would collapse most of the schedule onto one revision.
    """

    spec = f"{LEARNER_ENTRY_POINT}?" + urlencode(
        {
            "weights": str(weights_path.resolve()),
            "epsilon": round(epsilon, 4),
            "gamma": gamma,
        }
    )
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

    Read off `trading_decisions`, the share of turns that emitted anything, not
    off the wire order count. Those differ: `HIRE n` is one decision whose wire
    form is n rows, so a policy that only ever hires reports 13 wire orders per
    decision while trading on one turn in fifty. The floor is about how often the
    policy acts, so it is the decision share that belongs here.

    There is still no upper limit. Over-trading can only show up as lost money,
    which the money floors in `_arena_viability` already measure. The three
    cross-slot limits that used to live here (exact duplicates, repeated targets,
    buy/sell conflicts within one decision) were counting a failure mode that a
    single-order decision cannot have.
    """

    diagnostics = dict(result.get("order_diagnostics") or {})
    decisions = max(1, int(diagnostics.get("decisions", 0)))
    trading_rate = int(diagnostics.get("trading_decisions", 0)) / decisions
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

    state, chosen = buffer.sample_demonstrations(batch_size)
    optimizer.zero_grad(set_to_none=True)
    logits = online(torch.from_numpy(state).to(device))
    loss = nn.functional.cross_entropy(logits, torch.from_numpy(chosen).to(device))
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    return float(loss.detach())


def _behavior_clone_steps_per_update(config: dict[str, Any]) -> int:
    """Default demo-seeded runs to imitation without changing run identity."""

    default = 100 if int(config.get("demonstration_episodes", 0)) > 0 else 0
    return int(config.get("behavior_clone_steps_per_update", default))


def _reject_inert_network(network: OrderQNetwork) -> None:
    """Fail loudly when the exported network can only ever emit NOOP."""

    if net_module.has_inert_hidden_trunk(network):
        raise RuntimeError(
            "exported free-market network has all-zero hidden weights: every action "
            "ties to NOOP and the agent never trades. An all-zero network also has "
            "no gradient outside its output bias, so the run cannot recover -- the "
            "hidden trunk collapsed after the cold start, which starts it random"
        )


def _epsilon(config: dict[str, Any], transitions: int) -> float:
    """Probability that a rollout turn emits a uniformly random real order.

    This absorbed the old per-slot `act_rate`: only the product of the two ever
    mattered, and keeping them apart meant the effective exploration rate was a
    number nobody could read off the config.
    """

    fraction = min(
        1.0, transitions / max(1, int(config.get("epsilon_decay_transitions", 2_000_000)))
    )
    start = float(config.get("epsilon_start", 0.18))
    end = float(config.get("epsilon_end", 0.01))
    return start + fraction * (end - start)


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
        jobs = int(config.get("jobs", 1))

        base = [str(item) for item in config.get("opponents", [])]
        if config.get("league_opponents", True):
            base = list(dict.fromkeys([*base, *league_opponents(context, reader)]))
        if not base:
            raise ValueError("free-market collection needs at least one opponent")

        self_play_after = int(config.get("self_play_after", 0))
        episode_config = {"episodeSteps": int(config.get("episode_steps", 720))}

        written: list[str] = []
        for _ in range(count):
            epsilon = _epsilon(config, total_transitions)
            batch_index = int(
                reader.fetchone(
                    "SELECT count(*) FROM training_batches WHERE model_id = %s",
                    [context.model_id],
                )[0]
            )

            # Self-play against this model's own promoted checkpoints, once the
            # run has enough batches behind it to have promoted anything worth
            # playing. Held to a ratio: evaluation is always against the
            # benchmark, so a higher share trains on a distribution the
            # objective never measures.
            #
            # Gated on `batch_index`, which is what `self_play_after` counts.
            # The batch count is only known inside this loop, which is why the
            # whole opponent mix is assembled here rather than once above.
            opponents = list(base)
            if self_play_after and batch_index >= self_play_after:
                pool = self_play_pool(
                    context, reader, int(config.get("self_play_pool", 5)), SELF_SPEC
                )
                if pool:
                    opponents = compose_opponents(
                        base,
                        pool,
                        float(config.get("self_play_ratio", 0.3)),
                        pool_offset=batch_index,
                    )

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
            # One rate per seed pair, spread around the schedule's current value
            # so a single batch carries both exploratory and near-greedy data.
            # The *identity* policy keeps the base rate: folding sixteen actor
            # epsilons into it would mint eight revisions a batch, none of them
            # with enough games to rate.
            epsilons = actor_epsilons(
                epsilon,
                episodes // 2,
                float(config.get("epsilon_actor_alpha", 4.0)),
            )
            learner = _learner_policy(behavior_path, epsilon, gamma)
            rollouts = collect_batch(
                behavior_path,
                opponents,
                seeds,
                epsilons,
                gamma,
                trace_root,
                jobs,
                exploration_seed=seed + total_transitions,
                policy_revision_id=learner.revision_id,
                config=episode_config,
            )
            if not rollouts:
                raise RuntimeError("every free-market rollout episode failed")
            records = [rollout.record for rollout in rollouts]

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
                    "epsilon": epsilon,
                    "actor_epsilons": epsilons,
                    "seeds": seeds,
                    # The two numbers that separate this agent's failure modes
                    # from each other. All-zero orders is the NOOP collapse;
                    # orders on nearly every decision with money at the floor is
                    # the bankruptcy collapse. Recorded per batch so either is
                    # visible in SQL before it has eaten a whole run.
                    "orders_placed": orders_placed,
                    "orders_per_decision": orders_placed / max(1, transitions),
                    "order_diagnostics": order_diagnostics.as_dict(),
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

        online = net_module.build().to(device)
        target = net_module.build().to(device)
        optimizer = torch.optim.Adam(
            online.parameters(),
            lr=float(config.get("learning_rate", 1e-4)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        # Its own optimizer, not the TD one. Cross-entropy over 186 classes and a
        # Huber TD residual have very different gradient scales, and Adam's
        # second moment is a running average: sharing one optimizer lets the
        # clone phase set the step sizes the fine-tuning that follows starts from.
        clone_optimizer = torch.optim.Adam(
            online.parameters(),
            lr=float(config.get("behavior_clone_learning_rate", config.get("learning_rate", 1e-4))),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        gradient_steps = 0
        behavior_clone_steps = 0
        resumed = lineage_checkpoint(context, reader)
        resumed_from = resumed.checkpoint_id if resumed is not None else None
        if resumed is not None:
            state = torch.load(
                resumed.trainer_artifact_path, map_location="cpu", weights_only=False
            )
            online.load_state_dict(state["online"])
            target.load_state_dict(state["target"])
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
            online = net_module.from_numpy(initial_network(config), online)
            target.load_state_dict(online.state_dict())

        loss_fn = nn.SmoothL1Loss()
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
        clone_losses: list[float] = []
        started = time.perf_counter()
        clone_steps_to_run = (
            _behavior_clone_steps_per_update(config) * batches_per_update
            if buffer.pinned
            else 0
        )
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
            predicted = online(state_tensor).gather(1, chosen.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_state = torch.from_numpy(batch.next_state).to(device)
                # Double DQN: online picks the argmax, target scores it.
                best = online(next_state).argmax(dim=1, keepdim=True)
                bootstrap = target(next_state).gather(1, best).squeeze(1)
                goal = torch.from_numpy(batch.reward).to(device) + torch.from_numpy(
                    batch.discount
                ).to(device) * bootstrap

            loss = loss_fn(predicted, goal)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            gradient_steps += 1
            if gradient_steps % target_sync == 0:
                target.load_state_dict(online.state_dict())
        train_seconds = time.perf_counter() - started

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
        # The activity floor only binds once there is a released incumbent to
        # fall back to. A cold start is legitimately near-silent.
        require_activity = active is not None
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
        promote = hard_viability_passed and head_to_head_ok and not degenerate_head_to_head
        gate: dict[str, Any] = {
            "kind": f"{promotion_rule}_incumbent_improvement_and_hard_viability",
            "policy_kinds": POLICY_KINDS,
            "promotion_rule": promotion_rule,
            "seed_start": seed_start,
            "challenger_benchmarks": challenger_benchmarks,
            "resumed_from_checkpoint_id": resumed_from,
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
            "episodes_in_buffer": len(buffer.episodes),
            "pinned_demonstrations": len(buffer.pinned),
            "demo_fraction": demo_fraction,
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
