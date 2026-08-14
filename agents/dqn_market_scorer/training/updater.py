"""Persistent collector and incremental updater for the market scorer.

Forked from `agents/dqn_job_scorer/training/updater.py` with the job network
removed. What is left is one network, one optimizer, one loss -- the factored
market Double-DQN. The opponent mixing, lineage resume, and promotion gate are
carried over unchanged, because none of that was ever job-specific.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import torch
from torch import nn

from agents.dqn_market_scorer.market_policy import POLICY_KINDS
from agents.dqn_market_scorer.policy import WEIGHTS_PATH, MarketQNetwork
from agents.dqn_market_scorer.training import network as net_module
from agents.dqn_market_scorer.training.benchmarks import (
    configured_benchmarks,
    promotion_benchmark,
)
from agents.dqn_market_scorer.training.collect import (
    LEARNER_ENTRY_POINT,
    EpisodeFormatError,
    collect_batch,
    load_episode_batch,
    save_episode_batch,
)
from agents.dqn_market_scorer.training.device import move_optimizer, resolve_device
from agents.dqn_market_scorer.training.peers import (
    active_checkpoint_opponents,
    historical_checkpoint_opponents,
)
from agents.dqn_market_scorer.training.progress import _opponent_policy, _policy_identity
from agents.dqn_market_scorer.training.replay import ReplayBuffer
from kaggriculture_lab.model_trainers import EvaluationPolicy
from kaggriculture_lab.model_updates import (
    ModelEvaluation,
    ModelTrainingReader,
    ModelUpdateContext,
    ModelUpdateResult,
)

SELF_SPEC = "agents/dqn_market_scorer/main.py"
LEARNER_MANIFEST = "agents/dqn_market_scorer/match_agent.toml"
SOURCE_KIND = "market_rollout"
_PROMOTION_RULES = {"significant", "point_estimate"}


def _incumbent_path(config: dict[str, Any]) -> Path:
    return Path(str(config.get("incumbent_weights", WEIGHTS_PATH)))


def initial_network(config: dict[str, Any]) -> MarketQNetwork:
    """Build the first behavior policy for a new training instance.

    `donor_default` is the deliberate default rather than `random`: it starts the
    run scoring every candidate equally, so each argmax falls to index 0 -- the
    donor-default modifier and a zero reserve -- and round one rolls out the
    unmodified donor policy. Starting from a random market policy would spend
    the early batches learning its way back to a baseline it was handed free.

    Only the output layer is zeroed. An all-zero MLP makes both hidden
    activations identically zero for any input, which zeroes the gradient of
    every parameter but the output bias; `relu'(0) == 0` closes the path back to
    the biases too. The symmetry would never break, so the network would stay
    byte-identical to the donor for the life of the run while the gate reported
    a healthy promotion.
    """

    initialization = str(config.get("initialization", "donor_default"))
    if initialization == "donor_default":
        return MarketQNetwork.donor_default(seed=int(config.get("seed", 0)))
    if initialization == "random":
        return MarketQNetwork.random(seed=int(config.get("seed", 0)))
    if initialization != "incumbent":
        raise ValueError(f"unsupported market initialization: {initialization}")
    path = _incumbent_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"market incumbent checkpoint does not exist: {path}")
    return MarketQNetwork.load(path)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def league_opponents(context, reader) -> list[str]:
    """Agent specs of everyone competing in the round that just closed.

    The league breeds a fresh population every generation, so these specs carry
    parameters that the checked-in `main.py` files do not. Training only against
    those files would pin the opponent distribution to whatever happened to be
    committed, while the league is busy generating variation right next door.

    The learner's own family is excluded -- facing itself is self-play, which is
    mixed in separately and under an explicit ratio.
    """

    rows = reader.fetchall(
        """
        SELECT DISTINCT ar.agent_spec
        FROM generation_members gm
        JOIN agent_revisions ar USING (revision_id)
        WHERE gm.generation_id = %s AND ar.agent_spec IS NOT NULL
          AND ar.family_id != %s
        ORDER BY ar.agent_spec
        """,
        [context.generation_id, context.family_id],
    )
    return [str(row[0]) for row in rows]


def self_play_pool(context, reader, limit: int, self_spec: str = SELF_SPEC) -> list[str]:
    """The most recent promoted checkpoints, addressed as agent specs.

    A snapshot is reached through the agent's `weights` config key rather than as
    a bare `.npz` path: the registry resolves agent modules, not weight files.

    `self_spec` is a parameter because this helper is shared with forks that keep
    their checkpoints under their own `model_id` but load them with their own
    network class. Defaulting it to this module's `SELF_SPEC` would hand a caller
    its own weights behind *this* agent's entry point, and the feature-dimension
    check in the loader would fail every episode.
    """

    if limit < 1:
        return []
    rows = reader.fetchall(
        """
        SELECT inference_artifact_path FROM model_training_checkpoints
        WHERE model_id = %s AND promoted
        ORDER BY created_at DESC, checkpoint_id DESC LIMIT %s
        """,
        [context.model_id, limit],
    )
    return [f"{self_spec}?weights={row[0]}" for row in rows if Path(str(row[0])).is_file()]


@dataclass(frozen=True, slots=True)
class LineageCheckpoint:
    """Where the run actually is, as opposed to what it has released."""

    checkpoint_id: str
    trainer_artifact_path: Path
    inference_artifact_path: Path
    #: True when this is `context.active_checkpoint` -- either because nothing
    #: newer exists, or because the reset below sent the run back to it.
    released: bool
    #: Rejections since the last promotion, at the moment of the call.
    rejection_streak: int = 0
    #: Set when the streak tripped `resume_reset_after_rejections`.
    reset: bool = False


def rejections_since_promotion(reader, model_id: str) -> int:
    """Consecutive rejections the lineage is currently carrying.

    A recorded `lineage_reset` ends the streak as surely as a promotion does, and
    for the same reason: whatever was diverging has already been dropped, so
    counting past it charges the new trajectory for the old one's failures. The
    job scorer's copy of this has always read it that way. Without it the streak
    only ever grows, so a run that trips the limit once trips it on every update
    after -- which pins a released run to its incumbent forever, and puts a
    never-released one on a reinitialize-every-update treadmill.
    """

    row = reader.fetchone(
        """
        SELECT count(*) FROM model_updates
        WHERE model_id = %s AND status = 'REJECTED' AND started_at > coalesce(
            (SELECT max(started_at) FROM model_updates
             WHERE model_id = %s AND (
                 status = 'PROMOTED'
                 OR coalesce((gate_json ->> 'lineage_reset')::boolean, false)
             )),
            '-infinity'::timestamp
        )
        """,
        [model_id, model_id],
    )
    return 0 if row is None else int(row[0])


def reset_since_promotion(reader, model_id: str) -> bool:
    """Whether a lineage reset has been recorded with nothing promoted after it.

    The companion to `rejections_since_promotion`, and it exists because the
    streak alone cannot express "the restart is still in progress". A reset zeroes
    the streak by design, so one update later the counter says the lineage is
    healthy while the only evidence available says the opposite.
    """

    row = reader.fetchone(
        """
        SELECT count(*) FROM model_updates
        WHERE model_id = %s
          AND coalesce((gate_json ->> 'lineage_reset')::boolean, false)
          AND started_at > coalesce(
              (SELECT max(started_at) FROM model_updates
               WHERE model_id = %s AND status = 'PROMOTED'),
              '-infinity'::timestamp
          )
        """,
        [model_id, model_id],
    )
    return bool(row and int(row[0]) > 0)


def lineage_checkpoint(context, reader) -> LineageCheckpoint | None:
    """The newest checkpoint on the model's lineage, promoted or not.

    Release is gated; the learner is not. `context.active_checkpoint` is the
    *released* policy, and both resuming gradient descent from it and rolling
    out from it are wrong in the same way: the first discards every step a
    rejected update took, and the second freezes the behaviour distribution for
    as long as the gate says no.

    Two escapes. Training and collection both follow the lineage, so the data
    tracks the learner without shipping an ungated policy; and once the run has
    been rejected `resume_reset_after_rejections` times in a row, the lineage
    snaps back to the release, because carrying a diverged trajectory forward
    forever is how it stays diverged.

    That snapback **holds until a promotion**, not for a single update. It used to
    end immediately: the reset zeroes the streak, but the resetting update still
    writes a checkpoint, and the newest-by-`created_at` rule then handed the
    lineage straight back to whatever that update produced. `reset_since_promotion`
    is what keeps it in force; see the comment at the branch below for the
    measurement and for the freeze it deliberately accepts in exchange.

    **A run that has never promoted does not reset at all**, and this has now been
    decided both ways, so the history matters.

    It originally did not reset, because the snapback was written as "fall back to
    the release" and there was none -- the guard was silently inert exactly where
    it looked most needed. Measured on `market-free-selfplay-v3`: twelve
    consecutive rejections at a limit of five, all carrying one trajectory. So
    `None` was made the reset: the caller reads it as a cold start and
    reinitializes, the only "not carried forward" available with nothing released.

    That is worse, and a genuine cold start showed why. With nothing released the
    reset is not a snapback to something known good, it is a **wipe** -- every
    batch of learning discarded and the network reinitialized to `quiet_start`.
    A cold start needs many generations to reach its first promotion, so at any
    useful limit it never survives long enough to get there. Measured on
    `market-free-landgate-v10` at a limit of 2: zero promotions, two consecutive
    rejections, and the next update due to discard all sixty batches behind it and
    then repeat every two rejections.

    So the release is a precondition of resetting. This knowingly reinstates the
    v3 hole -- a never-released run can ride one diverged trajectory indefinitely
    -- and that is the accepted side of the trade, because v3's problem was a
    *trained* trajectory diverging and the wipe is only defensible when there is
    something better to fall back to. Watch the rejection streak on a run that has
    not promoted; a long streak with flat `mean_abs_q` is v3's failure returning.

    Falls back to the released checkpoint when the newer artifact is missing --
    trainer `.pt` files are large and a pruned artifact root must not silently
    restart the run from scratch. That case is deliberately *not* a reset: the
    artifact being absent says nothing about the trajectory.
    """

    def released() -> LineageCheckpoint | None:
        active = context.active_checkpoint
        if active is None:
            return None
        return LineageCheckpoint(
            active.checkpoint_id,
            Path(active.trainer_artifact_path),
            Path(active.inference_artifact_path),
            released=True,
        )

    row = reader.fetchone(
        """
        SELECT checkpoint_id, trainer_artifact_path, inference_artifact_path
        FROM model_training_checkpoints WHERE model_id = %s
        ORDER BY created_at DESC, checkpoint_id DESC LIMIT 1
        """,
        [context.model_id],
    )
    if row is None or not Path(str(row[1])).is_file():
        return released()
    active = context.active_checkpoint
    if active is not None and str(row[0]) == active.checkpoint_id:
        return released()

    limit = int(getattr(context, "config", {}).get("resume_reset_after_rejections", 0))
    streak = rejections_since_promotion(reader, context.model_id) if limit > 0 else 0
    # A reset holds until something promotes, rather than lasting one update.
    #
    # It used to last exactly one. The reset zeroes the streak -- correctly, the
    # old trajectory is gone -- but the resetting update still trains, still
    # writes a checkpoint row, and this query takes the newest by `created_at`.
    # So the moment it finishes, the lineage points at the candidate *it* just
    # produced, and both training and collection follow that. Measured on
    # `market-free-landgate-v10`: the reset fired, produced a candidate worth 357
    # against a released checkpoint worth 18,339, and the next twenty batches were
    # collected from the 357 policy at 3-8k against the 11-16k before it.
    #
    # Skipping only that one checkpoint would be worse -- the next newest is the
    # diverged row the reset existed to escape -- so the release holds instead.
    #
    # This does freeze the behaviour distribution while the gate says no, which
    # the note above argues against, and the argument is still right *in
    # general*. It stops applying once `resume_reset_after_rejections` consecutive
    # rejections have already proven the trajectory bad: a frozen release beats
    # following a policy the gate scored fifty times worse. It also repairs
    # itself, which the one-update version could not -- collection follows the
    # release too, so the ring refills with the released policy's episodes and the
    # next attempt trains from good weights on good data instead of from good
    # weights on the discarded policy's buffer.
    #
    # Both conditions require a release, so `fallback` gates the whole branch.
    # With nothing promoted there is no snapback available -- only a wipe back to
    # `quiet_start`, which discards the run's entire training history and which a
    # cold start cannot survive at any useful limit. See the docstring for the
    # measurement and for the v3 hole this knowingly reinstates.
    fallback = released()
    held = fallback is not None and reset_since_promotion(reader, context.model_id)
    if limit > 0 and fallback is not None and (streak >= limit or held):
        return replace(fallback, rejection_streak=streak, reset=True)
    return LineageCheckpoint(
        str(row[0]),
        Path(str(row[1])),
        Path(str(row[2])),
        released=False,
        rejection_streak=streak,
    )


def compose_opponents(
    base: list[str], pool: list[str], ratio: float, *, pool_offset: int = 0
) -> list[str]:
    """Mix self-play into `base` so it lands on `ratio` of the episodes.

    `collect_batch` walks the returned list round-robin, so its composition *is*
    the sampling distribution. The naive `[*base, *pool]` with a single benchmark
    and a pool of five makes self-play 5 of every 6 episodes while evaluation
    stays on the benchmark. Solve for the slot count instead:
    `slots / (len(base) + slots) == ratio`.
    """

    ratio = min(max(ratio, 0.0), 0.9)
    if not pool or not base or ratio <= 0.0:
        return list(base) or list(pool)
    # A narrow base cannot express a small ratio in whole slots -- one static
    # opponent at ratio 0.25 wants a third of a slot, which rounds to zero and
    # silently disables self-play. Repeat the base until one slot is
    # representable, then solve as usual.
    repeats = max(1, math.ceil((1 - ratio) / (ratio * len(base))))
    widened = base * repeats
    slots = max(1, round(len(widened) * ratio / (1 - ratio)))
    return [*widened, *(pool[(pool_offset + index) % len(pool)] for index in range(slots))]


def rotate_opponents(opponents: list[str], offset: int) -> list[str]:
    """Start each batch where the previous one stopped.

    `collect_batch` plays `episodes_per_batch` episodes against
    `opponents[i % len(opponents)]`. Once the list is longer than a batch --
    which it is as soon as league revisions join -- a fixed start would mean the
    tail never plays at all.
    """

    if not opponents:
        return opponents
    start = offset % len(opponents)
    return [*opponents[start:], *opponents[:start]]


def _learner_policy(weights_path: Path, epsilon: float, gamma: float) -> EvaluationPolicy:
    """Identity of the exploring behaviour policy for one batch.

    The spec actually played also carries a per-episode trace path and
    exploration seed; folding those into the identity would mint one revision
    per rollout episode. What defines the policy is its weights and how much it
    explores, so the identity spec carries only those -- and stays resolvable.

    Epsilon is bucketed to two decimals. It decays continuously, so carrying
    full precision means a distinct revision every batch and no revision with
    more than `episodes_per_batch` games to its name.
    """
    spec = f"{LEARNER_ENTRY_POINT}?" + urlencode(
        {
            "weights": str(weights_path.resolve()),
            "epsilon": round(epsilon, 2),
            "gamma": gamma,
        }
    )
    return EvaluationPolicy(
        _policy_identity("market-rollout-policy", spec),
        spec,
        LEARNER_MANIFEST,
        POLICY_KINDS["market_learner"],
    )


def _materialize_behavior_policy(
    network: MarketQNetwork, path: Path, *, refresh: bool = False
) -> None:
    """Write the worker-visible policy, optionally replacing a cached cold start.

    Checkpoint paths are content-addressed and therefore immutable, so writing
    one once is enough. ``initial`` is not content-addressed: it is a fixed name
    that two sessions sharing an artifact root would both claim, and the second
    one's cold start need not equal the first's. Rewrite it from the current
    initializer before each pre-checkpoint rollout rather than trusting the name.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not path.is_file():
        network.save(path)


class MarketCollector:
    """Roll out exploration episodes for one closed league round.

    Off-policy, so the league's population is irrelevant -- only its clock is.
    The round hook supplies the resolved config and active checkpoint, reads go
    through the SELECT-only reader, and writes go through the writer, which is
    already bound to this model.

    Every episode is recorded twice over: as decisions in the batch `.npz` the
    updater trains from, and as an ordinary league pairing under
    `pairing_kind = 'market_rollout'`, so what the learner actually played is
    answerable in SQL rather than only inferable from an artifact.
    """

    def collect(self, context, reader, writer) -> list[str]:
        config = context.config
        count = int(config.get("batches_per_round", 1))
        if count < 1:
            raise ValueError("batches_per_round must be positive")
        episodes = int(config.get("episodes_per_batch", 16))
        if episodes < 2 or episodes % 2:
            raise ValueError("episodes_per_batch must be a positive even number")
        # The lineage checkpoint, not the released one. Gating release is a
        # decision about what to ship; applying it to the behaviour policy as
        # well means a run that stops promoting also stops seeing new states.
        checkpoint = lineage_checkpoint(context, reader)
        network = (
            MarketQNetwork.load(checkpoint.inference_artifact_path)
            if checkpoint is not None
            else initial_network(config)
        )

        total_transitions = int(
            reader.fetchone(
                """
                SELECT coalesce(sum(CAST(metadata_json ->> 'market_transitions' AS BIGINT)), 0)
                FROM training_batches WHERE model_id = %s
                """,
                [context.model_id],
            )[0]
        )
        # Rollouts run in spawned workers and the pairing needs a stable policy
        # identity, so the behaviour policy goes to a file both the workers and
        # the league refer to. Keyed by the checkpoint it came from, not by
        # generation and round: `_learner_policy` hashes this path into the
        # revision id, and a per-round path mints a throwaway revision every
        # batch, none with enough games behind it to carry a rating.
        behavior_path = (
            Path(context.artifacts)
            / "rollout_policies"
            / f"{checkpoint.checkpoint_id if checkpoint is not None else 'initial'}.npz"
        )
        _materialize_behavior_policy(
            network,
            behavior_path,
            refresh=checkpoint is None,
        )
        trace_root = (
            Path(context.artifacts)
            / "episode_traces"
            / context.generation_id
            / f"round-{context.round_index:04d}"
        )

        seed = int(config.get("seed", 0))
        rng = np.random.default_rng(seed + total_transitions)
        written: list[str] = []
        for _ in range(count):
            fraction = min(
                1.0,
                total_transitions
                / max(1, int(config.get("epsilon_decay_transitions", 40_000))),
            )
            start = float(config.get("market_epsilon_start", 1.0))
            end = float(config.get("market_epsilon_end", 0.05))
            epsilon = start + fraction * (end - start)
            batch_index = int(
                reader.fetchone(
                    "SELECT count(*) FROM training_batches WHERE model_id = %s",
                    [context.model_id],
                )[0]
            )
            # One seed per pair, played twice: the learner takes seat 0 in one
            # orientation and seat 1 in the other, which is what lets the
            # pairing settle a seat-balanced margin.
            seeds = [
                int(value)
                for value in rng.integers(0, 2**31 - 1, size=episodes // 2)
                for _ in range(2)
            ]
            configured_opponents = config.get("opponents")
            base = (
                [str(item) for item in configured_opponents]
                if configured_opponents
                else [promotion_benchmark(config)]
            )
            if config.get("league_opponents", True):
                base += league_opponents(context, reader)
            base += active_checkpoint_opponents(
                context,
                reader,
                [str(item) for item in config.get("active_peer_types", [])],
            )
            # Static, league, and active-peer sources may resolve to the same
            # specification; a duplicate would unintentionally double its weight.
            base = list(dict.fromkeys(base))
            pool: list[str] = []
            if batch_index >= int(config.get("self_play_after", 20)):
                history_types = [str(item) for item in config.get("history_peer_types", [])]
                pool = (
                    historical_checkpoint_opponents(
                        context, reader, history_types, int(config.get("self_play_pool", 5))
                    )
                    if history_types
                    else self_play_pool(context, reader, int(config.get("self_play_pool", 5)))
                )
            opponents = rotate_opponents(
                compose_opponents(
                    base,
                    pool,
                    float(config.get("self_play_ratio", 0.3)),
                    pool_offset=batch_index,
                ),
                # Slots consumed, not episodes played: seat rotation makes
                # `collect_batch` draw one opponent per seed *pair*, so a batch
                # of 16 advances the list by 8.
                batch_index * (episodes // 2),
            )
            gamma = float(config.get("gamma", 0.997))
            learner = _learner_policy(behavior_path, epsilon, gamma)
            rollouts = collect_batch(
                behavior_path,
                opponents,
                seeds,
                epsilon,
                gamma,
                int(config.get("jobs", 1)),
                config={"episodeSteps": int(config.get("episode_steps", 720))},
                trace_root=trace_root,
                policy_revision_id=learner.revision_id,
            )
            if not rollouts:
                raise RuntimeError("every market rollout episode failed")
            records = [rollout.record for rollout in rollouts]
            artifact_ids: list[str] = []
            for rollout in rollouts:
                artifact_ids.extend(
                    writer.record_policy_episode(
                        context=context,
                        pairing_kind="market_rollout",
                        pairing_index=batch_index * (episodes // 2) + rollout.pair_index,
                        left=learner,
                        right=_opponent_policy(
                            rollout.opponent_spec,
                            role=POLICY_KINDS["training_opponent"],
                            prefix="market-rollout-opponent",
                        ),
                        # Seat order as played: orientation 0 is (learner,
                        # opponent), which is exactly the learner holding seat 0.
                        orientation=rollout.learner_seat,
                        result=rollout.match,
                    )
                )
            source_id = f"rollout-{batch_index:08d}"
            path = Path(context.artifacts) / "training_batches" / f"{source_id}.npz"
            save_episode_batch(records, path)
            market_transitions = sum(record.decisions for record in records)
            played = [rollout.opponent_spec for rollout in rollouts]
            batch = writer.record(
                source_kind=SOURCE_KIND,
                source_id=source_id,
                sample_count=len(records),
                artifact_path=path,
                artifact_sha256=_file_digest(path),
                metadata={
                    "episodes": len(records),
                    "market_transitions": market_transitions,
                    "turns": sum(record.steps for record in records),
                    "epsilon": epsilon,
                    "seeds": seeds,
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
            total_transitions += market_transitions
            written.append(source_id)
        return written


def _arena_evaluate(
    weights_path: Path,
    opponent: str,
    seeds: int,
    jobs: int,
    seed_start: int = 0,
) -> dict[str, Any]:
    from sim.arena import play_series
    from sim.metrics import summarize

    spec = f"{SELF_SPEC}?weights={weights_path}"
    results = play_series(spec, opponent, range(seed_start, seed_start + seeds), {}, jobs)
    summary = summarize(spec, opponent, results)
    return {
        "opponent": opponent,
        "pairs": summary.pairs,
        "win_rate": summary.win_rate,
        "margin": summary.margin_mean,
        "margin_ci": list(summary.margin_ci),
        "significant": summary.significant,
        "errors": summary.errors,
    }


def _benchmark_kind(index: int) -> str:
    """Name the evaluation slot. Index 0 is the release baseline, by convention.

    `configured_benchmarks` puts the promotion benchmark first, so the first
    slot is the one the gate reads and the rest are non-regression diagnostics.
    The suffix is kept on those so two benchmarks never collide in
    `model_evaluations`.
    """

    if index == 0:
        return POLICY_KINDS["promotion_baseline"]
    return f"{POLICY_KINDS['fixed_benchmark']}:{index}"


def _reject_degenerate(result: dict[str, Any], role: str) -> None:
    """Fail loudly when the candidate evaluates as the policy it is measured against.

    Paired seeds cancel exactly only when both seats run one policy, so a
    zero-width interval over many pairs is a harness fault -- a dropped
    `?weights=`, a missing checkpoint, an inert market layer -- not evidence
    about the candidate. Scored as an ordinary result it reads as "no
    significant improvement" and the session rejects every update forever while
    looking healthy; on the donor benchmark it reads as a clean tie and the
    initial checkpoint promotes.
    """

    if result["errors"] or int(result["pairs"]) <= 1:
        return
    low, high = (float(bound) for bound in result["margin_ci"])
    if low == 0.0 and high == 0.0:
        raise RuntimeError(
            f"{role} margin is identically zero over "
            f"{result['pairs']} paired seeds against {result['opponent']}: "
            "the candidate is playing as the same policy"
        )


def _reject_inert_network(network: MarketQNetwork) -> None:
    """Fail loudly when the exported network cannot express anything but a tie.

    A market layer whose hidden weights are all zero outputs a constant Q for
    every candidate, so every argmax ties to index 0 and the agent is the
    unmodified donor. That is what an all-zero initialization collapses to, and
    it survives the gate: it beats the donor's own opponents exactly as well as
    the donor does. Cheaper and more direct than inferring it from a benchmark
    interval, and independent of which benchmark is configured.
    """

    if _has_inert_hidden_trunk(network):
        raise RuntimeError(
            "exported market network has all-zero hidden weights: it scores every "
            "candidate identically, so it plays the unmodified donor policy. An "
            "all-zero initialization has no gradient outside its output bias -- "
            "use `donor_default` instead"
        )


def _has_inert_hidden_trunk(network: MarketQNetwork) -> bool:
    """Whether the hidden trunk is the legacy all-zero initialization."""

    hidden = network.layers[:-1]
    return bool(hidden and all(not weight.any() for weight, _ in hidden))


def benchmark_non_regression(challenger: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    """Apply the release gate to one fixed benchmark result pair."""

    return (
        not challenger["errors"]
        and not incumbent["errors"]
        and float(challenger["margin_ci"][1]) >= float(incumbent["margin_ci"][0])
    )


def policy_improvement(result: dict[str, Any], promotion_rule: str) -> bool:
    """Whether a paired evaluation clears the configured improvement bar."""

    if result["errors"]:
        return False
    if promotion_rule == "point_estimate":
        return float(result["margin"]) > 0.0
    return float(result["margin_ci"][0]) > 0.0


class MarketUpdater:
    def update(
        self, context: ModelUpdateContext, reader: ModelTrainingReader
    ) -> ModelUpdateResult:
        config = context.config
        device = resolve_device(str(config.get("device", "auto")))
        torch.manual_seed(int(config.get("seed", 0)))
        buffer = ReplayBuffer(
            int(config.get("buffer_episodes", 800)),
            float(config.get("gamma", 0.997)),
            int(config.get("n_step", 3)),
            int(config.get("seed", 0)),
            float(config.get("reservoir_fraction", 0.35)),
        )
        skipped_legacy_batches = 0
        for batch in context.all_batches:
            if batch.source_kind != SOURCE_KIND or not batch.artifact_path:
                continue
            path = Path(batch.artifact_path)
            if batch.artifact_sha256 and _file_digest(path) != batch.artifact_sha256:
                raise ValueError(f"rollout artifact digest mismatch: {path}")
            try:
                records = load_episode_batch(path)
            except EpisodeFormatError:
                # A schema change leaves old batches auditable but not trainable;
                # training them under the current replay semantics would be worse
                # than dropping them.
                skipped_legacy_batches += 1
                continue
            buffer.add(records)
        warmup = int(config.get("warmup_episodes", 16))
        if len(buffer.episodes) < warmup:
            raise ValueError(f"market training requires at least {warmup} replay episodes")

        online = net_module.build().to(device)
        target = net_module.build().to(device)
        optimizer = torch.optim.Adam(
            online.parameters(), lr=float(config.get("market_learning_rate", 1e-4))
        )
        gradient_steps = 0
        resumed = lineage_checkpoint(context, reader)
        resumed_from = resumed.checkpoint_id if resumed is not None else None
        if resumed is not None:
            state = torch.load(
                resumed.trainer_artifact_path, map_location="cpu", weights_only=False
            )
            online.load_state_dict(state["market_online"])
            target.load_state_dict(state["market_target"])
            optimizer.load_state_dict(state["market_optimizer"])
            move_optimizer(optimizer, device)
            gradient_steps = int(state.get("market_gradient_steps", 0))
            if "torch_rng_state" in state:
                torch.set_rng_state(state["torch_rng_state"])
            if "replay_rng_state" in state:
                buffer.rng.bit_generator.state = state["replay_rng_state"]
        else:
            online = net_module.from_numpy(initial_network(config), online)
            target.load_state_dict(online.state_dict())

        loss_fn = nn.SmoothL1Loss()
        losses: list[float] = []
        steps_to_run = (
            int(config.get("market_gradient_steps_per_batch", 400)) * len(context.new_batches)
            if buffer.market_transitions
            else 0
        )
        batch_size = int(config.get("market_batch_size", 256))
        target_sync = int(config.get("market_target_sync", 2000))
        started = time.perf_counter()
        for _ in range(steps_to_run):
            batch = buffer.sample(batch_size)
            features = torch.from_numpy(batch.features).to(device)
            next_features = torch.from_numpy(batch.next_features).to(device)
            next_mask = torch.from_numpy(batch.next_mask).to(device)
            reward = torch.from_numpy(batch.reward).to(device)
            discount = torch.from_numpy(batch.discount).to(device)
            with torch.no_grad():
                flat = next_features.reshape(-1, next_features.shape[-1])
                shape = next_features.shape[:3]
                online_q = online(flat).reshape(shape)
                # Double DQN, per factor: the online net picks each factor's
                # argmax, the target net scores it. The joint value is the sum
                # over factors, which is the decomposition the behaviour policy
                # also maximises factor-wise.
                best = online_q.masked_fill(~next_mask, float("-inf")).argmax(dim=2)
                target_q = target(flat).reshape(shape)
                factor_values = target_q.gather(2, best.unsqueeze(2)).squeeze(2)
                factor_values = torch.where(
                    next_mask.any(dim=2), factor_values, torch.zeros_like(factor_values)
                )
                goal = reward + discount * factor_values.sum(dim=1)
            predicted = (
                online(features.reshape(-1, features.shape[-1]))
                .reshape(features.shape[:2])
                .sum(dim=1)
            )
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
                "market_online": online.state_dict(),
                "market_target": target.state_dict(),
                "market_optimizer": optimizer.state_dict(),
                "market_gradient_steps": gradient_steps,
                "torch_rng_state": torch.get_rng_state(),
                "replay_rng_state": buffer.rng.bit_generator.state,
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
        benchmarks = configured_benchmarks(config)
        challenger_benchmarks = [
            _arena_evaluate(inference_path, benchmark, eval_seeds, jobs, seed_start)
            for benchmark in benchmarks
        ]
        # The primary benchmark is the donor this agent derives from, so a
        # zero-width interval here is the specific failure this trainer is most
        # exposed to: an inert market layer that plays the donor exactly. Left
        # unchecked it reads as a clean tie, and on the first checkpoint that is
        # one `policy_improvement` call away from promoting nothing at all.
        for result in challenger_benchmarks:
            _reject_degenerate(result, "benchmark")

        evaluations: list[ModelEvaluation] = []
        promote = False
        incumbent_path = (
            Path(context.active_checkpoint.inference_artifact_path)
            if context.active_checkpoint is not None
            else (
                _incumbent_path(config)
                if str(config.get("initialization", "donor_default")) == "incumbent"
                else None
            )
        )
        promotion_rule = str(config.get("promotion_rule", "significant"))
        if promotion_rule not in _PROMOTION_RULES:
            raise ValueError(
                f"promotion_rule must be one of {sorted(_PROMOTION_RULES)}: {promotion_rule}"
            )
        gate: dict[str, Any] = {
            "kind": f"{promotion_rule}_incumbent_improvement_and_benchmarks_non_regression",
            "policy_kinds": POLICY_KINDS,
            "promotion_rule": promotion_rule,
            "seed_start": seed_start,
            "challenger_benchmarks": challenger_benchmarks,
            "resumed_from_checkpoint_id": resumed_from,
            "rejection_streak": resumed.rejection_streak if resumed is not None else 0,
            "lineage_reset": resumed is not None and resumed.reset,
        }
        if incumbent_path is None:
            # The primary benchmark is the release baseline until a checkpoint
            # exists. A clean loss or exact donor tie is not a successful first
            # checkpoint; the configured promotion rule must clear zero here in
            # exactly the same way it later clears the active incumbent.
            benchmark_results = [
                policy_improvement(challenger_benchmarks[0], promotion_rule),
                *(not result["errors"] for result in challenger_benchmarks[1:]),
            ]
            promote = all(benchmark_results)
            gate.update(
                {
                    "initial_checkpoint": True,
                    "primary_benchmark_improved": benchmark_results[0],
                    "passed": promote,
                }
            )
            evaluations.extend(
                ModelEvaluation(_benchmark_kind(index), result, passed)
                for index, (result, passed) in enumerate(
                    zip(challenger_benchmarks, benchmark_results)
                )
            )
        else:
            incumbent_spec = f"{SELF_SPEC}?weights={incumbent_path}"
            head_to_head = _arena_evaluate(
                inference_path, incumbent_spec, promotion_seeds, jobs, seed_start
            )
            _reject_degenerate(head_to_head, "head-to-head")
            incumbent_benchmarks = [
                _arena_evaluate(incumbent_path, benchmark, eval_seeds, jobs, seed_start)
                for benchmark in benchmarks
            ]
            # Two rules, because "release" and "carry the run forward" are not
            # the same decision. `significant` is the release rule: the paired
            # lower bound must clear zero, which is what stops a coin flip from
            # becoming the incumbent. `point_estimate` only asks the paired mean
            # to be positive, which promotes noise about half the time it is
            # offered -- acceptable when the point of promoting is to keep the
            # rollout distribution moving rather than to ship. Either way every
            # fixed benchmark must stay statistically compatible with its
            # incumbent.
            head_to_head_ok = policy_improvement(head_to_head, promotion_rule)
            benchmark_results = [
                benchmark_non_regression(challenger, incumbent)
                for challenger, incumbent in zip(challenger_benchmarks, incumbent_benchmarks)
            ]
            benchmarks_ok = all(benchmark_results)
            promote = head_to_head_ok and benchmarks_ok
            gate.update(
                {
                    "head_to_head": head_to_head,
                    "incumbent_benchmarks": incumbent_benchmarks,
                    "head_to_head_significantly_better": head_to_head_ok,
                    "benchmark_non_regression": dict(zip(benchmarks, benchmark_results)),
                    "passed": promote,
                }
            )
            opponent_checkpoint_id = (
                context.active_checkpoint.checkpoint_id
                if context.active_checkpoint is not None
                else None
            )
            evaluations.append(
                ModelEvaluation(
                    "incumbent_head_to_head", head_to_head, head_to_head_ok, opponent_checkpoint_id
                )
            )
            evaluations.extend(
                ModelEvaluation(_benchmark_kind(index), challenger, passed)
                for index, (challenger, passed) in enumerate(
                    zip(challenger_benchmarks, benchmark_results)
                )
            )
        metrics = {
            "policy_kinds": POLICY_KINDS,
            "resumed_from_checkpoint_id": resumed_from,
            "resumed_from_released": resumed is not None and resumed.released,
            "rejection_streak": resumed.rejection_streak if resumed is not None else 0,
            "lineage_reset": resumed is not None and resumed.reset,
            "episodes_in_buffer": len(buffer.episodes),
            "skipped_legacy_batches": skipped_legacy_batches,
            "market_transitions_in_buffer": buffer.market_transitions,
            "gradient_steps": gradient_steps,
            "gradient_steps_this_update": steps_to_run,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "train_seconds": train_seconds,
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
