"""Persistent collector and incremental updater for the DQN job scorer."""

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

from agents.dqn_job_scorer.market_policy import MARKET_FEATURE_DIM
from agents.dqn_job_scorer.policy import WEIGHTS_PATH, PolicyBundle, QNetwork
from agents.dqn_job_scorer.training import network as net_module
from agents.dqn_job_scorer.training.collect import (
    LEARNER_ENTRY_POINT,
    EpisodeFormatError,
    collect_batch,
    load_episode_batch,
    save_episode_batch,
)
from agents.dqn_job_scorer.training.device import move_optimizer, resolve_device
from agents.dqn_job_scorer.training.peers import (
    active_checkpoint_opponents,
    historical_checkpoint_opponents,
)
from agents.dqn_job_scorer.training.progress import _opponent_policy, _policy_identity
from agents.dqn_job_scorer.training.replay import ReplayBuffer
from kaggriculture_lab.model_trainers import EvaluationPolicy
from kaggriculture_lab.model_updates import (
    ModelEvaluation,
    ModelTrainingReader,
    ModelUpdateContext,
    ModelUpdateResult,
)

SELF_SPEC = "agents/dqn_job_scorer/main.py"
LEARNER_MANIFEST = "agents/dqn_job_scorer/match_agent.toml"
_PROMOTION_RULES = {"significant", "point_estimate"}


def _incumbent_path(config: dict[str, Any]) -> Path:
    return Path(str(config.get("incumbent_weights", WEIGHTS_PATH)))


def initial_bundle(config: dict[str, Any]) -> PolicyBundle:
    """Build the first behavior policy for a new training instance.

    Fixed-curriculum runs can continue from an inference-only checkpoint while
    starting a fresh optimizer and replay buffer. Ordinary DQN experiments keep
    their historical random initialization unless they opt in explicitly.
    """

    initialization = str(config.get("initialization", "random"))
    if initialization == "random":
        return PolicyBundle.random(seed=int(config.get("seed", 0)))
    if initialization != "incumbent":
        raise ValueError(f"unsupported DQN initialization: {initialization}")
    path = _incumbent_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"DQN incumbent checkpoint does not exist: {path}")
    bundle = PolicyBundle.load(path)
    if bundle.market is None:
        bundle = PolicyBundle(
            bundle.job,
            QNetwork.random(
                seed=int(config.get("seed", 0)) + 1,
                feature_dim=MARKET_FEATURE_DIM,
            ),
        )
    return bundle


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


def self_play_pool(context, reader, limit: int) -> list[str]:
    """The most recent promoted checkpoints, addressed as agent specs.

    A snapshot is reached through the agent's `weights` config key rather than as
    a bare `.npz` path: the registry resolves agent modules, not weight files.
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
    return [
        f"{SELF_SPEC}?weights={row[0]}" for row in rows if Path(str(row[0])).is_file()
    ]


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


def _rejections_since_restart(reader, model_id: str) -> int:
    """Rejections since the run last got a clean start.

    A clean start is a promotion or a previous reset. Counting only from the
    last promotion would make the reset fire on every update once it fired once
    -- the streak never drops below the limit until something promotes, so the
    run would go back to re-deriving the same candidate from the release, which
    is the waste the lineage exists to avoid. Resetting buys another `limit`
    updates, not a permanent policy.
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


def _reset_since_restart(reader, model_id: str) -> bool:
    """Whether a reset has been recorded with nothing promoted after it.

    The companion to `_rejections_since_restart`, and it exists because the streak
    alone cannot express "the restart is still in progress". A reset ends the
    streak by design, so one update later the counter reads healthy while the only
    evidence available says otherwise. Kept identical to the market scorer's
    `reset_since_promotion`; these two copies diverging is what produced the
    streak bug the docstring above records.
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
    *released* policy, and both resuming gradient descent from it and rolling out
    from it are wrong in the same way: the first discards every step a rejected
    update took, and the second freezes the behaviour distribution for as long as
    the gate says no. dqn-self-play-v1 did both -- it reloaded the released
    weights thirteen times running while every rollout batch from 60 to 340 came
    out of one checkpoint frozen at 07:59, and the learner drifted to a policy
    that lost 100 of 100 to its own incumbent and 92 of 100 to `starter`.

    Two escapes. Training and collection both follow the lineage, so the data
    tracks the learner without shipping an ungated policy; and once the run has
    been rejected `resume_reset_after_rejections` times in a row, the lineage
    snaps back to the release, because carrying a diverged trajectory forward
    forever is how it stays diverged.

    **A run that has never promoted does not reset at all.** It has no release to
    snap back to, so the only "not carried forward" available is `None` -- a wipe
    back to a fresh network, discarding the run's whole training history. That was
    the behaviour until a genuine cold start showed what it costs: such a run needs
    many generations to reach a first promotion, so at any useful limit it is
    reinitialized long before it gets there. Measured on the free-market agent's
    `market-free-landgate-v10` at a limit of 2 -- zero promotions, two consecutive
    rejections, and the next update due to discard all sixty batches behind it.

    The cost, taken knowingly and kept in step with the market scorer's copy: a
    never-released run can ride one diverged trajectory indefinitely, which is
    what that copy measured as twelve consecutive rejections at a limit of five.
    A wipe is only defensible when there is something better to fall back to.

    Falls back to the released checkpoint when the newer artifact is missing --
    trainer `.pt` files are large and a pruned artifact root must not silently
    restart the run from random weights. That case is deliberately *not* a reset:
    the artifact being absent says nothing about the trajectory.
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
    streak = _rejections_since_restart(reader, context.model_id) if limit > 0 else 0
    # A reset holds until something promotes. It used to last exactly one update:
    # the reset ends the streak, but the resetting update still trains and still
    # writes a checkpoint row, and the newest-by-`created_at` rule above then
    # handed the lineage back to whatever that update produced -- for training
    # *and* for collection. Measured on the free-market agent, whose copy of this
    # had the same hole: a reset produced a candidate worth 357 against a released
    # checkpoint worth 18,339, and the next twenty batches were collected from the
    # 357 policy. Skipping just that checkpoint would be worse, since the next
    # newest is the diverged row the reset existed to escape.
    #
    # It is a real departure from `_rejections_since_restart`'s note that
    # "resetting buys another `limit` updates, not a permanent policy" -- the hold
    # *is* semi-permanent, ending only at a promotion. That note is right in
    # general and stops being right once `limit` consecutive rejections have
    # already proven the trajectory bad. The hold also repairs itself: collection
    # follows the release too, so the ring refills with the released policy's
    # episodes instead of the discarded one's.
    #
    # Both conditions require a release, so `fallback` gates the whole branch.
    # With nothing promoted there is no snapback available -- only a wipe back to
    # a fresh network, which discards the run's entire training history and which
    # a cold start cannot survive at any useful limit. See the docstring.
    fallback = released()
    held = fallback is not None and _reset_since_restart(reader, context.model_id)
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
    the sampling distribution. This used to be `[*base, *pool]`, which with a
    single benchmark and a pool of five made self-play 5 of every 6 episodes
    while evaluation stayed on the benchmark; dqn04 fell from +2,269 to -22,154
    on the round that switched on. Solve for the slot count instead:
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
    return [
        *widened,
        *(pool[(pool_offset + index) % len(pool)] for index in range(slots)),
    ]


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


def _learner_policy(
    weights_path: Path,
    epsilon: float,
    market_epsilon: float,
    gamma: float,
) -> EvaluationPolicy:
    """Identity of the exploring behaviour policy for one batch.

    The spec actually played also carries a per-episode trace path and
    exploration seed; folding those into the identity would mint one revision
    per rollout episode. What defines the policy is its weights and how much it
    explores, so the identity spec carries only those -- and stays resolvable.

    The exploration rates are bucketed to two decimals. They decay continuously,
    so carrying full precision means a distinct revision every batch and no
    revision with more than `episodes_per_batch` games to its name; the whole
    decay from 1.0 to the floor is ~85 buckets, and once at the floor it is one.
    The identity is therefore a rounding of the policy played, not the policy
    played -- close enough to name it, and the exact rates stay on the batch.
    """
    spec = f"{LEARNER_ENTRY_POINT}?" + urlencode(
        {
            "weights": str(weights_path.resolve()),
            "epsilon": round(epsilon, 2),
            "market_epsilon": round(market_epsilon, 2),
            "gamma": gamma,
        }
    )
    return EvaluationPolicy(
        _policy_identity("dqn-rollout-policy", spec),
        spec,
        LEARNER_MANIFEST,
        "rollout_learner",
    )


class DQNCollector:
    """Roll out exploration episodes for one closed league round.

    Off-policy, so the league's population is irrelevant -- only its clock is.
    The round hook supplies the resolved config and active checkpoint, reads go
    through the SELECT-only reader, and writes go through the writer, which is
    already bound to this model.

    Every episode is recorded twice over: as transitions in the batch `.npz` the
    updater trains from, and as an ordinary league pairing under
    `pairing_kind = 'dqn_rollout'`, so what the learner actually played is
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
        # well means a run that stops promoting also stops seeing new states,
        # which is the loop dqn-self-play-v1 could not get out of.
        checkpoint = lineage_checkpoint(context, reader)
        if checkpoint is not None:
            bundle = PolicyBundle.load(checkpoint.inference_artifact_path)
            if bundle.market is None:
                bundle = PolicyBundle(
                    bundle.job,
                    QNetwork.random(
                        seed=int(config.get("seed", 0)) + 1,
                        feature_dim=MARKET_FEATURE_DIM,
                    ),
                )
        else:
            bundle = initial_bundle(config)

        total_transitions = int(
            reader.fetchone(
                """
                SELECT coalesce(sum(CAST(metadata_json ->> 'transitions' AS BIGINT)), 0)
                FROM training_batches WHERE model_id = %s
                """,
                [context.model_id],
            )[0]
        )
        # Rollouts run in spawned workers and the pairing needs a stable policy
        # identity, so the behaviour policy goes to a file both the workers and
        # the league refer to. Keyed by the checkpoint it came from, not by
        # generation and round: `_learner_policy` hashes this path into the
        # revision id, and with `batches_per_round = 1` a per-round path minted a
        # throwaway revision every batch -- 312 of them over 4,992 episodes, none
        # with enough games behind it to carry a rating.
        behavior_path = (
            Path(context.artifacts)
            / "rollout_policies"
            / f"{checkpoint.checkpoint_id if checkpoint is not None else 'initial'}.npz"
        )
        behavior_path.parent.mkdir(parents=True, exist_ok=True)
        if not behavior_path.is_file():
            bundle.save(behavior_path)
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
                total_transitions / max(1, int(config.get("epsilon_decay_transitions", 200_000))),
            )
            epsilon = float(config.get("epsilon_start", 1.0)) + fraction * (
                float(config.get("epsilon_end", 0.05)) - float(config.get("epsilon_start", 1.0))
            )
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
                else [
                    str(config.get("benchmark", "starter"))
                    if checkpoint is not None
                    else "starter"
                ]
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
                history_types = [
                    str(item) for item in config.get("history_peer_types", [])
                ]
                pool = (
                    historical_checkpoint_opponents(
                        context,
                        reader,
                        history_types,
                        int(config.get("self_play_pool", 5)),
                    )
                    if history_types
                    else self_play_pool(
                        context, reader, int(config.get("self_play_pool", 5))
                    )
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
                # of 16 advances the list by 8. Offsetting by 16 would step over
                # half of it every batch.
                batch_index * (episodes // 2),
            )
            gamma = float(config.get("gamma", 0.997))
            market_epsilon = float(config.get("market_epsilon_start", 1.0)) + fraction * (
                float(config.get("market_epsilon_end", config.get("epsilon_end", 0.15)))
                - float(config.get("market_epsilon_start", 1.0))
            )
            learner = _learner_policy(behavior_path, epsilon, market_epsilon, gamma)
            rollouts = collect_batch(
                behavior_path,
                opponents,
                seeds,
                epsilon,
                gamma,
                int(config.get("jobs", 1)),
                config={"episodeSteps": int(config.get("episode_steps", 720))},
                market_epsilon=market_epsilon,
                trace_root=trace_root,
                policy_revision_id=learner.revision_id,
            )
            if not rollouts:
                raise RuntimeError("every DQN rollout episode failed")
            records = [rollout.record for rollout in rollouts]
            artifact_ids: list[str] = []
            for rollout in rollouts:
                artifact_ids.extend(
                    writer.record_policy_episode(
                        context=context,
                        pairing_kind="dqn_rollout",
                        pairing_index=batch_index * (episodes // 2) + rollout.pair_index,
                        left=learner,
                        right=_opponent_policy(
                            rollout.opponent_spec,
                            role="rollout_opponent",
                            prefix="dqn-rollout-opponent",
                        ),
                        # Seat order as played: orientation 0 is (learner,
                        # opponent), which is exactly the learner holding seat 0.
                        orientation=rollout.learner_seat,
                        result=rollout.match,
                    )
                )
            source_id = f"rollout-{batch_index:08d}"
            path = (
                Path(context.artifacts)
                / "training_batches"
                / f"{source_id}.npz"
            )
            save_episode_batch(records, path)
            transitions = sum(int(record.unit_mask.sum()) for record in records)
            market_transitions = sum(0 if record.market_step is None else len(record.market_step) for record in records)
            played = [rollout.opponent_spec for rollout in rollouts]
            batch = writer.record(
                source_kind="dqn_rollout",
                source_id=source_id,
                sample_count=len(records),
                artifact_path=path,
                artifact_sha256=_file_digest(path),
                metadata={
                    "episodes": len(records),
                    "transitions": transitions,
                    "market_transitions": market_transitions,
                    "epsilon": epsilon,
                    "market_epsilon": market_epsilon,
                    "seeds": seeds,
                    "behavior_checkpoint_id": (
                        checkpoint.checkpoint_id if checkpoint is not None else None
                    ),
                    "behavior_policy_revision_id": learner.revision_id,
                    "session_id": context.session_id,
                    "generation_id": context.generation_id,
                    "round_index": context.round_index,
                    # Which opponents actually played, so a collapse like dqn04's
                    # can be traced to the mix rather than guessed at.
                    "opponents": played,
                    "self_play_episodes": sum(
                        1 for spec in played if "?weights=" in spec
                    ),
                },
            )
            writer.link_artifacts(batch.batch_id, artifact_ids)
            total_transitions += transitions
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


def configured_benchmarks(config: dict[str, Any]) -> list[str]:
    """Return the fixed release suite, accepting the legacy singular key."""

    configured = config.get("benchmarks")
    if configured:
        return list(dict.fromkeys(str(item) for item in configured))
    return [
        str(
            config.get(
                "benchmark", "agents/configurable_crop_surrogate/main.py"
            )
        )
    ]


def _benchmark_kind(index: int, total: int) -> str:
    return "fixed_benchmark" if total == 1 else f"fixed_benchmark:{index}"


def _reject_degenerate_head_to_head(result: dict[str, Any]) -> None:
    """Fail loudly when challenger and incumbent evaluate as the same policy.

    Paired seeds cancel exactly only when both seats run one policy, so a
    zero-width interval over many pairs is a harness fault -- a dropped
    `?weights=`, a missing checkpoint -- not evidence about the candidate.
    Scored as an ordinary result it reads as "no significant improvement" and
    the session rejects every update forever while looking healthy.
    """

    if result["errors"] or int(result["pairs"]) <= 1:
        return
    low, high = (float(bound) for bound in result["margin_ci"])
    if low == 0.0 and high == 0.0:
        raise RuntimeError(
            "head-to-head margin is identically zero over "
            f"{result['pairs']} paired seeds against {result['opponent']}: "
            "challenger and incumbent are playing as the same policy"
        )


def benchmark_non_regression(
    challenger: dict[str, Any], incumbent: dict[str, Any]
) -> bool:
    """Apply the release gate to one fixed benchmark result pair."""

    return (
        not challenger["errors"]
        and not incumbent["errors"]
        and float(challenger["margin_ci"][1])
        >= float(incumbent["margin_ci"][0])
    )


class DQNUpdater:
    def update(
        self, context: ModelUpdateContext, reader: ModelTrainingReader
    ) -> ModelUpdateResult:
        config = context.config
        device = resolve_device(str(config.get("device", "auto")))
        torch.manual_seed(int(config.get("seed", 0)))
        buffer = ReplayBuffer(
            int(config.get("buffer_episodes", 200)),
            float(config.get("gamma", 0.997)),
            int(config.get("n_step", 3)),
            int(config.get("seed", 0)),
            float(config.get("reservoir_fraction", 0.35)),
        )
        skipped_legacy_batches = 0
        for batch in context.all_batches:
            if batch.source_kind != "dqn_rollout" or not batch.artifact_path:
                continue
            path = Path(batch.artifact_path)
            if batch.artifact_sha256 and _file_digest(path) != batch.artifact_sha256:
                raise ValueError(f"rollout artifact digest mismatch: {path}")
            try:
                records = load_episode_batch(path)
            except EpisodeFormatError:
                # Format 1 did not record the live action mask. Treating every
                # job as legal would reintroduce the corruption this migration
                # fixes, so old batches remain auditable but are not trainable.
                skipped_legacy_batches += 1
                continue
            buffer.add(records)
        warmup = int(config.get("warmup_episodes", 16))
        if len(buffer.episodes) < warmup:
            raise ValueError(f"DQN requires at least {warmup} replay episodes")

        online = net_module.build().to(device)
        target = net_module.build().to(device)
        market_online = net_module.build(feature_dim=MARKET_FEATURE_DIM).to(device)
        market_target = net_module.build(feature_dim=MARKET_FEATURE_DIM).to(device)
        optimizer = torch.optim.Adam(
            online.parameters(), lr=float(config.get("learning_rate", 1e-4))
        )
        market_optimizer = torch.optim.Adam(
            market_online.parameters(), lr=float(config.get("market_learning_rate", 1e-4))
        )
        gradient_steps = 0
        market_gradient_steps = 0
        resumed = lineage_checkpoint(context, reader)
        resumed_from = resumed.checkpoint_id if resumed is not None else None
        if resumed is not None:
            state = torch.load(
                resumed.trainer_artifact_path,
                map_location="cpu",
                weights_only=False,
            )
            online.load_state_dict(state["online"])
            target.load_state_dict(state["target"])
            optimizer.load_state_dict(state["optimizer"])
            move_optimizer(optimizer, device)
            gradient_steps = int(state.get("gradient_steps", 0))
            if "market_online" in state:
                market_online.load_state_dict(state["market_online"])
                market_target.load_state_dict(state["market_target"])
                market_optimizer.load_state_dict(state["market_optimizer"])
                move_optimizer(market_optimizer, device)
                market_gradient_steps = int(state.get("market_gradient_steps", 0))
            else:
                market_target.load_state_dict(market_online.state_dict())
            if "torch_rng_state" in state:
                torch.set_rng_state(state["torch_rng_state"])
            if "replay_rng_state" in state:
                buffer.rng.bit_generator.state = state["replay_rng_state"]
        else:
            bundle = initial_bundle(config)
            online = net_module.from_numpy(bundle.job, online)
            if bundle.market is not None:
                market_online = net_module.from_numpy(bundle.market, market_online)
            target.load_state_dict(online.state_dict())
            market_target.load_state_dict(market_online.state_dict())

        loss_fn = nn.SmoothL1Loss()
        losses: list[float] = []
        market_losses: list[float] = []
        steps_to_run = int(config.get("gradient_steps_per_batch", 400)) * len(
            context.new_batches
        )
        batch_size = int(config.get("batch_size", 512))
        target_sync = int(config.get("target_sync", 2000))
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
                online_q = online(flat).reshape(next_features.shape[:2])
                best = net_module.masked_argmax(online_q, next_mask)
                target_q = target(flat).reshape(next_features.shape[:2])
                bootstrap = target_q.gather(1, best.unsqueeze(1)).squeeze(1)
                bootstrap = torch.where(
                    next_mask.any(dim=1), bootstrap, torch.zeros_like(bootstrap)
                )
                goal = reward + discount * bootstrap
            predicted = online(features).squeeze(1)
            loss = loss_fn(predicted, goal)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            gradient_steps += 1
            if gradient_steps % target_sync == 0:
                target.load_state_dict(online.state_dict())

        market_steps_to_run = (
            int(config.get("market_gradient_steps_per_batch", 100)) * len(context.new_batches)
            if buffer.market_transitions
            else 0
        )
        market_batch_size = int(config.get("market_batch_size", 128))
        market_target_sync = int(config.get("market_target_sync", target_sync))
        for _ in range(market_steps_to_run):
            batch = buffer.sample_market(market_batch_size)
            features = torch.from_numpy(batch.features).to(device)
            next_features = torch.from_numpy(batch.next_features).to(device)
            next_mask = torch.from_numpy(batch.next_mask).to(device)
            reward = torch.from_numpy(batch.reward).to(device)
            discount = torch.from_numpy(batch.discount).to(device)
            with torch.no_grad():
                flat = next_features.reshape(-1, next_features.shape[-1])
                shape = next_features.shape[:3]
                online_q = market_online(flat).reshape(shape)
                best = online_q.masked_fill(~next_mask, float("-inf")).argmax(dim=2)
                target_q = market_target(flat).reshape(shape)
                factor_values = target_q.gather(2, best.unsqueeze(2)).squeeze(2)
                factor_values = torch.where(next_mask.any(dim=2), factor_values, torch.zeros_like(factor_values))
                goal = reward + discount * factor_values.sum(dim=1)
            predicted = market_online(features.reshape(-1, features.shape[-1])).reshape(features.shape[:2]).sum(dim=1)
            loss = loss_fn(predicted, goal)
            market_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(market_online.parameters(), 10.0)
            market_optimizer.step()
            market_losses.append(float(loss.detach()))
            market_gradient_steps += 1
            if market_gradient_steps % market_target_sync == 0:
                market_target.load_state_dict(market_online.state_dict())
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
                "gradient_steps": gradient_steps,
                "market_online": market_online.state_dict(),
                "market_target": market_target.state_dict(),
                "market_optimizer": market_optimizer.state_dict(),
                "market_gradient_steps": market_gradient_steps,
                "torch_rng_state": torch.get_rng_state(),
                "replay_rng_state": buffer.rng.bit_generator.state,
                "dataset_sha256": context.dataset_sha256,
                "device": str(device),
            },
            trainer_path,
        )
        PolicyBundle(
            net_module.to_numpy(online),
            net_module.to_numpy(market_online),
        ).save(inference_path)

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
        evaluations: list[ModelEvaluation] = []
        promote = False
        incumbent_path = (
            Path(context.active_checkpoint.inference_artifact_path)
            if context.active_checkpoint is not None
            else (
                _incumbent_path(config)
                if str(config.get("initialization", "random")) == "incumbent"
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
            "promotion_rule": promotion_rule,
            "seed_start": seed_start,
            "challenger_benchmarks": challenger_benchmarks,
            "resumed_from_checkpoint_id": resumed_from,
            "rejection_streak": resumed.rejection_streak if resumed is not None else 0,
            "lineage_reset": resumed is not None and resumed.reset,
        }
        if incumbent_path is None:
            benchmark_results = [not result["errors"] for result in challenger_benchmarks]
            promote = all(benchmark_results)
            gate.update({"initial_checkpoint": True, "passed": promote})
            evaluations.extend(
                ModelEvaluation(
                    _benchmark_kind(index, len(benchmarks)),
                    result,
                    passed,
                )
                for index, (result, passed) in enumerate(
                    zip(challenger_benchmarks, benchmark_results)
                )
            )
            head_to_head = None
        else:
            incumbent_spec = f"{SELF_SPEC}?weights={incumbent_path}"
            head_to_head = _arena_evaluate(
                inference_path,
                incumbent_spec,
                promotion_seeds,
                jobs,
                seed_start,
            )
            _reject_degenerate_head_to_head(head_to_head)
            incumbent_benchmarks = [
                _arena_evaluate(
                    incumbent_path, benchmark, eval_seeds, jobs, seed_start
                )
                for benchmark in benchmarks
            ]
            # Two rules, because "release" and "carry the run forward" are not
            # the same decision. `significant` is the release rule: the paired
            # lower bound must clear zero, which is what stops a +64-on-24-seeds
            # coin flip from becoming the incumbent. `point_estimate` only asks
            # the paired mean to be positive, which promotes noise about half the
            # time it is offered -- acceptable when the point of promoting is to
            # keep the rollout distribution and the self-play pool moving rather
            # than to ship. Either way every fixed benchmark must stay
            # statistically compatible with its incumbent.
            head_to_head_ok = not head_to_head["errors"] and (
                float(head_to_head["margin"]) > 0.0
                if promotion_rule == "point_estimate"
                else float(head_to_head["margin_ci"][0]) > 0.0
            )
            benchmark_results = [
                benchmark_non_regression(challenger, incumbent)
                for challenger, incumbent in zip(
                    challenger_benchmarks, incumbent_benchmarks
                )
            ]
            benchmarks_ok = all(benchmark_results)
            promote = head_to_head_ok and benchmarks_ok
            gate.update(
                {
                    "head_to_head": head_to_head,
                    "incumbent_benchmarks": incumbent_benchmarks,
                    "head_to_head_significantly_better": head_to_head_ok,
                    "benchmark_non_regression": dict(
                        zip(benchmarks, benchmark_results)
                    ),
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
                    "incumbent_head_to_head",
                    head_to_head,
                    head_to_head_ok,
                    opponent_checkpoint_id,
                )
            )
            evaluations.extend(
                ModelEvaluation(
                    _benchmark_kind(index, len(benchmarks)),
                    challenger,
                    passed,
                )
                for index, (challenger, passed) in enumerate(
                    zip(challenger_benchmarks, benchmark_results)
                )
            )
        metrics = {
            "resumed_from_checkpoint_id": resumed_from,
            "resumed_from_released": resumed is not None and resumed.released,
            "rejection_streak": resumed.rejection_streak if resumed is not None else 0,
            "lineage_reset": resumed is not None and resumed.reset,
            "episodes_in_buffer": len(buffer.episodes),
            "skipped_legacy_batches": skipped_legacy_batches,
            "transitions_in_buffer": buffer.transitions,
            "market_transitions_in_buffer": buffer.market_transitions,
            "gradient_steps": gradient_steps,
            "gradient_steps_this_update": steps_to_run,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "market_gradient_steps": market_gradient_steps,
            "market_gradient_steps_this_update": market_steps_to_run,
            "market_mean_loss": float(np.mean(market_losses)) if market_losses else None,
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


# The manifest names `train.py`, which constructs these itself. Nothing outside
# this agent resolves `training.updater` by path any more.
