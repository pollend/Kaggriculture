"""Roll out episodes and record the raw order decisions as compact training data.

Transitions cannot be reconstructed from `env.steps` after the fact -- the
replay stores filled orders, not the action set the policy chose from -- so
the acting agent records its own decisions and this module owns that recorder.

Records here are far smaller than the scorer's. Its candidate set is rebuilt per
decision from live shed contents, so it has to store the whole encoded grid per
decision. This agent's action is one index into a fixed list, so a decision is
fully described by the state vector and a single integer. That is what makes a
per-turn decision rate affordable: 720 decisions an episode costs less memory
than the scorer's 60.

The reward stream and its scales live in `training/reward.py`. They started as
the scorer's and are no longer: this agent decides land and cash reserves itself,
so the scales had to differ from the ones the scorer's market layer handled in
code.
"""

from __future__ import annotations

import traceback
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import numpy as np

from agents.dqn_market_free.diagnostics import OrderDiagnostics, diagnose_order_lists
from agents.dqn_market_free.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free.order_space import ACTION_COUNT, ACTIONS, OrderDecision
from agents.dqn_market_free import lots
from agents.dqn_market_free.training.reward import OPPONENT_WEIGHT, episode_rewards
from sim.workers import (
    MAP_TIMEOUT_SECONDS,
    discard_pool,
    single_threaded_workers,
    worker_pool,
)

if TYPE_CHECKING:
    from sim.runner import MatchResult

#: Fallback only, for the land potential's day when an observation carries none.
#: The live value comes off `configuration.turnsPerDay` in `main.py`; this is the
#: simulator's default and matches `donor.TOTAL_DAYS * 24 == 720` episode steps.
_TURNS_PER_DAY = 24


@dataclass
class EpisodeRecord:
    """One episode: the per-turn reward stream and the per-turn order decisions."""

    reward: np.ndarray  # float32 [T]    shaped, ready for training
    raw_reward: np.ndarray  # float32 [T]    unshaped money delta, for diagnostics
    money: np.ndarray  # float32 [T+1]
    order_state: np.ndarray  # float32 [D, STATE_DIM]
    order_chosen: np.ndarray  # int16   [D]  index into ACTIONS
    order_step: np.ndarray  # int16   [D]  index into `reward` each decision is paid from
    #: bool [D, ACTION_COUNT]. Which actions the simulator would have accepted at
    #: each decision. Stored rather than recomputed from `order_state`: every
    #: input is technically recoverable from the normalised features, but that
    #: would tie the mask to the exact scaling of six columns in `market_state`,
    #: which is source-frozen against the scorer -- and a drift there would
    #: corrupt the Double-DQN target silently. It compresses to almost nothing
    #: beside `order_state`.
    order_mask: np.ndarray

    @property
    def steps(self) -> int:
        return self.reward.shape[0]

    @property
    def decisions(self) -> int:
        return self.order_step.shape[0]

    @property
    def orders_placed(self) -> int:
        """Trading decisions across the episode. The headline health diagnostic.

        An unscaffolded market policy has two degenerate attractors and this
        number separates them: zero means it collapsed to NOOP, and something
        near `decisions` means it is ordering every single turn and is almost
        certainly bankrupt.

        Counted in decisions rather than wire orders, so a `HIRE 8` counts once.
        """

        return int(np.count_nonzero(self.order_chosen))

    @property
    def order_diagnostics(self) -> OrderDiagnostics:
        """Decision and order counts for the orders actually emitted."""

        return diagnose_order_lists(
            [ACTIONS[int(choice)].to_wire() for choice in self.order_chosen],
            [float(self.raw_reward[int(step)]) for step in self.order_step],
        )

    def nbytes(self) -> int:
        return sum(getattr(self, name).nbytes for name in EPISODE_ARRAY_FIELDS)


EPISODE_ARRAY_FIELDS = (
    "reward",
    "raw_reward",
    "money",
    "order_state",
    "order_chosen",
    "order_step",
    "order_mask",
)


def save_episode_batch(records: list[EpisodeRecord], path: str | Path) -> Path:
    """Persist variable-width episode records without pickle/object arrays."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "episode_count": np.asarray([len(records)], dtype=np.int32),
    }
    for index, record in enumerate(records):
        for name in EPISODE_ARRAY_FIELDS:
            payload[f"e{index}_{name}"] = getattr(record, name)
    np.savez_compressed(destination, **payload)
    return destination


def load_episode_batch(path: str | Path) -> list[EpisodeRecord]:
    """Read a rollout artifact written by this run.

    Rollout artifacts do not outlive the session that wrote them, so there is no
    format negotiation here -- only a width check, so an artifact left over from
    a differently-shaped state encoder fails on the spot rather than training
    against misaligned columns.
    """

    with np.load(Path(path), allow_pickle=False) as payload:
        count = int(payload["episode_count"][0])
        records = []
        for index in range(count):
            try:
                values = {
                    name: payload[f"e{index}_{name}"].copy() for name in EPISODE_ARRAY_FIELDS
                }
            except KeyError as missing:
                raise ValueError(
                    f"{path}: episode {index} is missing {missing}, so it predates a "
                    "schema change. Rollout artifacts do not outlive the session that "
                    "wrote them -- start a new training instance rather than reading it."
                ) from missing
            width = values["order_state"].shape[-1]
            if width != STATE_DIM:
                raise ValueError(
                    f"{path}: episode {index} has a {width}-wide state, expected {STATE_DIM}"
                )
            actions = values["order_mask"].shape[-1]
            if actions != ACTION_COUNT:
                raise ValueError(
                    f"{path}: episode {index} has a {actions}-wide action mask, "
                    f"expected {ACTION_COUNT}"
                )
            records.append(EpisodeRecord(**values))
        return records


@dataclass
class Recorder:
    """Accumulates one episode of decisions as the agent makes them."""

    money: list[float] = field(default_factory=list)
    # The land potential's two inputs, indexed exactly like `money` because the
    # reward stream is built from that list and `order_step` is a compacted index
    # into it. Kept as parallel lists rather than folded into one tuple list so a
    # trace that goes missing fails at `episode_rewards`' length check rather
    # than silently pairing a quadrant count with the wrong day.
    quadrants: list[int] = field(default_factory=list)
    days: list[int] = field(default_factory=list)
    # The lot potential's three inputs, indexed the same way again. Counts rather
    # than dicts, over `lots.CROP_ITEMS` and `lots.STOCK_ITEMS`, so an episode
    # carries 720 x 19 small integers instead of 720 dictionaries.
    #
    # `standing` is the one that is not in `private`: a seed leaving the seed bag
    # for a tile has not stopped being committed cost, and without the tile count
    # `PHI` would fall on the turn the agent plants -- charging for the action the
    # term exists to make cheaper.
    seeds: list[list[int]] = field(default_factory=list)
    standing: list[list[int]] = field(default_factory=list)
    stock: list[list[int]] = field(default_factory=list)
    order_steps: list[int] = field(default_factory=list)
    decisions: list[OrderDecision] = field(default_factory=list)

    def reset(self) -> None:
        self.money.clear()
        self.quadrants.clear()
        self.days.clear()
        self.seeds.clear()
        self.standing.clear()
        self.stock.clear()
        self.order_steps.clear()
        self.decisions.clear()

    def record_decision(self, *, decision: OrderDecision) -> None:
        """Store a decision against the reward index it will be paid from.

        The simulator's turn number is deliberately not what lands in
        `order_step`. The reward stream is built from `self.money`, which is
        appended by `record_turn` -- and the agent skips both on a malformed
        observation, so a turn the simulator counted may contribute no reward at
        all. Indexing by simulator step then slides every later decision's n-step
        window forward, silently, because `replay.sample` clamps the window with
        `min(end_step, record.steps)`. `clone.py` records the same compacted
        index for the same reason; the two record kinds share one buffer and
        must mean the same thing by it.

        `record_decision` runs before `record_turn` in `main.py`, so
        `len(self.money)` is exactly the slot this turn's reward will take.
        """

        self.order_steps.append(len(self.money))
        self.decisions.append(decision)

    def record_turn(self, *, money: float, obs: Any, farm: dict, private: dict) -> None:
        self.money.append(float(money))
        # Both already on hand and previously discarded. `unlocked_quadrants`
        # defaults to what a fresh farm carries, so a malformed observation
        # contributes no land potential rather than a spurious one, and the day
        # falls back to this turn's own index the way `market_state` does -- a
        # missing day must not read as "the whole season is still ahead".
        self.quadrants.append(len(farm.get("unlocked_quadrants", []) or ["NW"]))
        held_seeds, held_stock = lots.holdings(private)
        self.seeds.append(held_seeds)
        self.standing.append(lots.standing_plants(farm))
        self.stock.append(held_stock)
        turn = len(self.money) - 1
        self.days.append(int(obs.get("day", turn // _TURNS_PER_DAY) or 0))

    def finish(
        self,
        final_money: float,
        opponent_money: float,
        gamma: float,
        opponent_weight: float = OPPONENT_WEIGHT,
    ) -> EpisodeRecord:
        steps = len(self.money)
        if steps == 0:
            raise ValueError("recorder captured no steps")
        if not self.decisions:
            raise ValueError("recorder captured no order decisions")

        # Which turns put an order on the wire, indexed the same way the reward
        # stream is. A turn that emitted one and moved no money had it refused.
        emitted = [False] * len(self.money)
        for step, decision in zip(self.order_steps, self.decisions):
            if 0 <= step < len(emitted) and decision.chosen:
                emitted[step] = True

        shaped, raw, money = episode_rewards(
            self.money,
            final_money,
            opponent_money,
            gamma,
            emitted,
            opponent_weight=opponent_weight,
            quadrants=self.quadrants,
            days=self.days,
            seeds=self.seeds,
            standing=self.standing,
            stock=self.stock,
        )

        count = len(self.decisions)
        order_state = np.zeros((count, STATE_DIM), dtype=np.float32)
        order_chosen = np.zeros(count, dtype=np.int16)
        order_mask = np.ones((count, ACTION_COUNT), dtype=np.bool_)
        for index, decision in enumerate(self.decisions):
            order_state[index] = decision.state
            order_chosen[index] = decision.chosen
            if decision.mask is not None:
                order_mask[index] = decision.mask

        return EpisodeRecord(
            reward=shaped.astype(np.float32),
            raw_reward=raw.astype(np.float32),
            money=money,
            order_state=order_state,
            order_chosen=order_chosen,
            order_step=np.asarray(self.order_steps, dtype=np.int16),
            order_mask=order_mask,
        )


#: Spec target of the training-only wrapper in `match_agent.py`.
LEARNER_ENTRY_POINT = "agents.dqn_market_free.match_agent:make_agent"

#: Spec target of this agent's ordinary inference entry point.
#:
#: Here rather than in `training/updater`, where it used to live, because the
#: mirror gate on the opponent-relative reward leg has to read it and `updater`
#: already imports this module. Re-exported there under the same name, so nothing
#: that referred to `updater.SELF_SPEC` has to move.
SELF_SPEC = "agents/dqn_market_free/main.py"


@dataclass(frozen=True)
class OrderRollout:
    """One recorded rollout episode plus the match the league will store."""

    record: EpisodeRecord
    opponent_spec: str
    learner_seat: int
    match: MatchResult
    seed: int
    decisions: int
    #: Index of the seed pair this episode belongs to. Both orientations of one
    #: seed share it, which is what lets the league settle a seat-balanced
    #: margin for the pairing rather than two unrelated one-sided results.
    pair_index: int = 0


def learner_spec(
    weights_path: Path | str,
    *,
    gamma: float,
    exploration_seed: int,
    trace_path: Path | str | None = None,
    policy_revision_id: str | None = None,
    noise_cadence: str = "none",
    sigma_scale: float = 1.0,
    rare_sigma_gain: float = 0.0,
    rare_sigma_cap: float = 0.0,
    rare_sigma_novelty: str = "count",
    rare_sigma_path: Path | str | None = None,
    opponent_weight: float = 0.0,
) -> str:
    """Address the exploring learner as an ordinary agent spec.

    The per-episode trace path and revision id are deliberately separable: the
    spec without them is the *policy identity*, and folding a per-episode trace
    into it would mint one agent revision per episode.

    `gamma` sits in the identity spec although the policy never reads it -- only
    `Recorder.finish` does -- because a run that relabels its rewards is worth
    telling apart from one that does not.

    `opponent_weight` is the other reward-labelling parameter and is treated the
    *opposite* way, for the reason that separates the two: gamma is one number
    for a whole run, and this one is per-opponent, since the mirror is gated off.
    Folding a per-episode value into the identity would mint two revisions for
    one set of weights and split both `rating_snapshots` and the self-play pool
    down the middle. So it travels on the played spec beside `trace_path`, and
    `_learner_policy` in `training/updater` -- which is what the identity is
    actually built from -- does not carry it.
    """

    query: dict[str, Any] = {
        "weights": str(weights_path),
        "gamma": f"{gamma:.6f}",
        "exploration_seed": str(int(exploration_seed)),
    }
    # Emitted only on departure from the default, for the reason spelled out
    # above: a run with NoisyNet off leaves every spec string byte-identical to
    # what this agent recorded before it existed, so no revision churns.
    if noise_cadence != "none":
        query["noise_cadence"] = str(noise_cadence)
    if sigma_scale != 1.0:
        query["sigma_scale"] = f"{sigma_scale:.6f}"
    # Same rule, and it matters more here: the counts file is content-addressed
    # per checkpoint, so emitting these unconditionally would mint a fresh policy
    # identity every round even for a run that leaves the widening off.
    if rare_sigma_gain > 0.0 and rare_sigma_path is not None:
        query["rare_sigma_gain"] = f"{rare_sigma_gain:.6f}"
        query["rare_sigma_cap"] = f"{rare_sigma_cap:.6f}"
        query["rare_sigma_path"] = str(rare_sigma_path)
        # Same departure-from-the-default rule as the keys above: `"count"` is
        # what every spec before this key meant, so a run on that arm still emits
        # byte-identical spec strings and churns no revisions.
        if rare_sigma_novelty != "count":
            query["rare_sigma_novelty"] = str(rare_sigma_novelty)
    # Same "departure from the default" rule as the noise knobs above, and it
    # matters for the same reason plus one more: at `OPPONENT_WEIGHT = 0.0` this
    # is 0.0 on every episode, so a run with the relative leg switched off emits
    # spec strings byte-identical to what this agent recorded before the
    # parameter existed and churns no revisions at all.
    if opponent_weight:
        query["opponent_weight"] = f"{opponent_weight:.6f}"
    if trace_path is not None:
        query["trace_path"] = str(trace_path)
    if policy_revision_id is not None:
        query["policy_revision_id"] = str(policy_revision_id)
    return f"{LEARNER_ENTRY_POINT}?{urlencode(query)}"


def collect_episode(
    weights_path: Path,
    opponent_spec: str,
    seed: int,
    learner_seat: int,
    gamma: float,
    trace_dir: Path,
    exploration_seed: int,
    pair_index: int = 0,
    policy_revision_id: str | None = None,
    config: dict[str, Any] | None = None,
    noise_cadence: str = "none",
    sigma_scale: float = 1.0,
    rare_sigma_gain: float = 0.0,
    rare_sigma_cap: float = 0.0,
    rare_sigma_novelty: str = "count",
    rare_sigma_path: Path | str | None = None,
) -> OrderRollout | None:
    """Play one episode with an exploring learner and record it."""

    from kaggriculture_lab.episode_metrics import episode_metrics
    from sim.runner import OK_STATUSES, run_match

    trace_path = Path(trace_dir) / f"seed-{seed:010d}-seat-{learner_seat}.npz"
    # The mirror gate on the reward's opponent-relative leg. This is the only
    # place that holds both the opponent's spec and a route to `Recorder.finish`,
    # which runs in the match subprocess and cannot see who it played.
    #
    # A mirror is anything addressed at this agent's own inference entry point --
    # `_weights_spec` in `training/updater` builds every self-play opponent as
    # `SELF_SPEC?weights=...`, so the prefix is the test. Against a recent
    # checkpoint of the same slot the relative leg is noise by construction: the
    # opponent's purse carries this policy's own variance and nothing else.
    opponent_weight = 0.0 if opponent_spec.startswith(SELF_SPEC) else OPPONENT_WEIGHT
    spec = learner_spec(
        weights_path,
        gamma=gamma,
        exploration_seed=exploration_seed,
        opponent_weight=opponent_weight,
        trace_path=trace_path,
        policy_revision_id=policy_revision_id,
        noise_cadence=noise_cadence,
        sigma_scale=sigma_scale,
        rare_sigma_gain=rare_sigma_gain,
        rare_sigma_cap=rare_sigma_cap,
        rare_sigma_novelty=rare_sigma_novelty,
        rare_sigma_path=rare_sigma_path,
    )
    seats = (spec, opponent_spec) if learner_seat == 0 else (opponent_spec, spec)
    result = run_match(seats, seed, config=config, analyzer=episode_metrics)
    # The learner's own seat is what decides whether this episode is trainable.
    # `result.failed` is a match-level flag and would also discard episodes the
    # opponent lost by erroring, which are perfectly good training data.
    if result.seats[learner_seat].status not in OK_STATUSES or not trace_path.is_file():
        return None
    records = load_episode_batch(trace_path)
    if not records:
        return None
    record = records[0]
    return OrderRollout(
        record=record,
        opponent_spec=opponent_spec,
        learner_seat=learner_seat,
        match=result,
        seed=seed,
        decisions=record.decisions,
        pair_index=pair_index,
    )


def _job(args: tuple) -> OrderRollout | None:
    try:
        return collect_episode(*args)
    except Exception:
        # Swallowed so one bad seed cannot take down a whole batch, but never
        # silently: a worker that dies without a trace turns "collection
        # produced nothing" into an unfalsifiable guess. Worker stderr flows to
        # the trainer's stderr, which is where a round hook's failure is read.
        traceback.print_exc()
        return None


#: The BLAS pin and the rollout pool both moved to `sim.workers`, because the
#: promotion gate's arena pays exactly the same two costs on exactly the same
#: episodes. Re-exported under the old names: `training.updater` imports the pin
#: from here to wrap `play_series`.
_single_threaded_workers = single_threaded_workers


def actor_sigma_scales(actors: int, alpha: float = 32.0) -> list[float]:
    """One NoisyNet sigma multiplier per actor, spread geometrically.

    One temperature for a whole batch fills the buffer with near-copies of one
    trajectory. Measured on the epsilon spread this replaced: 81% mean pairwise
    action agreement between the sixteen episodes of a batch, and on 38% of turns
    all sixteen chose the identical action. The finding is about behavioural
    diversity within a batch and is orthogonal to the exploration *mechanism*, so
    a single sigma reproduces it exactly.

    Actor `i` of `N` runs at `alpha ** (-i/(N-1))`: actor 0 plays the learned
    sigma and the last actor is effectively greedy, so `trading_rate` and the
    money diagnostics always have a clean read of what the policy actually does.
    At alpha 32 and eight pairs that is 1.0 down to 0.031.

    This is Ape-X's per-actor temperature trick transplanted, *not* principled
    NoisyNet -- the paper has one sigma, learned, and no spread at all.

    The spread is downward only, and the two mechanisms that tried to put rungs
    *above* the setpoint are both retired: the periodic idle burst at v15, and
    `noisy_actor_hot` here. Per-episode attribution across `burst-v9` and
    `falloff-v18` says the same thing twice -- every land purchase in the run
    happened at sigma <= 0.871, 49% of seats went bankrupt at 3.159, and the hot
    rungs' land rate (4.6%) was indistinguishable from the ordinary rungs' (4.9%).
    Widening the whole policy does not reach rare behaviour, it bankrupts the
    wallet before the behaviour becomes affordable. `rare.sigma_scale_vector`
    widens single rows instead, which is a different mechanism and stays.
    """

    if actors <= 1:
        return [1.0]
    return [alpha ** (-(index / (actors - 1))) for index in range(actors)]


def collect_batch(
    weights_path: Path,
    opponents: list[str],
    seeds: list[int],
    gamma: float,
    trace_dir: Path,
    jobs: int,
    exploration_seed: int = 0,
    policy_revision_id: str | None = None,
    config: dict[str, Any] | None = None,
    noise_cadence: str = "none",
    sigma_scales: list[float] | None = None,
    rare_sigma_gain: float = 0.0,
    rare_sigma_cap: float = 0.0,
    rare_sigma_novelty: str = "count",
    rare_sigma_path: Path | str | None = None,
    rare_sigma_pairs: int = 0,
) -> list[OrderRollout]:
    """Play `len(seeds)` episodes, one opponent drawn per consecutive seed pair.

    Seeds arrive in consecutive pairs -- the same seed twice -- so the learner
    takes seat 0 in one orientation and seat 1 in the other. Drawing one
    opponent per pair rather than per episode is what lets the resulting pairing
    settle a seat-balanced margin.

    `sigma_scales` is indexed by pair for the same reason the opponent is: both
    orientations of a seed must be played by the same policy at the same
    temperature, or the paired margin the league settles is a comparison between
    two different agents.

    `rare_sigma_gain` goes to the first `rare_sigma_pairs` pairs -- the widest
    rungs of the ladder -- at full gain rather than one scaled by the rung's own
    sigma.

    The widest end because that is already the arm whose job is finding things,
    and because the near-greedy tail has to stay a clean read of the policy: the
    money series is only comparable across this change if some pairs are
    collecting what v18 collected.

    Worth being explicit about one piece of evidence that looks like it argues the
    other way. `market-free-burst-v9` attributed every episode to its rung across
    212 seats and found land bought only at sigma <= 0.871, never at 3.159, where
    49% went bankrupt. That measured *global* sigma -- every row widened at once,
    which is a policy shaken into noise, and it says nothing about widening one
    action's row while the other 185 stay where the controller put them. The
    failure it records is bankruptcy from indiscriminate trading, and this cannot
    produce that: `rare.sigma_scale_vector` leaves NOOP and every well-sampled row
    at exactly 1.0.
    """

    scales = sigma_scales or [1.0]
    payloads = []
    for index, seed in enumerate(seeds):
        pair_index = index // 2
        opponent = opponents[pair_index % len(opponents)]
        scale = scales[pair_index % len(scales)]
        pair_gain = rare_sigma_gain if pair_index < max(0, rare_sigma_pairs) else 0.0
        payloads.append(
            (
                weights_path,
                opponent,
                seed,
                index % 2,
                gamma,
                trace_dir,
                exploration_seed + index,
                pair_index,
                policy_revision_id,
                config,
                noise_cadence,
                scale,
                pair_gain,
                rare_sigma_cap,
                rare_sigma_novelty,
                rare_sigma_path,
            )
        )
    if jobs <= 1:
        return [rollout for rollout in (_job(payload) for payload in payloads) if rollout]
    with _single_threaded_workers(), worker_pool(jobs) as pool:
        try:
            return [
                rollout
                for rollout in pool.map(_job, payloads, timeout=MAP_TIMEOUT_SECONDS)
                if rollout
            ]
        except TimeoutError:
            # Same reason as the arena's: a stalled pool that goes back on the
            # free-list hands the next caller the same wedged workers.
            discard_pool(pool)
            raise
        except BrokenProcessPool:
            # A worker died hard (OOM kill, segfault in a BLAS backend). The
            # pool is unusable for good, so drop it rather than returning it to
            # the free-list, and replay the batch once on a fresh one. A second
            # failure is the caller's to see.
            discard_pool(pool)
            with worker_pool(jobs) as retry:
                return [
                    rollout
                    for rollout in retry.map(_job, payloads, timeout=MAP_TIMEOUT_SECONDS)
                    if rollout
                ]
