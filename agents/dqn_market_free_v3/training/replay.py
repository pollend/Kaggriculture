"""Episode ring buffer with a fixed demonstration share per minibatch.

A sample is one turn-level decision `(episode, decision)`. Unlike the scorer's
day-scale market MDP this one is turn-scale: a raw order is filled at this
turn's quote and is gone, so the reward window between consecutive decisions is
a single turn and the n-step horizon counts turns directly.

The problem this buffer solves is that solvent trajectories are rare: an
unweighted draw over self-collected data is almost entirely episodes that ended
near zero, and the only policy consistent with that is to do nothing. Two
mechanisms cover different halves of it:

* `demo_fraction` draws a fixed share of every minibatch from the pinned
  demonstrations, which *are* solvent by construction -- but only while something
  is pinned, and on the live run nothing is, so the share silently resolves to
  zero.
* `priority_alpha` ranks the agent's *own* episodes by shaped return, with the
  importance-sampling correction prioritisation obliges. Self-imitation: the best
  episodes the policy has produced are the only positive signal when there is no
  demonstrator.

The two compose -- demonstrations stay uniform inside their share, own episodes
are ranked inside theirs. The exponents are fixed and never annealed; see
`_own_priorities`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.dqn_market_free_v3.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free_v3.order_space import ACTION_COUNT
from agents.dqn_market_free_v3.policy import HIDDEN_SIZE as GRU_HIDDEN
from agents.dqn_market_free_v3.training.collect import HIDDEN_STRIDE, EpisodeRecord

#: Buffer defaults, named so the updater can fall back to them instead of
#: restating the numbers. Restated they drifted: the updater still read
#: `n_step=8` after the manifest moved to 24.
DEFAULT_CAPACITY = 400
#: One in-game day. The credit this carries is a crop cycle -- 48 to 96 turns
#: from BUY_SEED to the harvest that pays for it.
DEFAULT_N_STEP = 24
DEFAULT_RESERVOIR_FRACTION = 0.35
#: Rank exponent for the return-weighted draw over own episodes. 0.0 is the uniform
#: draw, byte for byte down to the bit-generator state, so a caller that does not
#: ask for prioritisation keeps exactly what it had. The live value is in
#: `agent.toml` rather than in the runner, whose overrides `ensure_session` digests.
DEFAULT_PRIORITY_ALPHA = 0.0
#: Importance-sampling exponent. Fixed, never annealed -- see `_own_priorities`.
DEFAULT_PRIORITY_BETA = 0.5


#: Turns of no-grad rollout ahead of the training window, and turns the gradient
#: flows over. Defaults only -- `agent.toml` is the authority and the updater
#: passes both explicitly.
DEFAULT_BURN_IN = 20
DEFAULT_TRAIN = 40


def _nstep_tables(
    record: EpisodeRecord, gamma: float, n_step: int
) -> tuple[np.ndarray, np.ndarray]:
    """`(discounted n-step return, bootstrap discount)`, both `[T]`, in O(T).

    Vectorised rather than looped, and that is a requirement rather than a
    flourish. v2 summed the reward window inside a per-row Python loop: 24 steps
    times 256 rows is ~6k iterations a minibatch. A chunk batch is 32 chunks x 40
    turns x 24 steps = ~31k iterations *per gradient step*, ~15M an update, and the
    sampler would cost more than the network.

    The discount is `gamma ** n_step` where a full window fits and 0.0 where the
    episode ends inside it -- v2's terminal convention, which pairs with
    `boot_mask` staying all-True on those rows so the masked max cannot produce
    `-inf * 0 = nan`.
    """

    rewards = record.reward
    turns = rewards.shape[0]
    if turns == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty
    weights = (gamma ** np.arange(n_step, dtype=np.float64)).astype(np.float64)
    padded = np.concatenate([rewards.astype(np.float64), np.zeros(n_step)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, n_step)[:turns]
    returns = (windows * weights).sum(axis=1).astype(np.float32)
    remaining = turns - np.arange(turns)
    discount = np.where(
        remaining > n_step, np.float32(gamma**n_step), np.float32(0.0)
    ).astype(np.float32)
    return returns, discount


def _turn_first(record: EpisodeRecord) -> np.ndarray:
    """`[T + 1]`: the first decision index at or after each turn.

    One `searchsorted` over the whole turn range rather than one per sampled row.
    `order_step` is non-decreasing, which is what makes this well defined; the
    extra trailing entry is the total decision count, so `first[t]:first[t + 1]`
    slices turn `t`'s decisions with no bounds test.
    """

    turns = record.steps
    return np.searchsorted(
        record.order_step, np.arange(turns + 1, dtype=np.int64), side="left"
    ).astype(np.int64)


@dataclass
class SequenceBatch:
    """`chunks` contiguous windows of turns, with their decisions flattened out.

    Two granularities, because the episode has two. Turns are dense: every chunk
    is the same length, so the recurrence is one batched rollout. Decisions are
    ragged -- one to eleven per turn -- and padding them to eleven would waste
    roughly seventy percent of the head's forward on rows masked out of the loss.
    They are flattened instead, with `decision_turn` gathering each row's hidden
    state and its goal out of the `[chunks, train]` grid.

    The only padding here is over turns, at the two ends of an episode, and
    `turn_valid` marks it. It gates the recurrence rather than zeroing the input:
    see `training/network.OrderQModule.recur` for why those are not the same thing.
    """

    #: float32 [B, burn_in + train + n_step, STATE_DIM]. The GRU's input across the
    #: whole chunk: the stale-state refresh, the graded window, and the tail the
    #: bootstraps live in.
    turn_input: np.ndarray
    #: float32 [B, H]. The stored hidden state at the chunk's first turn, as the
    #: acting policy produced it, and stale by however long the episode has sat in
    #: the buffer. Exactly zero where the chunk starts at turn 0.
    hidden0: np.ndarray
    #: bool [B, Lfull]. False on the left pad and past the episode's end.
    turn_valid: np.ndarray
    #: float32 [B]. The IS correction for the *episode* draw, applied once per
    #: chunk at the outermost mean.
    weight: np.ndarray

    #: float32 [N]. The reward this decision is credited with. **Zero on every
    #: sub-decision but a turn's last**, because no game time passes between the
    #: sub-decisions of one turn -- the whole turn's reward is paid once, at the
    #: boundary, to the decision that closed it.
    #:
    #: Per decision rather than per turn, and that is the correction this fork
    #: exists for. v2 gave every sub-decision of a turn the same reward, the same
    #: discount and the same bootstrap, so ten transitions were regressed onto one
    #: scalar and nothing in the target could say the seventh order was worse than
    #: the first. Its policy converged on a ten-slot zero-P&L wash for exactly that
    #: reason. See `sample_sequences`.
    reward: np.ndarray
    #: float32 [N]. `1.0` intra-turn -- no time elapses between sub-decisions --
    #: `gamma ** n_step` at a turn boundary, and `0.0` where the episode ended
    #: inside the horizon.
    discount: np.ndarray
    #: float32 [N, STATE_DIM]. The successor decision's own state: the next
    #: sub-decision of the same turn, or the first decision of the turn `n_step`
    #: ahead when this row closed its turn.
    next_state: np.ndarray
    #: bool [N, ACTION_COUNT]. All-True where the episode ended, which is v2's
    #: convention and load-bearing: masking a terminal successor to -inf and then
    #: multiplying by a zero discount yields nan, and one nan takes the whole batch
    #: down through the backward pass.
    next_mask: np.ndarray
    #: int64 [N]. Flat index into the `[B * (train + n_step)]` post-burn-in grid of
    #: the turn whose hidden state the successor is scored against. Equal to this
    #: row's own turn for an intra-turn successor -- the latent does not advance
    #: inside a turn -- and the turn `n_step` ahead at a boundary.
    next_turn: np.ndarray

    #: float32 [N, STATE_DIM], the training decisions themselves, flattened.
    state: np.ndarray
    #: int64 [N].
    chosen: np.ndarray
    #: bool [N, ACTION_COUNT].
    mask: np.ndarray
    #: int64 [N]. Flat index into `[B * train]`: which turn each row belongs to.
    decision_turn: np.ndarray
    #: float32 [N]. `1 / decisions on this row's turn`, the inner half of the
    #: two-level mean. See `chunk_loss_normalisation` in `agent.toml`.
    turn_share: np.ndarray
    #: How many training turns had their bootstrap clamped for want of a decision
    #: at the horizon. Must be zero; see `sample_sequences`.
    clamped: int = 0
    #: Whether the own-episode half of this draw was return-ranked. False means the
    #: draw fell back to uniform with no IS correction, which `sample_sequences`
    #: does whenever any episode is too short to hold a training window. Reported
    #: rather than inferred: `priority_stats` is computed over the *unfiltered*
    #: pool, so it keeps describing a healthy prioritization that this draw did not
    #: use, and one short episode anywhere in the buffer is enough to trigger it.
    ranked: bool = True


@dataclass
class OrderBatch:
    """One training minibatch."""

    state: np.ndarray  # float32 [B, STATE_DIM]
    chosen: np.ndarray  # int64   [B]  index into ACTIONS
    next_state: np.ndarray  # float32 [B, STATE_DIM]  state at the bootstrap
    reward: np.ndarray  # float32 [B]  discounted turn rewards over the horizon
    discount: np.ndarray  # float32 [B]  gamma**turns, 0 if the episode ended first
    mask: np.ndarray  # bool [B, ACTION_COUNT]  feasible actions at `state`
    #: bool [B, ACTION_COUNT] at `next_state`. All-True on rows whose episode
    #: ended, where there is no next state: masking those to -inf and then
    #: multiplying by a zero discount yields nan, not zero, and one nan takes the
    #: whole batch down through the backward pass.
    next_mask: np.ndarray
    #: float32 [B]. Importance-sampling correction for the return-weighted draw,
    #: normalised so the largest is exactly 1.0. Every weight is therefore <= 1, so
    #: the reweighting can only shrink a gradient -- which matters because
    #: `clip_grad_norm_` clips the batch norm and cannot see one row inflating
    #: itself. All-ones at `priority_alpha = 0`.
    #:
    #: Demonstration rows are always exactly 1.0, because the correction undoes the
    #: departure from uniform *within the own pool* and the pinned pool is drawn
    #: uniformly. Side effect: demo rows carry more gradient mass than their row
    #: share, so a run that re-enables demonstrations gets a share that does not
    #: match its config.
    weight: np.ndarray


class ReplayBuffer:
    """Recent episodes, a uniform sample of everything older, and pinned demos.

    A plain FIFO ring turns over quickly, so the network would only ever train on
    data from roughly its current policy -- catastrophic forgetting at a high replay
    ratio. So the buffer is split: a FIFO segment for recent on-policy data, and a
    reservoir segment holding a uniform sample of everything ever evicted (Vitter's
    Algorithm R) at fixed memory cost.

    Demonstrations added with `pinned=True` sit outside both and are never evicted:
    they are the only solvent data the run has until its own policy becomes solvent.
    """

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        gamma: float = 0.999,
        n_step: int = DEFAULT_N_STEP,
        seed: int = 0,
        reservoir_fraction: float = DEFAULT_RESERVOIR_FRACTION,
        priority_alpha: float = DEFAULT_PRIORITY_ALPHA,
        priority_beta: float = DEFAULT_PRIORITY_BETA,
    ):
        if not 0.0 <= reservoir_fraction < 1.0:
            raise ValueError("reservoir_fraction must be in [0, 1)")
        if priority_alpha < 0.0:
            raise ValueError("priority_alpha must be non-negative")
        if not 0.0 <= priority_beta <= 1.0:
            raise ValueError("priority_beta must be in [0, 1]")
        self.capacity = capacity
        self.gamma = gamma
        self.n_step = n_step
        self.reservoir_capacity = int(capacity * reservoir_fraction)
        self.recent_capacity = capacity - self.reservoir_capacity
        self.recent: list[EpisodeRecord] = []
        self.reservoir: list[EpisodeRecord] = []
        self.pinned: list[EpisodeRecord] = []
        self.evicted_seen = 0
        self.rng = np.random.default_rng(seed)
        self.priority_alpha = float(priority_alpha)
        self.priority_beta = float(priority_beta)
        #: Memoised `(pool, returns, probabilities, weights)`, or None when the
        #: own-episode pool has changed since it was built. Explicitly a dirty
        #: flag rather than a length key: at steady state the FIFO length is
        #: constant and `_offer_to_reservoir` replaces a reservoir slot *in
        #: place*, so a length-keyed cache would never invalidate and the run
        #: would train on the first update's priorities forever, with no error.
        self._priorities: (
            tuple[list[EpisodeRecord], np.ndarray, np.ndarray, np.ndarray] | None
        ) = None

    def add(self, records: list[EpisodeRecord], *, pinned: bool = False) -> None:
        if pinned:
            # Deliberately no invalidation: pinned rows are never ranked.
            self.pinned.extend(records)
            return
        for record in records:
            self.recent.append(record)
            if len(self.recent) > self.recent_capacity:
                self._offer_to_reservoir(self.recent.pop(0))
        self._priorities = None

    def plan_intake(self, counts: list[int]) -> tuple[list[int], list[int], int]:
        """Decide which episodes survive an intake before any are read from disk.

        Returns `(recent_ids, reservoir_ids, evicted_seen)` over a flat episode
        numbering: batch `b` owns ids `sum(counts[:b]) .. +counts[b]`.

        Neither decision needs the episode itself -- the FIFO segment keeps ids by
        position and Algorithm R draws without looking at the record -- so the
        caller can run this over `sample_count` and then load only the batches that
        still own a surviving id, instead of decompressing the whole run history.

        Draws are taken in the same order as the record-by-record path, so the
        resulting buffer is identical to the one `add` would have built.
        """

        recent: list[int] = []
        reservoir: list[int] = []
        evicted_seen = 0
        for episode_id in range(sum(counts)):
            recent.append(episode_id)
            if len(recent) <= self.recent_capacity:
                continue
            evicted = recent.pop(0)
            if self.reservoir_capacity == 0:
                continue
            evicted_seen += 1
            if len(reservoir) < self.reservoir_capacity:
                reservoir.append(evicted)
                continue
            slot = int(self.rng.integers(0, evicted_seen))
            if slot < self.reservoir_capacity:
                reservoir[slot] = evicted
        return recent, reservoir, evicted_seen

    def load_planned(
        self,
        recent_ids: list[int],
        reservoir_ids: list[int],
        evicted_seen: int,
        records: dict[int, EpisodeRecord],
    ) -> None:
        """Install the outcome of `plan_intake`, skipping ids that failed to load.

        A batch whose artifact is unreadable in the current format drops out
        here rather than shifting every later episode's identity.
        """

        self.recent = [records[key] for key in recent_ids if key in records]
        self.reservoir = [records[key] for key in reservoir_ids if key in records]
        self.evicted_seen = evicted_seen
        self._priorities = None

    def _offer_to_reservoir(self, record: EpisodeRecord) -> None:
        """Algorithm R over the eviction stream: a uniform sample of all history."""

        if self.reservoir_capacity == 0:
            return
        # Before the draws, not after: this replaces a slot in place, so the
        # lengths the cache could key on do not move even though the pool did.
        self._priorities = None
        self.evicted_seen += 1
        if len(self.reservoir) < self.reservoir_capacity:
            self.reservoir.append(record)
            return
        index = int(self.rng.integers(0, self.evicted_seen))
        if index < self.reservoir_capacity:
            self.reservoir[index] = record

    @property
    def episodes(self) -> list[EpisodeRecord]:
        return self.recent + self.reservoir + self.pinned

    @property
    def transitions(self) -> int:
        return sum(record.decisions for record in self.episodes)

    def nbytes(self) -> int:
        return sum(record.nbytes() for record in self.episodes)

    @staticmethod
    def _sampleable(records: list[EpisodeRecord]) -> list[EpisodeRecord]:
        return [record for record in records if record.decisions]

    @staticmethod
    def _sampleable_sequences(
        records: list[EpisodeRecord], train: int
    ) -> list[EpisodeRecord]:
        """Episodes long enough to hold one training window.

        `record.decisions` -- the flat sampler's predicate -- is the wrong test for
        a chunk draw: an episode with plenty of decisions packed into fewer than
        `train` turns would pass it and then index off the end of `turn_state`.
        """

        return [record for record in records if record.steps >= train]

    def _own_priorities(
        self,
    ) -> tuple[list[EpisodeRecord], np.ndarray, np.ndarray, np.ndarray]:
        """The own-episode pool, its returns, its sampling law, and the IS weights.

        Self-imitation: drawing the agent's own best episodes more often is the only
        positive signal available with no demonstrator pinned.

        Four properties worth keeping:

        * **Rank-based, not proportional**, because the shaped return is *signed* --
          `return ** alpha` is undefined on most of the buffer. Shifting by the
          minimum makes one episode's priority depend on the worst episode currently
          held; clamping at zero deletes the reservoir segment, which is the
          anti-forgetting mechanism. A rank is invariant to all of it.
        * **Static priorities.** A return is fixed when `Recorder.finish` writes it
          and is never a TD error, so there is nothing to update in place and no
          sum-tree: the whole distribution is one vector rebuilt when the pool moves.
        * **Episode weighting is transition weighting here**, because the horizon is
          fixed -- nothing terminates early on bankruptcy -- so equal episodes means
          equal turns. Note *turns*, not rows: v2 could say rows because it drew a
          decision uniformly within an episode, so a chatty episode's extra
          sub-decisions bought it no extra draws. v3 draws a chunk of turns, and a
          turn holds one to eleven rows, so the row count is back in the draw.
          `sample_sequences` cancels it in the loss rather than here -- see
          `chunk_loss_normalisation`.
        * **Beta is fixed, never annealed.** An anneal keys off `gradient_steps`,
          which a lineage reset snaps backwards while the buffer does not move, so
          the correction and the data go out of phase exactly when the run is
          already in trouble. It is also below 1.0 on purpose: at 1 the correction
          is exact and prioritisation buys nothing but variance.
        """

        if self._priorities is not None:
            return self._priorities
        pool = self._sampleable(self.recent + self.reservoir)
        count = len(pool)
        returns = np.asarray(
            [float(record.reward.sum()) for record in pool], dtype=np.float64
        )
        if count == 0 or self.priority_alpha <= 0.0:
            probabilities = np.full(count, 1.0 / count) if count else np.zeros(0)
            weights = np.ones(count, dtype=np.float32)
        else:
            # Stable, so equal returns rank by position in the pool and two runs
            # over the same buffer draw the same episodes. `argsort` defaults to
            # quicksort, which is not stable.
            order = np.argsort(-returns, kind="stable")
            rank = np.empty(count, dtype=np.float64)
            rank[order] = np.arange(1, count + 1, dtype=np.float64)
            priority = rank ** (-self.priority_alpha)
            # float64 throughout: `rng.choice` validates that `p` sums to 1, and
            # a float32 vector of 400 entries does not, reliably.
            probabilities = priority / priority.sum()
            weights = (1.0 / (count * probabilities)) ** self.priority_beta
            # Normalised by the largest, so every weight is <= 1 and the
            # correction can only ever shrink a gradient. Dividing by the mean
            # instead would let a rarely-drawn episode enlarge the effective
            # learning rate on the step it lands.
            weights = (weights / weights.max()).astype(np.float32)
        self._priorities = (pool, returns, probabilities, weights)
        return self._priorities

    def priority_stats(self) -> dict[str, float | None]:
        """What the return weighting is actually doing to the draw.

        `sampled_mean_return` is exact rather than empirical: it is the mean
        return under the sampling law, which is what the draw converges to over
        the couple of thousand minibatches an update takes. Equal to
        `own_mean_return` means `priority_alpha` is doing nothing, and that is
        the reading to watch for -- the mechanism this one supplements went inert
        without moving a single number anyone was looking at.
        """

        pool, returns, probabilities, weights = self._own_priorities()
        if not pool:
            return dict.fromkeys(
                (
                    "own_mean_return",
                    "sampled_mean_return",
                    "own_max_return",
                    "priority_top_decile_share",
                    "min_importance_weight",
                )
            )
        decile = max(1, len(pool) // 10)
        return {
            "own_mean_return": float(returns.mean()),
            "sampled_mean_return": float(probabilities @ returns),
            "own_max_return": float(returns.max()),
            # Share of the draw the best 10% of episodes take. 0.1 is uniform.
            "priority_top_decile_share": float(
                np.sort(probabilities)[::-1][:decile].sum()
            ),
            "min_importance_weight": float(weights.min()),
        }

    def sample(
        self,
        batch_size: int,
        demo_fraction: float = 0.0,
        *,
        prioritized: bool = True,
    ) -> OrderBatch:
        """Draw a minibatch, `demo_fraction` of it from the pinned demonstrations.

        The demonstration share is drawn uniformly; asking for it when nothing is
        pinned is not an error, the batch simply comes entirely from self-play. The
        own share is ranked by return when `priority_alpha` is positive, with the
        matching correction in `weight`.

        `prioritized=False` forces the uniform draw whatever the configuration. The
        flip-rate probe needs it: it measures the states the actor really visits, so
        a return-weighted draw would resolve the controller against a distribution
        no actor ever sees -- silently, since the probe reports a plausible number
        either way.
        """

        own, _returns, own_probabilities, own_weights = self._own_priorities()
        demos = self._sampleable(self.pinned)
        if not own and not demos:
            raise ValueError("free-market replay has no recorded decisions")

        wanted = int(round(batch_size * max(0.0, min(1.0, demo_fraction))))
        demo_rows = min(wanted, batch_size) if demos else 0
        own_rows = batch_size - demo_rows
        if not own:
            demo_rows, own_rows = batch_size, 0

        ranked = prioritized and self.priority_alpha > 0.0 and len(own) > 1
        if ranked:
            # One vectorised draw for the whole own share. Per-row `searchsorted`
            # against a cached CDF would keep the stream the same *shape* as the
            # uniform branch, but that buys nothing -- flipping alpha changes
            # which episode row 0 draws anyway -- and costs 256 Python-level RNG
            # calls per sample against one.
            own_picks = np.asarray(
                self.rng.choice(len(own), size=own_rows, p=own_probabilities)
            )
        else:
            # Byte-for-byte the draw this buffer took before return weighting
            # existed: `own_rows` separate scalar calls, in this order, ahead of
            # the demonstration draws. Not `integers(0, n, size=own_rows)`, which
            # is a different consumption of the same bit stream -- it happens to
            # agree for the bounds used here on the pinned numpy, but it is not
            # guaranteed across bounds or versions, and silently re-randomising
            # every resumed run is not worth the microseconds.
            own_picks = np.asarray(
                [int(self.rng.integers(0, len(own))) for _ in range(own_rows)],
                dtype=np.intp,
            )

        picks = [
            *((own, int(pick)) for pick in own_picks),
            *((demos, int(self.rng.integers(0, len(demos)))) for _ in range(demo_rows)),
        ]

        state = np.zeros((batch_size, STATE_DIM), dtype=np.float32)
        next_state = np.zeros((batch_size, STATE_DIM), dtype=np.float32)
        chosen = np.zeros(batch_size, dtype=np.int64)
        rewards = np.zeros(batch_size, dtype=np.float32)
        discounts = np.zeros(batch_size, dtype=np.float32)
        mask = np.ones((batch_size, ACTION_COUNT), dtype=np.bool_)
        next_mask = np.ones((batch_size, ACTION_COUNT), dtype=np.bool_)
        # A slice, not a per-row lookup: `picks` is own rows then demo rows, and
        # demo rows keep the 1.0 this was initialised to.
        #
        # Gated on `ranked`, not on `priority_alpha`. The weight has to describe
        # the draw that actually happened, and a `prioritized=False` caller took
        # the uniform one -- correcting it for a bias it does not carry would
        # reweight the probe's rows against a distribution it never sampled.
        weight = np.ones(batch_size, dtype=np.float32)
        if ranked and own_rows:
            weight[:own_rows] = own_weights[own_picks]

        for row, (pool, pick) in enumerate(picks):
            record = pool[pick]
            decision = int(self.rng.integers(0, record.decisions))
            state[row] = record.order_state[decision]
            chosen[row] = int(record.order_chosen[decision])
            mask[row] = record.order_mask[decision]

            # `n_step` is a number of *turns*, not of decisions.
            #
            # In v1 the two were the same thing -- one decision per turn -- so this
            # was `decision + n_step`. A v2 episode takes several decisions on a
            # trading turn and one on a quiet one, so counting decisions would make
            # the credit horizon a function of how chatty the policy currently is:
            # 24 decisions is 24 turns for a policy that has collapsed to STOP and
            # about 7 for one that trades, and the horizon would then *shrink* as the
            # policy learned to trade. `order_step` is non-decreasing, so the first
            # decision at or after the target turn is one search.
            start_step = int(record.order_step[decision])
            bootstrap = int(
                np.searchsorted(record.order_step, start_step + self.n_step, side="left")
            )
            bootstrap = min(bootstrap, record.decisions)
            # A turn that fills the wire can hold more decisions than `n_step` has
            # turns to spend, which would leave the bootstrap at or behind the row it
            # is bootstrapping from -- a self-referential target. The next decision is
            # the nearest honest one, and it costs the horizon rather than the row.
            bootstrap = max(bootstrap, decision + 1)
            end_step = (
                int(record.order_step[bootstrap])
                if bootstrap < record.decisions
                else record.steps
            )
            turn_discount = 1.0
            for step in range(start_step, min(end_step, record.steps)):
                rewards[row] += turn_discount * float(record.reward[step])
                turn_discount *= self.gamma
            if bootstrap < record.decisions:
                discounts[row] = turn_discount
                next_state[row] = record.order_state[bootstrap]
                next_mask[row] = record.order_mask[bootstrap]
            # else: the episode ended, discount stays 0 and `next_mask` stays
            # all-True so the masked max cannot produce -inf * 0 = nan.

        return OrderBatch(
            state=state,
            chosen=chosen,
            next_state=next_state,
            reward=rewards,
            discount=discounts,
            mask=mask,
            next_mask=next_mask,
            weight=weight,
        )

    def sample_sequences(
        self,
        chunks: int,
        *,
        burn_in: int = DEFAULT_BURN_IN,
        train: int = DEFAULT_TRAIN,
        demo_fraction: float = 0.0,
        prioritized: bool = True,
    ) -> SequenceBatch:
        """Draw `chunks` contiguous turn windows for truncated BPTT.

        Geometry, in turns::

            [ burn_in | train | n_step tail ]

        The episode is drawn by the same return-ranked law the flat sampler uses --
        a return is a property of the episode, not of how it is sliced -- and the
        chunk start is then uniform over that episode's stride grid.

        **Why the tail exists.** A training decision at turn `t` bootstraps from the
        first decision at turn `>= t + n_step`, so the last training turn reaches
        `n_step` turns past the window. The alternatives were to truncate the
        training window by `n_step` (which discards most of the gradient rows), to
        burn in a second time at the bootstrap (roughly seventeen times the cost,
        and a second independent staleness path), or to shorten the horizon near the
        boundary (which makes the credit horizon a function of a row's random
        position inside a chunk). Rolling the recurrence `n_step` turns further is
        ~3% more arithmetic and is the only option where the bootstrap's hidden
        state and the prediction's come out of the same rollout.

        **Why the start is on a grid.** `hidden0` has to be a state somebody stored,
        and `collect.HIDDEN_STRIDE` decides which turns those are. `burn_in` must be
        a multiple of it or `train_start - burn_in` lands between stored rows;
        that is validated here rather than assumed.

        A chunk that would start before turn 0 is left-padded instead, with
        `hidden0` at exactly zero -- the same state the episode really began from,
        so those chunks carry no staleness at all.
        """

        if burn_in < 0 or train <= 0:
            raise ValueError(f"burn_in must be >= 0 and train > 0; got {burn_in}, {train}")
        if self.n_step < 1:
            # `_roll_chunk` slices the tail as `turn_input[:, burn_in + train:]`,
            # which is empty at `n_step = 0`, and `recur` stacks an empty list --
            # a RuntimeError inside the first gradient step rather than a rejected
            # config. Refused here, beside the other two geometry checks.
            raise ValueError(f"n_step must be at least 1 for a chunk draw; got {self.n_step}")
        if burn_in % HIDDEN_STRIDE:
            raise ValueError(
                f"burn_in {burn_in} is not a multiple of the stored-hidden stride "
                f"{HIDDEN_STRIDE}; a chunk would have to start at a turn no hidden "
                "state was written for"
            )

        own, _returns, own_probabilities, own_weights = self._own_priorities()
        own = self._sampleable_sequences(own, train)
        demos = self._sampleable_sequences(self.pinned, train)
        if not own and not demos:
            raise ValueError(
                "free-market replay holds no episode long enough for a "
                f"{train}-turn training window"
            )
        # The ranked law was built over the full own pool; dropping short episodes
        # from the draw means its probabilities no longer sum to 1 over what is
        # left. Rather than renormalise a distribution against a different pool --
        # which would silently re-rank everything -- the ranked branch is used only
        # while no episode was dropped, and the uniform draw covers the rest.
        ranked = (
            prioritized
            and self.priority_alpha > 0.0
            and len(own) > 1
            and len(own) == len(own_probabilities)
        )

        wanted = int(round(chunks * max(0.0, min(1.0, demo_fraction))))
        demo_rows = min(wanted, chunks) if demos else 0
        own_rows = chunks - demo_rows
        if not own:
            demo_rows, own_rows = chunks, 0

        if ranked:
            own_picks = np.asarray(
                self.rng.choice(len(own), size=own_rows, p=own_probabilities)
            )
        else:
            own_picks = np.asarray(
                [int(self.rng.integers(0, len(own))) for _ in range(own_rows)],
                dtype=np.intp,
            )
        picks = [
            *((own, int(pick)) for pick in own_picks),
            *((demos, int(self.rng.integers(0, len(demos)))) for _ in range(demo_rows)),
        ]

        length = burn_in + train + self.n_step
        turn_input = np.zeros((chunks, length, STATE_DIM), dtype=np.float32)
        hidden0 = np.zeros((chunks, GRU_HIDDEN), dtype=np.float32)
        turn_valid = np.zeros((chunks, length), dtype=np.bool_)
        weight = np.ones(chunks, dtype=np.float32)
        if ranked and own_rows:
            weight[:own_rows] = own_weights[own_picks]

        states: list[np.ndarray] = []
        chosen: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        shares: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        discounts_out: list[np.ndarray] = []
        next_states: list[np.ndarray] = []
        next_masks: list[np.ndarray] = []
        next_turns: list[np.ndarray] = []
        clamped = 0
        post = train + self.n_step

        for row, (pool, pick) in enumerate(picks):
            record = pool[pick]
            turns = record.steps
            first = _turn_first(record)
            returns, discounts = self._nstep(record)

            starts = 1 + (turns - train) // HIDDEN_STRIDE
            train_start = int(self.rng.integers(0, starts)) * HIDDEN_STRIDE
            begin = train_start - burn_in
            # Where the chunk would start before the episode does, the missing
            # turns stay invalid and the state is the zero one the episode really
            # opened on -- not a stored row, and not a stale one either.
            stored = max(begin, 0) // HIDDEN_STRIDE
            if begin >= 0 and stored < record.turn_hidden.shape[0]:
                hidden0[row] = record.turn_hidden[stored].astype(np.float32)

            for slot in range(length):
                turn = begin + slot
                if 0 <= turn < turns:
                    turn_input[row, slot] = record.turn_state[turn]
                    turn_valid[row, slot] = True

            # One pass over the training window, emitting every decision of every
            # turn with its *own* target.
            #
            # The MDP a turn describes: sub-decisions are simultaneous in game
            # time, so a transition inside a turn carries no reward and does not
            # discount -- it just says "having queued that, what is the next order
            # worth?". Time passes only at the boundary, where the turn's whole
            # n-step return is paid to the decision that closed it.
            #
            # v2 gave all of a turn's decisions the same reward, discount and
            # bootstrap, which is one value replicated rather than a chain, and its
            # policy settled on a ten-slot zero-P&L wash because nothing in the
            # target distinguished the seventh order from the first.
            for index in range(train):
                turn = train_start + index
                if turn >= turns:
                    # Past the episode's end: the turn contributed no decisions, so
                    # there is nothing to emit.
                    continue
                lo, hi = int(first[turn]), int(first[turn + 1])
                if hi <= lo:
                    continue
                count = hi - lo
                own_slot = row * post + index

                # --- the boundary target, for this turn's last decision ---------
                turn_return = np.float32(returns[turn])
                horizon = np.float32(discounts[turn])
                boot_state_row = np.zeros(STATE_DIM, dtype=np.float32)
                boot_mask_row = np.ones(ACTION_COUNT, dtype=np.bool_)
                boot_slot = own_slot

                target = turn + self.n_step
                bootstrap = int(first[target]) if target <= turns else record.decisions
                if bootstrap < record.decisions:
                    boot_row = int(record.order_step[bootstrap])
                    if boot_row != target:
                        # A turn that recorded no decision at all pushes the
                        # bootstrap past `target`. `main.py` records a terminator
                        # every turn, so this cannot happen today; it is counted
                        # rather than asserted because `bootstrap_clamped_rows` must
                        # stay zero and a silent bias is worse than a loud one.
                        #
                        # Priced honestly rather than left on the fixed tables:
                        # pairing `returns[turn]` -- exactly `n_step` turns of
                        # reward -- with a bootstrap sitting further out is a
                        # lengthened horizon against a truncated return, which
                        # biases the target low. The return covers `[turn, boot_row)`
                        # and the discount spans the gap the bootstrap really sits at.
                        clamped += 1
                        window = record.reward[turn:boot_row].astype(np.float64)
                        weights = self.gamma ** np.arange(window.shape[0], dtype=np.float64)
                        turn_return = np.float32((window * weights).sum())
                        horizon = np.float32(self.gamma ** (boot_row - turn))
                    slot = boot_row - train_start
                    if 0 <= slot < post:
                        boot_state_row = record.order_state[bootstrap]
                        boot_mask_row = record.order_mask[bootstrap]
                        boot_slot = row * post + slot
                    else:
                        # Outside the rolled window, so there is no latent to score
                        # it against. Terminal convention: zero discount, all-True
                        # mask, and the row keeps only its own reward.
                        horizon = np.float32(0.0)
                else:
                    # The episode ended inside the horizon. `_nstep_tables` already
                    # zeroed the discount; nothing bootstraps and the mask stays
                    # all-True so the masked max cannot produce `-inf * 0 = nan`.
                    horizon = np.float32(0.0)

                # --- the rows ---------------------------------------------------
                states.append(record.order_state[lo:hi])
                chosen.append(record.order_chosen[lo:hi].astype(np.int64))
                masks.append(record.order_mask[lo:hi])
                owners.append(np.full(count, row * train + index, dtype=np.int64))
                shares.append(np.full(count, 1.0 / count, dtype=np.float32))

                # Intra-turn rows: no reward, no discounting, successor is the next
                # sub-decision against the *same* latent -- the GRU does not advance
                # inside a turn.
                row_reward = np.zeros(count, dtype=np.float32)
                row_discount = np.ones(count, dtype=np.float32)
                row_next_turn = np.full(count, own_slot, dtype=np.int64)
                row_next_state = np.empty((count, STATE_DIM), dtype=np.float32)
                row_next_mask = np.empty((count, ACTION_COUNT), dtype=np.bool_)
                if count > 1:
                    row_next_state[:-1] = record.order_state[lo + 1 : hi]
                    row_next_mask[:-1] = record.order_mask[lo + 1 : hi]
                # The last row closes the turn and takes the boundary target.
                row_reward[-1] = turn_return
                row_discount[-1] = horizon
                row_next_state[-1] = boot_state_row
                row_next_mask[-1] = boot_mask_row
                row_next_turn[-1] = boot_slot

                rewards.append(row_reward)
                discounts_out.append(row_discount)
                next_states.append(row_next_state)
                next_masks.append(row_next_mask)
                next_turns.append(row_next_turn)

        def stack(parts, width, dtype):
            if parts:
                return np.concatenate(parts).astype(dtype, copy=False)
            shape = (0,) if width is None else (0, width)
            return np.zeros(shape, dtype=dtype)

        return SequenceBatch(
            turn_input=turn_input,
            hidden0=hidden0,
            turn_valid=turn_valid,
            weight=weight,
            reward=stack(rewards, None, np.float32),
            discount=stack(discounts_out, None, np.float32),
            next_state=stack(next_states, STATE_DIM, np.float32),
            next_mask=stack(next_masks, ACTION_COUNT, np.bool_),
            next_turn=stack(next_turns, None, np.int64),
            state=stack(states, STATE_DIM, np.float32),
            chosen=stack(chosen, None, np.int64),
            mask=stack(masks, ACTION_COUNT, np.bool_),
            decision_turn=stack(owners, None, np.int64),
            turn_share=stack(shares, None, np.float32),
            clamped=clamped,
            ranked=bool(ranked),
        )

    def _nstep(self, record: EpisodeRecord) -> tuple[np.ndarray, np.ndarray]:
        """`_nstep_tables`, memoised on the record.

        Cached on the record rather than in the buffer because `reward` is written
        once by `Recorder.finish` and never changes, so unlike `_priorities` there
        is nothing that could invalidate it.
        """

        cached = getattr(record, "_nstep_cache", None)
        if cached is None:
            cached = _nstep_tables(record, self.gamma, self.n_step)
            object.__setattr__(record, "_nstep_cache", cached)
        return cached
