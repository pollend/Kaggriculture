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

from agents.dqn_market_free.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free.order_space import ACTION_COUNT
from agents.dqn_market_free.training.collect import EpisodeRecord

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
          equal rows.
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

            horizon = min(self.n_step, record.decisions - decision)
            bootstrap = decision + horizon
            start_step = int(record.order_step[decision])
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

    def sample_demonstrations(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly sample `(state, chosen)` from the pinned expert trajectories.

        Separate from `sample` because the behaviour-cloning phase wants a pure
        supervised target rather than a TD transition: no reward, no bootstrap,
        and no self-play rows diluting the classification signal.
        """

        eligible = self._sampleable(self.pinned)
        if not eligible:
            raise ValueError("free-market replay has no demonstration decisions")

        state = np.zeros((batch_size, STATE_DIM), dtype=np.float32)
        chosen = np.zeros(batch_size, dtype=np.int64)
        mask = np.ones((batch_size, ACTION_COUNT), dtype=np.bool_)
        picks = self.rng.integers(0, len(eligible), size=batch_size)
        for row, pick in enumerate(picks):
            record = eligible[int(pick)]
            decision = int(self.rng.integers(0, record.decisions))
            state[row] = record.order_state[decision]
            chosen[row] = int(record.order_chosen[decision])
            mask[row] = record.order_mask[decision]
        return state, chosen, mask
