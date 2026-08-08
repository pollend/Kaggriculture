"""Episode ring buffer that featurizes minibatches on demand.

Samples are `(episode, t, unit)` triples. The features the network sees are
rebuilt from the stored globals and job statics plus a freshly computed distance
column -- which is exactly what the agent computed live, and a test pins that
equivalence, since the whole memory saving rests on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.dqn_job_scorer import features as F
from agents.dqn_job_scorer.training.collect import EpisodeRecord


@dataclass
class Batch:
    """One training minibatch.

    `next_features` / `next_mask` are padded to the widest candidate set in the
    batch; masked slots must be excluded from the max, not merely zeroed, or a
    padded row would win whenever every real Q is negative.
    """

    features: np.ndarray        # float32 [B, FEATURE_DIM]        chosen action
    next_features: np.ndarray   # float32 [B, J, FEATURE_DIM]     candidates at t+n
    next_mask: np.ndarray       # bool    [B, J]
    reward: np.ndarray          # float32 [B]                     n-step discounted
    discount: np.ndarray        # float32 [B]                     gamma**n, 0 if terminal


class ReplayBuffer:
    """Fixed-capacity ring of whole episodes."""

    def __init__(self, capacity: int = 200, gamma: float = 0.997, n_step: int = 3, seed: int = 0):
        self.capacity = capacity
        self.gamma = gamma
        self.n_step = n_step
        self.episodes: list[EpisodeRecord] = []
        self.rng = np.random.default_rng(seed)

    def add(self, records: list[EpisodeRecord]) -> None:
        self.episodes.extend(records)
        if len(self.episodes) > self.capacity:
            del self.episodes[: len(self.episodes) - self.capacity]

    @property
    def transitions(self) -> int:
        return sum(int(record.unit_mask.sum()) for record in self.episodes)

    def nbytes(self) -> int:
        return sum(record.nbytes() for record in self.episodes)

    def _features_at(self, record: EpisodeRecord, t: int, unit: int) -> tuple[np.ndarray, np.ndarray]:
        """Rebuild the candidate feature matrix this unit faced at step t."""
        mask = record.job_mask[t]
        count = int(mask.sum())
        static = record.job_static[t, :count].astype(np.float32)
        targets = record.job_targets[t, :count]
        position = (int(record.unit_pos[t, unit, 0]), int(record.unit_pos[t, unit, 1]))
        distance = F.distance_features(targets, position, int(record.committed[t, unit]))
        return F.assemble(record.global_feats[t], static, distance), mask[:count]

    def sample(self, batch_size: int) -> Batch:
        episode_indices = self.rng.integers(0, len(self.episodes), size=batch_size)

        chosen_features = np.empty((batch_size, F.FEATURE_DIM), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        discounts = np.zeros(batch_size, dtype=np.float32)
        next_sets: list[np.ndarray] = []

        for row, episode_index in enumerate(episode_indices):
            record = self.episodes[int(episode_index)]
            steps = record.steps
            t = int(self.rng.integers(0, steps))
            units = np.flatnonzero(record.unit_mask[t])
            unit = int(self.rng.choice(units))

            features, _ = self._features_at(record, t, unit)
            chosen_features[row] = features[int(record.chosen[t, unit])]

            # n-step return, truncated at the episode end.
            horizon = min(self.n_step, steps - t)
            discount = 1.0
            total = 0.0
            for offset in range(horizon):
                total += discount * float(record.reward[t + offset])
                discount *= self.gamma
            rewards[row] = total

            bootstrap = t + horizon
            if bootstrap >= steps:
                discounts[row] = 0.0  # terminal: nothing left to bootstrap from
                next_sets.append(np.zeros((1, F.FEATURE_DIM), dtype=np.float32))
                continue

            discounts[row] = discount
            # The acting unit may not exist at t+n (hands are re-hired daily), so
            # fall back to the main farmer, which always exists.
            next_units = np.flatnonzero(record.unit_mask[bootstrap])
            next_unit = unit if unit in next_units else int(next_units[0])
            next_features, _ = self._features_at(record, bootstrap, next_unit)
            next_sets.append(next_features)

        width = max(candidates.shape[0] for candidates in next_sets)
        next_features = np.zeros((batch_size, width, F.FEATURE_DIM), dtype=np.float32)
        next_mask = np.zeros((batch_size, width), dtype=bool)
        for row, candidates in enumerate(next_sets):
            count = candidates.shape[0]
            next_features[row, :count] = candidates
            next_mask[row, :count] = discounts[row] > 0.0

        return Batch(chosen_features, next_features, next_mask, rewards, discounts)
