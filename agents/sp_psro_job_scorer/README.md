# SP-PSRO Job Scorer

This strategy keeps the DQN Job Scorer's inference policy and replaces its
training curriculum with a learned self-play population policy inspired by
Self-Play PSRO (arXiv:2207.06541). Each population member is one joint artifact:
a network that scores field jobs plus a network that chooses daily purchasing,
selling, and animal-production targets.

Every rollout samples either a frozen member or the live new policy from an
Exp3-style multiplicative-weights distribution and places it against a live best response.
Both seats are recorded. The best response
learns from its own trajectory and the new policy learns off-policy from the
population-side trajectory. Both networks use the existing potential-shaped DDQN reward;
losses and ties remain valid training data.

The market policy chooses a crop, animal, or mixed regime, one preferred animal
species, a labor-scaled animal count, one or two days of wheat, crop acreage,
hands, land, and per-product sell reserves. The resulting plan drives structure
construction, pickup, placement, feeding, care, harvest, and fertilizer jobs.
Existing animals of other species remain serviced. Emergency feed,
affordability, the order cap, and terminal liquidation remain deterministic.

After an update, the best response and a bounded checkpoint-reservoir
approximation of the new policy's time average are admitted together only when
their uniform pair has a positive paired-margin 95% lower bound against the old
population. Rejected learners keep training. Accepted learners are frozen, both
live learners are reset, and only accepted members participate in the bounded
recent-policy pool.

The population is training-only. Each accepted member is evaluated on held-out
seat-swapped games against the fixed validation suite, and the highest mean-win
member becomes the single active inference artifact and submission candidate.
