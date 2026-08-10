# SP-PSRO Job Scorer

> **No entrypoint.** The population workflow has no runnable launcher: DQN
> training is consolidated into `python train_dqn.py`, a single learner playing
> frozen snapshots of itself, with no population mixture, Exp3 update, or
> admission round. The mixture never earned its complexity in practice.
>
> The code below stays as the source for three pieces worth folding into the
> base DQN later: the absolute release gate (`passes_release_gate`,
> `training/updater.py`), the iterate-averaged candidate (`_record_nu_snapshot`
> / `_save_nu_average`), and durable, resumable gate evaluation recorded as
> league rounds (`_prepare_admission` / `_apply_recorded_evaluations`).
> Restoring the population itself needs a new entrypoint written against
> `configs/sp_psro_joint_league.toml`.

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
recent-policy pool. Admission is submitted as a seat-swapped league evaluation
round after the candidate checkpoint is committed; every orientation and replay
is durable, so an interrupted evaluation resumes only its missing games.

The population is training-only. Each accepted member is evaluated on held-out
seat-swapped games against the fixed validation suite. The live best response is
evaluated by the same external release gate even before population admission, so
an externally strong beta may become the inference artifact while it continues
training. Population admission and submission release are deliberately separate.
External validation is likewise a recorded league evaluation round. Its compact
result batch advances the pending checkpoint immediately without counting as a
gradient-training batch.
