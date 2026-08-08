# DQN Job Scorer

The DQN replaces only candidate-job scoring. Training data must be recorded while the
policy acts because normal Kaggle replays do not contain the candidate-job set.

```bash
.venv/bin/kaglab models collect dqn_job_scorer.q_network --count 8
.venv/bin/kaglab models update dqn_job_scorer.q_network
.venv/bin/kaglab models status
```

One batch contains 16 rollout episodes by default. After eight new batches, the
updater loads the rolling replay window, resumes the online/target networks and
optimizer, trains a challenger, and evaluates it on seat-swapped seeds against the
incumbent and the fixed crop benchmark. A statistically clear regression rejects the
challenger without changing the active checkpoint. NumPy inference weights are
exported explicitly; the submission agent never imports Torch or opens DuckDB.
