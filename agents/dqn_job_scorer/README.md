# DQN Job Scorer

Strategy attempt **#2**. A Q-network replaces the hand-tuned job scoring; everything else —
job enumeration, unit routing, market orders — stays on rules.

The agent is self-contained: standard library plus numpy, no imports from `kaggriculture_lab`,
`sim`, or any other agent. Its crop tables and board geometry are its own copy in `rules.py`
(`tests/test_game_facts.py` checks every agent's copy against `README.md`).

## Layout

| File | Role |
| --- | --- |
| `main.py` | `agent(obs)`; job selection, commitment, routing |
| `jobs.py` | legal-job enumerator |
| `features.py` | the featurizer — shared by training and inference, hence single-sourced |
| `policy.py` | numpy MLP forward pass, ε-greedy selection |
| `market.py` | selling, hiring, seed buying (rules, not learned) |
| `rules.py` | this agent's private crop tables and geometry |
| `evaluation.py` | post-episode league hook |
| `train.py` | lab-facing `on_round` collector and `on_batch` updater |
| `weights.npz` | trained parameters |
| `training/` | torch trainer — **never** imported by `main.py`, never bundled |
| `MODEL.html` | illustrated guide to inference, features, training, and deployment |

## How it decides

Each turn it enumerates every *legal* job — deliberately not
`ConfigurableAgent._jobs`, whose heuristic filters would trap the policy inside the agent it
is trying to beat — and scores each one with `Q(state, job)`. Features are 43 global + 26
per-job + 3 per-unit = 72 dims through a 128×128 MLP.

**Job commitment is load-bearing.** A unit keeps its chosen job until it arrives, the job
becomes illegal, or 20 turns pass. Without it the policy re-picks every turn, and a small
shift in Q values flips the target, so units walk east then west forever and plants die
unwatered. The first training run failed exactly this way: 62% of actions were movement, 10%
productive, and 39 of 70 plants became weeds. The commitment is also a *feature*, because
otherwise the state is not Markov — the same board can be one turn into a walk east or one
turn into a walk west and the network cannot tell.

Exploration therefore applies at selection points, not per step. At ε=0.87 a per-step ε
derails almost every walk before it completes.

## Training

The normal persistent workflow is driven by a league session whose
`training_manifests` includes this agent. Each closed league round calls `train.py`,
which collects one 16-episode rollout batch. After twenty new batches, `on_batch`
trains and gates a checkpoint. Batches, replay artifacts, watermarks, and the active
checkpoint remain available when the process or session resumes against the same
DuckDB catalog.

```bash
python -m kaggriculture_lab.cli league run \
  --session-id all-agents-training \
  --database artifacts/experiments.duckdb --artifacts artifacts

python -m agents.dqn_job_scorer.cli model status \
  --name default \
  --database artifacts/experiments.duckdb
```

The constants-driven runner can keep several named DQN instances in the same catalog.
Each name has independent rollout batches, checkpoints, random seed, and active pointer
under `artifacts/dqn_job_scorer/<name>/`. Set that roster entry's `train` flag to `False`
to freeze it without deleting or exporting anything.

The standalone research loop is still useful when an experiment should not share the
league database:

```bash
python -m agents.dqn_job_scorer.training.loop --rounds 250 --jobs 8 --ship
```

Double DQN, potential-based reward shaping, n-step returns, opponent curriculum
(`starter` → the crop heuristic → frozen self-play snapshots).

Two things to know before changing it:

- **Progress is the arena margin, not training return.** Every `--eval-every` rounds the
  snapshot plays seat-swapped seeds against a fixed baseline. Training money moves for
  reasons unrelated to winning: ε-greedy rollouts once scored 25k while the greedy policy
  scored 13k.
- **Promotion is by measured result, never recency.** `weights.npz` tracks the best-evaluated
  checkpoint. Training is not monotonic — an earlier run peaked at round 45 (+2,298 margin,
  93.8% wins) and collapsed to −21,496 by round 75, so shipping the last checkpoint ships the
  collapse. `--ship` is required to touch the live weights at all; without it, promotion only
  writes `runs/<name>/best.npz`.

For league training, promotion changes the catalog's active-checkpoint pointer and does
not overwrite `weights.npz`. Copy a reviewed active checkpoint into the submission only
with `python -m agents.dqn_job_scorer.cli model export-active --output
agents/dqn_job_scorer/weights.npz --confirm-release`.

## Evaluating

```bash
python -m sim.arena match agents/dqn_job_scorer/main.py \
       agents/configurable_crop_surrogate/main.py --seeds 200
```

Compare a specific checkpoint with `?weights=`:

```bash
python -m sim.arena match "agents/dqn_job_scorer/main.py?weights=runs/dqn03/best.npz" \
       agents/configurable_crop_surrogate/main.py --seeds 100
```

**Absolute money is not comparable across opponents.** The market is shared, so a productive
opponent gluts it and both players earn less. The crop heuristic scores ~32k against a
do-nothing agent and ~19k against a real one — same agent, same seeds. Only the paired margin
means anything.

## Submitting

Build an inspectable folder and a ready-to-upload archive containing only the
inference allowlist from `agent.toml`:

```bash
python agents/dqn_job_scorer/export_submission.py
```

The command recreates `agents/dqn_job_scorer/submission/` and writes
`agents/dqn_job_scorer/dqn_job_scorer_submission.tgz`. The archive has the
Kaggle-required `main.py` at its root and excludes `training/`, torch, and the
rest of the repository.

The generic lab bundler remains available:

```bash
python -m kaggriculture_lab.cli bundle --agent agents/dqn_job_scorer/agent.toml
```

`code_paths` is an allowlist, so `training/` and torch are excluded by construction.
`tests/test_bundle.py` unpacks the result and imports it with the repo off `sys.path` to prove
it stands alone.
