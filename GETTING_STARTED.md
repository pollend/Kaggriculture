# Getting started

How to run rounds in this repo. `README.md` is the competition's own rules document — game
mechanics, crop tables, the price function — and is left alone; this file is the workflow.

Every command below has been run as written.

## Setup

```bash
uv sync --python 3.12
.venv/bin/python -m pytest -q          # ~117 tests, a few minutes
```

`torch` is a dev dependency used only for training. Submitted agents are numpy-only.

## The three kinds of "round"

| You want to | Use | Round means |
| --- | --- | --- |
| Compare two agents head-to-head | `sim.arena` | one seed, played in both seats |
| Evolve a population of strategies | `kaglab league` | one pass of pairings within a generation |
| Train the DQN | league trainer hook or `agents.dqn_job_scorer.training.train` | collect episodes, then take gradient steps |

---

## 1. Arena rounds — comparing two agents

The fastest loop, and the one that settles arguments. An episode is ~0.9 s, so 100 seeds
(200 episodes, both seat orders) takes about 25 s on 8 workers.

```bash
.venv/bin/python -m sim.arena match \
  agents/dqn_job_scorer/main.py \
  agents/configurable_crop_surrogate/main.py --seeds 100
```

```
  record       : 58W 2L 0D
  win rate     :  96.7%   95% CI [88.5%, 99.1%]
  margin       :     +2,645   95% CI [+2,271, +3,019]   sign p=0.0000
  verdict      : RESOLVED
```

Read the **margin**, not the money. The market is shared, so a productive opponent gluts it
and both players earn less: the crop heuristic scores ~32k against a do-nothing agent and
~19k against a real one. Absolute money is meaningless without naming the opponent.

`NOT RESOLVED` means the interval spans zero — the arena tells you how many more seeds it
would take rather than calling noise a win.

Other subcommands:

```bash
python -m sim.arena league a.py b.py starter --seeds 50          # round-robin matrix
python -m sim.arena sweep main.py --vs starter --param k=1,2,3   # one agent across values
```

Point at a specific checkpoint with `?weights=`:

```bash
python -m sim.arena match \
  "agents/dqn_job_scorer/main.py?weights=runs/best/dqn03_r60_margin+2645_60seeds.npz" \
  agents/configurable_crop_surrogate/main.py --seeds 60
```

## 2. League rounds — evolving a population

A league is a persistent DuckDB session: generations of immutable agent revisions playing
paired matches, rated with Glicko-2, bred into the next generation.

```bash
DB=artifacts/experiments.duckdb

# Generation zero. Population is sampled by the agent family's own training adapter.
.venv/bin/python -m kaggriculture_lab.cli league init \
  --config configs/league_smoke.toml --session-id demo --database $DB

# Play every pairing in the current generation. Resumable: rerun to continue.
.venv/bin/python -m kaggriculture_lab.cli league run \
  --session-id demo --database $DB --artifacts artifacts
```

```json
{"generation_id": "demo:g0000", "hook_evaluations": {"COMPLETE": 48},
 "pairings": {"COMPLETE": 15}, "status": "COMPLETE"}
```

Then inspect and breed:

```bash
.venv/bin/python -m kaggriculture_lab.cli league status      --session-id demo --database $DB
.venv/bin/python -m kaggriculture_lab.cli league leaderboard --session-id demo --database $DB --limit 10
.venv/bin/python -m kaggriculture_lab.cli league next-generation --session-id demo --database $DB
```

`next-generation` builds a 20/50/20/10 elite / evolution / surrogate / random mix. The
surrogate share comes from the family's own `propose()`; a family without one falls back to
mutation, so a new agent works before it has any model.

Configs live in `configs/`. Start with `league_smoke.toml` (6 agents, 4 rounds, 120-step
episodes — finishes in under a minute) before `league.toml`.

### Episode hooks

Each agent's `agent.toml` may name an `episode_evaluator`, run after every episode it plays,
with read-only DuckDB access to *all* of its prior games:

```python
def evaluate_episode(context, reader):
    summary = reader.summary()                        # this episode
    history = reader.history(context.revision_id, 50) # its previous games
    ...
```

`agents/dqn_job_scorer/evaluation.py` uses this to compare recent games against older ones and
flag `should_reevaluate` when the checkpoint has regressed — a per-episode number is far too
noisy to see that, since a good checkpoint still loses plenty of individual games.

Re-run hooks over stored replays without replaying anything:

```bash
.venv/bin/python -m kaggriculture_lab.cli league evaluate --session-id demo --database $DB --force
```

## 3. Training rounds — the DQN

One round = collect N episodes in parallel, then take gradient steps against the replay buffer.

```bash
.venv/bin/python -u -m agents.dqn_job_scorer.training.train \
  --rounds 250 --episodes-per-round 16 --gradient-steps 400 \
  --eval-every 10 --eval-seeds 40 --duel-seeds 60 --jobs 8 \
  --output runs/dqn04 --ship
```

Roughly 18 s per round; every 10th round runs a real arena evaluation. Watch it with:

```bash
tail -f runs_dqn04.log | grep --line-buffered '"eval"'
```

Two behaviours worth knowing before you trust a number:

- **Progress is the arena margin, never training return.** ε-greedy rollouts once scored 25k
  while the greedy policy scored 13k. Only the `eval` block means anything.
- **Promotion is by measurement, not recency.** Every run starts from the configured reviewed
  incumbent; `--ship` only enables the explicit release copy. Training is not
  monotonic — one run peaked at +2,298 and collapsed to −21,496 thirty rounds later, so the
  last checkpoint is regularly the worst one. Without `--ship`, promotion only writes
  `runs/<name>/best.npz` and the live agent is never touched.

A real trend, for calibration:

```
r10  -10,120    0.0%
r30   -4,249    0.0%
r50  -26,616    0.0%   <- collapse, correctly rejected
r60   +2,402   95.8%   <- crossover
```

## 4. Submitting

```bash
.venv/bin/python -m kaggriculture_lab.cli bundle \
  --agent agents/dqn_job_scorer/agent.toml --output artifacts/submission.tar.gz

kaggle competitions submit kaggriculture -f artifacts/submission.tar.gz -m "dqn r60"
```

The bundle contains exactly the manifest's `code_paths` plus a generated top-level `main.py`.
Because that is an allowlist, training code and torch are excluded by construction.
`tests/test_bundle.py` unpacks the archive and imports it with the repo off `sys.path` to prove
it stands alone.

## Adding an agent

```bash
.venv/bin/python -m kaggriculture_lab.cli new-attempt --name my_idea --title "My Idea"
```

Agents are fully independent — no agent imports another, and none import `sim` or
`kaggriculture_lab` on the inference path. Copy the crop tables you need rather than sharing
them; `tests/test_game_facts.py` checks every copy against the rules in `README.md`, and
`tests/test_layering.py` enforces the boundaries. That is what lets you prototype one agent
without breaking the others.

## Layout

```
sim/                 game-agnostic match runner and paired statistics
kaggriculture_lab/   league, DuckDB storage, hooks, episode analysis
agents/<name>/       one self-contained agent each
  training/          that agent's trainer (never bundled)
tests/               layering, game-facts, bundle and behaviour guards
runs/                training output; runs/best/ holds proven checkpoints
```
