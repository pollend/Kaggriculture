# sim — head-to-head evaluation for kaggriculture agents

```bash
uv sync --python 3.12

.venv/bin/python -m pytest sim/test_arena.py -q
.venv/bin/python -m sim.arena match agents/configurable_crop_surrogate/main.py starter --seeds 100
```

## Convenience wrappers

The repository-root wrappers create or update `.venv` with the locked Python 3.12
environment before running the arena. With no arguments they run a 10-seed match of
our configurable agent against `starter`:

```bash
# Bash / macOS / Linux
./run-sim.sh

# PowerShell / Windows
./run-sim.ps1
```

Use `setup` to prepare `.venv` without starting a simulation, or pass any normal arena
command through the wrapper:

```bash
./run-sim.sh setup
./run-sim.sh match agents/configurable_crop_surrogate/main.py starter --seeds 50
./run-sim.sh sweep agents/configurable_crop_surrogate/main.py --vs starter --param harvest_age=2,3,4
```

The `competitors` shortcut builds a round-robin containing our configurable agent, every
`agents/competitor_*/main.py` snapshot, and `starter`. It defaults to 10 seeds; a trailing
`--seeds` overrides that default:

```bash
./run-sim.sh competitors --seeds 2 --jobs 8
./run-sim.ps1 competitors --seeds 2 --jobs 8
```

An episode is ~0.9 s, so a 100-seed match (200 episodes, both seat orders) runs in
about 25 s on 8 workers.

## Commands

```bash
python -m sim.arena match  main.py starter --seeds 100
python -m sim.arena league main.py v2.py starter --seeds 50
python -m sim.arena sweep  main.py --vs starter --param harvest_age=2,3,4,5 --seeds 50
```

Flags: `--jobs`, `--steps` (shorten `episodeSteps` for coarse sweeps),
`--replay-dir` (dump replay JSON for non-wins and errors only),
`--config KEY=VALUE` (any env configuration key, repeatable).

## Agent specs

| Spec | Meaning |
| --- | --- |
| `pass` `starter` | built-ins, deterministic |
| `random` | built-in, **not reproducible** — the arena warns |
| `kaggriculture_lab.opponents:make_seeded_random?seed=7` | our reproducible replacement |
| `main.py` | file path; uses `agent`, else the last callable |
| `pkg.mod:agent` | module attribute |
| `main.py?hire_cap=12` | config injected as `KAGG_AGENT_CONFIG` JSON before import |

Read injected config from inside an agent with `sim.registry.agent_config()`. It is read
at import time, so the submission stays a bare one-argument `agent(obs)`.

Built-in names are read from the environment itself (`make(env_name).agents`), so this
package names no particular game.

## Scope: this package knows nothing about farming

`sim` runs matches and does statistics. It understands seats, statuses, rewards, timings and
paired margins — not crops, money, or tiles. Anything that reads a farm lives in
`kaggriculture_lab/episode_metrics.py` and is passed in:

```python
from kaggriculture_lab.episode_metrics import episode_metrics
run_match(("a.py", "b.py"), seed=0, analyzer=episode_metrics)   # metrics populated
run_match(("a.py", "b.py"), seed=0)                             # metrics == {}
```

The league opts in because it persists those metrics for agents' evaluation hooks to read
back out of PostgreSQL. `tests/test_layering.py` enforces the direction: `sim` may not import
`kaggriculture_lab` or any agent.

## Why it is built this way

Three environment behaviours make a naive harness report wrong numbers:

- **A crashed agent looks like a weak one.** The seat's status goes `ERROR` on the step
  after the exception, but the *final* step reports `DONE` with a plausible reward. Reading
  `env.steps[-1]` misses the crash entirely. `runner` scans every step and drops errored
  seeds from the summary instead of scoring them.
- **`debug=True` re-raises agent exceptions** rather than recording them, so the arena runs
  with `debug=False` and reads tracebacks out of `env.logs`. Note `env.logs` is indexed by
  env step, lagging the observation `step` an agent saw by 2.
- **The built-in `random` agent builds an unseeded `random.Random()` per call**, so seeding
  the global module does not make it reproducible. Use `kaggriculture_lab.opponents:make_seeded_random`.

Every seed is played in both seat orientations, so the headline statistic is a *paired*
margin under common random numbers rather than a difference of two independent means. That
is what makes a few dozen seeds enough. `test_identical_agents_have_zero_margin` pins the
assumption it rests on: the same deterministic agent in both seats ties exactly, so no seat
bias leaks into any verdict.

`render` says `NOT RESOLVED` and estimates the seeds required rather than reporting an
unresolved margin as a win.

## Baselines are weak

`starter` farms a single tile and never moves, finishing a 30-day season with ~$3,480 from a
$3,000 start — about +$480. `random` bankrupts itself to $0. Both are useful as smoke tests
and as a floor, but neither discriminates between two competent agents. Use `league` against
your own previous versions once you have them.
