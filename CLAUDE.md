# Kaggriculture — agent guide

Everything the league records lives in one PostgreSQL database, and the running lab
exposes it over MCP. **Query it.** Ratings, per-episode outcomes, training batches,
checkpoint promotions, and hook results are all there, so questions like "is dqn02
regressing", "which opponent family does v5 lose to", or "why did that model update
fail" are answered from data rather than guessed at.

See `AGENTS.md` for how to write an agent and its trainer, `README.md` for the lab
commands, `GETTING_STARTED.md` for the simulator rules.

## The MCP endpoint

`kaglab web` serves both the dashboard and a stateless MCP Streamable HTTP endpoint:

| Interface | URL |
| --- | --- |
| Dashboard | `http://127.0.0.1:8000/` |
| MCP | `http://127.0.0.1:8000/mcp` |

```bash
./lab-stack.sh up                                   # PostgreSQL + dashboard + MCP
claude mcp add --transport http kaggriculture http://127.0.0.1:8000/mcp
```

`kaglab web` also writes the managed `kaggriculture_progress` entry into
`.codex/config.toml` for Codex. Nothing else needs to be started: training
processes are database writers only and never bind ports of their own, so one
endpoint sees every concurrent runner.

Transport specifics, if you are speaking JSON-RPC by hand: POST only,
`Content-Type: application/json`, 1 MiB body cap, JSON responses (no SSE stream),
protocol revisions `2025-11-25`, `2025-06-18`, `2025-03-26`. An `Origin` header, if
sent, must match `Host`.

## Dashboard pages

| Path | Shows |
| --- | --- |
| `/` | Every session, with pairing or round progress and the current leader |
| `/sessions/<session_id>` | Rating trend, head-to-head margin by opponent family, models, generations |
| `/sessions/<session_id>/failures` | Failures grouped by cause; tracebacks collapsed to their exception line |
| `/generations/<generation_id>` | Leaderboard, margin distribution, seat bias, per-agent behaviour, runtime, pairings |
| `/models/<model_id>` | Update gate timeline, metric trends, checkpoint lineage, batch watermark |
| `/episodes/<match_id>` | Both seats' money and behaviour, plus the embedded replay |

Pages update live over `/events/dashboard`, and both the stream and ordinary
requests share a render cache keyed by a database watermark, so an idle tab costs
one cheap query per tick rather than a full re-render.

## Tools

Four bounded inspection tools, plus one open SQL tool. Start from the top when you
do not know the session ID.

| Tool | Use it for |
| --- | --- |
| `list_training_sessions` | Discover sessions by `league`/`status`; generation, episode, and failure counts |
| `get_session_progress` | One session: generation history, rating deltas, leaders, hook evaluations, models |
| `get_generation_progress` | One generation: work completion, outcomes, seat balance, ranked agents, failures |
| `get_model_progress` | One model: batch watermark, active checkpoint, update gates, checkpoint history |
| `query_database` | Anything the four above do not answer — arbitrary SQL |

`query_database` takes `sql`, an optional `parameters` array (psycopg `%s`
placeholders), and `max_rows` (default 200, max 1000). It returns `columns`, `rows`,
`row_count`, and `truncated`.

**It is not read-only.** It connects as `kag_writer`, so `INSERT`/`UPDATE`/`DELETE`
and DDL all run, and the connection is autocommit — there is no transaction to roll
back. Statements with no result set return `affected_rows` instead of rows. This is
a Kaggle playground, not production; fixing up a stuck session or backfilling a
column directly is fine, but a `DELETE` without a `WHERE` is gone for good. Say what
you are about to mutate before you mutate it.

Schema discovery is live — prefer it over trusting this file:

```json
{"name": "query_database", "arguments": {
  "sql": "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' ORDER BY table_name, ordinal_position"}}
```

## Where the data lives

All tables are in the `public` schema of database `kaggriculture`.

**League runs.** `training_sessions` (one per league run, grouped by `league`) →
`generations` → `league_pairings` → `pairing_episodes` → `episodes`. Each pairing
plays both seat orientations; `pairing_results` holds the seat-balanced
`paired_margin` and `outcome_a`, which is what ratings actually consume.

**Per-episode facts.** `episode_players` is the workhorse: one row per seat with
`final_money`, `money_difference`, `win`, action-mix percentages, and `metrics_json`.
`episode_payloads.replay_gzip` is the gzipped replay JSON for the same `match_id`.
`episode_evaluations.result_json` holds whatever an agent's `evaluate_episode` hook
returned.

**Who played.** `agent_revisions` is one immutable policy identity (family, params,
code fingerprint, optional `checkpoint_id`); `agent_families` is the manifest behind
it. `generation_members` assigns revisions to a generation with a `role`
(elite/evolution/surrogate/random), `rating_snapshots` carries Glicko
rating/deviation/games with `is_frozen` marking fixed anchors, `generation_stats`
the win/loss/margin aggregates, and `lineage` who bred from whom.

**Training.** `training_models` (one per trainable model, with
`active_checkpoint_id` and `last_attempted_batch_seq`) → `training_batches` (the
`batch_seq` watermark) → `model_updates` (`status`, `metrics_json`, `gate_json`) →
`model_training_checkpoints` (`promoted`). `model_evaluations` records gate
matches, `model_round_hooks` records which round hooks already fired
(`round_index = -1` means generation scope). `episode_training_artifacts` links
rollout artifacts to batches through `training_batch_artifacts`.

**Population training.** `population_members` is the mature half of a
population-trained model: checkpoints that graduated out of a learner slot, keyed
on `declared_model_id` because that is what a roster of named instances shares
(`decl@learner_00` … `decl@learner_09`). Live members are `retired_at IS NULL`;
retirement is a membership change and never a delete, so what was dropped and at
what `score` stays answerable. `score` is a windowed median of benchmark money,
not a single arena draw. The learner side is not a table — it is per-update state
in `model_updates.metrics_json` under the `lr_*` keys plus `population`.

**Self-play.** `policy_mixtures` + `policy_mixture_members` define a weighted
opponent blend; `pairing_policy_assignments` records which member a pairing drew.

**Older cycle tables.** `runs`, `strategies`, `ratings`, `models`, `proposals` back
`kaglab cycle`, not the league. `model_checkpoints` is superseded by
`model_training_checkpoints` and no code writes it.

## What the rollups mean

`kaggriculture_lab/league_queries.py` holds the SQL behind both the dashboard and
the four inspection tools, so the two now answer every question the same way. The
definitions worth knowing, because the two used to disagree:

- **A round is complete when *every* model's hook completed** — `bool_and(status =
  'COMPLETE')` over `model_round_hooks` grouped by `(generation_id, round_index)`.
  One model finishing does not finish the round.
- **Pairing failure is `league_pairings.status = 'FAILED'`.** That column is only
  ever written `COMPLETE` or `FAILED`, so the old `pairing_errors` field counted a
  status that never occurs and was structurally zero. It is now `failed_pairings`.
- **A failed episode is `episodes.failed`,** not `episode_players.failed`. An
  episode that crashed is excluded whole from seat balance and per-agent money,
  which is how `failed_episodes` is counted everywhere else.
- **`should_reevaluate` is read only from `episode_evaluations` rows with
  `status = 'COMPLETE'`.** An evaluator that raised has no meaningful result.
- **Rating trend columns are `champion_conservative` and `median_conservative`.**

`list_training_sessions` also returns `completed_rounds` and `failed_pairings`, and
`get_generation_progress` returns the full margin distribution (`min_paired_margin`,
`max_paired_margin`, `drawn`, `mean_outcome_a`) rather than mean/median alone.

## Query patterns that come up

Rating trend for one family across a session:

```sql
SELECT g.generation_index, max(rs.rating - 2 * rs.deviation) AS conservative
FROM rating_snapshots rs
JOIN generations g USING (generation_id)
JOIN agent_revisions ar USING (revision_id)
WHERE g.session_id = %s AND ar.family_id = %s AND NOT rs.is_frozen
GROUP BY g.generation_index ORDER BY g.generation_index
```

Head-to-head money margin by opponent family:

```sql
SELECT opp.family_id, count(*) AS games, avg(ep.money_difference) AS mean_margin
FROM episode_players ep
JOIN agent_revisions me ON me.revision_id = ep.strategy_id
JOIN agent_revisions opp ON opp.revision_id = ep.opponent_id
WHERE me.family_id = %s AND NOT ep.failed
GROUP BY opp.family_id ORDER BY mean_margin
```

Why the last model updates failed:

```sql
SELECT update_id, status, cutoff_batch_seq, error_text, gate_json
FROM model_updates WHERE model_id = %s ORDER BY started_at DESC LIMIT 10
```

Notes when reading results: filter `NOT failed` on `episodes`/`episode_players`
unless you are specifically hunting crashes; compare at generation boundaries
(`generations.status = 'COMPLETE'`) since a mid-generation snapshot is partial; and
prefer `pairing_results.paired_margin` over raw per-seat money when comparing two
revisions, because it already cancels seat advantage.
