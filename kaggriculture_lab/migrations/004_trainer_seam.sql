-- v3 -> v4: one `trainer` seam replaces the `updater` + `collector` pair, and
-- round hooks get a ledger.
--
-- `model_round_hooks` is created by schema.sql, which is replayed before any
-- migration runs, so this file only has to reshape `training_models` and
-- backfill.

ALTER TABLE training_models RENAME COLUMN updater_path TO trainer_path;
ALTER TABLE training_models DROP COLUMN collector_path;

-- Every generation that closed before this migration existed never had its
-- generation-scope hook offered. Without this backfill the first post-migration
-- `league run` would fire it across all of history -- harmless for the crop
-- surrogate, which records idempotently, but the DQN would collect a rollout
-- batch attributed to a long-finished generation.
--
-- `hook_id` is a surrogate key; the ledger is looked up by the natural
-- (model_id, scope, generation_id, round_index) tuple, so these ids only have
-- to be unique.
INSERT INTO model_round_hooks (
    hook_id, model_id, session_id, generation_id, scope, round_index,
    status, batches_recorded, completed_at
)
SELECT
    'backfill_' || generations.generation_id || '_' || training_models.model_id,
    training_models.model_id,
    generations.session_id,
    generations.generation_id,
    'generation',
    -1,
    'COMPLETE',
    0,
    current_timestamp
FROM generations CROSS JOIN training_models
WHERE generations.status = 'COMPLETE';

INSERT OR IGNORE INTO schema_migrations (version) VALUES (4);
