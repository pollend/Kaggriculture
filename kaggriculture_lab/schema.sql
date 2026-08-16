-- Tables are declared in topological order: `database.py` replays this whole
-- file on every connect and relies on `CREATE TABLE IF NOT EXISTS`, so inline
-- REFERENCES can only name a table that has already appeared above. The one
-- genuine cycle (training_models/model_updates <-> model_training_checkpoints)
-- is closed by the guarded ALTER statements at the bottom.
--
-- Seat identity columns (`episodes.seat0_id`, `episode_players.strategy_id`,
-- `ratings.strategy_id`, ...) are deliberately unconstrained: they carry
-- revision ids in the league flow, strategy ids in the surrogate flow, and
-- `builtin:*` sentinels in both, so no single parent table exists.

CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp,
    status VARCHAR NOT NULL,
    seed BIGINT NOT NULL,
    config_json JSON NOT NULL,
    versions_json JSON NOT NULL,
    git_commit VARCHAR
);

CREATE TABLE IF NOT EXISTS strategies (
    strategy_id VARCHAR PRIMARY KEY,
    params_json JSON NOT NULL,
    source VARCHAR NOT NULL,
    created_run_id VARCHAR REFERENCES runs (run_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS agent_families (
    family_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    manifest_path VARCHAR NOT NULL,
    entry_point VARCHAR NOT NULL,
    training_adapter VARCHAR,
    episode_evaluator VARCHAR,
    parameter_schema_version INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

-- Incremental model training. Large datasets and checkpoints remain immutable
-- filesystem artifacts; this database owns identity, lineage, and decisions.
CREATE TABLE IF NOT EXISTS training_models (
    model_id VARCHAR PRIMARY KEY,
    family_id VARCHAR NOT NULL REFERENCES agent_families (family_id) ON DELETE CASCADE,
    agent_type VARCHAR,
    agent_name VARCHAR,
    declared_model_id VARCHAR,
    trainer_path VARCHAR NOT NULL,
    min_new_batches INTEGER NOT NULL,
    config_json JSON NOT NULL,
    active_checkpoint_id VARCHAR,
    last_attempted_batch_seq BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT current_timestamp,
    updated_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS model_updates (
    update_id VARCHAR PRIMARY KEY,
    model_id VARCHAR NOT NULL REFERENCES training_models (model_id) ON DELETE CASCADE,
    parent_checkpoint_id VARCHAR,
    cutoff_batch_seq BIGINT NOT NULL,
    new_batch_count INTEGER NOT NULL,
    dataset_sha256 VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    candidate_checkpoint_id VARCHAR,
    metrics_json JSON,
    gate_json JSON,
    error_text VARCHAR,
    started_at TIMESTAMP DEFAULT current_timestamp,
    completed_at TIMESTAMP,
    UNIQUE (model_id, cutoff_batch_seq)
);

CREATE TABLE IF NOT EXISTS model_training_checkpoints (
    checkpoint_id VARCHAR PRIMARY KEY,
    model_id VARCHAR NOT NULL REFERENCES training_models (model_id) ON DELETE CASCADE,
    update_id VARCHAR NOT NULL REFERENCES model_updates (update_id) ON DELETE CASCADE,
    parent_checkpoint_id VARCHAR REFERENCES model_training_checkpoints (checkpoint_id) ON DELETE SET NULL,
    trainer_artifact_path VARCHAR NOT NULL,
    inference_artifact_path VARCHAR NOT NULL,
    artifact_sha256 VARCHAR NOT NULL,
    dataset_sha256 VARCHAR NOT NULL,
    metrics_json JSON NOT NULL,
    gate_json JSON NOT NULL,
    promoted BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

-- The mature half of a population-trained model: released checkpoints that have
-- stood long enough to be worth copying, serving as the reseed source and the
-- self-play mirror.
--
-- Entry is no longer tied to a slot ending. A release enters once it has held its
-- slot for `population_entry_updates` consecutive updates unbeaten, and a
-- graduating slot contributes whatever the age. The lag is the point: a release
-- copyable the moment it is promoted makes whichever slot released last the
-- ancestor for whichever slot reseeds next, and the roster converges on clones.
--
-- Keyed on `declared_model_id` rather than `model_id` because that is what the
-- learners share. A roster of ten named instances of one declared model
-- (`decl@learner_00` .. `decl@learner_09`, see `effective_model_id`) is ten rows
-- in `training_models` and one population, and a member graduating out of one
-- slot is what the next slot reseeds from.
--
-- A member is live while `retired_at` is NULL. Retirement is a membership change
-- and never a delete: the row and its artifacts stay, so a run can be read back
-- afterwards to ask what was dropped and at what score.
CREATE TABLE IF NOT EXISTS population_members (
    member_id VARCHAR PRIMARY KEY,
    declared_model_id VARCHAR NOT NULL,
    model_id VARCHAR NOT NULL REFERENCES training_models (model_id) ON DELETE CASCADE,
    checkpoint_id VARCHAR NOT NULL REFERENCES model_training_checkpoints (checkpoint_id) ON DELETE CASCADE,
    -- Windowed median of the checkpoint's benchmark money, discounted by the
    -- policy's trading rate so a hoarder cannot outrank a trader, and refreshed
    -- on each upsert rather than frozen at graduation.
    -- A single arena series is a draw and not a number here: `falloff_v1`
    -- measured 61 of them on one unchanged checkpoint, running 0.93 to 427.
    score DOUBLE PRECISION NOT NULL,
    -- The strategy fingerprint the group's reserved berths are keyed on: crops
    -- planted, quadrants bought, the undiscounted money both were measured with,
    -- and the archive cell they bin to. Nullable, and NULL is not zero -- a
    -- member graduated before this existed has no fingerprint, and filing it into
    -- a cell on that basis would hand a reserved berth to an unmeasured row.
    behavior_json JSONB,
    graduated_at TIMESTAMP DEFAULT current_timestamp,
    retired_at TIMESTAMP,
    retired_reason VARCHAR
);

CREATE INDEX IF NOT EXISTS population_members_live_idx
    ON population_members (declared_model_id, retired_at, score);

CREATE TABLE IF NOT EXISTS agent_revisions (
    revision_id VARCHAR PRIMARY KEY,
    family_id VARCHAR NOT NULL REFERENCES agent_families (family_id) ON DELETE CASCADE,
    strategy_id VARCHAR REFERENCES strategies (strategy_id) ON DELETE SET NULL,
    entry_point VARCHAR NOT NULL,
    agent_spec VARCHAR NOT NULL,
    code_fingerprint VARCHAR NOT NULL,
    parameter_schema_version INTEGER NOT NULL,
    params_json JSON,
    checkpoint_id VARCHAR REFERENCES model_training_checkpoints (checkpoint_id) ON DELETE SET NULL,
    checkpoint_digest VARCHAR,
    source VARCHAR NOT NULL,
    episode_evaluator VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS training_sessions (
    session_id VARCHAR PRIMARY KEY,
    league VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    seed BIGINT NOT NULL,
    config_json JSON NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

-- `model_checkpoint_id` predates `model_training_checkpoints` and is never
-- written, so it carries no reference.
CREATE TABLE IF NOT EXISTS generations (
    generation_id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL REFERENCES training_sessions (session_id) ON DELETE CASCADE,
    generation_index INTEGER NOT NULL,
    status VARCHAR NOT NULL,
    model_checkpoint_id VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp,
    completed_at TIMESTAMP,
    UNIQUE (session_id, generation_index)
);

CREATE TABLE IF NOT EXISTS episodes (
    match_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    stage VARCHAR NOT NULL,
    seed BIGINT NOT NULL,
    seat0_id VARCHAR NOT NULL,
    seat1_id VARCHAR NOT NULL,
    seat0_spec VARCHAR NOT NULL,
    seat1_spec VARCHAR NOT NULL,
    steps INTEGER NOT NULL,
    wallclock DOUBLE PRECISION NOT NULL,
    failed BOOLEAN NOT NULL,
    replay_path VARCHAR
);

CREATE TABLE IF NOT EXISTS episode_players (
    match_id VARCHAR NOT NULL REFERENCES episodes (match_id) ON DELETE CASCADE,
    run_id VARCHAR NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    stage VARCHAR NOT NULL,
    seed BIGINT NOT NULL,
    seat INTEGER NOT NULL,
    strategy_id VARCHAR NOT NULL,
    opponent_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    failed BOOLEAN NOT NULL,
    reward DOUBLE PRECISION,
    final_money DOUBLE PRECISION,
    opponent_money DOUBLE PRECISION,
    money_difference DOUBLE PRECISION,
    win DOUBLE PRECISION,
    total_unit_actions BIGINT,
    productive_actions BIGINT,
    movement_actions BIGINT,
    idle_actions BIGINT,
    productive_action_percentage DOUBLE PRECISION,
    movement_percentage DOUBLE PRECISION,
    idle_percentage DOUBLE PRECISION,
    plants_lost_inferred BIGINT,
    final_shed_items BIGINT,
    final_carried_items BIGINT,
    metrics_json JSON NOT NULL,
    PRIMARY KEY (match_id, seat)
);

CREATE TABLE IF NOT EXISTS episode_payloads (
    match_id VARCHAR PRIMARY KEY REFERENCES episodes (match_id) ON DELETE CASCADE,
    replay_gzip BYTEA NOT NULL,
    payload_sha256 VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS episode_evaluations (
    evaluation_id VARCHAR PRIMARY KEY,
    match_id VARCHAR NOT NULL REFERENCES episodes (match_id) ON DELETE CASCADE,
    revision_id VARCHAR NOT NULL REFERENCES agent_revisions (revision_id) ON DELETE CASCADE,
    seat INTEGER NOT NULL,
    evaluator_path VARCHAR NOT NULL,
    evaluator_fingerprint VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    result_json JSON,
    error_text VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp,
    UNIQUE (match_id, revision_id, seat, evaluator_fingerprint)
);

CREATE TABLE IF NOT EXISTS episode_training_artifacts (
    artifact_id VARCHAR PRIMARY KEY,
    match_id VARCHAR NOT NULL REFERENCES episodes (match_id) ON DELETE CASCADE,
    revision_id VARCHAR NOT NULL REFERENCES agent_revisions (revision_id) ON DELETE CASCADE,
    -- Unconstrained: the agent mints this itself from the weights it rolled out
    -- with (`sp_psro_job_scorer/training/collect.py` builds `beta-<digest>`), so
    -- it identifies a policy, not a league revision.
    policy_revision_id VARCHAR NOT NULL,
    role VARCHAR NOT NULL,
    artifact_path VARCHAR NOT NULL,
    artifact_sha256 VARCHAR NOT NULL,
    sample_count BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS league_pairings (
    pairing_id VARCHAR PRIMARY KEY,
    generation_id VARCHAR NOT NULL REFERENCES generations (generation_id) ON DELETE CASCADE,
    round_index INTEGER NOT NULL,
    pairing_kind VARCHAR NOT NULL,
    seed BIGINT NOT NULL,
    revision_a VARCHAR NOT NULL REFERENCES agent_revisions (revision_id) ON DELETE CASCADE,
    revision_b VARCHAR NOT NULL REFERENCES agent_revisions (revision_id) ON DELETE CASCADE,
    status VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS pairing_episodes (
    pairing_id VARCHAR NOT NULL REFERENCES league_pairings (pairing_id) ON DELETE CASCADE,
    orientation INTEGER NOT NULL,
    match_id VARCHAR NOT NULL REFERENCES episodes (match_id) ON DELETE CASCADE,
    PRIMARY KEY (pairing_id, orientation)
);

CREATE TABLE IF NOT EXISTS pairing_results (
    pairing_id VARCHAR PRIMARY KEY REFERENCES league_pairings (pairing_id) ON DELETE CASCADE,
    paired_margin DOUBLE PRECISION,
    outcome_a DOUBLE PRECISION,
    failed BOOLEAN NOT NULL,
    episodes_completed INTEGER NOT NULL,
    metrics_json JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS policy_mixtures (
    mixture_id VARCHAR PRIMARY KEY,
    model_id VARCHAR NOT NULL REFERENCES training_models (model_id) ON DELETE CASCADE,
    generation_index INTEGER NOT NULL,
    max_members INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS policy_mixture_members (
    mixture_id VARCHAR NOT NULL REFERENCES policy_mixtures (mixture_id) ON DELETE CASCADE,
    revision_id VARCHAR NOT NULL REFERENCES agent_revisions (revision_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    weight DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (mixture_id, revision_id)
);

CREATE TABLE IF NOT EXISTS pairing_policy_assignments (
    pairing_id VARCHAR NOT NULL REFERENCES league_pairings (pairing_id) ON DELETE CASCADE,
    mixture_id VARCHAR REFERENCES policy_mixtures (mixture_id) ON DELETE SET NULL,
    -- Unconstrained: `model_trainers` labels the mixture-as-an-opponent
    -- `<mixture_id>:composite`, which stands for a weighted blend rather than
    -- any one revision, so it is never registered as one.
    composite_revision_id VARCHAR NOT NULL,
    selected_revision_id VARCHAR NOT NULL REFERENCES agent_revisions (revision_id) ON DELETE CASCADE,
    opponent_revision_id VARCHAR NOT NULL REFERENCES agent_revisions (revision_id) ON DELETE CASCADE,
    PRIMARY KEY (pairing_id, composite_revision_id)
);

CREATE TABLE IF NOT EXISTS generation_members (
    generation_id VARCHAR NOT NULL REFERENCES generations (generation_id) ON DELETE CASCADE,
    revision_id VARCHAR NOT NULL REFERENCES agent_revisions (revision_id) ON DELETE CASCADE,
    role VARCHAR NOT NULL,
    initial_rating DOUBLE PRECISION NOT NULL,
    initial_deviation DOUBLE PRECISION NOT NULL,
    initial_volatility DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (generation_id, revision_id)
);

CREATE TABLE IF NOT EXISTS rating_snapshots (
    generation_id VARCHAR NOT NULL REFERENCES generations (generation_id) ON DELETE CASCADE,
    revision_id VARCHAR NOT NULL REFERENCES agent_revisions (revision_id) ON DELETE CASCADE,
    rating DOUBLE PRECISION NOT NULL,
    deviation DOUBLE PRECISION NOT NULL,
    volatility DOUBLE PRECISION NOT NULL,
    games INTEGER NOT NULL,
    is_frozen BOOLEAN NOT NULL,
    PRIMARY KEY (generation_id, revision_id)
);

CREATE TABLE IF NOT EXISTS generation_stats (
    generation_id VARCHAR NOT NULL REFERENCES generations (generation_id) ON DELETE CASCADE,
    revision_id VARCHAR NOT NULL REFERENCES agent_revisions (revision_id) ON DELETE CASCADE,
    games INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    draws INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    mean_margin DOUBLE PRECISION,
    median_margin DOUBLE PRECISION,
    margin_ci_low DOUBLE PRECISION,
    margin_ci_high DOUBLE PRECISION,
    opponent_diversity INTEGER NOT NULL,
    PRIMARY KEY (generation_id, revision_id)
);

CREATE TABLE IF NOT EXISTS lineage (
    generation_id VARCHAR NOT NULL REFERENCES generations (generation_id) ON DELETE CASCADE,
    child_revision_id VARCHAR NOT NULL REFERENCES agent_revisions (revision_id) ON DELETE CASCADE,
    parent_a_revision_id VARCHAR REFERENCES agent_revisions (revision_id) ON DELETE SET NULL,
    parent_b_revision_id VARCHAR REFERENCES agent_revisions (revision_id) ON DELETE SET NULL,
    creation_method VARCHAR NOT NULL,
    PRIMARY KEY (generation_id, child_revision_id)
);

CREATE TABLE IF NOT EXISTS model_evaluations (
    evaluation_id VARCHAR PRIMARY KEY,
    update_id VARCHAR NOT NULL REFERENCES model_updates (update_id) ON DELETE CASCADE,
    model_id VARCHAR NOT NULL REFERENCES training_models (model_id) ON DELETE CASCADE,
    evaluation_kind VARCHAR NOT NULL,
    opponent_checkpoint_id VARCHAR REFERENCES model_training_checkpoints (checkpoint_id) ON DELETE SET NULL,
    metrics_json JSON NOT NULL,
    passed BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

-- One row per round hook a model's trainer has been offered. The league fires
-- `on_round` once per closed round and once per closed generation; the hook may
-- roll out episodes for minutes, so "already done" cannot be re-derived from
-- pairing status -- resuming a half-finished generation would re-fire every
-- earlier round. `round_index` uses -1 for generation scope rather than NULL:
-- A non-null sentinel keeps generation-scope hook identity unambiguous.
-- silently admit two generation rows.
CREATE TABLE IF NOT EXISTS model_round_hooks (
    hook_id VARCHAR PRIMARY KEY,
    model_id VARCHAR NOT NULL REFERENCES training_models (model_id) ON DELETE CASCADE,
    session_id VARCHAR NOT NULL REFERENCES training_sessions (session_id) ON DELETE CASCADE,
    generation_id VARCHAR NOT NULL REFERENCES generations (generation_id) ON DELETE CASCADE,
    scope VARCHAR NOT NULL,
    round_index INTEGER NOT NULL,
    status VARCHAR NOT NULL,
    batches_recorded INTEGER NOT NULL DEFAULT 0,
    error_text VARCHAR,
    started_at TIMESTAMP DEFAULT current_timestamp,
    completed_at TIMESTAMP,
    UNIQUE (model_id, scope, generation_id, round_index)
);

CREATE TABLE IF NOT EXISTS training_batches (
    model_id VARCHAR NOT NULL REFERENCES training_models (model_id) ON DELETE CASCADE,
    batch_seq BIGINT NOT NULL,
    batch_id VARCHAR NOT NULL UNIQUE,
    source_kind VARCHAR NOT NULL,
    source_id VARCHAR NOT NULL,
    artifact_path VARCHAR,
    artifact_sha256 VARCHAR,
    sample_count BIGINT NOT NULL,
    metadata_json JSON NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (model_id, batch_seq),
    UNIQUE (model_id, source_kind, source_id)
);

CREATE TABLE IF NOT EXISTS training_batch_artifacts (
    batch_id VARCHAR NOT NULL REFERENCES training_batches (batch_id) ON DELETE CASCADE,
    artifact_id VARCHAR NOT NULL REFERENCES episode_training_artifacts (artifact_id) ON DELETE CASCADE,
    PRIMARY KEY (batch_id, artifact_id)
);

CREATE TABLE IF NOT EXISTS session_manifests (
    session_id VARCHAR PRIMARY KEY REFERENCES training_sessions (session_id) ON DELETE CASCADE,
    manifest_path VARCHAR NOT NULL,
    manifest_sha256 VARCHAR NOT NULL,
    manifest_json JSON NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS ratings (
    run_id VARCHAR NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    strategy_id VARCHAR NOT NULL,
    rating DOUBLE PRECISION NOT NULL,
    games INTEGER NOT NULL,
    PRIMARY KEY (run_id, strategy_id)
);

CREATE TABLE IF NOT EXISTS models (
    model_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    kind VARCHAR NOT NULL,
    artifact_path VARCHAR NOT NULL,
    report_json JSON NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS proposals (
    run_id VARCHAR NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    strategy_id VARCHAR NOT NULL,
    rank INTEGER NOT NULL,
    acquisition VARCHAR NOT NULL,
    predicted_quality DOUBLE PRECISION,
    predicted_matchup DOUBLE PRECISION,
    uncertainty DOUBLE PRECISION,
    PRIMARY KEY (run_id, strategy_id)
);

-- Superseded by `model_training_checkpoints`; retained so historical rows stay
-- readable. No code writes it, so it takes no references.
CREATE TABLE IF NOT EXISTS model_checkpoints (
    checkpoint_id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    generation_id VARCHAR NOT NULL,
    family_id VARCHAR NOT NULL,
    parent_checkpoint_id VARCHAR,
    artifact_path VARCHAR NOT NULL,
    artifact_sha256 VARCHAR NOT NULL,
    dataset_sha256 VARCHAR NOT NULL,
    training_cutoff TIMESTAMP NOT NULL,
    feature_schema_json JSON NOT NULL,
    metrics_json JSON NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

-- Cycle-closing references. `training_models` and `model_updates` are created
-- before `model_training_checkpoints` (their rows must exist first), so these
-- three cannot be inline. PostgreSQL has no `ADD CONSTRAINT IF NOT EXISTS`, and
-- this file is replayed on every connect, so each is guarded individually --
-- one block per constraint, so a duplicate on the first does not skip the rest.
DO $$
BEGIN
    ALTER TABLE training_models
        ADD CONSTRAINT training_models_active_checkpoint_fkey
        FOREIGN KEY (active_checkpoint_id)
        REFERENCES model_training_checkpoints (checkpoint_id) ON DELETE SET NULL;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE model_updates
        ADD CONSTRAINT model_updates_parent_checkpoint_fkey
        FOREIGN KEY (parent_checkpoint_id)
        REFERENCES model_training_checkpoints (checkpoint_id) ON DELETE SET NULL;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE model_updates
        ADD CONSTRAINT model_updates_candidate_checkpoint_fkey
        FOREIGN KEY (candidate_checkpoint_id)
        REFERENCES model_training_checkpoints (checkpoint_id) ON DELETE SET NULL;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- PostgreSQL indexes the referenced side of a foreign key, never the
-- referencing side. Every child key below that is not already the leading
-- column of a PRIMARY KEY or UNIQUE constraint needs its own index: cascading
-- deletes scan them, and so do the dashboard and progress queries.
CREATE INDEX IF NOT EXISTS strategies_created_run_idx ON strategies (created_run_id);
CREATE INDEX IF NOT EXISTS training_models_family_idx ON training_models (family_id);
CREATE INDEX IF NOT EXISTS training_models_active_checkpoint_idx ON training_models (active_checkpoint_id);
CREATE INDEX IF NOT EXISTS model_updates_parent_checkpoint_idx ON model_updates (parent_checkpoint_id);
CREATE INDEX IF NOT EXISTS model_updates_candidate_checkpoint_idx ON model_updates (candidate_checkpoint_id);
CREATE INDEX IF NOT EXISTS model_training_checkpoints_model_idx ON model_training_checkpoints (model_id);
CREATE INDEX IF NOT EXISTS model_training_checkpoints_update_idx ON model_training_checkpoints (update_id);
CREATE INDEX IF NOT EXISTS model_training_checkpoints_parent_idx ON model_training_checkpoints (parent_checkpoint_id);
CREATE INDEX IF NOT EXISTS agent_revisions_family_idx ON agent_revisions (family_id);
CREATE INDEX IF NOT EXISTS agent_revisions_strategy_idx ON agent_revisions (strategy_id);
CREATE INDEX IF NOT EXISTS agent_revisions_checkpoint_idx ON agent_revisions (checkpoint_id);
CREATE INDEX IF NOT EXISTS episodes_run_idx ON episodes (run_id);
CREATE INDEX IF NOT EXISTS episode_players_run_idx ON episode_players (run_id);
CREATE INDEX IF NOT EXISTS episode_evaluations_revision_idx ON episode_evaluations (revision_id);
CREATE INDEX IF NOT EXISTS episode_training_artifacts_match_idx ON episode_training_artifacts (match_id);
CREATE INDEX IF NOT EXISTS episode_training_artifacts_revision_idx ON episode_training_artifacts (revision_id);
CREATE INDEX IF NOT EXISTS league_pairings_generation_idx ON league_pairings (generation_id);
CREATE INDEX IF NOT EXISTS league_pairings_revision_a_idx ON league_pairings (revision_a);
CREATE INDEX IF NOT EXISTS league_pairings_revision_b_idx ON league_pairings (revision_b);
CREATE INDEX IF NOT EXISTS pairing_episodes_match_idx ON pairing_episodes (match_id);
CREATE INDEX IF NOT EXISTS policy_mixtures_model_idx ON policy_mixtures (model_id);
CREATE INDEX IF NOT EXISTS policy_mixture_members_revision_idx ON policy_mixture_members (revision_id);
CREATE INDEX IF NOT EXISTS pairing_policy_assignments_mixture_idx ON pairing_policy_assignments (mixture_id);
CREATE INDEX IF NOT EXISTS pairing_policy_assignments_selected_idx ON pairing_policy_assignments (selected_revision_id);
CREATE INDEX IF NOT EXISTS pairing_policy_assignments_opponent_idx ON pairing_policy_assignments (opponent_revision_id);
CREATE INDEX IF NOT EXISTS generation_members_revision_idx ON generation_members (revision_id);
CREATE INDEX IF NOT EXISTS rating_snapshots_revision_idx ON rating_snapshots (revision_id);
CREATE INDEX IF NOT EXISTS generation_stats_revision_idx ON generation_stats (revision_id);
CREATE INDEX IF NOT EXISTS lineage_child_revision_idx ON lineage (child_revision_id);
CREATE INDEX IF NOT EXISTS lineage_parent_a_revision_idx ON lineage (parent_a_revision_id);
CREATE INDEX IF NOT EXISTS lineage_parent_b_revision_idx ON lineage (parent_b_revision_id);
CREATE INDEX IF NOT EXISTS model_evaluations_update_idx ON model_evaluations (update_id);
CREATE INDEX IF NOT EXISTS model_evaluations_model_idx ON model_evaluations (model_id);
CREATE INDEX IF NOT EXISTS model_evaluations_opponent_checkpoint_idx ON model_evaluations (opponent_checkpoint_id);
CREATE INDEX IF NOT EXISTS model_round_hooks_session_idx ON model_round_hooks (session_id);
CREATE INDEX IF NOT EXISTS model_round_hooks_generation_idx ON model_round_hooks (generation_id);
CREATE INDEX IF NOT EXISTS training_batch_artifacts_artifact_idx ON training_batch_artifacts (artifact_id);
CREATE INDEX IF NOT EXISTS models_run_idx ON models (run_id);
