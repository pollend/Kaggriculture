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
    created_run_id VARCHAR
);

CREATE TABLE IF NOT EXISTS episodes (
    match_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    stage VARCHAR NOT NULL,
    seed BIGINT NOT NULL,
    seat0_id VARCHAR NOT NULL,
    seat1_id VARCHAR NOT NULL,
    seat0_spec VARCHAR NOT NULL,
    seat1_spec VARCHAR NOT NULL,
    steps INTEGER NOT NULL,
    wallclock DOUBLE NOT NULL,
    failed BOOLEAN NOT NULL,
    replay_path VARCHAR
);

CREATE TABLE IF NOT EXISTS episode_players (
    match_id VARCHAR NOT NULL,
    run_id VARCHAR NOT NULL,
    stage VARCHAR NOT NULL,
    seed BIGINT NOT NULL,
    seat INTEGER NOT NULL,
    strategy_id VARCHAR NOT NULL,
    opponent_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    failed BOOLEAN NOT NULL,
    reward DOUBLE,
    final_money DOUBLE,
    opponent_money DOUBLE,
    money_difference DOUBLE,
    win DOUBLE,
    total_unit_actions BIGINT,
    productive_actions BIGINT,
    movement_actions BIGINT,
    idle_actions BIGINT,
    productive_action_percentage DOUBLE,
    movement_percentage DOUBLE,
    idle_percentage DOUBLE,
    plants_lost_inferred BIGINT,
    final_shed_items BIGINT,
    final_carried_items BIGINT,
    metrics_json JSON NOT NULL,
    PRIMARY KEY (match_id, seat)
);

CREATE TABLE IF NOT EXISTS ratings (
    run_id VARCHAR NOT NULL,
    strategy_id VARCHAR NOT NULL,
    rating DOUBLE NOT NULL,
    games INTEGER NOT NULL,
    PRIMARY KEY (run_id, strategy_id)
);

CREATE TABLE IF NOT EXISTS models (
    model_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    artifact_path VARCHAR NOT NULL,
    report_json JSON NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS proposals (
    run_id VARCHAR NOT NULL,
    strategy_id VARCHAR NOT NULL,
    rank INTEGER NOT NULL,
    acquisition VARCHAR NOT NULL,
    predicted_quality DOUBLE,
    predicted_matchup DOUBLE,
    uncertainty DOUBLE,
    PRIMARY KEY (run_id, strategy_id)
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT current_timestamp
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

CREATE TABLE IF NOT EXISTS agent_revisions (
    revision_id VARCHAR PRIMARY KEY,
    family_id VARCHAR NOT NULL,
    strategy_id VARCHAR,
    entry_point VARCHAR NOT NULL,
    agent_spec VARCHAR NOT NULL,
    code_fingerprint VARCHAR NOT NULL,
    parameter_schema_version INTEGER NOT NULL,
    params_json JSON,
    checkpoint_id VARCHAR,
    checkpoint_digest VARCHAR,
    source VARCHAR NOT NULL,
    episode_evaluator VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS training_sessions (
    session_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    seed BIGINT NOT NULL,
    config_json JSON NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS generations (
    generation_id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    generation_index INTEGER NOT NULL,
    status VARCHAR NOT NULL,
    model_checkpoint_id VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp,
    completed_at TIMESTAMP,
    UNIQUE (session_id, generation_index)
);

CREATE TABLE IF NOT EXISTS generation_members (
    generation_id VARCHAR NOT NULL,
    revision_id VARCHAR NOT NULL,
    role VARCHAR NOT NULL,
    initial_rating DOUBLE NOT NULL,
    initial_deviation DOUBLE NOT NULL,
    initial_volatility DOUBLE NOT NULL,
    PRIMARY KEY (generation_id, revision_id)
);

CREATE TABLE IF NOT EXISTS league_pairings (
    pairing_id VARCHAR PRIMARY KEY,
    generation_id VARCHAR NOT NULL,
    round_index INTEGER NOT NULL,
    pairing_kind VARCHAR NOT NULL,
    seed BIGINT NOT NULL,
    revision_a VARCHAR NOT NULL,
    revision_b VARCHAR NOT NULL,
    status VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS pairing_episodes (
    pairing_id VARCHAR NOT NULL,
    orientation INTEGER NOT NULL,
    match_id VARCHAR NOT NULL,
    PRIMARY KEY (pairing_id, orientation)
);

CREATE TABLE IF NOT EXISTS episode_payloads (
    match_id VARCHAR PRIMARY KEY,
    replay_gzip BLOB NOT NULL,
    payload_sha256 VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS pairing_results (
    pairing_id VARCHAR PRIMARY KEY,
    paired_margin DOUBLE,
    outcome_a DOUBLE,
    failed BOOLEAN NOT NULL,
    episodes_completed INTEGER NOT NULL,
    metrics_json JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS episode_evaluations (
    evaluation_id VARCHAR PRIMARY KEY,
    match_id VARCHAR NOT NULL,
    revision_id VARCHAR NOT NULL,
    seat INTEGER NOT NULL,
    evaluator_path VARCHAR NOT NULL,
    evaluator_fingerprint VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    result_json JSON,
    error_text VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp,
    UNIQUE (match_id, revision_id, seat, evaluator_fingerprint)
);

CREATE TABLE IF NOT EXISTS generation_stats (
    generation_id VARCHAR NOT NULL,
    revision_id VARCHAR NOT NULL,
    games INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    draws INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    mean_margin DOUBLE,
    median_margin DOUBLE,
    margin_ci_low DOUBLE,
    margin_ci_high DOUBLE,
    opponent_diversity INTEGER NOT NULL,
    PRIMARY KEY (generation_id, revision_id)
);

CREATE TABLE IF NOT EXISTS rating_snapshots (
    generation_id VARCHAR NOT NULL,
    revision_id VARCHAR NOT NULL,
    rating DOUBLE NOT NULL,
    deviation DOUBLE NOT NULL,
    volatility DOUBLE NOT NULL,
    games INTEGER NOT NULL,
    is_frozen BOOLEAN NOT NULL,
    PRIMARY KEY (generation_id, revision_id)
);

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

CREATE TABLE IF NOT EXISTS lineage (
    generation_id VARCHAR NOT NULL,
    child_revision_id VARCHAR NOT NULL,
    parent_a_revision_id VARCHAR,
    parent_b_revision_id VARCHAR,
    creation_method VARCHAR NOT NULL,
    PRIMARY KEY (generation_id, child_revision_id)
);

-- Incremental model training. Large datasets and checkpoints remain immutable
-- filesystem artifacts; this catalog owns identity, lineage, and decisions.
CREATE TABLE IF NOT EXISTS training_models (
    model_id VARCHAR PRIMARY KEY,
    family_id VARCHAR NOT NULL,
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

CREATE TABLE IF NOT EXISTS training_batches (
    model_id VARCHAR NOT NULL,
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

CREATE TABLE IF NOT EXISTS model_updates (
    update_id VARCHAR PRIMARY KEY,
    model_id VARCHAR NOT NULL,
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
    model_id VARCHAR NOT NULL,
    update_id VARCHAR NOT NULL,
    parent_checkpoint_id VARCHAR,
    trainer_artifact_path VARCHAR NOT NULL,
    inference_artifact_path VARCHAR NOT NULL,
    artifact_sha256 VARCHAR NOT NULL,
    dataset_sha256 VARCHAR NOT NULL,
    metrics_json JSON NOT NULL,
    gate_json JSON NOT NULL,
    promoted BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS model_evaluations (
    evaluation_id VARCHAR PRIMARY KEY,
    update_id VARCHAR NOT NULL,
    model_id VARCHAR NOT NULL,
    evaluation_kind VARCHAR NOT NULL,
    opponent_checkpoint_id VARCHAR,
    metrics_json JSON NOT NULL,
    passed BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

-- One row per round hook a model's trainer has been offered. The league fires
-- `on_round` once per closed round and once per closed generation; the hook may
-- roll out episodes for minutes, so "already done" cannot be re-derived from
-- pairing status -- resuming a half-finished generation would re-fire every
-- earlier round. `round_index` uses -1 for generation scope rather than NULL:
-- DuckDB's UNIQUE does not dedupe NULL-bearing keys, so a nullable column would
-- silently admit two generation rows.
CREATE TABLE IF NOT EXISTS model_round_hooks (
    hook_id VARCHAR PRIMARY KEY,
    model_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    generation_id VARCHAR NOT NULL,
    scope VARCHAR NOT NULL,
    round_index INTEGER NOT NULL,
    status VARCHAR NOT NULL,
    batches_recorded INTEGER NOT NULL DEFAULT 0,
    error_text VARCHAR,
    started_at TIMESTAMP DEFAULT current_timestamp,
    completed_at TIMESTAMP,
    UNIQUE (model_id, scope, generation_id, round_index)
);

CREATE TABLE IF NOT EXISTS session_manifests (
    session_id VARCHAR PRIMARY KEY,
    manifest_path VARCHAR NOT NULL,
    manifest_sha256 VARCHAR NOT NULL,
    manifest_json JSON NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS policy_mixtures (
    mixture_id VARCHAR PRIMARY KEY,
    model_id VARCHAR NOT NULL,
    generation_index INTEGER NOT NULL,
    max_members INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS policy_mixture_members (
    mixture_id VARCHAR NOT NULL,
    revision_id VARCHAR NOT NULL,
    ordinal INTEGER NOT NULL,
    weight DOUBLE NOT NULL,
    PRIMARY KEY (mixture_id, revision_id)
);

CREATE TABLE IF NOT EXISTS pairing_policy_assignments (
    pairing_id VARCHAR NOT NULL,
    mixture_id VARCHAR,
    composite_revision_id VARCHAR NOT NULL,
    selected_revision_id VARCHAR NOT NULL,
    opponent_revision_id VARCHAR NOT NULL,
    PRIMARY KEY (pairing_id, composite_revision_id)
);

CREATE TABLE IF NOT EXISTS episode_training_artifacts (
    artifact_id VARCHAR PRIMARY KEY,
    match_id VARCHAR NOT NULL,
    revision_id VARCHAR NOT NULL,
    policy_revision_id VARCHAR NOT NULL,
    role VARCHAR NOT NULL,
    artifact_path VARCHAR NOT NULL,
    artifact_sha256 VARCHAR NOT NULL,
    sample_count BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS training_batch_artifacts (
    batch_id VARCHAR NOT NULL,
    artifact_id VARCHAR NOT NULL,
    PRIMARY KEY (batch_id, artifact_id)
);

-- The version stamp lives in `ExperimentStore._initialize`, not here: this file
-- describes the current shape and is replayed on every open, so stamping from
-- inside it would mark an unmigrated database as current.
