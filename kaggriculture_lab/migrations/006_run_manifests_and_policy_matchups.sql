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

INSERT INTO schema_migrations (version) VALUES (6);
