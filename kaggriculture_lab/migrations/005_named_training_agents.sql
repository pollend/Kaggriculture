-- Give each trainable model an explicit agent type/name identity. Existing
-- model IDs remain the default instance so no persisted history is renamed.
ALTER TABLE training_models ADD COLUMN agent_type VARCHAR;
ALTER TABLE training_models ADD COLUMN agent_name VARCHAR;
ALTER TABLE training_models ADD COLUMN declared_model_id VARCHAR;

UPDATE training_models
SET agent_type = family_id,
    agent_name = 'default',
    declared_model_id = model_id
WHERE agent_type IS NULL;

INSERT INTO schema_migrations (version) VALUES (5);
