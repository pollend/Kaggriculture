"""Model administration belongs to each agent, not the generic lab CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.configurable_crop_surrogate.cli import main as crop_cli
from agents.dqn_job_scorer.cli import main as dqn_cli
from agents.receding_horizon_planner.cli import main as planner_cli
from kaggriculture_lab.cli import build_parser
from kaggriculture_lab.storage import ExperimentStore
from tests.postgres_support import postgres_database, seed_reference_rows


@pytest.mark.parametrize(
    ("command", "model_id"),
    [
        (crop_cli, "configurable_crop_surrogate.surrogate"),
        (dqn_cli, "dqn_job_scorer.adaptive_hybrid_q_v2"),
        (planner_cli, "receding_horizon_planner.surrogate"),
    ],
)
def test_agent_status_registers_only_its_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command, model_id: str
):
    database = postgres_database(tmp_path, "status")
    assert command([
        "model", "status", "--database-url", database.url,
    ]) == 0
    assert json.loads(capsys.readouterr().out) == [
        {
            "active_checkpoint_id": None,
            "last_attempted_batch_seq": 0,
            "latest_batch_seq": 0,
            "min_new_batches": 20 if model_id == "dqn_job_scorer.adaptive_hybrid_q_v2" else 2,
            "model_id": model_id,
            "new_batches": 0,
        }
    ]
    with ExperimentStore(database) as store:
        assert store.connection.execute("SELECT model_id FROM training_models").fetchall() == [(model_id,)]


@pytest.mark.parametrize("command", [crop_cli, dqn_cli, planner_cli])
def test_agent_update_defers_without_new_batches(tmp_path: Path, capsys: pytest.CaptureFixture[str], command):
    database = postgres_database(tmp_path, "deferred")
    assert (
        command(
            [
                "model",
                "update",
                "--database-url",
                database.url,
                "--artifacts",
                str(tmp_path / "artifacts"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "deferred"


def test_agent_model_commands_address_a_named_instance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    database = postgres_database(tmp_path, "named-status")
    assert dqn_cli(
        [
            "model", "status", "--name", "dqn05",
            "--database-url", database.url,
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["model_id"] == "dqn_job_scorer.adaptive_hybrid_q_v2@dqn05"
    with ExperimentStore(database) as store:
        row = store.connection.execute(
            "SELECT agent_type, agent_name, config_json FROM training_models"
        ).fetchone()
    assert row[:2] == ("dqn_job_scorer", "dqn05")
    assert json.loads(row[2])["name"] == "dqn05"


def test_dqn_export_requires_confirmation_and_copies_active_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    database = postgres_database(tmp_path, "export")
    assert dqn_cli([
        "model", "status", "--database-url", database.url,
    ]) == 0
    capsys.readouterr()

    source = tmp_path / "checkpoint.npz"
    source.write_bytes(b"reviewed weights")
    with ExperimentStore(database) as store:
        store.connection.execute(
            """
            INSERT INTO model_updates (
                update_id, model_id, cutoff_batch_seq, new_batch_count,
                dataset_sha256, status
            ) VALUES ('update', 'dqn_job_scorer.adaptive_hybrid_q_v2', 1, 1,
                      'dataset', 'PROMOTED')
            """
        )
        store.connection.execute(
            """
            INSERT INTO model_training_checkpoints (
                checkpoint_id, model_id, update_id, trainer_artifact_path,
                inference_artifact_path, artifact_sha256, dataset_sha256,
                metrics_json, gate_json, promoted
            ) VALUES ('checkpoint', 'dqn_job_scorer.adaptive_hybrid_q_v2', 'update', %s, %s,
                      'artifact', 'dataset', '{}', '{"passed": true}', true)
            """,
            [str(source), str(source)],
        )
        store.connection.execute(
            """
            UPDATE training_models SET active_checkpoint_id = 'checkpoint'
            WHERE model_id = 'dqn_job_scorer.adaptive_hybrid_q_v2'
            """
        )
        store.connection.execute(
            "UPDATE model_updates SET candidate_checkpoint_id = 'checkpoint'"
        )

    output = tmp_path / "released.npz"
    arguments = [
        "model",
        "export-active",
        "--database-url",
        database.url,
        "--output",
        str(output),
    ]
    with pytest.raises(ValueError, match="confirm-release"):
        dqn_cli(arguments)
    assert not output.exists()

    assert dqn_cli([*arguments, "--confirm-release"]) == 0
    assert output.read_bytes() == source.read_bytes()


def test_dqn_export_best_copies_highest_benchmark_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    database = postgres_database(tmp_path, "export-best")
    assert dqn_cli([
        "model", "status", "--database-url", database.url,
    ]) == 0
    capsys.readouterr()

    low_source = tmp_path / "checkpoint-low.npz"
    high_source = tmp_path / "checkpoint-high.npz"
    low_source.write_bytes(b"latest accepted weights")
    high_source.write_bytes(b"best benchmark weights")
    with ExperimentStore(database) as store:
        store.connection.execute(
            """
            INSERT INTO model_updates (
                update_id, model_id, cutoff_batch_seq, new_batch_count,
                dataset_sha256, status
            ) VALUES
                ('update-low', 'dqn_job_scorer.adaptive_hybrid_q_v2', 20, 20,
                 'dataset-low', 'PROMOTED'),
                ('update-high', 'dqn_job_scorer.adaptive_hybrid_q_v2', 10, 10,
                 'dataset-high', 'REJECTED')
            """
        )
        store.connection.execute(
            """
            INSERT INTO model_training_checkpoints (
                checkpoint_id, model_id, update_id, trainer_artifact_path,
                inference_artifact_path, artifact_sha256, dataset_sha256,
                metrics_json, gate_json, promoted
            ) VALUES
                ('checkpoint-low', 'dqn_job_scorer.adaptive_hybrid_q_v2',
                 'update-low', %s, %s, 'artifact-low', 'dataset-low', '{}',
                 '{"passed": true, "benchmark_score": {"candidate_money": 1200.0}}',
                 true),
                ('checkpoint-high', 'dqn_job_scorer.adaptive_hybrid_q_v2',
                 'update-high', %s, %s, 'artifact-high', 'dataset-high', '{}',
                 '{"passed": false, "benchmark_score": {"candidate_money": 9000.0}}',
                 false)
            """,
            [str(low_source), str(low_source), str(high_source), str(high_source)],
        )
        store.connection.execute(
            """
            UPDATE model_updates
            SET candidate_checkpoint_id = replace(update_id, 'update', 'checkpoint')
            """
        )
        store.connection.execute(
            """
            UPDATE training_models SET active_checkpoint_id = 'checkpoint-low'
            WHERE model_id = 'dqn_job_scorer.adaptive_hybrid_q_v2'
            """
        )

    output = tmp_path / "released-best.npz"
    arguments = [
        "model",
        "export-best",
        "--database-url",
        database.url,
        "--output",
        str(output),
    ]
    with pytest.raises(ValueError, match="confirm-release"):
        dqn_cli(arguments)
    assert not output.exists()

    assert dqn_cli([*arguments, "--confirm-release"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "checkpoint_id": "checkpoint-high",
        "metric": "benchmark-money",
        "output": str(output),
        "score": 9000.0,
    }
    assert output.read_bytes() == high_source.read_bytes()


def test_export_rejects_active_checkpoint_that_failed_release_gate(tmp_path: Path):
    from kaggriculture_lab.model_admin import export_active_model

    database = postgres_database(tmp_path, "blocked-export")
    dqn_cli([
        "model", "status", "--database-url", database.url,
    ])
    source = tmp_path / "checkpoint.npz"
    source.write_bytes(b"unreviewed weights")
    with ExperimentStore(database) as store:
        store.connection.execute(
            """
            INSERT INTO model_updates (
                update_id, model_id, cutoff_batch_seq, new_batch_count,
                dataset_sha256, status
            ) VALUES ('update', 'dqn_job_scorer.adaptive_hybrid_q_v2', 1, 1,
                      'dataset', 'PROMOTED')
            """
        )
        store.connection.execute(
            """
            INSERT INTO model_training_checkpoints (
                checkpoint_id, model_id, update_id, trainer_artifact_path,
                inference_artifact_path, artifact_sha256, dataset_sha256,
                metrics_json, gate_json, promoted
            ) VALUES ('checkpoint', 'dqn_job_scorer.adaptive_hybrid_q_v2', 'update', %s, %s,
                      'artifact', 'dataset', '{}', '{"passed": false}', true)
            """,
            [str(source), str(source)],
        )
        store.connection.execute(
            "UPDATE training_models SET active_checkpoint_id = 'checkpoint'"
        )
        store.connection.execute(
            "UPDATE model_updates SET candidate_checkpoint_id = 'checkpoint'"
        )
    with pytest.raises(ValueError, match="release gate"):
        export_active_model(
            "agents/dqn_job_scorer/agent.toml",
            database,
            tmp_path / "released.npz",
            confirm_release=True,
        )


def test_kaglab_has_no_models_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["models", "status"])
