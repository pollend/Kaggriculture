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
    database = tmp_path / "status.duckdb"
    assert command(["model", "status", "--database", str(database)]) == 0
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
    database = tmp_path / "deferred.duckdb"
    assert (
        command(
            [
                "model",
                "update",
                "--database",
                str(database),
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
    database = tmp_path / "named-status.duckdb"
    assert dqn_cli(
        ["model", "status", "--name", "dqn05", "--database", str(database)]
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
    database = tmp_path / "export.duckdb"
    assert dqn_cli(["model", "status", "--database", str(database)]) == 0
    capsys.readouterr()

    source = tmp_path / "checkpoint.npz"
    source.write_bytes(b"reviewed weights")
    with ExperimentStore(database) as store:
        store.connection.execute(
            """
            INSERT INTO model_updates (
                update_id, model_id, cutoff_batch_seq, new_batch_count,
                dataset_sha256, status, candidate_checkpoint_id
            ) VALUES ('update', 'dqn_job_scorer.adaptive_hybrid_q_v2', 1, 1,
                      'dataset', 'PROMOTED', 'checkpoint')
            """
        )
        store.connection.execute(
            """
            INSERT INTO model_training_checkpoints (
                checkpoint_id, model_id, update_id, trainer_artifact_path,
                inference_artifact_path, artifact_sha256, dataset_sha256,
                metrics_json, gate_json, promoted
            ) VALUES ('checkpoint', 'dqn_job_scorer.adaptive_hybrid_q_v2', 'update', ?, ?,
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

    output = tmp_path / "released.npz"
    arguments = [
        "model",
        "export-active",
        "--database",
        str(database),
        "--output",
        str(output),
    ]
    with pytest.raises(ValueError, match="confirm-release"):
        dqn_cli(arguments)
    assert not output.exists()

    assert dqn_cli([*arguments, "--confirm-release"]) == 0
    assert output.read_bytes() == source.read_bytes()


def test_kaglab_has_no_models_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["models", "status"])
