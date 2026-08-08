from pathlib import Path

import train_all
from kaggriculture_lab.identity import AgentManifest
from kaggriculture_lab.league import LeagueConfig


def test_training_roster_lists_every_manifest_backed_agent():
    configured = {str(item["type"]) for item in train_all.TRAINING_AGENTS}
    discovered = {
        AgentManifest.load(path).family_id
        for path in (train_all.REPO_ROOT / "agents").glob("*/agent.toml")
    }
    assert configured == discovered


def test_effective_config_contains_named_roster_and_all_agent_anchors(
    tmp_path: Path, monkeypatch
):
    roster = [
        {
            "type": "configurable_crop_surrogate",
            "name": "crop01",
            "primary": True,
            "train": False,
        },
        {"type": "dqn_job_scorer", "name": "dqn05", "train": True},
    ]
    monkeypatch.setattr(train_all, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(train_all, "TRAINING_AGENTS", roster)

    config = LeagueConfig.from_toml(train_all._effective_config())
    config.validate()
    assert [(item.type, item.name, item.train) for item in config.training_agents] == [
        ("configurable_crop_surrogate", "crop01", False),
        ("dqn_job_scorer", "dqn05", True),
    ]
    expected = {
        str(path.relative_to(train_all.REPO_ROOT))
        for path in (train_all.REPO_ROOT / "agents").glob("*/agent.toml")
        if str(path.relative_to(train_all.REPO_ROOT)) != config.manifest
    }
    assert set(config.anchor_manifests) == expected
