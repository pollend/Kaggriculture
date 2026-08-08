from __future__ import annotations

import ast
import hashlib
import json
import tarfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaggriculture_lab.cli import _cmd_bundle
from kaggriculture_lab.identity import AgentManifest
from kaggriculture_lab.league import LeagueConfig, create_session, schedule_generation
from kaggriculture_lab.storage import ExperimentStore
from sim.runner import run_match

ROOT = Path(__file__).resolve().parent.parent
COMPETITORS = tuple(sorted((ROOT / "agents").glob("competitor_*")))

DISALLOWED_IMPORTS = {
    "ctypes",
    "http",
    "importlib",
    "marshal",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "urllib",
}
DISALLOWED_CALLS = {"__import__", "compile", "eval", "exec", "open"}
DISALLOWED_ATTRIBUTES = {
    "connect",
    "fork",
    "popen",
    "remove",
    "rmdir",
    "spawn",
    "system",
    "unlink",
    "urlopen",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(folder: Path) -> dict:
    return json.loads((folder / "metadata.json").read_text())


def test_competitor_snapshot_is_complete_and_hashes_are_unique():
    assert len(COMPETITORS) == 12
    entrypoint_hashes = set()
    for folder in COMPETITORS:
        metadata = _metadata(folder)
        entrypoint = folder / metadata["entrypoint_file"]
        notebook = folder / metadata["notebook_file"]

        assert folder.name.startswith("competitor_")
        assert entrypoint == folder / "main.py"
        assert entrypoint.is_file()
        assert notebook.is_file()
        assert (folder / "kernel-metadata.json").is_file()
        assert (folder / "README.md").is_file()
        assert metadata["license"] == "Apache-2.0"
        assert metadata["source_url"].startswith("https://www.kaggle.com/code/")
        assert metadata["entrypoint_sha256"] == _sha256(entrypoint)
        assert metadata["notebook_sha256"] == _sha256(notebook)
        json.loads(notebook.read_text())
        entrypoint_hashes.add(metadata["entrypoint_sha256"])

    assert len(entrypoint_hashes) == len(COMPETITORS)


@pytest.mark.parametrize("folder", COMPETITORS, ids=lambda path: path.name)
def test_competitor_entrypoint_passes_static_safety_gate(folder: Path):
    source = (folder / "main.py").read_text()
    tree = ast.parse(source)
    has_agent = False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "agent":
            has_agent = True
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            has_agent |= any(isinstance(target, ast.Name) and target.id == "agent" for target in targets)
        elif isinstance(node, ast.Import):
            assert not ({alias.name.split(".", 1)[0] for alias in node.names} & DISALLOWED_IMPORTS)
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".", 1)[0] not in DISALLOWED_IMPORTS
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in DISALLOWED_CALLS
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in DISALLOWED_ATTRIBUTES

    assert has_agent


def test_competitor_manifests_are_fixed_unparameterized_agents():
    family_ids = set()
    for folder in COMPETITORS:
        manifest = AgentManifest.load(folder / "agent.toml")
        assert manifest.family_id.startswith("competitor_")
        assert manifest.entry_point.startswith("agents.competitor_")
        assert manifest.training_adapter is None
        assert manifest.episode_evaluator is None
        assert manifest.code_paths == (str((folder / "main.py").relative_to(ROOT)),)
        family_ids.add(manifest.family_id)
    assert len(family_ids) == len(COMPETITORS)


def test_configured_competitors_seed_persistent_league_as_anchors(tmp_path: Path):
    config = replace(
        LeagueConfig.from_toml(ROOT / "configs/league.toml"),
        population_size=4,
        rounds_per_generation=2,
        hall_of_fame_rounds=1,
        episode_steps=24,
        jobs=1,
    )
    database = tmp_path / "league.duckdb"
    session_id = create_session(config, database, workspace=ROOT, session_id="competitor-anchors")

    with ExperimentStore(database) as store:
        roles = dict(
            store.connection.execute(
                """
                SELECT role, count(*) FROM generation_members
                WHERE generation_id = 'competitor-anchors:g0000' GROUP BY role
                """
            ).fetchall()
        )
        assert roles == {"anchor": 12, "population": 4}
        assert schedule_generation(store, session_id) == 6
        anchor_ids = {
            row[0]
            for row in store.connection.execute(
                """
                SELECT revision_id FROM generation_members
                WHERE generation_id = 'competitor-anchors:g0000' AND role = 'anchor'
                """
            ).fetchall()
        }
        population_pairings = store.connection.execute(
            """
            SELECT revision_a, revision_b FROM league_pairings
            WHERE generation_id = 'competitor-anchors:g0000' AND pairing_kind = 'population'
            """
        ).fetchall()
        assert all(left not in anchor_ids and right not in anchor_ids for left, right in population_pairings)
        hall_opponents = {
            revision
            for row in store.connection.execute(
                """
                SELECT revision_a, revision_b FROM league_pairings
                WHERE generation_id = 'competitor-anchors:g0000' AND pairing_kind = 'hall_of_fame'
                """
            ).fetchall()
            for revision in row
        }
        assert hall_opponents & anchor_ids


@pytest.mark.parametrize("folder", COMPETITORS, ids=lambda path: path.name)
def test_competitor_completes_full_season_in_both_seats_and_is_repeatable(folder: Path):
    spec = str((folder / "main.py").relative_to(ROOT))
    forward = run_match((spec, "pass"), seed=17)
    repeated = run_match((spec, "pass"), seed=17)
    reverse = run_match(("pass", spec), seed=17)

    assert not forward.failed
    assert not repeated.failed
    assert not reverse.failed
    assert forward.seats[0].reward == repeated.seats[0].reward
    assert forward.seats[0].final_money == repeated.seats[0].final_money


@pytest.mark.parametrize(
    "manifest_path",
    ["agents/configurable_crop_surrogate/agent.toml", "agents/dqn_job_scorer/agent.toml"],
)
def test_submission_bundle_excludes_competitors(manifest_path: str, tmp_path: Path):
    """Vendored competitor agents must never leak into one of our submissions."""
    archive_path = tmp_path / "submission.tar.gz"
    _cmd_bundle(SimpleNamespace(agent=manifest_path, output=str(archive_path)))
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
    assert not any("competitor_" in name for name in names)
    # And it bundles only the agent it was asked for.
    owner = Path(manifest_path).parent.name
    others = [
        name for name in names
        if name.startswith("agents/") and Path(name).parts[1] not in {owner, "__init__.py"}
    ]
    assert not others, f"bundle for {owner} also contained {others}"
