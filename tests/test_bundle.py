"""A submission bundle must stand alone.

The real failure this catches is silent: an agent imports the lab or its own
trainer, works perfectly in the repo, and then dies on Kaggle where neither
exists. Unpacking the tarball into an empty directory and importing it with the
repo off `sys.path` is the only check that actually proves it.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path

import pytest

from kaggriculture_lab.cli import main as cli_main
from kaggriculture_lab.identity import AgentManifest

REPO = Path(__file__).resolve().parent.parent

# Agents we own and ship. Vendored competitor agents are not built here.
OWNED_AGENTS = [
    "agents/configurable_crop_surrogate/agent.toml",
    "agents/dqn_job_scorer/agent.toml",
]

# Anything here in a bundled submission means it would fail on Kaggle, or drag
# a training dependency into a inference-only artifact.
FORBIDDEN = {"torch", "kaggriculture_lab", "sim", "sklearn", "pandas", "duckdb", "joblib"}


def _build(manifest_path: str, tmp_path: Path) -> Path:
    archive = tmp_path / "submission.tar.gz"
    assert cli_main(["bundle", "--agent", manifest_path, "--output", str(archive)]) == 0
    unpacked = tmp_path / "unpacked"
    unpacked.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(unpacked, filter="data")
    return unpacked


@pytest.mark.parametrize("manifest_path", OWNED_AGENTS)
def test_bundle_imports_and_plays_standalone(manifest_path, tmp_path):
    unpacked = _build(manifest_path, tmp_path)
    assert (unpacked / "main.py").is_file(), "Kaggle requires a top-level main.py"

    probe = textwrap.dedent(f"""
        import sys
        sys.path = [p for p in sys.path if p != {str(REPO)!r}]
        sys.path.insert(0, ".")
        import main

        assert callable(main.agent), "main.agent is not callable"

        leaked = sorted({{m.split(".")[0] for m in sys.modules}} & {FORBIDDEN!r})
        assert not leaked, f"submission pulled in {{leaked}}"

        obs = {{
            "player": 0, "step": 0, "day": 0, "hour": 0,
            "farms": [
                {{"money": 3000.0, "tiles": [[None] * 10 for _ in range(10)],
                  "farmer": [4, 4], "hands": [],
                  "unlocked_quadrants": ["NW"], "hires_today": 0}},
                {{"money": 3000.0, "tiles": [[None] * 10 for _ in range(10)],
                  "farmer": [4, 4], "hands": [],
                  "unlocked_quadrants": ["NW"], "hires_today": 0}},
            ],
            "market": {{"inventory": {{}}, "prices": {{}}}},
            "town": {{"unlocked_shops": []}},
            "private": {{"shed": {{}}, "seeds": {{}}, "inventories": [{{}}]}},
        }}
        action = main.agent(obs)
        assert set(action) == {{"farmer", "hands", "market"}}, action
        assert len(action["market"]) <= 10
        print("ok")
    """)

    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=unpacked, capture_output=True, text=True, env={"PATH": ""}
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.parametrize("manifest_path", OWNED_AGENTS)
def test_manifest_code_paths_all_exist(manifest_path):
    """A stale `code_paths` entry silently omits a module from the submission."""
    manifest = AgentManifest.load(manifest_path)
    missing = [path for path in manifest.code_paths if not (REPO / path).is_file()]
    assert not missing, f"{manifest_path} lists files that do not exist: {missing}"


def test_bundle_excludes_training_code(tmp_path):
    """`code_paths` is an allowlist, so the trainer is excluded by construction."""
    unpacked = _build("agents/dqn_job_scorer/agent.toml", tmp_path)
    assert not (unpacked / "agents/dqn_job_scorer/training").exists()
