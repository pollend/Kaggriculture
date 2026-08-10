"""Submission-shaped entry point for the RF market experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agents.mixed_farm_rf_market.agent import make_agent

_DEFAULT_WEIGHTS = Path(__file__).with_name("weights.joblib")
_config = json.loads(os.environ.get("KAGG_AGENT_CONFIG", "{}"))
if "weights" not in _config and _DEFAULT_WEIGHTS.is_file():
    _config["weights"] = str(_DEFAULT_WEIGHTS)
agent = make_agent(_config)


def __getattr__(name: str) -> Any:
    if name == "evaluate_episode":
        from agents.mixed_farm_rf_market.evaluation import evaluate_episode

        return evaluate_episode
    raise AttributeError(name)
