"""Behaviour anchor for the layering refactor.

Moving modules between packages must not change what any agent does. "The
margin looks about the same" is not evidence, so this replays a fixed snapshot
against a fixed opponent over a fixed seed set and demands the exact same
rewards that were recorded before the move.

Regenerate deliberately (and only when a behaviour change is intended):

    .venv/bin/python -m tests.test_refactor_anchor --regenerate
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim.arena import play_series

ANCHOR_PATH = Path(__file__).parent / "data" / "refactor_anchor.json"


def _load() -> dict:
    if not ANCHOR_PATH.is_file():
        pytest.skip(f"no anchor recorded at {ANCHOR_PATH}")
    return json.loads(ANCHOR_PATH.read_text())


def _replay(anchor: dict) -> dict[str, dict[str, list]]:
    results = play_series(anchor["agent_a"], anchor["agent_b"], range(anchor["seeds"]), {}, jobs=8)
    return {
        str(seed): {
            "forward": [seat.reward for seat in forward.seats],
            "reverse": [seat.reward for seat in reverse.seats],
        }
        for seed, (forward, reverse) in sorted(results.items())
    }


def test_refactor_preserves_behaviour_exactly():
    anchor = _load()
    weights = anchor["agent_a"].partition("weights=")[2]
    if weights and not Path(weights).is_file():
        pytest.skip(f"anchor checkpoint missing: {weights}")

    replayed = _replay(anchor)

    drifted = {
        seed: {"expected": anchor["rewards"][seed], "actual": replayed[seed]}
        for seed in anchor["rewards"]
        if replayed.get(seed) != anchor["rewards"][seed]
    }
    assert not drifted, f"behaviour changed on {len(drifted)} seed(s): {json.dumps(drifted, indent=1)[:2000]}"


if __name__ == "__main__":  # pragma: no cover -- maintenance entry point
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    if not args.regenerate:
        raise SystemExit("pass --regenerate to overwrite the recorded anchor")

    existing = json.loads(ANCHOR_PATH.read_text())
    existing["rewards"] = _replay(existing)
    ANCHOR_PATH.write_text(json.dumps(existing, indent=1))
    print(f"rewrote {ANCHOR_PATH}")
