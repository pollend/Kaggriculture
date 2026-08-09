"""Checks that the arena itself is trustworthy, before any agent is tuned with it.

Run with:  .venv/bin/python -m pytest sim/test_arena.py -v
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from sim.arena import play_series
from sim.metrics import sign_test, summarize, wilson
from sim.runner import run_match

REPO = Path(__file__).resolve().parent.parent


# --- the environment behaves the way the arena assumes -----------------------


def test_same_seed_replays_identically():
    """Deterministic agents on one seed must reproduce exactly."""
    first = run_match(("starter", "pass"), seed=11)
    second = run_match(("starter", "pass"), seed=11)
    assert [s.reward for s in first.seats] == [s.reward for s in second.seats]


def test_seed_actually_changes_the_episode():
    """A seed that changed nothing would silently collapse the sample size."""
    shops = []
    for seed in (0, 1, 2):
        match = run_match(("pass", "pass"), seed=seed, keep_replay=True)
        final = match.replay["steps"][-1][0]["observation"]
        shops.append(tuple(final["town"]["unlocked_shops"]))
    assert len(set(shops)) > 1


def test_seeded_random_is_reproducible():
    """Our replacement opponent must be repeatable; the built-in `random` is not."""
    a = run_match(("kaggriculture_lab.opponents:make_seeded_random?seed=5", "pass"), seed=2)
    b = run_match(("kaggriculture_lab.opponents:make_seeded_random?seed=5", "pass"), seed=2)
    assert [s.reward for s in a.seats] == [s.reward for s in b.seats]


def test_spec_config_preserves_plus_and_percent():
    """`parse_qsl` would read `+` as a space and corrupt checkpoint paths."""
    from sim import registry

    target, config = registry.split_spec("main.py?weights=runs/best/round0045_margin+2071.npz")
    assert target == "main.py"
    assert config["weights"] == "runs/best/round0045_margin+2071.npz"

    _, config = registry.split_spec("main.py?path=a%2Fb.npz&n=3&flag=true")
    assert config == {"path": "a/b.npz", "n": 3, "flag": True}


def test_builtin_random_is_flagged_nondeterministic():
    from sim import registry

    assert not registry.is_deterministic("random")
    assert registry.is_deterministic("kaggriculture_lab.opponents:make_seeded_random?seed=1")
    assert registry.is_deterministic("starter")


# --- failures are reported as failures, not as losses ------------------------


def test_crash_is_an_error_not_a_loss(tmp_path):
    """The env reports DONE on the final step even for a seat that crashed."""
    crasher = tmp_path / "crasher.py"
    crasher.write_text(textwrap.dedent("""
        def agent(obs):
            if obs["step"] == 300:
                raise ValueError("deliberate crash")
            return {"farmer": ["PASS"], "hands": [], "market": []}
    """))

    match = run_match((str(crasher), "starter"), seed=0)
    seat = match.seats[0]

    assert seat.failed
    assert seat.status == "ERROR"
    assert seat.fail_step is not None
    assert "deliberate crash" in (seat.traceback or "")
    # The naive reading would have called this a 3000-money loss.
    assert seat.reward is not None
    assert match.margin is None


def test_errored_seeds_are_dropped_from_the_summary(tmp_path):
    crasher = tmp_path / "crasher.py"
    crasher.write_text(textwrap.dedent("""
        def agent(obs):
            if obs["step"] == 100:
                raise ValueError("boom")
            return {"farmer": ["PASS"], "hands": [], "market": []}
    """))

    results = play_series(str(crasher), "starter", range(2), {}, jobs=1)
    summary = summarize(str(crasher), "starter", results)

    assert summary.pairs == 0
    assert summary.dropped_seeds == 2
    assert summary.errors[str(crasher)] == 4  # 2 seeds x 2 orientations


# --- no seat bias leaks into the verdict -------------------------------------


@pytest.mark.parametrize("spec", ["pass", "starter"])
def test_identical_agents_have_zero_margin(spec):
    """Same deterministic agent in both seats must tie exactly on every seed.

    A non-zero margin here would mean the env favours a seat, which would
    contaminate every head-to-head number the arena produces.
    """
    results = play_series(spec, spec, range(5), {}, jobs=1)
    summary = summarize(spec, spec, results)

    assert summary.pairs == 5
    assert summary.margin_mean == 0.0
    assert summary.draws == 5


def test_starter_beats_pass_decisively():
    """Sanity: the metric has signal on a difference we know exists."""
    results = play_series("starter", "pass", range(6), {}, jobs=1)
    summary = summarize("starter", "pass", results)

    assert summary.wins == 6
    assert summary.margin_mean > 0
    assert summary.significant


# --- parallelism and worker state isolation ----------------------------------


def test_parallel_matches_serial():
    serial = play_series("starter", "kaggriculture_lab.opponents:make_seeded_random?seed=3", range(4), {}, jobs=1)
    parallel = play_series("starter", "kaggriculture_lab.opponents:make_seeded_random?seed=3", range(4), {}, jobs=4)

    for seed in serial:
        for a, b in zip(serial[seed], parallel[seed]):
            assert [s.reward for s in a.seats] == [s.reward for s in b.seats]


def test_module_state_is_reset_between_resolves(tmp_path):
    """Each resolve re-executes the agent module, so globals start clean.

    Submissions are a bare `agent(obs)`, so planner state has to live in module
    globals. If the arena reused one module across episodes in a worker, leaked
    state would make results depend on run order -- a silent correctness hole.
    Re-executing also matches Kaggle, which runs each episode in a fresh process.
    """
    from sim import registry

    counting = tmp_path / "counting.py"
    counting.write_text(textwrap.dedent("""
        CALLS = 0

        def agent(obs):
            global CALLS
            CALLS += 1
            return {"farmer": ["PASS"], "hands": [], "market": []}
    """))

    first = registry.resolve(str(counting))
    for _ in range(17):
        first({})
    assert first.__globals__["CALLS"] == 17

    second = registry.resolve(str(counting))
    assert second.__globals__["CALLS"] == 0
    # The two instances must not share a namespace either.
    second({})
    assert first.__globals__["CALLS"] == 17


def test_repeated_episodes_in_one_process_are_identical(tmp_path):
    """End-to-end version of the above, including a fresh-process comparison."""
    stateful = tmp_path / "stateful.py"
    stateful.write_text(textwrap.dedent("""
        CALLS = 0  # deliberately never reset on obs["step"] == 0

        def agent(obs):
            global CALLS
            CALLS += 1
            farm = obs["farms"][obs["player"]]
            tile = farm["tiles"][farm["farmer"][1]][farm["farmer"][0]]
            market = []
            if obs["private"]["shed"].get("CARROT", 0):
                market.append(["SELL", "CARROT", obs["private"]["shed"]["CARROT"]])
            if not obs["private"]["seeds"].get("CARROT", 0) and farm["money"] >= 20:
                market.append(["BUY_SEED", "CARROT", 1])
            op = ["PASS"]
            if tile is None and obs["private"]["seeds"].get("CARROT", 0):
                op = ["PLANT", "CARROT"]
            elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                op = ["HARVEST"] if obs["day"] - tile["planted_day"] >= 3 else ["WATER"]
            return {"farmer": op, "hands": [], "market": market}
    """))

    in_process = [run_match((str(stateful), "pass"), seed=0).seats[0].reward for _ in range(2)]

    fresh = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(f"""
            from sim.runner import run_match
            print(run_match(({str(stateful)!r}, "pass"), seed=0).seats[0].reward)
        """)],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    fresh_reward = float(fresh.stdout.strip())

    assert in_process[0] > 3000  # the probe agent actually earns, so it can differ
    assert in_process[0] == in_process[1] == fresh_reward


# --- statistics --------------------------------------------------------------


def test_wilson_bounds_are_sane():
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and 0 < lo < 1
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi


def test_sign_test_matches_known_values():
    assert sign_test([]) == 1.0
    assert sign_test([0, 0, 0]) == 1.0
    assert sign_test([1, 1, 1, 1, 1]) == pytest.approx(2 / 32)
    assert sign_test([1, -1]) == pytest.approx(1.0)


def test_seeds_needed_only_when_unresolved():
    results = play_series("starter", "pass", range(4), {}, jobs=1)
    resolved = summarize("starter", "pass", results)
    assert resolved.significant
    assert resolved.seeds_needed is None

    tied = summarize("pass", "pass", play_series("pass", "pass", range(3), {}, jobs=1))
    assert not tied.significant
    assert tied.seeds_needed is None  # no observed effect to resolve


# --- config plumbing ---------------------------------------------------------


def test_steps_override_shortens_the_episode():
    match = run_match(("pass", "pass"), seed=0, config={"episodeSteps": 120})
    assert match.steps == 120


def test_agent_config_is_injected(tmp_path):
    configured = tmp_path / "configured.py"
    configured.write_text(textwrap.dedent("""
        from sim.registry import agent_config

        _CFG = agent_config()

        def agent(obs):
            op = "WATER" if _CFG.get("mode") == "water" else "PASS"
            return {"farmer": [op], "hands": [], "market": []}
    """))

    match = run_match((f"{configured}?mode=water", "pass"), seed=0)
    assert not match.seats[0].failed
