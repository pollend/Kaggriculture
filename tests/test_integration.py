from pathlib import Path

from agents.configurable_crop_surrogate.experiment import ExperimentConfig, ExperimentRunner
from agents.configurable_crop_surrogate.strategy import generate_population
from kaggriculture_lab.episode_metrics import episode_metrics
from kaggriculture_lab.storage import ExperimentStore
from sim.runner import run_match


def test_named_agent_completes_match_and_emits_metrics():
    result = run_match(
        ("agents/configurable_crop_surrogate/main.py", "starter"),
        seed=3,
        config={"episodeSteps": 120},
        analyzer=episode_metrics,
    )
    assert not result.failed
    assert all(seat.final_money is not None for seat in result.seats)
    assert result.seats[0].metrics["total_unit_actions"] > 0


def test_runner_emits_no_game_metrics_without_an_analyzer():
    """`sim` stays game-agnostic: metrics only appear when a caller asks."""
    result = run_match(("starter", "pass"), seed=3, config={"episodeSteps": 120})
    assert not result.failed
    assert result.seats[0].metrics == {}
    assert result.seats[0].final_money is None
    assert result.seats[0].reward is not None  # rewards are generic and still present


def test_tournament_schedules_are_paired_and_stable(tmp_path: Path):
    config = ExperimentConfig(
        population_size=6,
        stage1_seeds=1,
        stage1_keep=4,
        stage2_opponents=2,
        stage2_seeds=1,
        stage2_keep=3,
        stage3_seeds=1,
        candidate_pool=20,
        proposal_count=4,
    )
    with ExperimentStore(tmp_path / "test.duckdb") as store:
        runner = ExperimentRunner(config, store, tmp_path, run_id="schedule-test")
        population = generate_population(6, config.seed)
        first = runner._stage1(population)
        assert len(first) == 6 * 3 * 2
        assert first[0].strategy_ids == tuple(reversed(first[1].strategy_ids))
        assert first[0].seed == first[1].seed
        second = runner._stage2(population)
        assert len(second) == 6 * 2  # degree 2 graph, each edge played in both seats
