"""What the recurrence changed, and what it must not have.

`tests/test_dqn_market_free_v2.py` covers the multi-order delta this agent
inherits. This module covers v3's: a GRU that ticks once per turn, a head that
scores `concat(hidden, state)`, and the numpy mirror of both that ships to Kaggle.

The parity tests are the centre of gravity. v2 had two implementations of an MLP
and a disagreement cost one wrong decision; v3 has two implementations of a
*recurrence*, and a disagreement compounds over 720 ticks into a submission that
plays a different policy than the one the gate measured.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.testing import assert_array_equal

from agents.dqn_market_free_v3d import donor, herd, lots
from agents.dqn_market_free_v3d.market_state import (
    CROP_MATURITY_DIM,
    HERD_VIABILITY_DIM,
    HISTORY_BLOCK_DIM,
    INTRA_TURN_DIM,
    INTRA_TURN_SLICE,
    MARKET_FLOW_DIM,
    MARKET_STATE_DIM,
    PRODUCTS,
    REFUSAL_BLOCK_DIM,
)
from agents.dqn_market_free_v3d.order_space import (
    _BUY_ANIMAL_ROWS,
    _ROW_QUANTITY,
    ACTION_COUNT,
    ACTIONS,
    MASKED_SCORE,
    Order,
    PendingTurn,
    animal_room,
    feasible_actions,
    land_expansion_pressure,
    land_score,
    sell_depth_cap,
    should_buy_land,
)
from agents.dqn_market_free_v3d.policy import (
    _LAYER_KEY,
    HEAD_IN,
    HIDDEN_SIZE,
    OrderQNetwork,
    _sigmoid,
)
from agents.dqn_market_free_v3d.training import network as net_module
from tests.test_dqn_market_free import _mask_state

STATE_DIM = MARKET_STATE_DIM


def test_activity_gate_lineage_resumes_from_best_promoted_benchmark_checkpoint(tmp_path):
    from agents.dqn_market_free_v3d.training import updater as updater_module

    low_trainer = tmp_path / "low.pt"
    low_inference = tmp_path / "low.npz"
    high_trainer = tmp_path / "high.pt"
    high_inference = tmp_path / "high.npz"
    for path in [low_trainer, low_inference, high_trainer, high_inference]:
        path.write_bytes(b"weights")

    class Reader:
        def fetchall(self, sql, parameters):
            assert "benchmark_score" in sql
            assert "benchmark_ratchet" in sql
            assert "AND promoted" in sql
            assert "ORDER BY score DESC" in sql
            assert parameters == ["model"]
            return [
                ("high", str(high_trainer), str(high_inference), 9000.0),
                ("low", str(low_trainer), str(low_inference), 1200.0),
            ]

        def fetchone(self, sql, parameters):
            raise AssertionError("best-benchmark lineage should not use fallback")

    context = SimpleNamespace(
        model_id="model",
        config={
            "promotion_rule": "activity_gate",
            "benchmarks": [
                "agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py"
            ],
        },
        active_checkpoint=SimpleNamespace(checkpoint_id="low"),
    )

    checkpoint = updater_module.lineage_checkpoint(context, Reader())

    assert checkpoint is not None
    assert checkpoint.checkpoint_id == "high"
    assert checkpoint.trainer_artifact_path == high_trainer
    assert checkpoint.inference_artifact_path == high_inference
    assert checkpoint.released is False


def test_activity_gate_lineage_can_resume_from_best_promoted_rollout_checkpoint(tmp_path):
    from agents.dqn_market_free_v3d.training import updater as updater_module

    low_trainer = tmp_path / "low.pt"
    low_inference = tmp_path / "low.npz"
    high_trainer = tmp_path / "high.pt"
    high_inference = tmp_path / "high.npz"
    for path in [low_trainer, low_inference, high_trainer, high_inference]:
        path.write_bytes(b"weights")

    class Reader:
        def fetchall(self, sql, parameters):
            assert "metrics_json ->> 'recent_mean_final_money'" in sql
            assert "metrics_json ->> 'own_mean_final_money'" in sql
            assert "benchmark_score" not in sql
            assert "AND promoted" in sql
            assert "ORDER BY score DESC" in sql
            assert parameters == ["model"]
            return [
                ("high", str(high_trainer), str(high_inference), 9000.0),
                ("low", str(low_trainer), str(low_inference), 1200.0),
            ]

        def fetchone(self, sql, parameters):
            raise AssertionError("best-rollout lineage should not use fallback")

    context = SimpleNamespace(
        model_id="model",
        config={
            "promotion_rule": "activity_gate",
            "promotion_score_signal": "rollout",
        },
        active_checkpoint=SimpleNamespace(checkpoint_id="low"),
    )

    checkpoint = updater_module.lineage_checkpoint(context, Reader())

    assert checkpoint is not None
    assert checkpoint.checkpoint_id == "high"
    assert checkpoint.trainer_artifact_path == high_trainer
    assert checkpoint.inference_artifact_path == high_inference
    assert checkpoint.released is False


def test_activity_gate_lineage_skips_missing_best_artifacts(tmp_path):
    from agents.dqn_market_free_v3d.training import updater as updater_module

    kept_trainer = tmp_path / "kept.pt"
    kept_inference = tmp_path / "kept.npz"
    kept_trainer.write_bytes(b"weights")
    kept_inference.write_bytes(b"weights")

    class Reader:
        def fetchall(self, sql, parameters):
            return [
                ("missing", str(tmp_path / "missing.pt"), str(tmp_path / "missing.npz"), 9000.0),
                ("kept", str(kept_trainer), str(kept_inference), 5000.0),
            ]

        def fetchone(self, sql, parameters):
            raise AssertionError("second-best existing checkpoint should avoid fallback")

    context = SimpleNamespace(
        model_id="model",
        config={
            "promotion_rule": "activity_gate",
            "benchmarks": [
                "agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py"
            ],
        },
        active_checkpoint=SimpleNamespace(checkpoint_id="kept"),
    )

    checkpoint = updater_module.lineage_checkpoint(context, Reader())

    assert checkpoint is not None
    assert checkpoint.checkpoint_id == "kept"
    assert checkpoint.released is True


def test_self_play_pool_uses_promoted_benchmark_rank(tmp_path):
    from agents.dqn_market_free_v3d.training import updater as updater_module

    high = tmp_path / "high.npz"
    low = tmp_path / "low.npz"
    for path in [high, low]:
        path.write_bytes(b"weights")

    class Reader:
        sql = ""
        parameters = []

        def fetchall(self, sql, parameters):
            self.sql = " ".join(sql.split())
            self.parameters = list(parameters)
            return [(str(high),), (str(low),)]

    reader = Reader()
    context = SimpleNamespace(model_id="model", config={})

    pool = updater_module._self_play_pool(context, reader, 2)

    assert "AND promoted" in reader.sql
    assert "benchmark_score" in reader.sql
    assert "benchmark_ratchet" in reader.sql
    assert "ORDER BY COALESCE" in reader.sql
    assert reader.parameters == ["model", 2]
    assert pool == [updater_module._weights_spec(high), updater_module._weights_spec(low)]


def test_self_play_pool_can_use_promoted_rollout_rank(tmp_path):
    from agents.dqn_market_free_v3d.training import updater as updater_module

    high = tmp_path / "high.npz"
    low = tmp_path / "low.npz"
    for path in [high, low]:
        path.write_bytes(b"weights")

    class Reader:
        sql = ""
        parameters = []

        def fetchall(self, sql, parameters):
            self.sql = " ".join(sql.split())
            self.parameters = list(parameters)
            return [(str(high),), (str(low),)]

    reader = Reader()
    context = SimpleNamespace(
        model_id="model",
        config={"promotion_score_signal": "rollout"},
    )

    pool = updater_module._self_play_pool(context, reader, 2)

    assert "metrics_json ->> 'recent_mean_final_money'" in reader.sql
    assert "metrics_json ->> 'own_mean_final_money'" in reader.sql
    assert "benchmark_score" not in reader.sql
    assert reader.parameters == ["model", 2]
    assert pool == [updater_module._weights_spec(high), updater_module._weights_spec(low)]


def test_ruin_is_no_longer_free():
    """Losing the stake used to cost 0.12 against a 6.1 upside, so gambling was optimal."""

    import numpy as np

    from agents.dqn_market_free_v3d.training import reward as reward_module

    def season(final_money, turns=720):
        target, raw, _ = reward_module.episode_rewards([3000.0] * turns, final_money)
        return float(target.sum()), float(raw.sum())

    ruin_target, ruin_raw = season(0.0)
    # The cash stream still telescopes to -0.12; the charge is on top of it.
    assert ruin_raw == pytest.approx(-0.12, abs=1e-3)
    assert ruin_target == pytest.approx(-1.12, abs=1e-3)

    # Profitable seasons are untouched -- the charge must not tax winning.
    for final, expected in ((10_000.0, 0.28), (50_000.0, 1.88)):
        target, raw = season(final)
        assert target == pytest.approx(expected, abs=1e-2)
        assert target == pytest.approx(raw, abs=1e-6)

    # Ending exactly on the stake is the zero point, in both streams.
    assert season(3000.0)[0] == pytest.approx(0.0, abs=1e-6)

    # Linear in the shortfall, not a cliff: 2,999 must not cost what 0 costs.
    assert season(2999.0)[0] > -0.01
    assert season(1500.0)[0] == pytest.approx(-0.56, abs=1e-2)

    # `raw` is the cash and only the cash, so `own_mean_return` still grades money.
    assert season(0.0)[1] == pytest.approx(season(0.0)[1])


def test_the_ruin_charge_lands_only_on_the_terminal_step():
    from agents.dqn_market_free_v3d.training import reward as reward_module

    target, raw, _ = reward_module.episode_rewards([3000.0] * 50, 0.0)
    # Every interior step is untouched cash; the whole charge is on the last.
    assert list(target[:-1]) == pytest.approx(list(raw[:-1]))
    assert float(target[-1] - raw[-1]) == pytest.approx(-1.0, abs=1e-5)


def test_wheat_the_herd_needs_cannot_be_sold():
    """Upkeep is in kind, so starving the herd moves no money column and nothing saw it."""

    import numpy as np

    from agents.dqn_market_free_v3d import donor, order_space

    # Two placed animals eat `FEED_STOCK_DAYS` each.
    farm = {"tiles": [[{"kind": "COOP", "animal": "GOOSE"}, {"kind": "PASTURE", "animal": "COW"}]]}
    assert order_space.feed_floor(farm, {}) == 2 * donor.FEED_STOCK_DAYS
    # No herd, no floor -- the rule must not touch a farm without animals.
    assert order_space.feed_floor({"tiles": [[{"kind": "COOP"}]]}, {}) == 0
    # A malformed board returns 0 rather than raising: `feasible_actions` never raises.
    assert order_space.feed_floor(None, None) == 0
    assert order_space.feed_floor({"tiles": "nonsense"}, {}) == 0


def test_both_ramps_advance_on_the_mirror_pools_fill():
    """The scaffolding comes away when the run has replacements, not on a calendar.

    `pool_size` counts this model's promoted checkpoints above `self_play_min_score`,
    so pool fill reads "how many of my own policies beat the stake". A batch count
    keeps advancing while a run goes nowhere, which is when the donor and the Kaggle
    prior should still be there.
    """

    from agents.dqn_market_free_v3d.training import updater as updater_module

    config = {
        "self_play_ratio": 0.65,
        "self_play_ratio_final": 1.0,
        "demo_fraction": 0.25,
        "demo_fraction_final": 0.0,
        "self_play_ratio_ramp_batches": 0,
    }
    ratio = lambda n: updater_module._scheduled_self_play_ratio(config, 0, n, 25)
    demo = lambda n: updater_module._scheduled_demo_fraction(config, 0, n, 25)

    # Empty pool: full scaffolding, whatever the batch count.
    assert ratio(0) == pytest.approx(0.65)
    assert demo(0) == pytest.approx(0.25)
    # Half replaced: half way along both ramps.
    assert ratio(12) == pytest.approx(0.65 + 0.35 * (12 / 25))
    assert demo(12) == pytest.approx(0.25 - 0.25 * (12 / 25))
    # Fully replaced: donor gone, prior gone.
    assert ratio(25) == pytest.approx(1.0)
    assert demo(25) == pytest.approx(0.0)
    assert ratio(99) == pytest.approx(1.0)

    # Both halves move together -- neither outlasts the other.
    for n in range(0, 26):
        ratio_progress = (ratio(n) - 0.65) / 0.35
        demo_progress = (0.25 - demo(n)) / 0.25
        assert ratio_progress == pytest.approx(demo_progress)


def test_the_batch_span_brakes_the_ramp_but_does_not_drive_it():
    from agents.dqn_market_free_v3d.training import updater as updater_module

    config = {
        "self_play_ratio": 0.65,
        "self_play_ratio_final": 1.0,
        "self_play_ratio_ramp_batches": 600,
    }
    # Pool already full, but only 60 of 600 batches collected: the brake holds.
    held = updater_module._scheduled_self_play_ratio(config, 60, 25, 25)
    assert held == pytest.approx(0.65 + 0.35 * 0.1)
    # Batches served but the run produced nothing: the pool holds it back instead.
    assert updater_module._scheduled_self_play_ratio(config, 600, 0, 25) == pytest.approx(0.65)
    # Both satisfied.
    assert updater_module._scheduled_self_play_ratio(config, 600, 25, 25) == pytest.approx(1.0)


def test_the_ramps_are_a_no_op_when_unconfigured():
    """Every other fork leaves the final keys unset and must keep constants."""

    from agents.dqn_market_free_v3d.training import updater as updater_module

    for config in (
        {"self_play_ratio": 0.65, "demo_fraction": 0.25},
        {"self_play_ratio": 0.65, "demo_fraction": 0.25, "self_play_ratio_final": ""},
        {"self_play_ratio": 0.65, "demo_fraction": 0.25, "demo_fraction_final": None},
    ):
        assert updater_module._scheduled_self_play_ratio(config, 9_999, 25, 25) == pytest.approx(0.65)
        assert updater_module._scheduled_demo_fraction(config, 9_999, 25, 25) == pytest.approx(0.25)


def test_self_play_min_score_floors_the_mirror_pool(tmp_path):
    """A checkpoint that cannot beat the stake never becomes a teacher.

    This is the surviving half of the deleted shared archive's
    `mature_reserved_min_money = 4000.0`. Without it `_self_play_pool` reads one
    `model_id` with no floor, so a slot that stops trading trains against copies of
    itself at the 3,000 stake and cannot leave.
    """

    from agents.dqn_market_free_v3d.training import updater as updater_module

    weights = tmp_path / "high.npz"
    weights.write_bytes(b"weights")

    class Reader:
        sql = ""
        parameters = []

        def fetchall(self, sql, parameters):
            self.sql = " ".join(sql.split())
            self.parameters = list(parameters)
            return [(str(weights),)]

    reader = Reader()
    context = SimpleNamespace(model_id="model", config={"self_play_min_score": 3000.0})

    pool = updater_module._self_play_pool(context, reader, 2)

    assert ">= %s" in reader.sql
    # The floor binds before the LIMIT, so it drops rows rather than reordering them.
    assert reader.parameters == ["model", 3000.0, 2]
    assert pool == [updater_module._weights_spec(weights)]


def test_self_play_min_score_floors_the_pool_after_a_reseed(tmp_path):
    from agents.dqn_market_free_v3d.training import updater as updater_module

    weights = tmp_path / "high.npz"
    weights.write_bytes(b"weights")

    class Reader:
        parameters = []

        def fetchall(self, sql, parameters):
            self.parameters = list(parameters)
            return [(str(weights),)]

    reader = Reader()
    context = SimpleNamespace(model_id="model", config={"self_play_min_score": 3000.0})

    updater_module._self_play_pool(context, reader, 2, after_checkpoint_id="ckpt")

    assert reader.parameters == ["model", "ckpt", 3000.0, 2]


def test_self_play_min_score_defaults_to_no_floor(tmp_path):
    """Every other fork leaves this unset; the SQL must be byte-identical for them."""

    from agents.dqn_market_free_v3d.training import updater as updater_module

    weights = tmp_path / "high.npz"
    weights.write_bytes(b"weights")

    class Reader:
        sql = ""
        parameters = []

        def fetchall(self, sql, parameters):
            self.sql = " ".join(sql.split())
            self.parameters = list(parameters)
            return [(str(weights),)]

    for config in ({}, {"self_play_min_score": 0.0}, {"self_play_min_score": ""}):
        reader = Reader()
        context = SimpleNamespace(model_id="model", config=config)
        updater_module._self_play_pool(context, reader, 2)
        assert ">= %s" not in reader.sql, config
        assert reader.parameters == ["model", 2], config


def test_non_activity_gate_lineage_keeps_shared_newest_checkpoint_policy(monkeypatch):
    from agents.dqn_market_free_v3d.training import updater as updater_module

    calls = []
    expected = object()

    def fake_newest(context, reader):
        calls.append((context, reader))
        return expected

    monkeypatch.setattr(updater_module, "_newest_lineage_checkpoint", fake_newest)
    context = SimpleNamespace(
        model_id="model",
        config={"promotion_rule": "legacy"},
        active_checkpoint=None,
    )
    reader = object()

    assert updater_module.lineage_checkpoint(context, reader) is expected
    assert calls == [(context, reader)]


def test_best_promoted_rollout_score_reads_recent_and_own_money():
    from agents.dqn_market_free_v3d.training.updater import _best_promoted_checkpoint_score

    class Reader:
        def fetchone(self, sql, parameters):
            assert "metrics_json ->> 'recent_mean_final_money'" in sql
            assert "metrics_json ->> 'own_mean_final_money'" in sql
            assert "benchmark_score" not in sql
            assert "AND promoted" in sql
            assert parameters == ["model"]
            return (8123.5, "checkpoint-rollout")

    assert _best_promoted_checkpoint_score(Reader(), "model", "rollout") == (
        8123.5,
        "checkpoint-rollout",
    )


def test_activation_reasons_block_non_idle_benchmark_regressions():
    from agents.dqn_market_free_v3d.training.updater import _activation_reasons

    assert _activation_reasons(
        activity_passed=True, candidate_money=5000.0, best_money=6000.0
    ) == ["below_best_promoted_benchmark_money"]
    assert (
        _activation_reasons(
            activity_passed=True, candidate_money=6000.0, best_money=6000.0
        )
        == []
    )
    assert _activation_reasons(
        activity_passed=False, candidate_money=7000.0, best_money=6000.0
    ) == ["activity_gate_failed"]


def test_activation_reasons_can_report_rollout_regressions():
    from agents.dqn_market_free_v3d.training.updater import _activation_reasons

    assert _activation_reasons(
        activity_passed=True,
        candidate_money=5000.0,
        best_money=6000.0,
        score_signal="rollout",
    ) == ["below_best_promoted_rollout_money"]
    assert _activation_reasons(
        activity_passed=True,
        candidate_money=None,
        best_money=6000.0,
        score_signal="rollout",
    ) == ["missing_rollout_score"]


def _module(seed: int = 0, noisy: str = "none", **fill) -> net_module.OrderQModule:
    """A built module with non-trivial GRU biases.

    Torch initialises `bias_hh` from `U(-1/sqrt(H), 1/sqrt(H))`, which at H = 128
    is +-0.088 -- small enough that a wrong reset-gate association is inside a
    1e-5 tolerance on a short sequence. Every parity test here therefore widens
    the biases first, so the terms that distinguish the right form from the wrong
    one are actually present in the arithmetic being compared.
    """

    torch.manual_seed(seed)
    module = net_module.build(noisy=noisy)
    with torch.no_grad():
        module.cell.bias_hh.uniform_(-1.5, 1.5)
        module.cell.bias_ih.uniform_(-1.5, 1.5)
        for name, value in fill.items():
            getattr(module.cell, name).fill_(value)
    return module


def test_activity_gate_lineage_can_follow_the_latest_checkpoint(monkeypatch):
    from agents.dqn_market_free_v3d.training import updater as updater_module

    calls = []
    expected = object()

    def fake_newest(context, reader):
        calls.append((context, reader))
        return expected

    monkeypatch.setattr(updater_module, "_newest_lineage_checkpoint", fake_newest)

    class Reader:
        def fetchall(self, sql, parameters):
            raise AssertionError("latest lineage must not rank promoted checkpoints")

    context = SimpleNamespace(
        model_id="model",
        config={"promotion_rule": "activity_gate", "lineage_source": "latest"},
        active_checkpoint=None,
    )
    reader = Reader()

    assert updater_module.lineage_checkpoint(context, reader) is expected
    assert calls == [(context, reader)]

    context.config["lineage_source"] = "newest"
    with pytest.raises(ValueError, match="lineage_source"):
        updater_module.lineage_checkpoint(context, reader)


def test_activation_reasons_accept_a_candidate_inside_the_tolerance():
    from agents.dqn_market_free_v3d.training.updater import _activation_reasons

    assert (
        _activation_reasons(
            activity_passed=True, candidate_money=5000.0, best_money=6000.0, tolerance=1500.0
        )
        == []
    )
    assert _activation_reasons(
        activity_passed=True, candidate_money=4000.0, best_money=6000.0, tolerance=1500.0
    ) == ["below_best_promoted_benchmark_money"]
    # A negative tolerance is a strict ratchet, never a stricter one.
    assert (
        _activation_reasons(
            activity_passed=True, candidate_money=6000.0, best_money=6000.0, tolerance=-500.0
        )
        == []
    )


def test_promotion_tolerance_adds_the_arena_ci_only_under_the_benchmark_signal():
    from agents.dqn_market_free_v3d.training.updater import _promotion_tolerance

    arena = {"margin_ci": [-1400.0, -600.0]}
    config = {"promotion_tolerance_money": 100.0, "promotion_tolerance_ci": True}
    assert _promotion_tolerance(config, "benchmark", arena) == pytest.approx(500.0)
    # The rollout score is collection money; an arena CI says nothing about it.
    assert _promotion_tolerance(config, "rollout", arena) == pytest.approx(100.0)
    assert _promotion_tolerance({}, "benchmark", arena) == 0.0
    assert _promotion_tolerance(config, "benchmark", None) == pytest.approx(100.0)
    assert _promotion_tolerance(config, "benchmark", {"margin_ci": ["x", 1]}) == pytest.approx(100.0)
    assert _promotion_tolerance({"promotion_tolerance_money": -50.0}, "benchmark", arena) == 0.0


def test_promotion_tolerance_is_capped_at_a_fraction_of_the_bar_it_relaxes():
    """The regression that let `rewardtogo-v2` promote candidates scoring 0.0.

    `margin_ci` is the paired margin, so its width tracks the opponent's ~140,000
    and not the candidate's three figures. Uncapped, the tolerance was 3,000-5,000
    coins against bars of 1,500-13,400 and the gate stopped rejecting anything.
    """

    from agents.dqn_market_free_v3d.training.updater import (
        _activation_reasons,
        _promotion_tolerance,
    )

    # The real numbers off `rewardtogo_07` at cutoff 160.
    arena = {"margin_ci": [-148928.76, -142929.34]}
    config = {"promotion_tolerance_ci": True}
    bar = 2864.01

    uncapped = _promotion_tolerance(config, "benchmark", arena)
    assert uncapped == pytest.approx(2999.71, abs=0.01)
    assert uncapped > bar, "the whole point: the raw half-width exceeds the bar"

    capped = _promotion_tolerance(config, "benchmark", arena, bar)
    assert capped == pytest.approx(0.10 * bar)
    assert _activation_reasons(
        activity_passed=True, candidate_money=0.0, best_money=bar, tolerance=capped
    ) == ["below_best_promoted_benchmark_money"]
    # A candidate genuinely inside the noise of the bar still promotes.
    assert (
        _activation_reasons(
            activity_passed=True, candidate_money=bar - 200.0, best_money=bar, tolerance=capped
        )
        == []
    )

    # The fraction is a dial, and no bar means nothing to relax.
    assert _promotion_tolerance(
        {**config, "promotion_tolerance_fraction": 0.5}, "benchmark", arena, bar
    ) == pytest.approx(0.5 * bar)
    assert _promotion_tolerance(config, "benchmark", arena, None) == uncapped
    assert _promotion_tolerance(config, "benchmark", arena, 0.0) == uncapped


def test_priority_stats_report_season_profit_as_the_sum_over_turns():
    """`own_mean_return` is `reward.sum()`: the per-turn cash deltas telescope to
    the season's profit over `MONEY_SCALE`."""

    from agents.dqn_market_free_v3d.training.replay import ReplayBuffer
    from agents.dqn_market_free_v3d.training.reward import MONEY_SCALE

    buffer = ReplayBuffer(capacity=8, seed=0)
    buffer.add([_priority_episode(0, 8_000.0, 0.2), _priority_episode(1, 13_000.0, 0.4)])

    stats = buffer.priority_stats()
    assert stats["own_mean_return"] == pytest.approx(0.3)
    assert stats["own_max_return"] == pytest.approx(0.4)
    assert stats["own_max_return"] == pytest.approx(
        (13_000.0 - buffer.recent[1].money[0]) / MONEY_SCALE
    )
    assert stats["sampled_mean_return"] == pytest.approx(0.3)  # alpha 0: uniform


def test_the_match_agent_reads_final_money_through_the_declared_player(tmp_path):
    """`money[t]` came off `farms[player]` in `main.py`; the terminal read has to
    use the same index, or every reward-to-go label in the season is off by one
    constant. Seat 1 declaring `player: 0` is the case, and 0 is falsy."""

    from agents.dqn_market_free_v3d.match_agent import MatchPolicyAgent
    from agents.dqn_market_free_v3d.training.collect import load_episode_batch

    class Recorder:
        def __init__(self):
            self.finished = []

        def finish(self, final_money, gamma):
            self.finished.append(final_money)
            return _priority_episode(0, final_money, 0.1)

    agent = MatchPolicyAgent.__new__(MatchPolicyAgent)
    agent.recorder = Recorder()
    agent.trace_path = tmp_path / "trace.npz"
    agent.gamma = 0.999
    agent.role = "learner"
    agent.policy_revision_id = "rev"

    farms = [{"money": 111.0}, {"money": 222.0}]
    step = [
        SimpleNamespace(observation={"farms": farms, "player": 0}),
        SimpleNamespace(observation={"farms": farms, "player": 0}),
    ]
    agent.finalize_episode([step], seat=1)
    assert agent.recorder.finished == [111.0]
    assert len(load_episode_batch(agent.trace_path)) == 1

    # No declaration, or one out of range, falls back to the seat.
    step[1].observation = {"farms": farms}
    agent.finalize_episode([step], seat=1)
    step[1].observation = {"farms": farms, "player": 7}
    agent.finalize_episode([step], seat=1)
    assert agent.recorder.finished[1:] == [222.0, 222.0]


def test_best_promoted_score_can_read_a_recent_window():
    from agents.dqn_market_free_v3d.training.updater import _best_promoted_checkpoint_score

    class Reader:
        def __init__(self):
            self.calls = []

        def fetchone(self, sql, parameters):
            self.calls.append((sql, parameters))
            return (4200.0, "recent")

    reader = Reader()
    assert _best_promoted_checkpoint_score(reader, "model", "benchmark", window=6) == (
        4200.0,
        "recent",
    )
    sql, parameters = reader.calls[0]
    assert parameters == ["model", 6]
    assert "AND promoted" in sql
    assert "ORDER BY created_at DESC" in sql
    assert "ORDER BY score DESC" in sql

    reader = Reader()
    _best_promoted_checkpoint_score(reader, "model", "benchmark")
    assert reader.calls[0][1] == ["model"]


def test_self_play_pool_tops_up_with_the_newest_checkpoints(tmp_path):
    from agents.dqn_market_free_v3d.training import updater as updater_module

    promoted = tmp_path / "promoted.npz"
    newest = tmp_path / "newest.npz"
    missing = tmp_path / "missing.npz"
    promoted.write_bytes(b"weights")
    newest.write_bytes(b"weights")

    class Reader:
        def fetchall(self, sql, parameters):
            if "AND promoted" in sql:
                return [(str(promoted),)]
            assert "ORDER BY created_at DESC" in sql
            assert parameters == ["model", 3]
            return [(str(missing),), (str(promoted),), (str(newest),)]

    context = SimpleNamespace(
        model_id="model",
        config={
            "promotion_rule": "activity_gate",
            "self_play_mature_backfill": False,
            "self_play_latest_backfill": True,
        },
        active_checkpoint=None,
    )

    pool = updater_module._mixed_self_play_pool(context, Reader(), 3)

    assert pool == [
        updater_module._weights_spec(str(promoted)),
        updater_module._weights_spec(str(newest)),
    ]

    context.config["self_play_latest_backfill"] = False
    assert updater_module._mixed_self_play_pool(context, Reader(), 3) == [
        updater_module._weights_spec(str(promoted))
    ]


def test_bootstrap_clamp_defaults_and_disable():
    from agents.dqn_market_free_v3d.training.updater import _bootstrap_clamp

    assert _bootstrap_clamp({}) == (-8.0, 8.0)
    assert _bootstrap_clamp({"bootstrap_clamp": False}) is None
    assert _bootstrap_clamp({"bootstrap_clamp_low": -1.0, "bootstrap_clamp_high": 2.0}) == (
        -1.0,
        2.0,
    )
    with pytest.raises(ValueError):
        _bootstrap_clamp({"bootstrap_clamp_low": 3.0, "bootstrap_clamp_high": 2.0})


def test_relabel_recomputes_a_stored_record_from_its_money_trace():
    from agents.dqn_market_free_v3d.training.reward import MONEY_SCALE, relabel

    record = _priority_episode(0, 8_000.0, 12.0)
    record.money = np.asarray(
        [3000.0, *np.linspace(2500.0, 7000.0, record.steps - 1), 8000.0], dtype=np.float32
    )
    stale = record.reward.copy()
    # No animals on the board, so the potential is zero and the label is the cash.
    record.turn_state = np.zeros_like(record.turn_state)

    relabel(record)

    assert not np.array_equal(record.reward, stale)
    assert record.reward[0] == pytest.approx((2500.0 - 3000.0) / MONEY_SCALE)
    assert record.reward[-1] == pytest.approx((8000.0 - 7000.0) / MONEY_SCALE)
    assert record.reward.sum() == pytest.approx(5000.0 / MONEY_SCALE, rel=1e-4)
    assert np.array_equal(record.reward, record.raw_reward)


def test_ladder_pack_labels_follow_the_current_reward(monkeypatch, tmp_path):
    from agents.dqn_market_free_v3d.training import ladder as ladder_module
    from agents.dqn_market_free_v3d.training.collect import save_episode_batch
    from agents.dqn_market_free_v3d.training.reward import episode_rewards

    record = _priority_episode(0, 52_000.0, 0.7)
    record.money = np.asarray(
        [3000.0, *np.linspace(3000.0, 40000.0, record.steps - 1), 52_000.0],
        dtype=np.float32,
    )
    (tmp_path / "pack.json").write_text(json.dumps({"provenance": [], "stats": {}}))
    save_episode_batch([record], tmp_path / "part-0000.npz")
    monkeypatch.setattr(ladder_module, "rehidden", lambda record, gru: record)

    records, _ = ladder_module.load_ladder_demonstrations(tmp_path, gru=None, limit=1)

    expected, _, _ = episode_rewards(
        list(record.money[:-1]), float(record.money[-1]), 0.999, turn_state=record.turn_state
    )
    assert records[0].reward.tolist() == pytest.approx(expected.tolist())


# --------------------------------------------------------------------------
# Widths
# --------------------------------------------------------------------------


def test_state_width_names_its_own_arithmetic():
    assert MARKET_STATE_DIM == (
        238
        - HISTORY_BLOCK_DIM
        + REFUSAL_BLOCK_DIM
        + MARKET_FLOW_DIM
        + INTRA_TURN_DIM
        + HERD_VIABILITY_DIM
        + CROP_MATURITY_DIM
    )
    # Three of block D's thirty came back and twenty-seven did not.
    assert REFUSAL_BLOCK_DIM == 3
    # Block F: flow, own emitted and the day window per product, three tick-phase
    # scalars, and two headroom bands per product.
    assert MARKET_FLOW_DIM == 3 * len(PRODUCTS) + 3 + 2 * len(PRODUCTS)
    assert MARKET_STATE_DIM == 296
    # Per own crop: tiles standing, mean maturity, soonest first yield. Plus the one
    # value scalar. Pinned against the roster so a sixth crop cannot silently widen it.
    assert CROP_MATURITY_DIM == 3 * len(donor.CROPS) + 1
    assert HISTORY_BLOCK_DIM == 30, "the block the GRU replaced was 30 columns wide"
    # One affordability column per animal, plus the three species triples. Pinned
    # against the roster rather than the literal 10 so a fourth animal cannot widen
    # the vector without this failing.
    assert HERD_VIABILITY_DIM == 1 + 3 * len(donor.ANIMALS)
    assert HEAD_IN == MARKET_STATE_DIM + HIDDEN_SIZE
    assert INTRA_TURN_SLICE == slice(MARKET_STATE_DIM - INTRA_TURN_DIM, MARKET_STATE_DIM)


def test_no_gru_key_can_inflate_the_head_layer_count():
    """`load` counts head layers by regex, and the GRU keys must not match it.

    Asserted rather than left to inspection because the consequence is silent:
    a `gru_*` key that matched would make `data[f"w{count - 1}"]` raise KeyError,
    which `main.load_network` swallows into a fallback to `zeros()`. Challenger
    and incumbent would then play as one policy and every paired margin would
    come out exactly 0.0.
    """

    for key in ("gru_wx", "gru_wh", "gru_bx", "gru_bh", "hidden_size"):
        assert not _LAYER_KEY.fullmatch(key)


# --------------------------------------------------------------------------
# numpy / torch parity
# --------------------------------------------------------------------------


def test_numpy_and_torch_recurrences_agree_over_a_sequence():
    module = _module()
    network = net_module.to_numpy(module)
    rng = np.random.default_rng(0)
    batch, length, decisions = 4, 64, 3
    turns = rng.standard_normal((batch, length, STATE_DIM)).astype(np.float32)
    states = rng.standard_normal((batch, length, decisions, STATE_DIM)).astype(np.float32)
    # Not zeros. At `h0 = 0` the hidden projection and `b_hh` are both invisible on
    # the first step and enter only faintly afterwards, so a transposed `wh` shows
    # up as slow drift rather than as a failure.
    hidden = rng.standard_normal((batch, HIDDEN_SIZE)).astype(np.float32)

    with torch.no_grad():
        rolled = module.recur(torch.from_numpy(turns), torch.from_numpy(hidden))
        expected = module.head_forward(torch.from_numpy(states), rolled[:, :, None, :])
    observed, final = network.forward_sequence(turns, states, hidden)

    assert observed.shape == (batch, length, decisions, ACTION_COUNT)
    assert np.allclose(final, rolled[:, -1].numpy(), atol=1e-5)
    assert np.allclose(observed, expected.numpy(), atol=1e-4)


def _reset_before_bias(gru, x, h):
    """The plausible wrong GRU: the reset gate applied before `b_hn` is added."""

    size = gru.hidden_size
    gx = x @ gru.wx + gru.bx
    hh = h @ gru.wh
    r = _sigmoid(gx[:, :size] + hh[:, :size] + gru.bh[:size])
    z = _sigmoid(gx[:, size : 2 * size] + hh[:, size : 2 * size] + gru.bh[size : 2 * size])
    n = np.tanh(gx[:, 2 * size :] + r * hh[:, 2 * size :] + gru.bh[2 * size :])
    return (1.0 - z) * n + z * h


def test_the_reset_gate_is_applied_after_the_hidden_bias():
    """The one parity break that is invisible at default biases.

    torch computes `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`. The plausible
    wrong form -- reset before the bias -- differs by exactly `(1 - r) * b_hn`,
    which is zero only while `b_hn` is zero, that is, never after the first
    optimizer step. So `b_hn` is driven large here and `b_in` to zero, and the
    test asserts both that numpy matches torch *and* that the wrong reference does
    not. The second half is what stops this passing vacuously.
    """

    module = _module(seed=1, bias_hh=3.0, bias_ih=0.0)
    network = net_module.to_numpy(module)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((16, STATE_DIM)).astype(np.float32)
    h = rng.standard_normal((16, HIDDEN_SIZE)).astype(np.float32)

    with torch.no_grad():
        expected = module.cell(torch.from_numpy(x), torch.from_numpy(h)).numpy()
    assert np.allclose(network.gru.step(x, h), expected, atol=1e-5)
    assert not np.allclose(_reset_before_bias(network.gru, x, h), expected, atol=1e-3)


def _swapped_legs(gru, x, h):
    """The plausible wrong GRU: `z` treated as the write gate instead of the keep
    gate, i.e. `z * n + (1 - z) * h`."""

    size = gru.hidden_size
    gx = x @ gru.wx + gru.bx
    gh = h @ gru.wh + gru.bh
    r = _sigmoid(gx[:, :size] + gh[:, :size])
    z = _sigmoid(gx[:, size : 2 * size] + gh[:, size : 2 * size])
    n = np.tanh(gx[:, 2 * size :] + r * gh[:, 2 * size :])
    return z * n + (1.0 - z) * h


def test_the_update_gate_keeps_rather_than_writes():
    """`h' = (1 - z) * n + z * h`. Swapping the legs is a different network.

    Held to the same standard as the reset-gate test: match torch, and *fail* to
    match the wrong reference. A large positive update-gate bias -- `bias_iz`, the
    middle third of `bias_ih` -- pins `z` near 1, so the correct form carries `h`
    almost unchanged while the swapped one almost discards it, which is what makes
    the two separable at all.
    """

    module = _module(seed=2, bias_ih=0.0, bias_hh=0.0)
    with torch.no_grad():
        module.cell.bias_ih[HIDDEN_SIZE : 2 * HIDDEN_SIZE].fill_(8.0)
    network = net_module.to_numpy(module)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((8, STATE_DIM)).astype(np.float32)
    h = rng.standard_normal((8, HIDDEN_SIZE)).astype(np.float32)

    with torch.no_grad():
        expected = module.cell(torch.from_numpy(x), torch.from_numpy(h)).numpy()
    assert np.allclose(network.gru.step(x, h), expected, atol=1e-5)
    assert not np.allclose(_swapped_legs(network.gru, x, h), expected, atol=1e-3)


def test_the_recurrence_does_not_drift_over_a_full_episode():
    """720 ticks. A per-step bias a short test cannot see compounds here."""

    module = _module(seed=4)
    network = net_module.to_numpy(module)
    rng = np.random.default_rng(11)
    turns = rng.standard_normal((2, 720, STATE_DIM)).astype(np.float32)
    hidden = rng.standard_normal((2, HIDDEN_SIZE)).astype(np.float32)

    with torch.no_grad():
        expected = module.recur(torch.from_numpy(turns), torch.from_numpy(hidden))
        expected = expected[:, -1].numpy()
    observed = hidden.copy()
    for step in range(turns.shape[1]):
        observed = network.gru.step(turns[:, step], observed)

    error = float(np.abs(observed - expected).max())
    assert error < 1e-4, f"hidden drifted {error:.3e} over 720 ticks"


def test_the_stateful_path_is_the_same_arithmetic_as_the_batched_one():
    """The submission's path and the training path must not diverge from each other.

    `forward_sequence` loops over the identical `GruCell.step` and `_head` the
    stateful inference path calls, so the two are the same *expressions* -- but not
    bit-identical results, and the reason is worth writing down rather than
    absorbing into a loose tolerance. Inference scores a rank-1 state against a
    rank-2 weight and the batched path scores a rank-3 one, so BLAS dispatches a
    dot kernel in one case and a gemm in the other, and the two accumulate in
    different orders. The gap is pure float32 rounding, ~2e-7 here.

    The tolerance is therefore set just above that and nowhere near the 1e-4 the
    numpy-versus-torch tests use: anything larger than rounding means one of the
    two paths has grown arithmetic the other does not have, which is exactly how a
    submission and the training runs that produced it come to disagree while every
    cross-framework parity test still passes.
    """

    network = net_module.to_numpy(_module(seed=6))
    rng = np.random.default_rng(13)
    length, decisions = 32, 3
    turns = rng.standard_normal((1, length, STATE_DIM)).astype(np.float32)
    states = rng.standard_normal((1, length, decisions, STATE_DIM)).astype(np.float32)

    network.reset_hidden()
    rows = []
    for step in range(length):
        network.tick(turns[0, step])
        rows.append(np.stack([network(states[0, step, k]) for k in range(decisions)]))
    batched, _final = network.forward_sequence(turns, states, network.gru.zeros(1))

    error = float(np.abs(np.stack(rows) - batched[0]).max())
    assert error < 1e-6, f"the two paths differ by {error:.3e}, beyond float32 rounding"


def test_the_masked_choice_agrees_between_numpy_and_torch_over_a_sequence():
    """Integer equality on the argmax, not a tolerance on Q.

    A 1e-4 gap that flips one argmax is a real behavioural divergence that every
    float comparison above passes on.
    """

    module = _module(seed=8)
    network = net_module.to_numpy(module)
    rng = np.random.default_rng(17)
    batch, length, decisions = 3, 48, 2
    turns = rng.standard_normal((batch, length, STATE_DIM)).astype(np.float32)
    states = rng.standard_normal((batch, length, decisions, STATE_DIM)).astype(np.float32)
    hidden = rng.standard_normal((batch, HIDDEN_SIZE)).astype(np.float32)
    mask = rng.random((batch, length, decisions, ACTION_COUNT)) > 0.5
    mask[..., 0] = True  # NOOP is always available, as `feasible_actions` guarantees

    with torch.no_grad():
        rolled = module.recur(torch.from_numpy(turns), torch.from_numpy(hidden))
        expected = module.head_forward(torch.from_numpy(states), rolled[:, :, None, :]).numpy()
    observed, _final = network.forward_sequence(turns, states, hidden)

    torch_choice = np.where(mask, expected, MASKED_SCORE).argmax(axis=-1)
    numpy_choice = np.where(mask, observed, MASKED_SCORE).argmax(axis=-1)
    disagreed = int((torch_choice != numpy_choice).sum())
    assert disagreed == 0, f"{disagreed} of {torch_choice.size} decisions disagreed"


@pytest.mark.parametrize("noisy", ["head", "all"])
def test_noisy_numpy_and_torch_agree_over_a_sequence(noisy):
    """One epsilon draw, held across the whole turn.

    `main.py` redraws once per turn rather than once per sub-decision, so the
    parity test has to hold the draw the same way or it is comparing two different
    policies.
    """

    module = _module(seed=9, noisy=noisy)
    network = net_module.to_numpy(module)
    generator = torch.Generator().manual_seed(4)
    net_module.reset_noise(module, generator)
    network._noise = [
        (layer.epsilon_in.numpy().copy(), layer.epsilon_out.numpy().copy())
        if isinstance(layer, net_module.NoisyLinear)
        else None
        for layer in net_module._decompose(module)[1]
    ]

    rng = np.random.default_rng(19)
    turns = rng.standard_normal((2, 24, STATE_DIM)).astype(np.float32)
    states = rng.standard_normal((2, 24, 2, STATE_DIM)).astype(np.float32)
    hidden = rng.standard_normal((2, HIDDEN_SIZE)).astype(np.float32)

    with torch.no_grad():
        rolled = module.recur(torch.from_numpy(turns), torch.from_numpy(hidden))
        expected = module.head_forward(torch.from_numpy(states), rolled[:, :, None, :]).numpy()
    observed, _final = network.forward_sequence(turns, states, hidden)
    assert np.allclose(observed, expected, atol=1e-4)
    # And the noise is doing something, so this is not a comparison of two
    # unperturbed networks.
    network.clear_noise()
    quiet, _ = network.forward_sequence(turns, states, hidden)
    assert not np.allclose(quiet, observed, atol=1e-6)


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_the_recurrence_is_never_noisy():
    """The invariant the TD step's `reset_noise` ordering rests on.

    v3 runs the *graded* GRU rollout before two `reset_noise` calls, which v2's
    ordering comment forbids -- legally, but only because nothing in the recurrent
    graph references an epsilon buffer. Put a `NoisyLinear` in the cell and the
    ordering silently becomes wrong.
    """

    for noisy in ("none", "head", "all"):
        module = net_module.build(noisy=noisy)
        assert not any(
            isinstance(layer, net_module.NoisyLinear) for layer in module.cell.modules()
        )


def test_an_export_raises_on_a_parameter_it_does_not_know_how_to_write():
    """`_decompose` accounts for parameters by identity, not by isinstance.

    The module is no longer a flat `Sequential`, so walking the children and
    recognising each one is no longer a proof of completeness -- a sub-module hung
    off `OrderQModule` and forgotten in the exporter would never be visited at all.
    """

    module = net_module.build()
    module.stray = torch.nn.Linear(4, 4)
    with pytest.raises(TypeError, match="cannot export parameters"):
        net_module.to_numpy(module)


def test_a_quiet_start_is_silent_but_not_inert():
    network = OrderQNetwork.quiet_start(seed=5)
    network.reset_hidden()
    rng = np.random.default_rng(23)
    network.tick(rng.standard_normal(STATE_DIM).astype(np.float32))
    scores = network(rng.standard_normal(STATE_DIM).astype(np.float32))
    assert np.allclose(scores, 0.0), "every action must tie, so the argmax falls to NOOP"
    # The head trunk is live, which is what `zeros()` cannot say...
    assert not net_module.has_inert_hidden_trunk(network)
    # ...and so is the recurrence.
    assert not net_module.has_inert_recurrence(network)
    assert net_module.has_inert_recurrence(OrderQNetwork.zeros())


def test_a_noisy_initial_network_keeps_sigma_after_save(tmp_path):
    """A noisy run's first checkpoint must not silently become deterministic."""

    from agents.dqn_market_free_v3d.training.updater import initial_network

    network = initial_network(
        {
            "seed": 5,
            "noisy": True,
            "noisy_layers": "head",
            "noisy_sigma0": 0.5,
        }
    )
    assert network.is_noisy
    assert [pair is not None for pair in network.sigma] == [False, False, True]

    path = tmp_path / "initial.npz"
    network.save(path)
    loaded = OrderQNetwork.load(path)
    assert loaded.is_noisy
    assert [pair is not None for pair in loaded.sigma] == [False, False, True]
    assert np.allclose(loaded.layers[-1][0], 0.0)
    assert np.allclose(loaded.layers[-1][1], 0.0)


def test_an_all_zero_network_stays_silent_for_the_whole_episode():
    """The inference fallback ties on turn 720, not only on turn 1.

    At zero weights `h' = (1 - 0.5) * tanh(0) + 0.5 * h`, so a hidden state that
    starts at zero stays there and the head keeps seeing the same dead columns.
    """

    network = OrderQNetwork.zeros()
    network.reset_hidden()
    rng = np.random.default_rng(29)
    for _ in range(50):
        network.tick(rng.standard_normal(STATE_DIM).astype(np.float32))
    assert np.allclose(network.hidden, 0.0)
    assert np.allclose(network(rng.standard_normal(STATE_DIM).astype(np.float32)), 0.0)


def test_scoring_before_a_reset_raises_rather_than_scoring_against_zeros():
    """A network that silently scored against zeros would be indistinguishable
    from a trained policy that had learned to ignore its memory."""

    network = OrderQNetwork.quiet_start(seed=2)
    with pytest.raises(RuntimeError, match="reset_hidden"):
        network(np.zeros(STATE_DIM, dtype=np.float32))
    with pytest.raises(RuntimeError, match="reset_hidden"):
        network.tick(np.zeros(STATE_DIM, dtype=np.float32))


def test_the_stateful_and_batched_paths_reject_each_others_input():
    network = OrderQNetwork.quiet_start(seed=3)
    network.reset_hidden()
    with pytest.raises(ValueError, match="single-decision path"):
        network(np.zeros((4, STATE_DIM), dtype=np.float32))
    with pytest.raises(ValueError, match="single-sample path"):
        network.tick(np.zeros((4, STATE_DIM), dtype=np.float32))


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------


def test_a_checkpoint_round_trips_and_a_pre_v3_one_is_refused_by_name(tmp_path):
    network = net_module.to_numpy(_module(seed=12, noisy="head"))
    path = tmp_path / "weights.npz"
    network.save(path)
    loaded = OrderQNetwork.load(path)

    for name in ("wx", "wh", "bx", "bh"):
        assert_array_equal(getattr(loaded.gru, name), getattr(network.gru, name))
    assert len(loaded.layers) == len(network.layers)
    assert loaded.is_noisy

    # A file with the head but no recurrence is a pre-v3 checkpoint, and must be
    # refused by name rather than by width -- `STATE_DIM + 30` is v2's head width
    # exactly, so a future `HIDDEN_SIZE` of 30 would let one load as a v3 network
    # with no memory at all.
    payload = dict(np.load(path))
    for name in ("gru_wx", "gru_wh", "gru_bx", "gru_bh"):
        payload.pop(name)
    stale = tmp_path / "v2.npz"
    np.savez_compressed(stale, **payload)
    with pytest.raises(ValueError, match="carries no GRU"):
        OrderQNetwork.load(stale)


def test_half_a_gru_is_refused(tmp_path):
    network = net_module.to_numpy(_module(seed=14))
    path = tmp_path / "weights.npz"
    network.save(path)
    payload = dict(np.load(path))
    payload.pop("gru_wh")
    partial = tmp_path / "partial.npz"
    np.savez_compressed(partial, **payload)
    with pytest.raises(ValueError, match="partial GRU"):
        OrderQNetwork.load(partial)


# The herd, and the ladder that made half of it unsellable
# --------------------------------------------------------------------------


def _pending(**kwargs) -> tuple[PendingTurn, tuple]:
    state = _mask_state(**kwargs)
    obs, farm, private, field = state
    return PendingTurn(obs, farm, private, field, None), state


def _herd_state(shed: dict | None = None, day: int = 0, tiles: dict | None = None):
    """A board with one unlocked quadrant, and whatever the case puts on it."""

    obs, farm, private, _field = _mask_state(shed=shed, day=day)
    for position, tile in (tiles or {}).items():
        x, y = position
        farm["tiles"][y][x] = tile
    obs = {**obs, "player": 0, "farms": [farm]}
    return obs, farm, private


def _animal_roles(roles: dict) -> dict:
    return {
        position: item
        for position, (kind, item) in roles.items()
        if kind == "ANIMAL"
    }


def test_the_donors_speculative_herd_survives_the_overlay():
    """The strip is gone, and this is the test that used to pin it.

    It asserted `len(roles) == WARM_SLOTS` -- one animal role on a zero-animal
    board, the donor's four thrown away -- on the grounds that 52% of episodes never
    bought the animal and those episodes ended 16% poorer.

    Both halves of that failed. The waste it named was already refused one layer
    down: `donor._field_jobs` emits `BUILD_` only at `left >= animal_rule["first"]
    + 2` and never during liquidation, so a role the season cannot pay back yields
    no job. And the 52% was measured on a learner that was not buying animals, which
    is circular -- the strip deletes the housing, the housing is what makes a
    purchase land. `market-free-splitladder-v1` averages 9.7 cows, 7.4 sheep and 8.9
    geese an episode.

    Measured over 24 seeds in both orientations with the donor's market rules on
    both sides: 50,240 with the strip against 83,525 without, and pilkwang held to
    86,111 instead of running to 122,515.
    """

    obs, farm, private = _herd_state()
    base = _animal_roles(donor._role_plan(obs, farm))
    assert len(base) >= donor.CORE_HERD, "the donor still plans a speculative herd"

    roles = _animal_roles(
        herd.herd_influenced_roles(obs, farm, private, donor._role_plan(obs, farm))
    )
    assert len(roles) >= len(base), "the overlay adds housing, it does not take it away"


def test_the_late_season_build_is_refused_by_the_donor_not_by_the_overlay():
    """Where the protection the strip claimed actually lives.

    This is the guard that makes deleting the strip safe, so it is pinned here
    rather than left to `donor.py` -- a speculative role late in the season is free
    precisely because no `BUILD_` job comes out of it.
    """

    obs, farm, private = _herd_state()
    roles = herd.field_roles(obs, farm, private)
    animal_tiles = [position for position, role in roles.items() if role[0] == "ANIMAL"]
    assert animal_tiles, "the fixture has animal roles to build on"

    def _builds(day):
        jobs = donor._field_jobs(
            {**obs, "day": day}, farm, private, roles, False
        )
        return [job for job in jobs if str(job["action"][0]).startswith("BUILD_")]

    assert _builds(int(obs.get("day", 0) or 0)), "early, the build is planned"
    assert not _builds(donor.TOTAL_DAYS - 1), "late, the donor refuses it itself"


def test_a_bought_animal_gets_a_structure_whatever_the_species():
    """Every species, one turn, symmetrically. The goose is the case that regressed.

    v1 housed a shed-held goose and nothing else: `donor._role_plan` emits COW and
    SHEEP only, so a bought bird's coop came from `herd.py` while a bought cow landed
    on a pasture the donor had already built speculatively. The learner priced that
    asymmetry correctly and stopped buying geese.
    """

    for animal in herd.ANIMALS:
        obs, farm, private = _herd_state(shed={animal: 2})
        roles = _animal_roles(
            herd.herd_influenced_roles(obs, farm, private, donor._role_plan(obs, farm))
        )
        housed = [item for item in roles.values() if item == animal]
        assert len(housed) >= 2, f"{animal} bought twice, housed {len(housed)}"


def test_a_waiting_goose_takes_a_bare_pasture_when_no_ground_is_left():
    """A goose bought onto a board of standing pasture gets a role, not a shed.

    `market-free-crop-v1` generation 12 bought 5.0 geese an episode and placed at
    most 1.3: every reserved slot carried a bare PASTURE, pass one roled each one
    COW, and the claim pass only takes dirt or weeds. The bare pasture that no cow
    is waiting for must be handed to the goose so the donor's dig / BUILD_COOP /
    PLACE chain can run. A pasture a waiting cow is queued for stays the cow's.
    """

    obs, farm, private = _herd_state(shed={"GOOSE": 1})
    slots, _zones = donor._reserved_animal_slots(farm)
    for x, y in slots:
        farm["tiles"][y][x] = {"kind": "PASTURE"}
    roles = _animal_roles(herd.field_roles(obs, farm, private))
    assert "GOOSE" in roles.values(), "goose left in the shed on an all-pasture board"

    # One bare pasture, one waiting cow, one waiting goose: the cow keeps it.
    obs, farm, private = _herd_state(shed={"GOOSE": 1, "COW": 1})
    slots, _zones = donor._reserved_animal_slots(farm)
    for x, y in slots:
        farm["tiles"][y][x] = {"kind": "PLANT", "crop": "WHEAT"}
    x, y = slots[0]
    farm["tiles"][y][x] = {"kind": "PASTURE"}
    roles = _animal_roles(herd.field_roles(obs, farm, private))
    assert roles.get((x, y)) == "COW"


def test_a_placed_animal_keeps_its_own_species_role():
    """The role repair, from the direction that destroys an animal.

    A COW role standing on a COOP is the priority-3 `replace_incompatible_structure`
    dig. The dig spares a tile that already carries an animal, but the role is what
    `_seed_needs`, the state's `role_targets` block and the liquidation path read.
    """

    obs, farm, private = _herd_state(
        tiles={(1, 1): {"kind": "COOP", "animal": "GOOSE", "yield_units": 0}}
    )
    roles = _animal_roles(
        herd.herd_influenced_roles(obs, farm, private, donor._role_plan(obs, farm))
    )
    assert roles[(1, 1)] == "GOOSE"


def test_a_bare_structure_is_not_torn_down_or_double_counted():
    """A built coop keeps a goose role and absorbs the warm slot.

    Every animal passes through a bare structure between `BUILD_` and `PLACE`, and
    the empty-structure state is also what the simulator's escape rule leaves behind.
    Neither may read as "build another one".
    """

    obs, farm, private = _herd_state(tiles={(1, 1): {"kind": "COOP"}})
    planned = donor._role_plan(obs, farm)
    roles = _animal_roles(herd.herd_influenced_roles(obs, farm, private, planned))

    assert roles[(1, 1)] == "GOOSE", "the built coop keeps its species"
    # The claim is absorption, not emptiness: the coop satisfies `WARM_SLOTS`, so the
    # overlay adds nothing beyond it. Asserted against the donor's plan rather than
    # against `{}` -- the plan used to be stripped here and is not any more.
    assert set(roles) == set(_animal_roles(planned)) | {(1, 1)}


def test_the_warm_slot_closes_when_nothing_can_still_produce():
    """Past every deadline a structure is a hand-turn spent on an asset that cannot pay."""

    obs, farm, private = _herd_state(day=donor.TOTAL_DAYS - 4)
    assert herd.warm_slot_species(obs, int(obs["day"])) is None
    roles = _animal_roles(
        herd.herd_influenced_roles(obs, farm, private, donor._role_plan(obs, farm))
    )
    assert roles == {}


def test_the_warm_slot_follows_the_market_the_herd_is_selling_into():
    """One cow is worth more than one goose; the third cow is not.

    The scorer prices the *herd's* remaining output, so MILK walks down its own
    `linear 1.60` curve while EGG's `log 0.20` barely moves. This is the fleet effect
    the Kaggle loss was made of and it cannot be seen one animal at a time.
    """

    assert herd.warm_slot_species({}, 0, {}) == "COW"
    assert herd.warm_slot_species({}, 0, {"COW": 1}) == "GOOSE"


def test_remaining_yield_events_agrees_with_the_ledger():
    """`herd` ships in the Kaggle bundle and `lots` does not, so the arithmetic is
    duplicated. It must not drift: the state column and the waste charge are the
    same number."""

    for animal in herd.ANIMALS:
        for day in range(donor.TOTAL_DAYS):
            assert herd.remaining_yield_events(animal, day) == lots.unplaced_events(
                animal, day
            )


def test_a_buy_animal_beyond_free_housing_is_shut():
    """The leak this gate exists to close.

    `donor.ANIMAL_SLOTS` is 15 tiles board-wide and 4 in the opening quadrant, but
    `shedCapacity` is 100, so affordability alone let a turn buy ~100 head that
    `herd.herd_influenced_roles` can never find ground for. Measured at generation
    17 of `market-free-refusal-v1`, the five worst learner slots were each spending
    ~37,000 coins an episode on animals and recovering ~4,000, at an
    `animal_lots_matched_share` of 0.11.
    """

    obs, farm, private, field = _mask_state(money=100_000)
    assert len(donor._reserved_animal_slots(farm)[0]) == 4, "the opening quadrant"
    assert animal_room(obs, farm, private) == {a: 4 for a in herd.ANIMALS}

    # More head waiting in the shed than the board can ever house.
    crowded = {**private, "shed": {**(private.get("shed") or {}), "COW": 50}}
    assert animal_room(obs, farm, crowded) == {}, "no proof of a slot, no open row"
    pending = PendingTurn(obs, farm, crowded, field, None)
    mask = feasible_actions(obs, farm, crowded, field, None, pending=pending)
    for animal in herd.ANIMALS:
        assert not mask[_BUY_ANIMAL_ROWS[animal]].any(), animal


def test_the_rung_is_capped_at_free_housing_not_shut_wholesale():
    """A cap, not a ban. Two free tiles still buy two head."""

    obs, farm, private, field = _mask_state(money=100_000)
    crowded = {**private, "shed": {**(private.get("shed") or {}), "COW": 2}}
    assert animal_room(obs, farm, crowded) == {a: 2 for a in herd.ANIMALS}

    pending = PendingTurn(obs, farm, crowded, field, None)
    mask = feasible_actions(obs, farm, crowded, field, None, pending=pending)
    rows = _BUY_ANIMAL_ROWS["COW"]
    for index, quantity in enumerate(_ROW_QUANTITY[rows]):
        assert bool(mask[rows][index]) == (int(quantity) <= 2), quantity


def test_a_turn_cannot_stuff_more_animals_than_it_has_tiles():
    """The `PendingTurn` half. Without it the mask is right once and wrong four
    times, because the wire holds ten orders and the board holds four tiles."""

    obs, farm, private, field = _mask_state(money=100_000)
    pending = PendingTurn(obs, farm, private, field, None)
    rows = _BUY_ANIMAL_ROWS["COW"]
    assert feasible_actions(obs, farm, private, field, None, pending=pending)[rows].all()

    pending.apply(Order(op="BUY_ANIMAL", item="COW", quantity=4))
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    # Every species, not just the one bought: COW and SHEEP share the pasture and a
    # bare COOP is re-roled by `herd._species_for`, so the ground is gone for all.
    for animal in herd.ANIMALS:
        assert not mask[_BUY_ANIMAL_ROWS[animal]].any(), animal


@pytest.mark.parametrize("animal", sorted(herd.ANIMALS))
def test_an_animal_that_cannot_produce_before_the_season_ends_is_unbuyable(animal):
    """The calendar leg. `market_state.py:812-815` names this exact hole: past day
    21 a cow cannot milk once, the row stayed open, and the waste term charged the
    full 400 coins. Tied to `remaining_yield_events` rather than to a literal, so it
    cannot drift from the charge `lots.unplaced_events` prices."""

    dead = next(
        day
        for day in range(1, donor.TOTAL_DAYS + 1)
        if herd.remaining_yield_events(animal, day) == 0
    )
    obs, farm, private, field = _mask_state(money=100_000)

    alive = {**obs, "day": dead - 1}
    assert animal_room(alive, farm, private).get(animal, 0) > 0, "open the day before"

    expired = {**obs, "day": dead}
    assert animal in animal_room(alive, farm, private)
    assert animal not in animal_room(expired, farm, private)
    pending = PendingTurn(expired, farm, private, field, None)
    mask = feasible_actions(expired, farm, private, field, None, pending=pending)
    assert not mask[_BUY_ANIMAL_ROWS[animal]].any()


def test_buying_land_widens_the_animal_rows():
    """The gate is recomputed per turn off `_reserved_animal_slots`, so unlocking a
    quadrant reopens rows rather than freezing them at the NW count."""

    obs, farm, private, field = _mask_state(money=100_000)
    # The flag alone is not the purchase: `_reserved_animal_slots` skips a "LOCKED"
    # tile whatever the quadrant list says, so the ground has to open too.
    tiles = [list(row) for row in farm["tiles"]]
    for y in range(5):
        for x in range(5, 10):
            tiles[y][x] = None
    wide = {**farm, "tiles": tiles, "unlocked_quadrants": ["NW", "NE"]}

    assert len(donor._reserved_animal_slots(farm)[0]) == 4
    assert len(donor._reserved_animal_slots(wide)[0]) == 4 + donor.ANIMAL_SLOTS["NE"]
    assert animal_room(obs, wide, private)["COW"] > animal_room(obs, farm, private)["COW"]


def test_land_pressure_reads_the_plan_not_just_the_board():
    """A full farm is not automatically a farm that needs more ground."""

    stocked = {
        "MELON": 40,
        "WHEAT": 40,
        "CARROT": 40,
        "TOMATO": 40,
        "STRAWBERRY": 40,
    }
    obs, farm, private, _ = _mask_state(
        money=100_000.0, day=5, plants=25, seeds=stocked
    )
    roles = herd.field_roles(obs, farm, private)
    thin_roles = {(0, 0): ("CROP", "WHEAT"), (1, 0): ("CROP", "WHEAT")}

    assert land_expansion_pressure(obs, farm, roles) == 1.0
    assert land_expansion_pressure(obs, farm, thin_roles) == pytest.approx(2 / 25)
    assert land_score(obs, farm, private, farm["money"], 0, thin_roles) < 0.15
    assert not should_buy_land(obs, farm, private, thin_roles)
    assert should_buy_land(obs, farm, private, roles)


def test_land_pressure_counts_animal_space_as_productive_demand():
    """Animal roles do not consume seed, but they still make acreage scarce."""

    obs, farm, private, _ = _mask_state(money=100_000.0, day=5, plants=0)
    animal_roles = {
        (0, 0): ("ANIMAL", "COW"),
        (1, 0): ("ANIMAL", "SHEEP"),
        (2, 0): ("ANIMAL", "GOOSE"),
    }

    assert land_expansion_pressure(obs, farm, animal_roles) == pytest.approx(3 / 25)


def test_thin_markets_lose_the_rungs_that_crater_them():
    """WOOL and MILK are the thinnest markets in the game; WHEAT and EGG are not."""

    pending, (obs, farm, private, field) = _pending(
        money=3000, shed={"WHEAT": 45, "EGG": 45, "WOOL": 45, "MILK": 45}
    )
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)

    def top(item: str) -> int:
        return max(
            ACTIONS[index].quantity
            for index in np.flatnonzero(mask)
            if ACTIONS[index].op == "SELL" and ACTIONS[index].item == item
        )

    assert top("WHEAT") == 45 and top("EGG") == 45
    assert top("WOOL") < 45 and top("MILK") < top("WOOL")


def test_the_cap_tightens_as_the_turn_sells_into_it():
    """The cap is per order and reprices itself off the inventory its own sales made,
    which is what makes a second budget unnecessary."""

    pending, (obs, farm, private, field) = _pending(money=3000, shed={"MILK": 90})

    def cap() -> int:
        return sell_depth_cap(
            "MILK", int(pending.market_inventory.get("MILK", donor.MARKET_I0)), obs
        )

    before = cap()
    # Rung-quantised, so one order at the cap need not move it a whole rung. Four
    # do, and four is what a turn with slots to spare would emit.
    for _ in range(4):
        pending.apply(Order("SELL", "MILK", before))
    assert cap() < before


def test_liquidation_lifts_the_cap():
    """Stock the season ends holding is worth zero however cheaply it went, and the
    endgame is short of wire slots rather than of price."""

    pending, (obs, farm, private, field) = _pending(money=3000, shed={"WOOL": 45})
    field = {**field, "liquidation": True}
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert max(
        ACTIONS[index].quantity
        for index in np.flatnonzero(mask)
        if ACTIONS[index].op == "SELL" and ACTIONS[index].item == "WOOL"
    ) == 45


def test_one_hire_action_per_turn():
    """`HIRE 7` then `HIRE 3` is ten `["HIRE"]` entries, which is `HIRE 10`'s wire.

    The ladder exists so that each reachable hand count has exactly one row. A
    second hire action in the same turn rebuilds the alias one sub-decision later,
    and -- worse -- makes the turn unrecoverable from its own replay, because
    `clone._chosen_sequence` can only map that wire back to a single row.
    """

    pending, (obs, farm, private, field) = _pending(money=40_000)
    pending.apply(Order("HIRE", None, 3))
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert not any(ACTIONS[index].op == "HIRE" for index in np.flatnonzero(mask))


# --------------------------------------------------------------------------
# Sequence replay and the TD step
# --------------------------------------------------------------------------


def _episode(seed: int = 0, turns: int = 200, rate: int = 3):
    """A synthetic episode with a variable number of decisions per turn."""

    from agents.dqn_market_free_v3d.training.collect import HIDDEN_STRIDE, EpisodeRecord

    rng = np.random.default_rng(seed)
    counts = rng.integers(1, rate + 1, size=turns)
    steps = np.repeat(np.arange(turns), counts).astype(np.int16)
    decisions = int(steps.shape[0])
    stored = int(np.ceil(turns / HIDDEN_STRIDE))
    return EpisodeRecord(
        reward=rng.standard_normal(turns).astype(np.float32),
        raw_reward=rng.standard_normal(turns).astype(np.float32),
        money=rng.standard_normal(turns + 1).astype(np.float32),
        order_state=rng.standard_normal((decisions, STATE_DIM)).astype(np.float32),
        order_chosen=rng.integers(0, ACTION_COUNT, size=decisions).astype(np.int16),
        order_step=steps,
        order_mask=np.ones((decisions, ACTION_COUNT), dtype=np.bool_),
        turn_state=rng.standard_normal((turns, STATE_DIM)).astype(np.float32),
        turn_hidden=rng.standard_normal((stored, HIDDEN_SIZE)).astype(np.float16),
        hidden_stride=np.asarray([HIDDEN_STRIDE], dtype=np.int16),
    )


def _buffer(count: int = 8, **kwargs):
    from agents.dqn_market_free_v3d.training.replay import ReplayBuffer

    buffer = ReplayBuffer(capacity=64, seed=1, **kwargs)
    buffer.add([_episode(seed=index) for index in range(count)])
    return buffer


def _priority_episode(
    index: int,
    final_money: float,
    episode_return: float,
    turns: int = 48,
):
    from agents.dqn_market_free_v3d.training.collect import HIDDEN_STRIDE, EpisodeRecord
    from agents.dqn_market_free_v3d.training.reward import MONEY_SCALE

    rng = np.random.default_rng(index)
    stored = int(np.ceil(turns / HIDDEN_STRIDE))
    # A money trace that opens on the stake and climbs to `final_money`, with the
    # reward the real function of it: `episode_return` is the season profit in
    # reward units, so `reward.sum()` is exactly what `priority_stats` reports.
    stake = final_money - episode_return * MONEY_SCALE
    money = np.linspace(stake, final_money, turns + 1).astype(np.float32)
    return EpisodeRecord(
        reward=(np.diff(money) / MONEY_SCALE).astype(np.float32),
        raw_reward=(np.diff(money) / MONEY_SCALE).astype(np.float32),
        money=money,
        order_state=rng.standard_normal((turns, STATE_DIM)).astype(np.float32),
        order_chosen=np.full(turns, index, dtype=np.int16),
        order_step=np.arange(turns, dtype=np.int16),
        order_mask=np.ones((turns, ACTION_COUNT), dtype=np.bool_),
        turn_state=rng.standard_normal((turns, STATE_DIM)).astype(np.float32),
        turn_hidden=rng.standard_normal((stored, HIDDEN_SIZE)).astype(np.float16),
        hidden_stride=np.asarray([HIDDEN_STRIDE], dtype=np.int16),
    )


def test_priority_draw_ranks_episodes_by_final_money_not_shaped_return():
    from agents.dqn_market_free_v3d.training.replay import ReplayBuffer

    final_money = [500.0, 1_500.0, 3_000.0, 6_000.0, 12_000.0]
    # Deliberately anti-correlated with final money. If the sampler still reads
    # shaped return, this test flips.
    returns = [5.0, 2.0, 0.0, -1.0, -4.0]
    buffer = ReplayBuffer(capacity=16, seed=4, priority_alpha=0.7)
    buffer.add(
        [
            _priority_episode(index, money, ret)
            for index, (money, ret) in enumerate(zip(final_money, returns))
        ]
    )

    _pool, scores, probabilities, weights = buffer._own_priorities()
    by_money = np.argsort(scores)
    assert (np.diff(probabilities[by_money]) > 0.0).all()
    assert weights.max() == pytest.approx(1.0)

    stats = buffer.priority_stats()
    assert stats["sampled_mean_priority_money"] > stats["own_mean_priority_money"]
    assert stats["sampled_mean_return"] < stats["own_mean_return"]

    drawn = np.concatenate([buffer.sample(128).chosen for _ in range(20)])
    empirical = float(np.mean([final_money[int(index)] for index in drawn]))
    assert empirical == pytest.approx(stats["sampled_mean_priority_money"], abs=500.0)
    assert empirical > float(np.mean(final_money))


def test_generation_weight_scales_chunk_loss_without_reward_scaling():
    from agents.dqn_market_free_v3d.training.replay import ReplayBuffer

    record = _episode(seed=5)
    record.episode_weight = 0.4
    buffer = ReplayBuffer(capacity=4, seed=2)
    buffer.add([record])

    batch = buffer.sample_sequences(3, burn_in=20, train=40)

    assert batch.weight.tolist() == pytest.approx([0.4, 0.4, 0.4])


def test_fill_buffer_weights_records_by_generation_final_money(tmp_path):
    from agents.dqn_market_free_v3d.training.collect import save_episode_batch
    from agents.dqn_market_free_v3d.training.replay import ReplayBuffer
    from agents.dqn_market_free_v3d.training.updater import SOURCE_KIND, fill_buffer

    weak_path = tmp_path / "weak.npz"
    strong_path = tmp_path / "strong.npz"
    save_episode_batch([_priority_episode(0, 20_000.0, 1.0)], weak_path)
    save_episode_batch([_priority_episode(1, 20_000.0, 1.0)], strong_path)
    batches = [
        SimpleNamespace(
            source_kind=SOURCE_KIND,
            artifact_path=weak_path,
            artifact_sha256=None,
            sample_count=1,
            metadata={"generation_id": "g0001", "mean_final_money": 1_000.0},
        ),
        SimpleNamespace(
            source_kind=SOURCE_KIND,
            artifact_path=strong_path,
            artifact_sha256=None,
            sample_count=1,
            metadata={"generation_id": "g0002", "mean_final_money": 9_000.0},
        ),
    ]
    buffer = ReplayBuffer(
        capacity=4,
        seed=0,
        generation_weight_power=0.5,
        generation_weight_floor=0.35,
    )

    skipped = fill_buffer(buffer, batches)

    assert skipped == 0
    assert [record.episode_weight for record in buffer.recent] == pytest.approx(
        [0.35, 1.0]
    )
    stats = buffer.priority_stats()
    assert stats["min_generation_weight"] == pytest.approx(0.35)
    assert stats["max_generation_weight"] == pytest.approx(1.0)
    assert stats["min_terminal_quality_weight"] == pytest.approx(1.0)
    assert stats["max_terminal_quality_weight"] == pytest.approx(1.0)


def test_fill_buffer_combines_generation_and_terminal_quality_weights(tmp_path):
    from agents.dqn_market_free_v3d.training.collect import save_episode_batch
    from agents.dqn_market_free_v3d.training.replay import ReplayBuffer
    from agents.dqn_market_free_v3d.training.updater import SOURCE_KIND, fill_buffer

    low_path = tmp_path / "low.npz"
    mid_path = tmp_path / "mid.npz"
    high_path = tmp_path / "high.npz"
    save_episode_batch([_priority_episode(0, 3_000.0, 1.0)], low_path)
    save_episode_batch([_priority_episode(1, 11_500.0, 1.0)], mid_path)
    save_episode_batch([_priority_episode(2, 20_000.0, 1.0)], high_path)
    batches = [
        SimpleNamespace(
            source_kind=SOURCE_KIND,
            artifact_path=low_path,
            artifact_sha256=None,
            sample_count=1,
            metadata={"generation_id": "g0001", "mean_final_money": 5_000.0},
        ),
        SimpleNamespace(
            source_kind=SOURCE_KIND,
            artifact_path=mid_path,
            artifact_sha256=None,
            sample_count=1,
            metadata={"generation_id": "g0001", "mean_final_money": 5_000.0},
        ),
        SimpleNamespace(
            source_kind=SOURCE_KIND,
            artifact_path=high_path,
            artifact_sha256=None,
            sample_count=1,
            metadata={"generation_id": "g0002", "mean_final_money": 20_000.0},
        ),
    ]
    buffer = ReplayBuffer(
        capacity=8,
        seed=0,
        generation_weight_power=0.5,
        generation_weight_floor=0.35,
    )

    skipped = fill_buffer(buffer, batches, terminal_quality_weighting=True)

    assert skipped == 0
    assert [record.generation_weight for record in buffer.recent] == pytest.approx(
        [0.5, 0.5, 1.0]
    )
    assert [
        record.terminal_quality_weight for record in buffer.recent
    ] == pytest.approx([0.10, 0.55, 1.0])
    assert [record.episode_weight for record in buffer.recent] == pytest.approx(
        [0.05, 0.275, 1.0]
    )
    stats = buffer.priority_stats()
    assert stats["min_episode_weight"] == pytest.approx(0.05)
    assert stats["max_episode_weight"] == pytest.approx(1.0)


def test_fill_buffer_leaves_terminal_quality_weight_off_by_default(tmp_path):
    """Insolvent seasons are the negative examples; the default must not erase them."""

    from agents.dqn_market_free_v3d.training.collect import save_episode_batch
    from agents.dqn_market_free_v3d.training.replay import ReplayBuffer
    from agents.dqn_market_free_v3d.training.updater import SOURCE_KIND, fill_buffer

    low_path = tmp_path / "low.npz"
    save_episode_batch([_priority_episode(0, 3_000.0, 1.0)], low_path)
    batches = [
        SimpleNamespace(
            source_kind=SOURCE_KIND,
            artifact_path=low_path,
            artifact_sha256=None,
            sample_count=1,
            metadata={"generation_id": "g0001", "mean_final_money": 5_000.0},
        )
    ]
    buffer = ReplayBuffer(capacity=8, seed=0)

    fill_buffer(buffer, batches)

    assert [record.terminal_quality_weight for record in buffer.recent] == [1.0]
    assert [record.episode_weight for record in buffer.recent] == [1.0]


def test_terminal_quality_weight_thresholds():
    from agents.dqn_market_free_v3d.training.updater import _terminal_quality_weight

    assert _terminal_quality_weight(3_000.0) == pytest.approx(0.10)
    assert _terminal_quality_weight(11_500.0) == pytest.approx(0.55)
    assert _terminal_quality_weight(20_000.0) == pytest.approx(1.0)


def test_a_chunk_bootstraps_the_turn_n_step_ahead_and_never_clamps():
    """Every turn records at least a NOOP terminator, so `searchsorted` cannot
    skip the horizon. `clamped` is the counter that would say otherwise, and it
    is the difference between an n-step return and a silently shortened one."""

    batch = _buffer().sample_sequences(6, burn_in=20, train=40)
    assert batch.clamped == 0
    assert batch.turn_input.shape == (6, 20 + 40 + 24, STATE_DIM)
    rows = batch.state.shape[0]
    # Every target array is per decision now, not per turn.
    for name in ("reward", "discount", "next_turn", "chosen", "turn_share"):
        assert getattr(batch, name).shape == (rows,), name
    assert batch.next_state.shape == (rows, STATE_DIM)
    assert batch.next_mask.shape == (rows, ACTION_COUNT)
    # Each row's owning turn is inside its own chunk's window, and every
    # bootstrap lands inside the rolled post-burn-in grid.
    assert int(batch.decision_turn.max()) < 6 * 40
    assert int(batch.next_turn.max()) < 6 * (40 + 24)
    # A closing row whose horizon is live points at a *later* turn's latent: that
    # is the cross-turn Bellman backup members-v2 did not have.
    boundary = batch.discount != 1.0
    live = boundary & (batch.discount > 0.0)
    assert live.any()
    assert np.all(
        batch.next_turn[live] % (40 + 24) > batch.decision_turn[live] % 40
    )


def test_a_turns_sub_decisions_get_individuated_targets():
    """The correction this fork exists for.

    v2 gave every sub-decision of a turn the same reward, discount and bootstrap,
    so ten transitions were regressed onto one scalar and nothing in the target
    could say the seventh order was worse than the first. Its policy converged on
    a ten-slot zero-P&L wash for exactly that reason.

    The MDP a turn describes: sub-decisions are simultaneous in game time, so an
    intra-turn transition pays no reward and does not discount -- it asks "having
    queued that, what is the next order worth?". Time passes only at the boundary,
    where the turn's whole n-step return goes to the decision that closed it.
    """

    batch = _buffer().sample_sequences(6, burn_in=20, train=40)
    intra = batch.discount == 1.0
    boundary = ~intra

    assert intra.any() and boundary.any(), "need both kinds of row to test either"
    # Intra-turn rows: no reward, no discounting, successor scored against the
    # same latent -- the GRU does not advance inside a turn.
    assert np.all(batch.reward[intra] == 0.0)
    assert np.array_equal(
        batch.next_turn[intra] // (40 + 24), batch.decision_turn[intra] // 40
    )
    assert np.array_equal(
        batch.next_turn[intra] % (40 + 24), batch.decision_turn[intra] % 40
    )
    # Exactly one boundary row per turn, and it is the turn's last decision.
    turns, counts = np.unique(batch.decision_turn[boundary], return_counts=True)
    assert np.all(counts == 1)
    assert turns.size == np.unique(batch.decision_turn).size
    # Boundary discounts are gamma**n_step, or 0 where the episode ended inside
    # the horizon. Never 1.0 -- that would mean a turn boundary cost no time.
    live = batch.discount[boundary][batch.discount[boundary] > 0.0]
    assert np.allclose(live, 0.999**24, atol=1e-6)
    assert np.any(batch.reward[boundary] != 0.0)


def test_two_sub_decisions_of_a_turn_can_disagree():
    """The property v2 structurally could not have.

    Same turn, different successor states, therefore different targets. Asserted
    on the successors rather than on Q, because the sampler is what was wrong --
    v2's rows were identical before the network ever saw them.
    """

    batch = _buffer().sample_sequences(8, burn_in=20, train=40)
    by_turn: dict[int, list[int]] = {}
    for index, turn in enumerate(batch.decision_turn.tolist()):
        by_turn.setdefault(turn, []).append(index)
    multi = [rows for rows in by_turn.values() if len(rows) > 1]
    assert multi, "need a turn holding several sub-decisions"

    differing = 0
    for rows in multi:
        first, second = rows[0], rows[1]
        if not np.array_equal(batch.next_state[first], batch.next_state[second]):
            differing += 1
    assert differing == len(multi), (
        "every multi-decision turn must give its rows distinct successors; "
        "identical successors is exactly v2's defect"
    )


def test_a_turns_decisions_share_exactly_one_unit_of_loss_weight():
    """`turn_share` is the inner half of the two-level mean.

    Without it a turn holding eleven sub-decisions would contribute eleven times
    the gradient of one holding a single terminator -- which is a bias toward
    exactly the behaviour `decisions_per_turn` exists to watch for, and it arrived
    with the chunk draw rather than existing in v2.
    """

    batch = _buffer().sample_sequences(4, burn_in=20, train=40)
    totals = np.bincount(
        batch.decision_turn, weights=batch.turn_share, minlength=4 * 40
    )
    populated = totals[totals > 0]
    assert populated.size
    np.testing.assert_allclose(populated, 1.0, atol=1e-6)


def test_a_burn_in_off_the_stored_grid_is_refused():
    """`hidden0` has to be a state somebody stored, and `HIDDEN_STRIDE` decides
    which turns those are."""

    from agents.dqn_market_free_v3d.training.collect import HIDDEN_STRIDE

    with pytest.raises(ValueError, match="multiple of the stored-hidden stride"):
        _buffer().sample_sequences(2, burn_in=HIDDEN_STRIDE + 1, train=40)


def test_flat_replay_sample_sums_the_horizon_and_bootstraps_past_it():
    from agents.dqn_market_free_v3d.training.replay import ReplayBuffer

    record = _episode(seed=11, turns=80, rate=1)
    record.reward[:] = 0.25
    buffer = ReplayBuffer(capacity=4, seed=0, gamma=1.0, n_step=24)
    buffer.add([record])

    batch = buffer.sample(64)

    live = batch.discount > 0.0
    assert live.any() and (~live).any()
    # A full horizon inside the episode: 24 turns of 0.25 and a live bootstrap.
    assert np.all(batch.reward[live] == pytest.approx(24 * 0.25))
    assert np.all(batch.discount[live] == 1.0)
    # Near the end the horizon is cut by the episode and nothing bootstraps.
    assert np.all(batch.reward[~live] <= 24 * 0.25 + 1e-6)


def test_an_episode_too_short_for_a_training_window_is_not_drawn():
    """`record.decisions` -- the flat sampler's predicate -- would pass an episode
    with plenty of decisions packed into too few turns, and then index off the end
    of `turn_state`."""

    from agents.dqn_market_free_v3d.training.replay import ReplayBuffer

    buffer = ReplayBuffer(capacity=8, seed=0)
    buffer.add([_episode(seed=1, turns=30)])
    with pytest.raises(ValueError, match="long enough"):
        buffer.sample_sequences(2, burn_in=20, train=40)


def test_the_left_pad_of_a_chunk_cannot_move_the_hidden_state():
    """A padded step must be a no-op, not a zero input.

    A GRU fed zeros from a zero state settles on `(1 - z) * tanh(b_in + b_hn)`,
    not on zero, so a chunk that begins before its episode does would start its
    burn-in from a bias artefact that grows with every pad step. The gate in
    `recur` is what prevents that, and this pins it by filling the pad with
    garbage and asserting the answer does not move.
    """

    module = _module(seed=21)
    turn_input = torch.zeros(2, 12, STATE_DIM)
    valid = torch.ones(2, 12, dtype=torch.bool)
    valid[:, :5] = False
    torch.manual_seed(0)
    turn_input[:, 5:] = torch.randn(2, 7, STATE_DIM)
    hidden = torch.zeros(2, HIDDEN_SIZE)

    with torch.no_grad():
        padded = module.recur(turn_input, hidden, valid)[:, -1]
        # The same real turns with no pad at all.
        bare = module.recur(turn_input[:, 5:], hidden)[:, -1]
        # And the pad filled with something else entirely.
        noisy_pad = turn_input.clone()
        noisy_pad[:, :5] = torch.randn(2, 5, STATE_DIM) * 10.0
        polluted = module.recur(noisy_pad, hidden, valid)[:, -1]

    assert torch.allclose(padded, bare, atol=1e-6)
    assert torch.allclose(padded, polluted, atol=1e-6)


def test_the_gradient_reaches_the_recurrence_over_the_training_window():
    """The whole point of the fork, asserted rather than assumed.

    Also checks the burn-in is genuinely excluded: `_roll_chunk` runs it under
    `no_grad`, so the graph the loss backprops through is `train` steps deep and
    not `burn_in + train`.
    """

    from agents.dqn_market_free_v3d.training.updater import _roll_chunk

    module = _module(seed=22)
    batch = _buffer().sample_sequences(3, burn_in=20, train=40)
    device = torch.device("cpu")
    train_hidden, post = _roll_chunk(module, batch, device, 20, 40, grad=True)
    assert train_hidden.requires_grad
    assert not post.requires_grad, "the bootstrap latents must not carry gradient"

    rows = torch.from_numpy(batch.decision_turn)
    states = torch.from_numpy(batch.state)
    hidden = train_hidden.reshape(-1, module.hidden_size)[rows]
    module(torch.cat((states, hidden), dim=1)).sum().backward()

    for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        grad = getattr(module.cell, name).grad
        assert grad is not None, f"{name} received no gradient"
        assert float(grad.abs().sum()) > 0.0, f"{name} gradient is identically zero"


def test_a_full_td_step_runs_and_moves_every_parameter():
    """The whole loop body, not just its pieces.

    Shape errors in the chunk machinery are concentrated in the gather indices --
    `next_turn` into the flattened post-burn-in grid, `decision_turn` into the
    training grid -- and every one of them is a plausible off-by-one that still
    broadcasts. Running the real arithmetic once is the only thing that
    catches those.

    Asserts every parameter moved, the GRU's four tensors included -- which is the
    fork's premise stated as a test rather than as a comment.

    Two optimizer steps rather than one, because this module is built at torch's
    own initialisation and a real run is not: `initial_network` cold-starts from
    `quiet_start`, whose output layer is zero, so on its first step `dL/da` is
    identically zero and nothing upstream of it moves at all. Running two here
    keeps this test honest against the trajectory a real run takes.
    """

    from agents.dqn_market_free_v3d.order_space import MASKED_SCORE
    from agents.dqn_market_free_v3d.training.updater import _roll_chunk

    burn_in, train, chunks = 20, 40, 4
    online = _module(seed=31)
    target = _module(seed=31)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    loss_fn = torch.nn.SmoothL1Loss(reduction="none")
    buffer = _buffer()
    device = torch.device("cpu")
    before = {name: p.detach().clone() for name, p in online.named_parameters()}

    for _ in range(2):
        batch = buffer.sample_sequences(chunks, burn_in=burn_in, train=train)
        train_hidden, post_online = _roll_chunk(
            online, batch, device, burn_in, train, grad=True
        )
        with torch.no_grad():
            _t, post_target = _roll_chunk(
                target, batch, device, burn_in, train, grad=False
            )
            boot = torch.from_numpy(batch.next_turn)
            boot_state = torch.from_numpy(batch.next_state)
            boot_mask = torch.from_numpy(batch.next_mask)
            best = online(torch.cat((boot_state, post_online[boot]), dim=1))
            best = best.masked_fill(~boot_mask, MASKED_SCORE).argmax(dim=1, keepdim=True)
            bootstrap = (
                target(torch.cat((boot_state, post_target[boot]), dim=1))
                .gather(1, best)
                .squeeze(1)
            )
            goal = torch.from_numpy(batch.reward) + (
                torch.from_numpy(batch.discount) * bootstrap
            )

        rows = torch.from_numpy(batch.decision_turn)
        hidden = train_hidden.reshape(-1, online.hidden_size)[rows]
        predicted = (
            online(torch.cat((torch.from_numpy(batch.state), hidden), dim=1))
            .gather(1, torch.from_numpy(batch.chosen).unsqueeze(1))
            .squeeze(1)
        )
        per_row = loss_fn(predicted, goal)
        share = torch.from_numpy(batch.turn_share)
        per_turn = torch.zeros(chunks * train, dtype=per_row.dtype).index_add_(
            0, rows, per_row * share
        )
        per_chunk = per_turn.view(chunks, train).mean(dim=1)
        loss = (per_chunk * torch.from_numpy(batch.weight)).mean()

        assert torch.isfinite(loss), "a nan reached the loss"
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
        optimizer.step()

    unmoved = [
        name
        for name, p in online.named_parameters()
        if torch.equal(p.detach(), before[name])
    ]
    assert not unmoved, f"parameters never moved: {unmoved}"


# --------------------------------------------------------------------------
# Wallet-sized product buys
# --------------------------------------------------------------------------


def test_a_product_buy_rung_the_wallet_cannot_cover_is_shut():
    """One affordable unit must not unlock the whole quantity ladder.

    The simulator does not refuse an order it cannot fill -- it part-fills it and
    drains the wallet to zero -- so an open rung the wallet cannot cover is a real
    loss, not a rejected order. On the Kaggle ladder 31.5% of `BUY_PRODUCT` orders
    were sized past the wallet against opponents' 3.9%, with 0.0% refused.

    `BUY_SEED` has always masked per rung; this pins the two symmetric.
    """

    from agents.dqn_market_free_v3d import donor
    from agents.dqn_market_free_v3d.order_space import (
        _BUY_PRODUCT_ROWS,
        _ROW_QUANTITY,
        feasible_actions,
    )
    from tests.test_dqn_market_free import _mask_state

    item = "WHEAT"
    rows = _BUY_PRODUCT_ROWS[item]
    quantities = _ROW_QUANTITY[rows]

    # A wallet that covers a handful of units and nowhere near the top rung.
    obs, farm, private, field = _mask_state(money=60.0)
    mask = feasible_actions(obs, farm, private, field, None)

    stock = int(
        ((obs.get("market") or {}).get("inventory") or {}).get(item, donor.MARKET_I0)
        or donor.MARKET_I0
    )
    cost = np.cumsum(
        [
            float(donor._price_at(item, stock - offset, obs))
            for offset in range(1, int(quantities.max()) + 1)
        ]
    )

    open_rows = mask[rows]
    assert open_rows.any(), "a wallet covering several units must leave some rung open"
    assert not open_rows.all(), "a 60-coin wallet cannot cover a 45-unit buy"
    # Every open rung is affordable and every shut one is not: the mask is exactly
    # the affordability predicate, not an approximation of it.
    for index, quantity in enumerate(quantities):
        affordable = cost[int(quantity) - 1] <= 60.0
        assert bool(open_rows[index]) == bool(affordable), (
            f"rung {quantity} open={bool(open_rows[index])} affordable={affordable}"
        )


def test_the_mask_prices_a_product_buy_the_way_pending_turn_does():
    """The mask and the wallet accounting must not disagree about one order's price.

    `PendingTurn.purchase_cost` is what actually debits the wallet when the order
    is queued. If the mask priced the same rung differently the policy would be
    offered orders the turn then cannot pay for -- the part-fill again, arriving by
    a different route.
    """

    from agents.dqn_market_free_v3d import donor
    from agents.dqn_market_free_v3d.order_space import PendingTurn
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(money=5000.0)
    pending = PendingTurn(obs, farm, private, field, None)
    item = "WHEAT"
    stock = int(
        ((obs.get("market") or {}).get("inventory") or {}).get(item, donor.MARKET_I0)
        or donor.MARKET_I0
    )
    for quantity in (1, 4, 12, 45):
        mask_cost = float(
            np.cumsum(
                [
                    float(donor._price_at(item, stock - offset, obs))
                    for offset in range(1, quantity + 1)
                ]
            )[-1]
        )
        assert mask_cost == pytest.approx(pending.purchase_cost(item, quantity))


def test_product_buys_are_capped_by_remaining_shed_capacity():
    """The turn model must not store units the simulator will discard."""

    from agents.dqn_market_free_v3d.order_space import (
        _BUY_PRODUCT_ROWS,
        _ROW_QUANTITY,
        Order,
        PendingTurn,
        feasible_actions,
    )
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(
        money=5000.0, shed={"MELON": donor.SHED_CAPACITY - 1}
    )
    pending = PendingTurn(obs, farm, private, field, None)
    rows = _BUY_PRODUCT_ROWS["WHEAT"]
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert _ROW_QUANTITY[rows][mask[rows]].tolist() == [1]

    pending.apply(Order("BUY_PRODUCT", "WHEAT", 45))
    assert pending.occupancy == donor.SHED_CAPACITY
    assert pending.shed["WHEAT"] == 1
    assert pending.events[-1][2] == 1


def test_sell_accounting_commits_only_stock_the_shed_holds():
    from agents.dqn_market_free_v3d.order_space import Order, PendingTurn
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(money=3000.0, shed={"WHEAT": 1})
    pending = PendingTurn(obs, farm, private, field, None)
    start_inventory = pending.market_inventory.get("WHEAT", donor.MARKET_I0)

    pending.apply(Order("SELL", "WHEAT", 45))

    assert pending.shed["WHEAT"] == 0
    assert pending.occupancy == 0
    assert pending.events[-1][2] == 1
    assert pending.market_inventory["WHEAT"] == start_inventory + 1


def test_floor_price_sales_do_not_advance_pending_market_inventory():
    from agents.dqn_market_free_v3d.order_space import Order, PendingTurn
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(money=3000.0, shed={"WHEAT": 1})
    obs["market"]["inventory"]["WHEAT"] = 10_000_000_000_000
    pending = PendingTurn(obs, farm, private, field, None)

    pending.apply(Order("SELL", "WHEAT", 1))

    assert pending.events[-1] == ("SELL", "WHEAT", 1, 1.0)
    assert pending.market_inventory["WHEAT"] == 10_000_000_000_000


# --------------------------------------------------------------------------
# The turn-opening encode, reused
# --------------------------------------------------------------------------


def _encode_case(field: dict, shed: dict, carried: dict):
    """`(reused, fresh, reusable)` for one turn, at its first sub-decision.

    `reused` is what `main.py` builds when it stands the turn-opening encode in
    for the first sub-decision; `fresh` is the full encode off `pending.view` that
    v2 did unconditionally. `reusable` is the gate `main.py` actually consults.
    """

    from agents.dqn_market_free_v3d.main import _shed_unmoved
    from agents.dqn_market_free_v3d.market_state import (
        intra_turn_block,
        market_state_features,
    )

    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    farm = {
        "money": 3000.0,
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
        "tiles": tiles,
        "hands": [],
        "farmer": [4, 4],
    }
    private = {"shed": dict(shed), "seeds": {}, "inventories": [dict(carried)]}
    obs = {
        "player": 0,
        "step": 30,
        "day": 1,
        "hour": 6,
        "farms": [farm, farm],
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
        "private": private,
    }
    roles = herd.field_roles(obs, farm, private)
    pending = PendingTurn(obs, farm, private, field, None, plantable={}, roles=roles)

    turn_state = market_state_features(
        obs, farm, private, roles, field, turn=None
    )
    reused = turn_state.copy()
    reused[INTRA_TURN_SLICE] = intra_turn_block(pending)

    obs_view, farm_view, private_view = pending.view(obs, farm, private)
    fresh = market_state_features(
        obs_view, farm_view, private_view, roles, field, turn=pending
    )
    return reused, fresh, _shed_unmoved(pending, private)


def test_the_reused_turn_state_equals_a_fresh_encode():
    """The gate on `main.py`'s encode-reuse, in both directions.

    Reusing the turn-opening vector for a turn's first sub-decision is what keeps
    v3 paying v2's encoder bill on the common single-decision turn -- a second full
    encode costs a `projected_supply_map`. It is only sound while `pending.view` is
    the identity, and `PendingTurn.shed` is the permissive
    `max(raw, _post_field_storage)` view, so a turn whose field plan DROPs carried
    produce breaks that before any order has been applied.

    Both halves are asserted. Where the gate opens the two encodes must agree
    *exactly* -- not within a tolerance, since one is a copy of the other. Where it
    closes they must genuinely differ, or the gate would be free to be wrong in the
    quiet direction and nothing would notice.
    """

    idle, idle_fresh, idle_reusable = _encode_case(
        {"farmer": ["PASS"], "hands": []}, {"WHEAT": 3}, {"WHEAT": 12}
    )
    assert idle_reusable
    assert_array_equal(idle, idle_fresh)

    dropped, dropped_fresh, dropped_reusable = _encode_case(
        {"farmer": ["DROP"], "hands": []}, {"WHEAT": 3}, {"WHEAT": 12}
    )
    assert not dropped_reusable, (
        "a DROP turn moves the shed into `pending.view` before the first decision"
    )
    assert not np.array_equal(dropped, dropped_fresh), (
        "the reused vector must be refused precisely because it differs"
    )


def test_the_reuse_gate_names_every_column_that_moves():
    """What the gate is protecting, stated as a count rather than a claim.

    The divergence is not confined to block E -- if it were, the block-E overwrite
    would already cover it. It reaches the per-product shed column, the shed-load
    pair, and the block-C constraint columns downstream of `shed_total`, none of
    which the overwrite touches.
    """

    dropped, dropped_fresh, _ = _encode_case(
        {"farmer": ["DROP"], "hands": []}, {"WHEAT": 3}, {"WHEAT": 12}
    )
    moved = np.flatnonzero(dropped != dropped_fresh)
    assert moved.size, "the DROP case must actually move columns"
    outside = [index for index in moved if not INTRA_TURN_SLICE.start <= index < INTRA_TURN_SLICE.stop]
    assert outside, (
        "the whole point is that columns outside block E move; the overwrite "
        "already handles block E"
    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Open bug carried over from v3c: clone.record_from_replay and main.py disagree "
        "on ~12 cells of the herd-viability block on animal-purchase turns (253, 255 at "
        "seed 17 under v3c's 280-wide vector). NOT strict in v3d, and that is the honest "
        "state rather than a green: widening the vector by CROP_MATURITY_DIM changes the "
        "argmax, so seed 17 no longer reaches a turn that buys an animal and the "
        "disagreement is not reproduced here. The parity bug is unverified in v3d, not "
        "fixed -- restoring strict would assert a fix nobody made."
    ),
)
def test_cloned_state_matches_the_live_agent():
    """`clone.py` reproduces `main.py`'s walk, vector for vector and tick for tick.

    The three properties `main.py`'s tick comment claims, asserted rather than
    described: the tick lands after `history.observe` so the turn's money has
    settled last turn's order, before the decision loop so every sub-decision of a
    turn scores against one latent, and on the untouched observation with block E
    carrying the *previous* turn's finished `PendingTurn`.

    Compared on the whole vector and the whole hidden trajectory, not one block.
    A recurrence that drifts does not degrade a feature -- it poisons every state
    after the drift, and the two walks are written out by hand in two files that
    build `PendingTurn` in a different order relative to the encode.
    """

    from kaggle_environments import make

    from agents.dqn_market_free_v3d.main import MarketFreeAgent
    from agents.dqn_market_free_v3d.training.clone import record_from_replay
    from agents.dqn_market_free_v3d.training.collect import HIDDEN_STRIDE, Recorder
    from tests.test_dqn_market_free import SNAPSHOT_PATH

    recorder = Recorder()
    # A random head rather than `quiet_start`: this needs the agent to trade, so
    # that block E carries something on the recurrent input and the two walks are
    # actually being asked to agree about a moving state.
    network = OrderQNetwork.random(seed=3)
    learner = MarketFreeAgent(
        network, rng=np.random.default_rng(0), recorder=recorder
    )
    # Seeded. Everything else in this test is seeded -- the head and the agent RNG
    # above -- and an unseeded `make` draws its own weed/shop RNG, which made this
    # the one test in the file that passed or failed on the run.
    env = make("kaggriculture", configuration={"seed": 17}, debug=True)
    env.run([learner, str(SNAPSHOT_PATH)])
    replay = env.toJSON()

    live = recorder.finish(
        final_money=float(env.state[0]["observation"]["farms"][0]["money"]),
        gamma=0.999,
    )
    cloned, _stats = record_from_replay(replay, seat=0, gru=network.gru)
    assert cloned is not None

    # The recurrent input: one row per turn, encoded off the untouched observation
    # with the previous turn's `PendingTurn` in block E.
    assert cloned.turn_state.shape == live.turn_state.shape
    np.testing.assert_allclose(cloned.turn_state, live.turn_state, atol=1e-6)

    # The hidden trajectory the chunks start from. Stored strided, pre-tick, and
    # row 0 is the zero state the episode opened on.
    assert cloned.turn_hidden.shape == live.turn_hidden.shape
    assert not np.any(live.turn_hidden[0]), "row 0 is the state the episode opened on"
    np.testing.assert_allclose(
        cloned.turn_hidden.astype(np.float32),
        live.turn_hidden.astype(np.float32),
        atol=1e-3,
    )
    assert np.abs(live.turn_hidden[1:]).sum() > 0.0, "the recurrence has to move"
    assert live.turn_hidden.shape[0] == -(-live.turn_state.shape[0] // HIDDEN_STRIDE)

    # And the per-decision states, which is where the encode-reuse gate lives:
    # `clone.py` takes `pending.view` for every sub-decision, so any turn where
    # `main.py` reused the turn-opening vector has to have been a turn where the
    # two are the same thing.
    assert cloned.order_state.shape == live.order_state.shape
    np.testing.assert_allclose(cloned.order_state, live.order_state, atol=1e-6)


def test_block_e_lands_where_the_slice_says_it_does():
    """The slice against a *built* vector, not against its own definition.

    `test_state_width_names_its_own_arithmetic` restates `INTRA_TURN_SLICE`'s
    definition and so cannot fail for the reason that matters: block E is appended
    last, and `main.py` overwrites exactly that span when it reuses the
    turn-opening encode. Append anything after it and the overwrite starts writing
    into someone else's columns while every width assertion still passes.
    """

    from agents.dqn_market_free_v3d.market_state import (
        intra_turn_block,
        market_state_features,
    )

    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    farm = {
        "money": 3000.0,
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
        "tiles": tiles,
        "hands": [],
        "farmer": [4, 4],
    }
    private = {"shed": {"WHEAT": 5}, "seeds": {}, "inventories": [{}]}
    obs = {
        "player": 0,
        "step": 30,
        "day": 1,
        "hour": 6,
        "farms": [farm, farm],
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
        "private": private,
    }
    field = {"farmer": ["PASS"], "hands": []}
    roles = herd.field_roles(obs, farm, private)
    pending = PendingTurn(obs, farm, private, field, None, plantable={}, roles=roles)
    pending.apply(Order("SELL", "WHEAT", 2))

    built = market_state_features(obs, farm, private, roles, field, turn=pending)
    block = np.asarray(intra_turn_block(pending), dtype=np.float32)
    assert_array_equal(built[INTRA_TURN_SLICE], block)
    assert np.abs(block).sum() > 0.0, "a committed sale has to show in block E"


def test_the_agent_resets_its_recurrence_between_episodes():
    """`step == 0` clears the memory, so a reused agent object cannot carry it over.

    `main.agent` keeps a module-level `_AGENT` that survives across episodes inside
    one interpreter, so the constructor's reset is not enough on its own. A network
    that resumed the previous episode's hidden state would read on every metric
    exactly like a policy that had learned to ignore its memory, and there is no
    way to separate those after the fact.

    Asserted as an equality against a network that never played the first episode,
    which is the only form that can fail: "the state changed" is true whether or
    not the reset ran.
    """

    from kaggle_environments import make

    from agents.dqn_market_free_v3d.main import MarketFreeAgent
    from tests.test_dqn_market_free import SNAPSHOT_PATH

    def opening_hidden(warm: bool) -> np.ndarray:
        network = OrderQNetwork.random(seed=5)
        assert not network.is_noisy, "a noisy network would advance `rng` across episodes"
        learner = MarketFreeAgent(network, rng=np.random.default_rng(0))
        if warm:
            # A full episode first, so the recurrence is carrying 720 turns of
            # memory when the next episode's opening turn arrives.
            env = make("kaggriculture", configuration={"seed": 17}, debug=True)
            env.run([learner, str(SNAPSHOT_PATH)])
            assert np.abs(network.hidden).sum() > 0.0, "the recurrence has to have moved"
            assert learner.last_turn is not None
        env = make("kaggriculture", configuration={"seed": 17}, debug=True)
        state = env.reset()
        learner(state[0]["observation"], env.configuration)
        return network.hidden.copy()

    assert_array_equal(opening_hidden(warm=True), opening_hidden(warm=False))

def test_the_burn_in_probe_compares_two_burn_ins_of_the_same_window():
    """`q_discrepancy` reads downward, and it has to.

    A chunk holds exactly `burn_in` turns of history, so a *longer* refresh cannot
    be built from one. An earlier version clamped `4 * burn_in` to the chunk width
    and so rolled straight into the graded window, reporting how far Q moves across
    a day of play -- large for a healthy time-sensitive GRU and constant
    in `burn_in_turns`, the dial it was documented to steer.

    Pinned two ways: the probe must not touch turns past the burn-in boundary, and
    at a burn-in the recurrence has fully forgotten, the two legs must agree.
    """

    from agents.dqn_market_free_v3d.training.updater import _recurrence_metrics

    module = _module(seed=7)
    buffer = _buffer()
    metrics = _recurrence_metrics(module, buffer, torch.device("cpu"), 20, 40)

    assert set(metrics) >= {
        "hidden_recovery_l2",
        "hidden_recovery_cosine",
        "q_discrepancy",
        "hidden_norm",
    }
    assert np.isfinite(metrics["q_discrepancy"])
    assert metrics["q_discrepancy"] >= 0.0

    # And the discriminating case. A GRU with its reset and update gates pinned
    # shut is memoryless -- `h_t` is a function of `x_t` alone -- so two burn-ins
    # of *different lengths* leave the window's latents bit-identical and the
    # discrepancy is exactly zero. Two latents from different *turns* are not, so
    # this is the case that separates a burn-in comparison from a comparison
    # across the window, which is what the clamp used to compute.
    memoryless = _module(seed=8)
    with torch.no_grad():
        memoryless.cell.weight_hh.zero_()
        memoryless.cell.bias_hh.zero_()
        memoryless.cell.bias_ih[:HIDDEN_SIZE] = -30.0
        memoryless.cell.bias_ih[HIDDEN_SIZE : 2 * HIDDEN_SIZE] = -30.0
    forgetful = _recurrence_metrics(memoryless, buffer, torch.device("cpu"), 20, 40)
    assert forgetful["q_discrepancy"] == pytest.approx(0.0, abs=1e-6)
    # The same network still moves Q across turns, so the zero above is a property
    # of the comparison rather than of a network that scores everything alike.
    assert forgetful["hidden_norm"] > 0.0


def test_the_clone_step_clips_only_what_it_will_step():
    """Sigma is excluded from `clone_optimizer`, so its gradient is never cleared.

    A clip over the whole module would keep counting those stale gradients, the
    total norm would grow across the phase, and the clip factor would shrink the
    head's and the GRU's real gradients a little further every step -- an effective
    learning rate that decays for a reason nothing reports.
    """

    from agents.dqn_market_free_v3d.training import network as net
    from agents.dqn_market_free_v3d.training.updater import _behavior_clone_step

    module = _module(seed=11, noisy="head")
    buffer = _buffer(count=4)
    buffer.add([_episode(seed=90 + index) for index in range(2)], pinned=True)
    optimizer = torch.optim.Adam(
        [
            parameter
            for name, parameter in module.named_parameters()
            if "sigma" not in name
        ],
        lr=1e-3,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }
    for _ in range(3):
        loss = _behavior_clone_step(
            module, optimizer, buffer, 2, torch.device("cpu"), 20, 40, 5.0
        )
        assert np.isfinite(loss)

    moved = {
        name
        for name, parameter in module.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    }
    assert any(name.startswith("cell.") for name in moved), "the GRU must train"
    assert not any("sigma" in name for name in moved), (
        "sigma is excluded from the clone optimizer and must not move"
    )
    assert isinstance(net.NOISY_LAYERS, tuple)


# --------------------------------------------------------------------------
# The wash
# --------------------------------------------------------------------------


def test_a_same_item_round_trip_is_worth_exactly_nothing():
    """The premise of the mask, priced rather than asserted.

    Buying walks the market inventory down and selling it back walks the same
    price levels up, so the pair nets exactly zero at every quantity. That is why
    v2 could spend 47% of its turns on it for free.
    """

    from agents.dqn_market_free_v3d import donor

    start = donor.MARKET_I0
    obs = {"market": {"inventory": {"WHEAT": start}, "prices": {}}}
    for quantity in (8, 24, 32, 45):
        cost = sum(
            float(donor._price_at("WHEAT", start - offset, obs))
            for offset in range(1, quantity + 1)
        )
        # The sell starts from the post-buy inventory, which is the whole point.
        proceeds = sum(
            float(donor._price_at("WHEAT", (start - quantity) + offset, obs))
            for offset in range(quantity)
        )
        assert proceeds - cost == pytest.approx(0.0), quantity


def test_a_turn_that_sold_an_item_cannot_buy_it_back():
    """Shut structurally, because a price cannot be tuned against a free action.

    A penalty has to be calibrated against the behaviour it suppresses and can be
    learned around at the margin. v1 could not wash at all -- one order per turn --
    and v1 is the lineage this fork has to beat.
    """

    from agents.dqn_market_free_v3d.order_space import (
        _BUY_PRODUCT_ROWS,
        _SELL_ROWS,
        Order,
        PendingTurn,
        feasible_actions,
    )
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(money=5000.0, shed={"WHEAT": 60})
    pending = PendingTurn(obs, farm, private, field, None)

    opening = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert opening[_SELL_ROWS["WHEAT"]].any(), "the turn must be able to sell at all"
    assert opening[_BUY_PRODUCT_ROWS["WHEAT"]].any(), "and to buy at all"

    pending.apply(Order("SELL", "WHEAT", 6))
    after = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert not after[_BUY_PRODUCT_ROWS["WHEAT"]].any(), "buying back is the wash"
    # The other item is untouched: this shuts a round trip, not the market.
    assert after[_BUY_PRODUCT_ROWS["FERTILIZER"]].any()


def test_a_turn_that_bought_an_item_cannot_sell_it_back():
    from agents.dqn_market_free_v3d.order_space import (
        _SELL_ROWS,
        Order,
        PendingTurn,
        feasible_actions,
    )
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(money=5000.0, shed={"WHEAT": 60})
    pending = PendingTurn(obs, farm, private, field, None)
    pending.apply(Order("BUY_PRODUCT", "WHEAT", 4))
    after = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert not after[_SELL_ROWS["WHEAT"]].any()


def test_a_mixed_turn_is_still_legal():
    """Selling one item and buying another is ordinary trading, not a wash."""

    from agents.dqn_market_free_v3d.order_space import (
        _BUY_PRODUCT_ROWS,
        _SELL_ROWS,
        Order,
        PendingTurn,
        feasible_actions,
    )
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(
        money=5000.0, shed={"WHEAT": 60, "MELON": 20}
    )
    pending = PendingTurn(obs, farm, private, field, None)
    pending.apply(Order("SELL", "MELON", 3))
    after = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert after[_BUY_PRODUCT_ROWS["WHEAT"]].any(), "a different item stays buyable"
    assert after[_SELL_ROWS["WHEAT"]].any(), "and a different item stays sellable"


def test_the_unpended_mask_is_unaffected_by_the_wash_rule():
    """No `PendingTurn` means no turn in flight, so nothing has been queued."""

    from agents.dqn_market_free_v3d.order_space import _BUY_PRODUCT_ROWS, feasible_actions
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(money=5000.0, shed={"WHEAT": 60})
    mask = feasible_actions(obs, farm, private, field, None)
    assert mask[_BUY_PRODUCT_ROWS["WHEAT"]].any()


def test_a_zero_quote_turn_is_charged_again_when_nothing_moved():
    """v2 waved this through and it was one of the three things that made the
    wash reward-optimal. With same-item round trips unreachable, a zero quote and
    a zero delta is a refusal again -- v1's reading."""

    from agents.dqn_market_free_v3d.market_state import order_refused

    assert bool(order_refused(True, 0.0, 0.0)), "refused: nothing quoted, nothing moved"
    assert not bool(order_refused(True, 42.0, 0.0)), "filled: the wallet moved"
    assert bool(order_refused(True, 0.0, 500.0)), "refused: quoted 500, moved nothing"
    assert not bool(order_refused(True, 500.0, 500.0))
    assert not bool(order_refused(False, 0.0, 0.0)), "nothing emitted, nothing to refuse"


def test_a_buy_animal_rung_the_wallet_cannot_cover_is_shut():
    """The last place a single affordable unit opened the whole ladder."""

    from agents.dqn_market_free_v3d import donor
    from agents.dqn_market_free_v3d.order_space import (
        _BUY_ANIMAL_ROWS,
        _ROW_QUANTITY,
        feasible_actions,
    )
    from tests.test_dqn_market_free import _mask_state

    animal = "COW"
    unit = float(donor.ANIMALS[animal]["cost"])
    # Enough for one head and not two.
    obs, farm, private, field = _mask_state(money=unit * 1.5)
    mask = feasible_actions(obs, farm, private, field, None)
    rows = _BUY_ANIMAL_ROWS[animal]
    for index, quantity in enumerate(_ROW_QUANTITY[rows]):
        affordable = int(quantity) * unit <= unit * 1.5
        assert bool(mask[rows][index]) == bool(affordable), quantity


def _lot_batch(**metadata):
    """A stand-in for `TrainingBatch` carrying only what the aggregator reads."""

    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class _Batch:
        metadata: dict[str, Any] = field(default_factory=dict)

    return _Batch(metadata=dict(metadata))


def test_batch_lot_metrics_average_the_batches_the_update_consumed():
    """`reward.py` names these columns as its instrument; the update now carries them."""

    from agents.dqn_market_free_v3d.training.updater import _batch_lot_metrics

    metrics = _batch_lot_metrics(
        (
            _lot_batch(animal_waste_cost=1000.0, animal_gain_credit=100.0),
            _lot_batch(animal_waste_cost=3000.0, animal_gain_credit=300.0),
        )
    )
    assert metrics["animal_waste_cost"] == pytest.approx(2000.0)
    assert metrics["animal_gain_credit"] == pytest.approx(200.0)


def test_batch_lot_metrics_skip_batches_that_carry_no_lot_columns():
    """An evaluation-sourced batch has no rollouts behind it.

    Folding it in as a zero would report the herd term decaying every time one
    landed, which is the failure this skip exists to prevent.
    """

    from agents.dqn_market_free_v3d.training.updater import _batch_lot_metrics

    metrics = _batch_lot_metrics(
        (
            _lot_batch(animal_waste_cost=1000.0),
            _lot_batch(transitions=1234),  # evaluation batch: no lot columns
            _lot_batch(),
            object(),  # no `metadata` attribute at all
        )
    )
    assert metrics["animal_waste_cost"] == pytest.approx(1000.0)


def test_batch_lot_metrics_omit_a_column_no_batch_measured():
    """Absent is not zero. 'Not measured' and 'measured as nothing' differ."""

    from agents.dqn_market_free_v3d.training.updater import _batch_lot_metrics

    metrics = _batch_lot_metrics((_lot_batch(animal_waste_cost=500.0),))
    assert "animal_waste_cost" in metrics
    assert "seed_waste_cost" not in metrics
    assert _batch_lot_metrics(()) == {}


def test_batch_lot_metrics_reject_values_that_are_not_finite_numbers():
    """A None or a NaN must not poison the mean for the whole update."""

    from agents.dqn_market_free_v3d.training.updater import _batch_lot_metrics

    metrics = _batch_lot_metrics(
        (
            _lot_batch(animal_waste_cost=1000.0, unrecovered_animal_share=None),
            _lot_batch(animal_waste_cost=float("nan"), unrecovered_animal_share="0.5"),
            _lot_batch(animal_waste_cost=2000.0, unrecovered_animal_share=float("inf")),
        )
    )
    assert metrics["animal_waste_cost"] == pytest.approx(1500.0)
    # "0.5" is a JSON-legal number that arrived as a string; it still counts.
    assert metrics["unrecovered_animal_share"] == pytest.approx(0.5)


def test_batch_lot_keys_are_exactly_what_the_collector_writes():
    """Every lifted key has to exist in batch metadata or it silently never appears.

    The aggregator omits a column no batch carried, so a typo here is invisible at
    runtime -- the metric would simply never show up.

    **Batch metadata has two sources, not one.** Most columns come from the
    `BehaviorCounts` export the collector builds from the rollout replays, but the
    per-record counters -- `wash_turns`, `lagged_wash_turns` -- are summed off
    `EpisodeRecord` in `updater.py` and never pass through `BehaviorCounts` at all.
    Pinning against the export alone failed `lagged_wash_turns`, a column that has
    been written on every batch since it landed (409.6 on `networth_01`'s last
    update). Both surfaces are checked, which still catches the typo this test
    exists for: a misspelling matches neither.
    """

    from agents.dqn_market_free_v3d.diagnostics import BehaviorCounts
    from agents.dqn_market_free_v3d.training.collect import EpisodeRecord
    from agents.dqn_market_free_v3d.training.updater import _BATCH_LOT_KEYS

    exported = BehaviorCounts().as_dict()
    # The collector writes the `_per_episode` variants under the bare name.
    available = set(exported) | {
        key[: -len("_per_episode")] for key in exported if key.endswith("_per_episode")
    }
    available |= {name for name in dir(EpisodeRecord) if not name.startswith("_")}
    missing = sorted(key for key in _BATCH_LOT_KEYS if key not in available)
    assert not missing, f"lifted keys the collector never writes: {missing}"


# --------------------------------------------------------------------------
# Block D': the restored refusal latches
# --------------------------------------------------------------------------


def _refusal_state(history):
    """A built vector for a fixed board, so only the latches vary."""

    from agents.dqn_market_free_v3d.market_state import market_state_features

    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    farm = {
        "money": 3000.0,
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
        "tiles": tiles,
        "hands": [],
        "farmer": [4, 4],
    }
    private = {"shed": {"WHEAT": 5}, "seeds": {}, "inventories": [{}]}
    obs = {
        "player": 0,
        "step": 30,
        "day": 1,
        "hour": 6,
        "farms": [farm, farm],
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
        "private": private,
    }
    field = {"farmer": ["PASS"], "hands": []}
    roles = herd.field_roles(obs, farm, private)
    return market_state_features(obs, farm, private, roles, field, history=history)


def test_the_refusal_block_lands_between_the_herd_block_and_block_e():
    """Three columns, and they must not disturb block E's span.

    `main.py` overwrites exactly `INTRA_TURN_SLICE` when it reuses the
    turn-opening encode, so anything appended after block E would be silently
    rewritten. This asserts the restored block sits *before* it and that block E
    is still the tail.
    """

    from agents.dqn_market_free_v3d.market_state import MarketHistory, refusal_block

    assert INTRA_TURN_SLICE.stop == MARKET_STATE_DIM
    assert INTRA_TURN_SLICE.start == MARKET_STATE_DIM - INTRA_TURN_DIM

    history = MarketHistory()
    state = _refusal_state(history)
    assert state.shape == (MARKET_STATE_DIM,)
    # Block F sits between the two now, so the latches are that much further back.
    flow_start = INTRA_TURN_SLICE.start - CROP_MATURITY_DIM - MARKET_FLOW_DIM
    block = slice(flow_start - REFUSAL_BLOCK_DIM, flow_start)
    assert_array_equal(
        state[block], np.asarray(refusal_block(history), dtype=np.float32)
    )


def test_a_refusal_run_moves_the_block_and_a_fill_clears_it():
    """The latches are integrators, which is the whole argument for the columns.

    A GRU can be asked to learn a 24-turn price difference; asking it to hold a
    running count of consecutive refusals across 720 ticks is the case recurrences
    are worst at. Nothing charges for a refusal any more, but the columns were never
    load-bearing for the charge: without them no policy defined on the state can tell
    it has entered a refusal loop, so none can climb out.
    """

    from agents.dqn_market_free_v3d.market_state import MarketHistory

    farm = {"money": 1000.0}
    obs = {"player": 0, "farms": [farm, farm], "market": {"prices": {}, "inventory": {}}}
    history = MarketHistory()
    history.observe(obs, farm)

    idle = _refusal_state(history)
    flow_start = INTRA_TURN_SLICE.start - CROP_MATURITY_DIM - MARKET_FLOW_DIM
    block = slice(flow_start - REFUSAL_BLOCK_DIM, flow_start)
    assert_array_equal(idle[block], np.zeros(REFUSAL_BLOCK_DIM, dtype=np.float32))

    # Three turns that each emit an order and move no money: refused, by the same
    # `order_refused` rule the reward charges on.
    for _ in range(3):
        history.record_decision(Order("SELL", "WHEAT", 1))
        history.observe(obs, farm)
    assert history.consecutive_refusals == 3
    refused = _refusal_state(history)
    assert refused[block][0] == 0.0, "nothing filled"
    assert refused[block][1] > 0.0, "turns_since_fill has run up"
    assert refused[block][2] == pytest.approx(3 / 8)

    # A turn whose money moves is a fill, and it resets both counters.
    history.record_decision(Order("SELL", "WHEAT", 1))
    history.observe(obs, {"money": 1400.0})
    assert history.consecutive_refusals == 0
    filled = _refusal_state(history)
    assert filled[block][0] == 1.0
    assert_array_equal(filled[block][1:], np.zeros(2, dtype=np.float32))


def test_a_v3c_checkpoint_cannot_load_into_v3d():
    """280 against 296. The fork is a cold start and `load` has to say so.

    v3d widened the vector by `CROP_MATURITY_DIM`, so every v3c checkpoint -- and both
    demonstration packs, which cache `turn_state` at the old width -- is a hard reject
    rather than a shape coincidence."""

    from agents.dqn_market_free_v3d.policy import STATE_DIM

    assert STATE_DIM == MARKET_STATE_DIM == 296
    import agents.dqn_market_free_v3c.market_state as v3c_state

    assert v3c_state.MARKET_STATE_DIM == 280, "the arm being compared against"


def _market_obs(step: int, inventory: dict, shops=()):
    """The same fixed board as `_refusal_state`, with a market that moves."""

    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    farm = {
        "money": 3000.0,
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
        "tiles": tiles,
        "hands": [],
        "farmer": [4, 4],
    }
    private = {"shed": {"WHEAT": 5}, "seeds": {}, "inventories": [{}]}
    stock = {item: float(donor.MARKET_I0) for item in PRODUCTS}
    stock.update(inventory)
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [farm, farm],
        "market": {"prices": {}, "inventory": stock},
        "town": {"unlocked_shops": list(shops)},
        "private": private,
    }, farm


def test_the_flow_block_lands_between_the_refusal_block_and_block_e():
    """Block F must not disturb block E's span.

    `main.py` overwrites exactly `INTRA_TURN_SLICE` when it reuses the turn-opening
    encode, so a block appended after E would be silently rewritten.
    """

    from agents.dqn_market_free_v3d.market_state import MarketHistory, market_flow_block

    history = MarketHistory()
    state = _refusal_state(history)
    obs, farm = _market_obs(30, {})
    history.observe(obs, farm)
    flow_stop = INTRA_TURN_SLICE.start - CROP_MATURITY_DIM
    block = slice(flow_stop - MARKET_FLOW_DIM, flow_stop)
    state = _refusal_state(history)
    assert state.shape == (MARKET_STATE_DIM,)
    bases, curves = _curves_at(obs, {item: float(donor.MARKET_I0) for item in PRODUCTS})
    assert_array_equal(
        state[block],
        np.asarray(market_flow_block(history, obs, bases, curves), dtype=np.float32),
    )


def test_the_flow_column_is_what_the_agent_did_not_do():
    """The whole premise: inventory moved, minus the town, minus us, is them.

    Two turns four apart so the shop tick lands, with one shop buying wheat. The
    market ends 6 wheat heavier; the town took 12 of them; this agent queued 4. The
    remaining 14 are the opponent's, and no part of the observation says so.
    """

    from agents.dqn_market_free_v3d.market_state import MarketHistory
    from agents.dqn_market_free_v3d.order_space import Order

    wheat = PRODUCTS.index("WHEAT")
    history = MarketHistory()
    obs, farm = _market_obs(4, {}, shops=("YARN_STORE", "BAKERY"))
    history.observe(obs, farm)
    history.record_decision(Order("SELL", "WHEAT", 4))

    # BAKERY buys wheat once every four turns; step 4 is a tick, so 1 unit of wheat
    # leaves for the shop. The town centre only fires on multiples of 24.
    later, farm = _market_obs(5, {"WHEAT": donor.MARKET_I0 + 17.0}, shops=("YARN_STORE", "BAKERY"))
    history.observe(later, farm)

    assert history._flow[-1]["WHEAT"] == 18.0, "17 on the books plus the 1 the shop took"
    assert history.last_emitted["WHEAT"] == 4.0
    # 18 - 4 = 14 units this agent did not sell, which is the signal the block exists
    # to carry. The two halves are separate columns because the engine never reports
    # a fill, so the subtraction is the network's to make.
    assert history._flow[-1]["WHEAT"] - history.last_emitted["WHEAT"] == 14.0
    assert history._flow[-1]["MELON"] == 0.0, "a product nobody touched does not move"
    del wheat


def test_the_flow_block_does_not_vary_within_a_turn():
    """Own emitted is read off the history, never off the `PendingTurn` in hand.

    The turn encode passes the *previous* turn and each sub-decision passes the
    *current* one. If block F read `turn` it would disagree between the two, and
    `main.py`'s reuse of the turn-opening encode -- which rewrites block E's span and
    nothing else -- would silently ship a stale column.
    """

    from agents.dqn_market_free_v3d.market_state import (
        MarketHistory,
        market_state_features,
    )
    from agents.dqn_market_free_v3d.order_space import Order, PendingTurn

    history = MarketHistory()
    obs, farm = _market_obs(8, {"WHEAT": donor.MARKET_I0 + 3.0})
    private = obs["private"]
    history.observe(obs, farm)
    history.record_decision(Order("SELL", "WHEAT", 2))
    later, farm = _market_obs(9, {"WHEAT": donor.MARKET_I0 + 9.0})
    private = later["private"]
    history.observe(later, farm)

    roles = herd.field_roles(later, farm, private)
    field = {"farmer": ["PASS"], "hands": []}
    pending = PendingTurn(later, farm, private, field, None)
    flow_stop = INTRA_TURN_SLICE.start - CROP_MATURITY_DIM
    block = slice(flow_stop - MARKET_FLOW_DIM, flow_stop)
    opening = market_state_features(
        later, farm, private, roles, field, history=history, turn=None
    )
    decision = market_state_features(
        later, farm, private, roles, field, history=history, turn=pending
    )
    assert_array_equal(opening[block], decision[block])
    assert not np.array_equal(opening[INTRA_TURN_SLICE], decision[INTRA_TURN_SLICE])


def test_headroom_counts_units_to_the_band_and_a_glut_closes_it():
    """F3 is the impact curve inverted, which is the question a sell order asks.

    WOOL is an `above_func = "sq"` product: at equilibrium it takes tens of units to
    push the quote through 0.9 of base, and once the market is already 60 deep the
    quote has cratered and there is no room left at all.
    """

    from agents.dqn_market_free_v3d.market_state import (
        HEADROOM_BANDS,
        MarketHistory,
        market_flow_block,
    )

    history = MarketHistory()
    obs, _ = _market_obs(12, {})
    quiet = {item: float(donor.MARKET_I0) for item in PRODUCTS}
    glutted = dict(quiet, WOOL=float(donor.MARKET_I0) + 60.0)
    wool = PRODUCTS.index("WOOL")
    head = 3 * len(PRODUCTS) + 3

    assert HEADROOM_BANDS == (0.9, 0.75)
    loose = head + len(PRODUCTS)
    tight = market_flow_block(history, obs, *_curves_at(obs, quiet))
    after = market_flow_block(history, obs, *_curves_at(obs, glutted))

    # 19 units before wool's quote falls through 0.9 of base, 30 before 0.75. The
    # two bands are a slope, not a duplicate column.
    assert round(tight[head + wool] * 45) == 19
    assert round(tight[loose + wool] * 45) == 30

    # 60 units in and wool is already under both bands, so there is no room left at
    # all -- which is the state a policy needs to see *before* it queues the sale,
    # and the exact cliff that costs this family money.
    assert after[head + wool] == 0.0
    assert after[loose + wool] == 0.0

    # The glut is one product's, and the columns say so.
    melon = PRODUCTS.index("MELON")
    assert after[head + melon] == tight[head + melon] == 1.0


def _curves_at(obs, stock):
    """`market_flow_block`'s two price arguments, as `market_state_features` builds them."""

    from agents.dqn_market_free_v3d.market_state import _FILL_OFFSETS, price_curve

    bases = {item: float(donor._market_parameters(obs, item)[0]) for item in PRODUCTS}
    curves = {item: price_curve(item, stock[item] + _FILL_OFFSETS, obs) for item in PRODUCTS}
    return bases, curves


def test_a_dump_cannot_push_the_flow_column_out_of_distribution():
    """Every other column in this vector sits in [0, 2]; F1 has to be bounded too.

    The failure this pre-empts is `SEED_FEATURE_CAP`'s, which was found the other way
    round -- after an unbounded column had already degraded the Q estimate for the
    states that produced it.
    """

    from agents.dqn_market_free_v3d.market_state import (
        FLOW_COLUMN_CLIP,
        MarketHistory,
        market_flow_block,
    )
    from agents.dqn_market_free_v3d.order_space import Order

    history = MarketHistory()
    obs, farm = _market_obs(6, {})
    history.observe(obs, farm)
    # Both seats empty a full shed into the same product on one turn.
    for _ in range(4):
        history.record_decision(Order("SELL", "MELON", 45))
    dumped, farm = _market_obs(7, {"MELON": donor.MARKET_I0 + 400.0})
    history.observe(dumped, farm)

    melon = PRODUCTS.index("MELON")
    stock = {item: float(donor.MARKET_I0) for item in PRODUCTS}
    stock["MELON"] += 400.0
    block = market_flow_block(history, dumped, *_curves_at(dumped, stock))
    assert history._flow[-1]["MELON"] == 400.0, "the raw quantity is not clipped"
    assert history.last_emitted["MELON"] == 180.0
    assert block[melon] == FLOW_COLUMN_CLIP
    assert block[len(PRODUCTS) + melon] == FLOW_COLUMN_CLIP
    assert block[2 * len(PRODUCTS) + melon] == FLOW_COLUMN_CLIP
    assert max(abs(value) for value in block) == FLOW_COLUMN_CLIP


def test_a_lot_resolves_on_its_last_sale_not_its_first():
    """`resolution_turn` is where the split credit's second half lands.

    A lot sold across several turns is only settled by the last of them, and a lot
    that never sold stayed sellable until the buzzer -- so it resolves at the end.
    """

    from agents.dqn_market_free_v3d.lots import _Lot, resolution_turn

    sold = _Lot(item="WHEAT", buy_turn=5, units=3, unit_cost=10.0, capacity=6)
    sold.first_sale_turn = 40
    sold.last_sale_turn = 96
    assert resolution_turn(sold, 719) == 96

    never = _Lot(item="MELON", buy_turn=5, units=1, unit_cost=80.0, capacity=6)
    assert never.last_sale_turn is None
    assert resolution_turn(never, 719) == 719


def test_split_credit_moves_half_the_charge_without_changing_its_size():
    """The pair stays bounded by the lot's own cost; only the indices change.

    At a share of 1.0 the walk has to reproduce the old single-index behaviour
    exactly, which is what makes the share a dial rather than a rewrite.
    """

    from agents.dqn_market_free_v3d.lots import _Lot, _split

    wasted = _Lot(item="MELON", buy_turn=12, units=1, unit_cost=80.0, capacity=6)
    paid = _Lot(item="WHEAT", buy_turn=20, units=2, unit_cost=10.0, capacity=12)
    paid.proceeds = 50.0
    paid.first_sale_turn = 30
    paid.last_sale_turn = 64

    entries = [(lot.buy_turn, max(0.0, lot.cost - lot.proceeds)) for lot in (wasted, paid)]
    split = _split(entries, (wasted, paid), 719)
    assert split[0] == (12, 719, 80.0), "a never-sold lot resolves at the buzzer"
    assert split[1] == (20, 64, 0.0), "a lot paid back in full owes nothing"

    # The coins are untouched by the split -- the whole change is which index pays.
    assert [coins for _, _, coins in split] == [coins for _, coins in entries]
    for buy_turn, resolved, _ in split:
        assert resolved >= buy_turn, "a lot cannot resolve before it was bought"


# ---------------------------------------------------------------------------
# The Kaggle ladder mirror as demonstrations
# ---------------------------------------------------------------------------


def _ladder_replay(steps: int = 96):
    """A short episode in the ladder replay's own shape.

    `env.toJSON()` is the same `kaggle-environments` document the Kaggle episode
    mirror stores, which is the whole reason `record_from_replay` reads both
    without knowing which it got. Generated rather than committed as a fixture:
    a real mirrored replay is ~30 MB and `data/` is gitignored.
    """

    from kaggle_environments import make

    from agents.dqn_market_free_v3d.main import MarketFreeAgent
    from tests.test_dqn_market_free import SNAPSHOT_PATH

    network = OrderQNetwork.random(seed=5)
    learner = MarketFreeAgent(network, rng=np.random.default_rng(0))
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": 17}, debug=True)
    env.run([learner, str(SNAPSHOT_PATH)])
    return env.toJSON(), network


def test_rehidden_reproduces_the_walk_it_replaced():
    """The invariant the demonstration cache rests on.

    `training/ladder.py` converts a replay once and re-rolls the hidden states per
    update, because the policy that has to have produced them moves every update
    and inflating 30 MB of JSON to redo the walk does not. That is only sound if
    rolling the GRU over the *stored* `turn_state` lands exactly where the walk
    did -- same zero start, same pre-tick store, same stride, same fp16. If it
    drifts, every mid-episode chunk resumes from a state the policy never
    occupied, and nothing about the record's widths or stride would say so.
    """

    from agents.dqn_market_free_v3d.training.clone import record_from_replay, rehidden

    replay, network = _ladder_replay()
    record, _stats = record_from_replay(replay, seat=0, gru=network.gru)
    assert record is not None
    assert np.abs(record.turn_hidden[1:].astype(np.float32)).sum() > 0.0

    # Under the same GRU: bit-identical, because it is the same arithmetic.
    same = rehidden(record, network.gru)
    np.testing.assert_array_equal(same.turn_hidden, record.turn_hidden)
    assert int(same.hidden_stride[0]) == int(record.hidden_stride[0])

    # And under a different one it actually moves, so the assertion above is
    # testing the roll rather than a column that was copied through.
    other = rehidden(record, OrderQNetwork.random(seed=11).gru)
    assert other.turn_hidden.shape == record.turn_hidden.shape
    assert not np.array_equal(other.turn_hidden, record.turn_hidden)
    assert not np.any(other.turn_hidden[0]), "row 0 is the state the episode opened on"


def test_rehidden_refuses_a_pack_from_another_schema():
    from agents.dqn_market_free_v3d.training.clone import record_from_replay, rehidden

    replay, network = _ladder_replay()
    record, _stats = record_from_replay(replay, seat=0, gru=network.gru)
    record.turn_state = record.turn_state[:, :-1]
    with pytest.raises(ValueError, match="another schema"):
        rehidden(record, network.gru)


def test_ladder_pack_round_trips_and_reports_only_what_it_loaded(tmp_path):
    """A pack is a prefix, and its metrics have to describe the prefix.

    Shards are written in descending ladder score and `demonstration_episodes`
    reads the head, so a pack built at 320 and loaded at 96 must not report the
    320-episode totals -- a `demonstrations.episodes` that disagreed with the
    number of pinned records is a metrics row nobody can reconcile.
    """

    from agents.dqn_market_free_v3d.training import ladder
    from agents.dqn_market_free_v3d.training.clone import record_from_replay

    replay, network = _ladder_replay()
    records = []
    totals = ladder.CloneStats()
    provenance = []
    for seat in (0, 1):
        record, stats = record_from_replay(replay, seat, 0.999, gru=network.gru)
        assert record is not None
        records.append(record)
        totals = ladder._merge(totals, stats)
        provenance.append({"seat": seat, "stats": ladder._as_dict(stats)})
    ladder.save_episode_batch(records[:1], tmp_path / "part-0000.npz")
    ladder.save_episode_batch(records[1:], tmp_path / "part-0001.npz")
    (tmp_path / ladder.SIDECAR).write_text(
        json.dumps({"stats": ladder._as_dict(totals), "provenance": provenance})
    )

    loaded, stats = ladder.load_ladder_demonstrations(tmp_path, gru=network.gru, limit=1)
    assert len(loaded) == 1
    assert stats.episodes == 1
    assert stats.decisions == provenance[0]["stats"]["decisions"]

    whole, stats = ladder.load_ladder_demonstrations(tmp_path, gru=network.gru, limit=8)
    assert len(whole) == 2
    assert stats.episodes == 2
    assert stats.decisions == totals.decisions
    # Re-rolled on the way out rather than trusted from disk.
    np.testing.assert_array_equal(whole[0].turn_hidden, records[0].turn_hidden)


def test_ladder_seat_finals_read_the_observation_player():
    """Through `observation["player"]`, never the seat index.

    They agree in every replay seen so far. Disagreeing would rank a seat by the
    other seat's wallet, which silently selects the demonstrations backwards.
    """

    from agents.dqn_market_free_v3d.training import ladder

    replay = {
        "steps": [
            [
                {"observation": {"player": 1, "farms": [{"money": 10}, {"money": 900}]}},
                {"observation": {"player": 0, "farms": [{"money": 10}, {"money": 900}]}},
            ]
        ]
    }
    assert ladder.seat_finals(replay) == [900.0, 10.0]


def test_ladder_mirror_walk_is_ranked_and_skips_absent_replays(tmp_path):
    """The manifests list every episode a day published; the mirror holds a slice."""

    from agents.dqn_market_free_v3d.training import ladder

    for date, rows in (
        ("2026-08-01", [("1", 3000.0), ("2", 2000.0)]),
        ("2026-08-02", [("3", 2500.0)]),
    ):
        day = tmp_path / date
        day.mkdir()
        lines = ["episode_id,avg_score"] + [f"{episode},{score}" for episode, score in rows]
        (day / "manifest.csv").write_text("\n".join(lines) + "\n")
        for episode, _score in rows:
            # Episode "2" is listed and not mirrored, which is the ordinary case.
            if episode != "2":
                (day / f"{episode}.json.gz").write_bytes(b"")

    walked = list(ladder.iter_mirror_episodes(tmp_path))
    assert [item.episode_id for item in walked] == ["1", "3"]
    assert [item.episode_id for item in ladder.iter_mirror_episodes(tmp_path, days=1)] == ["3"]


# --------------------------------------------------------------------------
# The reward: the per-turn cash delta over `reward.MONEY_SCALE`.
#
# Every reward-constant test in this repo lives in `test_dqn_market_free.py` and
# imports the **v1** package, so until these landed nothing guarded this fork's
# copy. What they pin is the argument the constant moved on, not the number: the
# return has to track money, and it has to keep slope where the policy is trying
# to get to.
# --------------------------------------------------------------------------


def _flat_season(final_money: float, turns: int = 720, stake: float = 3_000.0):
    """A money trace that holds the stake and lands on `final_money`."""

    return [stake] * turns, final_money


def _lot_traces(turns: int, seed_counts=None, stock_counts=None, held_from: int = 0):
    """Lot traces for a season, sized as `episode_rewards` expects them.

    `held_from` is the turn the holding appears on. It defaults to 0 -- a farm that
    owned the stock all along -- which is what a flat comparison wants, but any test
    that reads the sum needs the purchase inside the season: the stream telescopes
    over the basis, so stock present on turn 0 is stock the season is charged for
    losing without ever being credited for buying.
    """

    from agents.dqn_market_free_v3d import lots

    crops = len(lots.CROP_ITEMS)
    items = len(lots.STOCK_ITEMS)
    seeds, standing, stock = [], [], []
    for turn in range(turns):
        owns = turn >= held_from
        seeds.append(list(seed_counts) if owns and seed_counts else [0] * crops)
        standing.append([0] * crops)
        stock.append(list(stock_counts) if owns and stock_counts else [0] * items)
    return seeds, standing, stock


def _animal_traces(turns: int, animal_counts=None, held_from: int = 0):
    from agents.dqn_market_free_v3d import lots

    animals = len(lots.ANIMAL_ITEMS)
    owned = []
    for turn in range(turns):
        owns = turn >= held_from
        owned.append(list(animal_counts) if owns and animal_counts else [0] * animals)
    return owned


def test_reward_is_zero_wherever_cash_does_not_move():
    from agents.dqn_market_free_v3d.training.reward import MONEY_SCALE, episode_rewards

    money, final = _flat_season(9_000.0)
    reward, raw, money_array = episode_rewards(money, final, 1.0)

    profit = (final - 3_000.0) / MONEY_SCALE
    assert reward.shape == raw.shape == (len(money),)
    assert money_array.shape == (len(money) + 1,)
    # A flat trace pays nothing until the last turn, where the season lands.
    assert np.all(reward[:-1] == 0.0)
    assert float(reward[-1]) == pytest.approx(profit)
    assert float(reward.sum()) == pytest.approx(profit, abs=1e-6)
    assert np.array_equal(reward, raw)


def test_reward_is_charged_on_the_turn_the_cash_moves():
    from agents.dqn_market_free_v3d.training.reward import MONEY_SCALE, episode_rewards

    money, final = _flat_season(9_000.0)
    # An outlay at turn 9 lowers money[10]; turn 9 is charged, turn 10 is not.
    spent = list(money)
    spent[10:] = [value - 1_000.0 for value in spent[10:]]
    charged, _, _ = episode_rewards(spent, final, 0.99)
    assert float(charged[9]) == pytest.approx(-1_000.0 / MONEY_SCALE)
    assert float(charged[10]) == 0.0
    # gamma is not baked into the label; replay discounts over the horizon.
    undiscounted, _, _ = episode_rewards(spent, final, 1.0)
    assert np.array_equal(charged, undiscounted)


def test_reward_is_the_cash_delta_of_each_turn():
    from agents.dqn_market_free_v3d.training.reward import MONEY_SCALE, episode_rewards

    target, raw, money = episode_rewards([3000.0, 2000.0, 5000.0], 6000.0, 0.999)

    assert money.tolist() == [3000.0, 2000.0, 5000.0, 6000.0]
    assert target.tolist() == pytest.approx(
        [-1000.0 / MONEY_SCALE, 3000.0 / MONEY_SCALE, 1000.0 / MONEY_SCALE]
    )
    assert raw.tolist() == pytest.approx(target.tolist())


def test_reward_sums_rank_winning_and_losing_seasons():
    from agents.dqn_market_free_v3d.training.reward import MONEY_SCALE, episode_rewards

    from agents.dqn_market_free_v3d.training.reward import ruin_penalty

    kept, final_kept = _flat_season(124_000.0)
    lost, final_lost = _flat_season(1_000.0)
    won, won_raw, _ = episode_rewards(kept, final_kept, 1.0)
    lose, lose_raw, _ = episode_rewards(lost, final_lost, 1.0)

    # `raw` is the cash and only the cash, which is what `own_mean_return` grades on.
    assert float(won_raw.sum()) == pytest.approx(121_000.0 / MONEY_SCALE)
    assert float(lose_raw.sum()) == pytest.approx(-2_000.0 / MONEY_SCALE)
    assert float(won_raw.sum() - lose_raw.sum()) == pytest.approx(123_000.0 / MONEY_SCALE)

    # `target` carries the terminal ruin charge, and the losing season finished at
    # 1,000 -- under the 3,000 stake, so it pays two thirds of it.
    assert float(won.sum()) == pytest.approx(121_000.0 / MONEY_SCALE)
    assert float(lose.sum()) == pytest.approx(
        -2_000.0 / MONEY_SCALE + ruin_penalty(final_lost)
    )
    # The charge widens the ranking rather than inverting it, which is the point.
    assert float(won.sum() - lose.sum()) > 123_000.0 / MONEY_SCALE


def test_reward_ignores_the_retired_shaping_inputs():
    from agents.dqn_market_free_v3d.training.reward import episode_rewards

    money, final = _flat_season(9_000.0)
    seeds, standing, stock = _lot_traces(len(money), seed_counts=[40, 0, 0, 0, 0])
    plain, raw, _ = episode_rewards(money, final, 0.999)
    traced, traced_raw, _ = episode_rewards(
        money,
        final,
        0.999,
        emitted=[False] * len(money),
        seeds=seeds,
        standing=standing,
        stock=stock,
        basis_required=True,
        idle_penalty=100.0,
    )

    assert traced.tolist() == plain.tolist()
    assert traced_raw.tolist() == raw.tolist()

# --------------------------------------------------------------------------
# The behaviour-clone loss -- see `updater._large_margin_loss`.
#
# `market-free-splitladder-v1` ran a whole session at `max_abs_q` 20-27 because
# the clone was a cross-entropy over Q-values. These pin the property that fixed
# it: the loss has a fixed point, and it is inside the reward's range.
# --------------------------------------------------------------------------


def test_the_clone_loss_stops_once_the_demonstrated_action_leads():
    """Zero loss *and* zero gradient, which cross-entropy never reaches.

    This is the whole difference. Cross-entropy's optimum is the chosen logit at
    infinity, so it keeps pushing a Q head that is already ranking correctly, and
    `grad_clip_norm` bounds the step rather than the destination.
    """

    from agents.dqn_market_free_v3d.training.updater import (
        BEHAVIOR_CLONE_MARGIN,
        _large_margin_loss,
    )

    logits = torch.tensor([[5.0, 1.0, 2.0]], requires_grad=True)
    loss = _large_margin_loss(logits, torch.tensor([0]), BEHAVIOR_CLONE_MARGIN)
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert float(logits.grad.abs().sum()) == 0.0


def test_the_clone_loss_is_the_ranking_gap_plus_the_margin_in_q_units():
    """It states a preference ordering, not a target distribution."""

    from agents.dqn_market_free_v3d.training.updater import _large_margin_loss

    behind = torch.tensor([[1.0, 2.0, 0.0]])
    assert float(_large_margin_loss(behind, torch.tensor([0]), 0.3)) == pytest.approx(1.3)
    # Exactly at the margin is the boundary, and it is closed.
    edge = torch.tensor([[2.3, 2.0, 0.0]])
    assert float(_large_margin_loss(edge, torch.tensor([0]), 0.3)) == pytest.approx(0.0)


def test_the_clone_loss_cannot_be_dragged_up_by_a_masked_rival():
    """Infeasible rows arrive at `MASKED_SCORE`; the margin must not resurrect one."""

    from agents.dqn_market_free_v3d.order_space import MASKED_SCORE
    from agents.dqn_market_free_v3d.training.updater import _large_margin_loss

    masked = torch.tensor([[1.0, float(MASKED_SCORE), 0.0]])
    assert float(_large_margin_loss(masked, torch.tensor([0]), 0.3)) == 0.0


def test_the_clone_margin_is_sized_inside_the_reward_range():
    """A margin above the return's own range would set the Q scale itself.

    `outcome` is a tanh and every other term in `reward.py` is bounded, so the
    whole return lives inside roughly +/-2.
    """

    from agents.dqn_market_free_v3d.training.updater import BEHAVIOR_CLONE_MARGIN

    assert 0.0 < BEHAVIOR_CLONE_MARGIN < 1.0


def test_order_viability_records_behavior_without_gating_on_it():
    from agents.dqn_market_free_v3d.training.updater import _order_viability

    result = {
        "pairs": 100,
        "errors": {},
        "order_diagnostics": {
            "decisions": 72000,
            "orders": 72000,
            "trading_rate": 0.5,
        },
        "behavior": {
            "seed_lots": 100,
            "seed_lots_matched_share": 0.01,
            "animal_lots": 20,
            "animal_lots_matched_share": 0.05,
            "unrecovered_animal_share": 0.95,
        },
    }

    gate = _order_viability(result, {}, require_activity=True)

    assert gate["passed"]
    assert gate["checks"]["min_orders_per_decision"]
    assert "productive_lots" not in gate["checks"]
    assert "seed_lot_match" not in gate["checks"]
    assert "animal_lot_match" not in gate["checks"]
    assert "animal_recovery" not in gate["checks"]


def test_order_viability_reports_refused_order_rate_without_gating_on_it():
    from agents.dqn_market_free_v3d.training.updater import _order_viability

    result = {
        "pairs": 100,
        "errors": {},
        "order_diagnostics": {
            "decisions": 72000,
            "orders": 72000,
            "trading_rate": 0.10,
            "rejected_order_rate": 0.20,
        },
    }

    gate = _order_viability(result, {}, require_activity=True)

    assert gate["passed"]
    assert gate["checks"]["min_orders_per_decision"]
    assert "max_rejected_order_rate" not in gate["checks"]
    assert gate["rejected_order_rate"] == pytest.approx(0.20)


def test_order_viability_keeps_activity_as_the_only_order_gate():
    from agents.dqn_market_free_v3d.training.updater import _order_viability

    result = {
        "pairs": 100,
        "errors": {},
        "order_diagnostics": {
            "decisions": 72000,
            "orders": 72000,
            "trading_rate": 0.10,
            "rejected_order_rate": 0.01,
        },
    }

    gate = _order_viability(result, {}, require_activity=True)

    assert gate["passed"]
    assert gate["checks"]["min_orders_per_decision"]
    assert "max_rejected_order_rate" not in gate["checks"]


def test_order_viability_rejects_idle_benchmark_results():
    from agents.dqn_market_free_v3d.training.updater import _order_viability

    result = {
        "pairs": 100,
        "errors": {},
        "order_diagnostics": {
            "decisions": 72000,
            "orders": 0,
            "trading_rate": 0.0,
        },
    }

    gate = _order_viability(result, {}, require_activity=True)

    assert not gate["passed"]
    assert not gate["checks"]["min_orders_per_decision"]
    assert gate["trading_rate"] == 0.0


# --------------------------------------------------------------------------
# Sell rungs against held stock -- see `order_space.stock_ceiling`.
# --------------------------------------------------------------------------


def test_the_stock_ceiling_is_the_smallest_rung_that_clears_the_shed():
    """Everything above it is the same fill under a different name."""

    from agents.dqn_market_free_v3d.order_space import QUANTITIES, stock_ceiling

    for held in range(0, 50):
        ceiling = stock_ceiling(held)
        assert ceiling in QUANTITIES
        assert ceiling >= min(held, QUANTITIES[-1])
        below = [level for level in QUANTITIES if level < ceiling]
        assert all(level < held for level in below), "only rungs under the shed survive below it"
    # Past the ladder it saturates rather than falling off.
    assert stock_ceiling(10_000) == QUANTITIES[-1]


def test_the_stock_clip_never_shuts_selling_outright():
    """A seat holding anything must keep a way to sell it.

    The smallest rung is 1 and the caller has already skipped items with nothing
    held, so the clip can narrow the ladder but never empty it -- the same
    guarantee `sell_depth_cap` makes with its `QUANTITIES[0]` floor.
    """

    from agents.dqn_market_free_v3d.order_space import QUANTITIES, stock_ceiling

    assert all(stock_ceiling(held) >= QUANTITIES[0] for held in range(1, 200))


def test_a_turn_sells_each_item_at_most_once():
    """One SELL per item per turn.

    A *preference*, not a proof like the wash rule above -- 45 + 45 units is a
    reachable outcome no single rung reproduces, because `QUANTITIES` stops at 45 --
    so it is carried on a measured ceiling instead. Over 300 ladder seats a second
    same-item SELL is 1.9% of sell orders and 1.4% of units; over
    `market-free-splitladder-v1` it was 72.7% of them.
    """

    from agents.dqn_market_free_v3d.order_space import (
        _SELL_ROWS,
        NOOP_INDEX,
        Order,
        PendingTurn,
        feasible_actions,
    )
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(
        money=5000.0, shed={"WHEAT": 60, "MELON": 20}
    )
    pending = PendingTurn(obs, farm, private, field, None)

    opening = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert opening[_SELL_ROWS["WHEAT"]].any(), "the turn must be able to sell at all"

    pending.apply(Order("SELL", "WHEAT", 6))
    after = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert not after[_SELL_ROWS["WHEAT"]].any(), "wheat is done for the turn"
    # A cap on repetition, not on trading: a different item and the terminator both
    # survive, so nothing the seat holds becomes unsellable.
    assert after[_SELL_ROWS["MELON"]].any(), "a different item stays sellable"
    assert after[NOOP_INDEX], "the terminator is never shut"


def test_sell_rungs_are_clipped_to_the_shed():
    """`stock_ceiling` in the mask: rungs past the holding are the same fill.

    The rung that clears the shed stays open -- that is the one that sells it all --
    and everything above it goes, because the shed bounds the fill and they would
    put the identical units on the wire.
    """

    from agents.dqn_market_free_v3d.order_space import (
        _ROW_QUANTITY,
        _SELL_ROWS,
        PendingTurn,
        feasible_actions,
        stock_ceiling,
    )
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(money=5000.0, shed={"WHEAT": 5})
    pending = PendingTurn(obs, farm, private, field, None)
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)

    rows = _SELL_ROWS["WHEAT"]
    ceiling = stock_ceiling(5)
    assert ceiling == 6, "the smallest rung that clears five units"
    assert mask[rows].any(), "five units stay sellable"
    for open_row, quantity in zip(mask[rows], _ROW_QUANTITY[rows]):
        if open_row:
            assert int(quantity) <= ceiling, quantity


def test_flip_rate_curve_measures_down_to_the_configured_sigma_floor():
    """The probe ladder spans exactly what `resolve_sigma_scale` may return.

    `noisy_sigma_scale_floor` *replaces* the implicit `1 / scale_limit` lower clamp
    rather than tightening it, and the resolver only ever answers with a scale the
    curve mentions. Clipping the ladder on the limit alone therefore made the floor
    dead config in one direction and let the probe measure unreachable rungs in the
    other: `market-free-cashherd-v1` ran `limit = 2` with `floor = 0.25`, measured
    {0.5, 1, 2}, and five of ten slots sat above the 0.05 flip-to-order setpoint at
    the bottom rung with nowhere to descend to.
    """

    import torch
    import torch.nn as nn

    from agents.dqn_market_free_v3d.training import updater

    class _Stub(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.linear = nn.Linear(4, width)

        def forward(self, features):  # noqa: D102 - test stub
            return self.linear(features)

    torch.manual_seed(0)
    module = _Stub(updater.ACTION_COUNT)
    features = torch.randn(16, 4)
    mask = torch.ones(16, updater.ACTION_COUNT, dtype=torch.bool)

    def rungs(limit: float, floor: float) -> list[float]:
        curve, _ = updater.flip_rate_curve(
            module, features, mask, limit, draws=1, scale_floor=floor
        )
        return sorted(curve)

    # The floor reaches below the implicit clamp, which is the case that was broken.
    assert min(rungs(2.0, 0.25)) == pytest.approx(0.25)
    # And above it, where the resolver may not return the lower rungs, they are not
    # measured either -- the ladder tracks the resolver in both directions.
    assert min(rungs(8.0, 0.5)) == pytest.approx(0.5)
    # Unset is byte-for-byte the old behaviour.
    assert rungs(2.0, 0.0) == [0.5, 1.0, 2.0]
    assert min(rungs(8.0, 0.0)) == pytest.approx(0.125)

    # Every rung the widened ladder adds is one the resolver can actually answer
    # with, which is the property that makes the floor mean anything.
    curve = {scale: 0.4 for scale in rungs(2.0, 0.25)}
    curve[0.25] = 0.01
    assert updater.resolve_sigma_scale(curve, 0.01, 2.0, 0.25) == pytest.approx(0.25)


# --------------------------------------------------------------------------
# The bought-stock ledger legs in `lots.py`. Descriptive now -- the reward reads
# the money trace alone -- but `diagnostics` still reports them, and the hole they
# were built to expose is worth keeping visible.
#
# What these guard is the hole those columns closed. `market-free-splitladder-v1`
# moved 213,280 coins of gross revenue an episode-seat and kept 1,843 of them, a
# 0.9% net margin against a stored ladder champion's 67.3%, because 89% of that
# turnover was WHEAT and FERTILIZER round-tripped through the market and a
# `BUY_PRODUCT` lot was read by no reward leg at all.
# --------------------------------------------------------------------------


def _bought_lot(
    item: str = "WHEAT",
    units: int = 45,
    unit_cost: float = 40.0,
    absorbed: int = 0,
    proceeds: float = 0.0,
    buy_turn: int = 10,
):
    """One `BUY_PRODUCT` lot, posed directly rather than matched out of a trace.

    Bought stock resells one for one, so `capacity == units` is what `lots.match`
    opens it with and `absorbed` is the unit count that later left through a SELL.
    """

    from agents.dqn_market_free_v3d.lots import _Lot

    lot = _Lot(
        item=item,
        buy_turn=buy_turn,
        units=units,
        unit_cost=unit_cost,
        capacity=units,
        bought=True,
    )
    lot.absorbed = absorbed
    lot.proceeds = proceeds
    if absorbed:
        lot.first_sale_turn = buy_turn + 1
        lot.last_sale_turn = buy_turn + 1
    return lot


def _ledger_of(*lots_in):
    from agents.dqn_market_free_v3d.lots import LotLedger

    return LotLedger(lots=list(lots_in))


def test_a_bought_lot_that_lost_on_the_round_trip_charges_its_buy_turn():
    """The behaviour the pair exists to rank: buy 45 wheat, sell it back for less.

    Both legs read the *realised* trip -- proceeds against the cost of the units
    that actually left -- so a losing trip charges and a winning one credits, and
    the two are clamped so a policy cannot collect on both.
    """

    lost = _bought_lot(absorbed=45, proceeds=1_500.0)  # cost of sold units 1,800
    won = _bought_lot(absorbed=45, proceeds=2_400.0)

    assert _ledger_of(lost).stock_loss() == [(10, 300.0)]
    assert _ledger_of(lost).stock_gain() == [(10, 0.0)]
    assert _ledger_of(won).stock_loss() == [(10, 0.0)]
    assert _ledger_of(won).stock_gain() == [(10, 600.0)]


def test_bought_stock_consumed_rather_than_resold_is_charged_nothing():
    """The reason this is a realised-margin leg and not `unrecovered()`.

    Wheat bought to feed the herd and fertilizer bought to spread on the tiles
    both end with `proceeds == 0`, so the seed and animal definition -- `cost -
    proceeds` over the whole lot -- would charge the farm for using the input it
    bought. Counting only `unit_cost * absorbed` prices the round trip and nothing
    else.
    """

    from agents.dqn_market_free_v3d.lots import _unrecovered

    fed = _bought_lot(item="WHEAT", units=45, unit_cost=40.0, absorbed=0)
    spread = _bought_lot(item="FERTILIZER", units=20, unit_cost=85.0, absorbed=0)

    for lot in (fed, spread):
        ledger = _ledger_of(lot)
        assert ledger.stock_loss() == [(10, 0.0)]
        assert ledger.stock_gain() == [(10, 0.0)]
    # And the definition that would have got this wrong, stated so the difference
    # is not re-derived by hand the next time someone reaches for the seed helper.
    assert _unrecovered([fed]) == [(10, 1_800.0)]


def test_the_bought_stock_legs_are_clamped_per_lot_and_never_netted():
    """Each lot moves at most the cost of what it sold, in either direction.

    The cap is what bounds the pair by resale turnover, and is why a linear-coin
    term cannot quietly displace `outcome` as the objective.
    """

    windfall = _bought_lot(absorbed=45, proceeds=20_000.0)
    ledger = _ledger_of(windfall)
    sold_cost = 45 * 40.0
    assert ledger.stock_gain() == [(10, sold_cost)], "capped at the cost of what it sold"
    assert ledger.stock_loss() == [(10, 0.0)]

    partial = _bought_lot(units=45, absorbed=10, proceeds=100.0)
    # Only the ten units that left are priced; the 35 still in the shed are not.
    assert _ledger_of(partial).stock_loss() == [(10, 300.0)]


def test_the_bought_stock_legs_are_only_the_lots_no_other_leg_reads():
    """`bought_lots()` and the two existing accessors partition the ledger.

    WHEAT is both a crop and a `BUY_PRODUCT` item, so a lot landing in two legs
    would be billed twice for the same coins.
    """

    from agents.dqn_market_free_v3d.lots import _Lot

    seed = _Lot(item="WHEAT", buy_turn=3, units=2, unit_cost=10.0, capacity=12)
    cow = _Lot(item="MILK", buy_turn=4, units=1, unit_cost=400.0, capacity=8, animal="COW")
    ledger = _ledger_of(seed, cow, _bought_lot())

    assert ledger.seed_lots() == [seed]
    assert ledger.animal_lots() == [cow]
    assert [lot.item for lot in ledger.bought_lots()] == ["WHEAT"]
    assert len(ledger.seed_lots()) + len(ledger.animal_lots()) + len(
        ledger.bought_lots()
    ) == len(ledger.lots)


def test_split_legs_carries_the_bought_stock_pair():
    """Six keys, and the bought pair resolves like every other lot."""

    ledger = _ledger_of(_bought_lot(absorbed=45, proceeds=1_500.0))
    legs = ledger.split_legs(719)
    assert set(legs) == {
        "unrecovered",
        "surplus",
        "animal_unrecovered",
        "animal_surplus",
        "stock_loss",
        "stock_gain",
    }
    assert legs["stock_loss"] == [(10, 11, 300.0)], "resolved on its last sale"
    assert legs["stock_gain"] == [(10, 11, 0.0)]

    never = _ledger_of(_bought_lot())
    assert never.split_legs(719)["stock_loss"] == [(10, 719, 0.0)]


def test_a_fertilizer_lot_is_reconciled_against_the_shed():
    """The gap the third `_reconcile` pass closed.

    FERTILIZER is in `STOCK_ITEMS` but is neither a crop nor an animal's product, so
    neither existing ceiling iterated it: a lot whose units were spread on the tiles
    kept its full room forever, stayed at the head of the FIFO and absorbed a later
    sale a later purchase had paid for.
    """

    from agents.dqn_market_free_v3d import lots

    fert = lots.STOCK_ITEMS.index("FERTILIZER")
    assert lots._TRADED_ONLY == ("FERTILIZER",)

    crops = len(lots.CROP_ITEMS)
    turns = 5
    seeds = [[0] * crops for _ in range(turns + 1)]
    standing = [[0] * crops for _ in range(turns + 1)]
    stock = [[0] * len(lots.STOCK_ITEMS) for _ in range(turns + 1)]
    ops: list[str | None] = [None] * turns
    events: list[list] = [[] for _ in range(turns)]
    money = [3_000.0] * (turns + 1)

    # Turn 0 buys 10 fertilizer for 850; turn 1 spreads all ten on the tiles --
    # stock down, money flat, matching no rule in `match`.
    stock[1][fert] = 10
    ops[0] = "BUY_PRODUCT"
    events[0] = [("BUY_PRODUCT", "FERTILIZER", 10, -850.0)]
    money[1] = 2_150.0
    for turn in range(2, turns + 1):
        stock[turn][fert] = 0

    # Turn 3 buys 4 more and turn 4 sells them for 400.
    stock[4][fert] = 4
    ops[3] = "BUY_PRODUCT"
    events[3] = [("BUY_PRODUCT", "FERTILIZER", 4, -340.0)]
    money[4] = 1_810.0
    stock[5][fert] = 0
    ops[4] = "SELL"
    events[4] = [("SELL", "FERTILIZER", 4, 400.0)]
    money[5] = 2_210.0

    ledger = lots.match(seeds, stock, money, ops, standing, events=events)
    bought = ledger.bought_lots()
    assert [lot.buy_turn for lot in bought] == [0, 3]
    spread, resold = bought
    assert spread.absorbed == 0, "the spread lot must not absorb the later sale"
    assert resold.absorbed == 4 and resold.proceeds == 400.0
    # And therefore the ledger books the trip that happened, not the one that did not.
    assert ledger.stock_loss() == [(0, 0.0), (3, 0.0)]
    assert ledger.stock_gain() == [(0, 0.0), (3, 60.0)]


def test_this_policys_seasons_sit_in_the_huber_quadratic_zone_and_ladder_seasons_do_not():
    """`SmoothL1Loss` has a beta of 1.0; `MONEY_SCALE` is sized so the seasons this
    policy actually plays land inside it with room to differ.

    At 250_000 a 5k and a 10k season were 0.008 and 0.028 -- a 0.02-wide band whose
    gradient was the residual itself. At 25_000 they are 0.08 and 0.28, a 28k season
    reaches the edge of the quadratic zone, and a 124k ladder season is deliberately
    in the linear zone: it is an outlier to this policy and Huber treats it as one.
    """

    from agents.dqn_market_free_v3d.training.reward import MONEY_SCALE, episode_rewards

    crops = 5
    turns = 720
    money = [3_000.0 + 168.0 * turn - 40.0 * (turn % 25) for turn in range(turns)]
    shaped, _, _ = episode_rewards(
        money,
        124_000.0,
        0.999,
        emitted=[True] * turns,
        seeds=[[0] * crops] * turns,
        standing=[[0] * crops] * turns,
        stock=[[0] * 9] * turns,
    )
    # The season's value -- what Q at turn 0 has to reach through the bootstrap
    # chain -- is the sum of the stream, and a ladder season sits in the linear
    # zone under the clamp.
    assert float(shaped.sum()) == pytest.approx(121_000.0 / MONEY_SCALE)
    assert float(shaped[-1]) == pytest.approx((124_000.0 - money[-1]) / MONEY_SCALE)
    assert 1.0 < float(np.abs(shaped.sum())) < 8.0
    # One n-step window of it (24 turns, one day) is well inside the quadratic zone.
    assert float(np.abs(shaped[:24].sum())) < 1.0

    for final in (5_000.0, 10_000.0, 28_000.0):
        flat, _, _ = episode_rewards([3_000.0] * turns, final, 0.999)
        assert float(np.abs(flat.sum())) <= 1.0
    assert float(episode_rewards([3_000.0] * turns, 10_000.0, 0.999)[0].sum()) - float(
        episode_rewards([3_000.0] * turns, 5_000.0, 0.999)[0].sum()
    ) == pytest.approx(0.2)


def test_the_lagged_wash_counter_sees_what_the_mask_does_not():
    """`wash_turns` reads zero all session; the round trip moved one turn out.

    Measured over 20 stored `market-free-splitladder-v1` seats: 0.0% of resold
    units at lag 0, 17.3% of wheat and 49.8% of fertilizer at lag 1.
    """

    from agents.dqn_market_free_v3d.order_space import ACTIONS
    from agents.dqn_market_free_v3d.training.collect import (
        LAGGED_WASH_WINDOW,
        EpisodeRecord,
    )

    def index_of(op: str, item: str) -> int:
        for index, action in enumerate(ACTIONS):
            if action.op == op and action.item == item:
                return index
        raise AssertionError(f"no {op} {item} in the action space")

    buy = index_of("BUY_PRODUCT", "WHEAT")
    sell = index_of("SELL", "WHEAT")

    def record(pairs):
        chosen = np.asarray([choice for _, choice in pairs], dtype=np.int16)
        steps = np.asarray([turn for turn, _ in pairs], dtype=np.int16)
        empty = np.zeros(0, dtype=np.float32)
        return EpisodeRecord(
            reward=np.zeros(800, dtype=np.float32),
            raw_reward=np.zeros(800, dtype=np.float32),
            money=np.zeros(801, dtype=np.float32),
            order_state=np.zeros((len(pairs), 1), dtype=np.float32),
            order_chosen=chosen,
            order_step=steps,
            order_mask=np.ones((len(pairs), len(ACTIONS)), dtype=bool),
            turn_state=np.zeros((800, 1), dtype=np.float32),
            turn_hidden=np.zeros((1, 1), dtype=np.float32),
            hidden_stride=np.asarray([20], dtype=np.int16),
        )

    next_turn = record([(0, buy), (1, sell)])
    assert next_turn.wash_turns == 0, "the mask holds: the legs are not on one turn"
    assert next_turn.lagged_wash_turns == 1

    # Beyond the window it is inventory management, not a round trip.
    far = record([(0, buy), (LAGGED_WASH_WINDOW + 1, sell)])
    assert far.lagged_wash_turns == 0
    edge = record([(0, buy), (LAGGED_WASH_WINDOW, sell)])
    assert edge.lagged_wash_turns == 1

    # A sale with no purchase behind it is the farm selling what it grew.
    assert record([(0, sell), (1, sell)]).lagged_wash_turns == 0


# --------------------------------------------------------------------------
# Member pools: the round-robin schedule, the prune, and buffer isolation.
#
# What these guard is the replacement for the learner pool. `rewardtogo-v2` gave
# each slot one lineage, so slot 09's diverged Q head (`max_abs_q` 81.8) sat at
# 3,000 benchmark money for sixteen generations with nothing to fall back to.
# --------------------------------------------------------------------------


def test_the_round_robin_agrees_between_collection_and_the_update():
    """`collect` reads `round_index`, `update` reads `cutoff_batch_seq`, and the two
    have to name the same member or a member trains on a sibling's batches."""

    from agents.dqn_market_free_v3d.training.members import (
        member_for_cutoff,
        member_for_sequence,
    )

    members, batches = 5, 4
    # Contiguous, and it wraps at the generation boundary onto member 0 again.
    assert [member_for_sequence(r, batches, members) for r in range(24)] == (
        [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [0] * 4
    )
    # An update at cutoff C consumed sequences [C - batches, C), so it trains the
    # member that collected sequence C - 1.
    for cutoff in range(batches, 20 * batches, batches):
        assert member_for_cutoff(cutoff, batches, members) == member_for_sequence(
            cutoff - 1, batches, members
        )
    # One member is the old single-lineage slot, exactly.
    assert {member_for_sequence(r, 20, 1) for r in range(60)} == {0}


def test_the_prune_drops_a_diverged_lineage_and_forks_a_strong_one():
    """Pure top-N is the point: a lineage that loses both its checkpoints loses its
    position, and a strong lineage takes the vacancy as a *fork* -- new lineage id,
    and a batch floor at the current cutoff so it does not inherit its parent's
    replay history."""

    from agents.dqn_market_free_v3d.training.members import rank_and_prune

    candidates = [
        {"checkpoint_id": f"c{i}", "lineage_id": lin, "score": score, "batch_floor": 0}
        for i, (lin, score) in enumerate(
            [("A", 13000.0), ("A", 12000.0), ("B", 9000.0), ("B", 400.0),
             ("C", 8000.0), ("C", 7000.0), ("D", 300.0), ("D", 100.0),
             ("E", 50.0), ("E", None)]
        )
    ]
    kept = rank_and_prune(candidates, 5, model_id="m", generation=7, cutoff=200)

    assert [entry.position for entry in kept] == [0, 1, 2, 3, 4]
    assert [entry.score for entry in kept] == [13000.0, 12000.0, 9000.0, 8000.0, 7000.0]
    # D and E scored worst on both their checkpoints and are gone entirely.
    assert {entry.forked_from for entry in kept} == {None, "A", "C"}
    # The duplicates are forks, not clones: fresh identity, fresh floor.
    forks = [entry for entry in kept if entry.forked_from is not None]
    assert len(forks) == 2
    for fork in forks:
        assert fork.batch_floor == 200
        assert fork.lineage_id not in {"A", "B", "C", "D", "E"}
    # Survivors keep their own identity and their own history.
    survivors = [entry for entry in kept if entry.forked_from is None]
    assert {entry.lineage_id for entry in survivors} == {"A", "B", "C"}
    assert all(entry.batch_floor == 0 for entry in survivors)
    # Deterministic: ten concurrent hook calls must agree.
    assert [entry.lineage_id for entry in rank_and_prune(
        list(reversed(candidates)), 5, model_id="m", generation=7, cutoff=200
    )] == [entry.lineage_id for entry in kept]


def test_an_inert_member_ranks_below_every_trading_sibling():
    """Money alone selects *for* silence -- a member that never trades keeps its
    stake, so it outranks a sibling that traded and lost. That is what collapsed
    `market-free-members-v2`: the release gate refused these on its activity floor
    97 times in 105 while the prune kept promoting them into the roster."""

    from agents.dqn_market_free_v3d.training.members import rank_and_prune

    candidates = [
        {"checkpoint_id": "rich_idle", "lineage_id": "A", "score": 2806.0,
         "active": False, "batch_floor": 0},
        {"checkpoint_id": "poor_trader", "lineage_id": "B", "score": 850.0,
         "active": True, "batch_floor": 0},
        {"checkpoint_id": "poorer_trader", "lineage_id": "C", "score": 516.0,
         "active": True, "batch_floor": 0},
    ]
    kept = rank_and_prune(candidates, 2, model_id="m", generation=1, cutoff=10)

    # Both traders outrank the idle member despite banking a third of its money.
    assert [entry.lineage_id for entry in kept] == ["B", "C"]


def test_ranking_falls_back_to_money_when_every_member_is_inert():
    """Tiering rather than penalising is what makes this degrade gracefully. A cold
    slot starts with every member inert, and the ranking there must stay the old
    pure-money order rather than going arbitrary."""

    from agents.dqn_market_free_v3d.training.members import rank_and_prune

    candidates = [
        {"checkpoint_id": f"c{i}", "lineage_id": lin, "score": score,
         "active": False, "batch_floor": 0}
        for i, (lin, score) in enumerate([("A", 140.0), ("B", 2806.0), ("C", 516.0)])
    ]
    kept = rank_and_prune(candidates, 2, model_id="m", generation=1, cutoff=10)

    assert [entry.score for entry in kept] == [2806.0, 516.0]


def test_an_unmeasured_member_is_not_demoted_as_if_it_were_inert():
    """`active is None` means no rank arena ran. Absent evidence must not cost a
    member its position -- that is the same mistake as dropping a lineage over one
    failed arena, which `_rank_key`'s unscored tier already refuses to make."""

    from agents.dqn_market_free_v3d.training.members import rank_and_prune

    candidates = [
        {"checkpoint_id": "unmeasured", "lineage_id": "A", "score": 900.0,
         "active": None, "batch_floor": 0},
        {"checkpoint_id": "measured_inert", "lineage_id": "B", "score": 2806.0,
         "active": False, "batch_floor": 0},
    ]
    kept = rank_and_prune(candidates, 1, model_id="m", generation=1, cutoff=10)

    assert [entry.lineage_id for entry in kept] == ["A"]


def test_a_lineage_cap_trades_selection_pressure_for_distinct_lineages():
    """The cap is the escape hatch for a slot collapsing onto one lineage. At 1 it
    reduces the prune to "keep the better of parent and child" per lineage, which is
    why it is not the default -- the members then never compete."""

    from agents.dqn_market_free_v3d.training.members import rank_and_prune

    candidates = [
        {"checkpoint_id": f"c{i}", "lineage_id": lin, "score": score, "batch_floor": 0}
        for i, (lin, score) in enumerate(
            [("A", 13000.0), ("A", 12000.0), ("B", 9000.0), ("B", 400.0),
             ("C", 8000.0), ("C", 7000.0), ("D", 300.0), ("D", 100.0),
             ("E", 50.0), ("E", None)]
        )
    ]
    capped = rank_and_prune(
        candidates, 5, model_id="m", generation=7, cutoff=200, lineage_cap=1
    )
    assert {entry.lineage_id for entry in capped} == {"A", "B", "C", "D", "E"}
    assert all(entry.forked_from is None for entry in capped)
    # Even the diverged lineage survives a cap of 1. That is the inert version.
    assert 300.0 in {entry.score for entry in capped}


def test_a_member_index_round_trips_through_disk(tmp_path):
    from agents.dqn_market_free_v3d.training.members import (
        MemberEntry,
        find_member,
        read_index,
        write_index,
    )

    assert read_index(tmp_path) == []
    entries = [
        MemberEntry(position=1, lineage_id="b", score=2.0, batch_floor=40),
        MemberEntry(position=0, lineage_id="a", score=1.0),
    ]
    write_index(tmp_path, entries)
    loaded = read_index(tmp_path)
    assert [entry.position for entry in loaded] == [0, 1]
    assert find_member(loaded, 1).lineage_id == "b"
    assert find_member(loaded, 1).batch_floor == 40
    assert find_member(loaded, 9) is None
    # A corrupt sidecar reads as a cold start rather than taking a generation down.
    (tmp_path / "members" / "index.json").write_text("{not json")
    assert read_index(tmp_path) == []


def test_the_buffer_sees_only_this_members_own_batches():
    """Two filters, two jobs: the lineage keeps a member off its siblings' data so
    the prune ranks learning rather than luck, and the floor keeps a fork off its
    parent's data so it starts empty."""

    from agents.dqn_market_free_v3d.training.updater import _batch_lineage

    def batch(sequence, lineage):
        return SimpleNamespace(
            sequence=sequence,
            metadata=None if lineage is None else {"member_lineage_id": lineage},
        )

    all_batches = [
        batch(10, "mine"), batch(20, "sibling"), batch(30, "mine"),
        batch(40, None), batch(50, "mine"),
    ]
    assert _batch_lineage(all_batches[0]) == "mine"
    assert _batch_lineage(all_batches[3]) == ""

    floor = 25
    usable = [
        item
        for item in all_batches
        if item.sequence > floor and _batch_lineage(item) == "mine"
    ]
    assert [item.sequence for item in usable] == [30, 50]


def test_a_member_resumes_only_from_a_complete_checkpoint_pair(tmp_path):
    """Weights without the trainer state would silently restart Adam mid-lineage."""

    from agents.dqn_market_free_v3d.training.members import MemberEntry
    from agents.dqn_market_free_v3d.training.updater import _member_checkpoint

    weights, trainer = tmp_path / "m.npz", tmp_path / "m.pt"
    assert _member_checkpoint(None) is None
    entry = MemberEntry(position=0, lineage_id="a", checkpoint_id="c",
                        inference_path=str(weights), trainer_path=str(trainer))
    assert _member_checkpoint(entry) is None
    weights.write_bytes(b"w")
    assert _member_checkpoint(entry) is None, "weights alone must not resume"
    trainer.write_bytes(b"t")
    resumed = _member_checkpoint(entry)
    assert resumed is not None and resumed.checkpoint_id == "c"
    assert resumed.released is False


# --------------------------------------------------------------------------
# Cross-slot rescue: a slot that is going nowhere may import a sibling's
# checkpoint. Both gates exist to stop the roster collapsing onto one policy,
# and the fork is what stops an imported lineage reading another slot's batches.
# --------------------------------------------------------------------------


def _foreign(score, checkpoint_id="f0", lineage_id="far", source="decl@other"):
    return {
        "checkpoint_id": checkpoint_id,
        "inference_path": "/x.npz",
        "trainer_path": "/x.pt",
        "cutoff_batch_seq": 40,
        "lineage_id": lineage_id,
        "batch_floor": 999,
        "score": score,
        "active": True,
        "foreign": True,
        "source_model_id": source,
    }


def _own(score, checkpoint_id="o0", lineage_id="home", floor=0):
    return {
        "checkpoint_id": checkpoint_id,
        "inference_path": "/y.npz",
        "trainer_path": "/y.pt",
        "cutoff_batch_seq": 40,
        "lineage_id": lineage_id,
        "batch_floor": floor,
        "score": score,
        "active": True,
    }


def test_a_healthy_slot_imports_nothing():
    """The margin is what keeps a slot that is doing fine from churning."""

    from agents.dqn_market_free_v3d.training.members import admit_foreign

    own = [_own(10_000.0), _own(9_000.0, "o1", "home2")]
    # 11,000 is a lead, but not the 25% one the margin asks for.
    assert admit_foreign(
        own, [_foreign(11_000.0)], min_score=3_000.0, margin=0.25, max_adopt=1
    ) == []
    admitted = admit_foreign(
        own, [_foreign(13_000.0)], min_score=3_000.0, margin=0.25, max_adopt=1
    )
    assert [item["score"] for item in admitted] == [13_000.0]


def test_two_dead_slots_cannot_swap_corpses():
    """Without the absolute floor every score near zero beats some other score near
    zero, and the prune churns lineages for no gain."""

    from agents.dqn_market_free_v3d.training.members import admit_foreign

    own = [_own(0.0), _own(12.0, "o1", "home2")]
    assert admit_foreign(
        own, [_foreign(38.0)], min_score=3_000.0, margin=0.25, max_adopt=1
    ) == []
    # A genuinely solvent sibling clears it.
    admitted = admit_foreign(
        own, [_foreign(18_298.0)], min_score=3_000.0, margin=0.25, max_adopt=1
    )
    assert [item["score"] for item in admitted] == [18_298.0]


def test_a_slot_with_nothing_of_its_own_still_reaches_the_floor():
    from agents.dqn_market_free_v3d.training.members import admit_foreign

    assert admit_foreign(
        [], [_foreign(2_000.0)], min_score=3_000.0, margin=0.25, max_adopt=1
    ) == []
    assert len(
        admit_foreign([], [_foreign(5_000.0)], min_score=3_000.0, margin=0.25, max_adopt=1)
    ) == 1


def test_max_adopt_leaves_the_slot_a_lineage_of_its_own():
    """The roster's value is that its slots are independent replicates; a prune that
    could fill every position from one leader spends that hedge in one step."""

    from agents.dqn_market_free_v3d.training.members import admit_foreign

    foreign = [
        _foreign(18_000.0, "f0", "far0"),
        _foreign(17_000.0, "f1", "far1"),
        _foreign(16_000.0, "f2", "far2"),
    ]
    admitted = admit_foreign(
        [_own(100.0)], foreign, min_score=3_000.0, margin=0.25, max_adopt=1
    )
    assert [item["checkpoint_id"] for item in admitted] == ["f0"]
    assert admit_foreign(
        [_own(100.0)], foreign, min_score=3_000.0, margin=0.25, max_adopt=0
    ) == []


def test_an_adopted_checkpoint_forks_instead_of_keeping_its_lineage():
    """Its lineage id names batches under another slot's `model_id`. Keeping it would
    leave the replay filter matching nothing and the update would train on
    demonstrations alone."""

    from agents.dqn_market_free_v3d.training.members import rank_and_prune

    survivors = rank_and_prune(
        [_own(100.0, "o0", "home"), _foreign(18_000.0)],
        2,
        model_id="decl@slot_00",
        generation=3,
        cutoff=80,
    )
    adopted = [entry for entry in survivors if entry.forked_from == "far"]
    assert len(adopted) == 1
    assert adopted[0].lineage_id != "far", "an imported lineage must be renamed"
    assert adopted[0].batch_floor == 80, "a fork starts at the current cutoff"
    assert adopted[0].checkpoint_id == "f0"
    # The slot keeps its own lineage alongside it, history intact.
    home = [entry for entry in survivors if entry.lineage_id == "home"]
    assert len(home) == 1 and home[0].batch_floor == 0


def test_the_manifests_empty_max_adopt_resolves_to_the_default():
    """`""` is the manifest's unset value, as with `member_lineage_cap`. `int("")`
    raises, and it would do so inside the generation hook."""

    import tomllib
    from pathlib import Path

    manifest = tomllib.loads(
        Path("agents/dqn_market_free_v3d/agent.toml").read_text()
    )["models"][0]["config"]
    configured = manifest.get("cross_slot_reseed_max_adopt", "")
    members = int(manifest.get("members_per_slot", 1))
    resolved = max(1, members - 1) if configured in (None, "") else int(configured)
    assert resolved == max(1, members - 1)
    # And the manifest stays the control arm.
    assert manifest["cross_slot_reseed"] is False
    assert manifest["cross_slot_reseed_min_score"] >= 3_000.0


# --------------------------------------------------------------------------
# Demonstration selection: `limit` seats spread over competitors rather than
# taken off the head of the ladder order.
# --------------------------------------------------------------------------


def _prov(name, money, index):
    return {"name": name, "final_money": float(money), "episode_id": f"e{index}"}


def test_stratified_selection_spreads_over_competitors():
    """`ladder-001`'s head was half two players -- 34 of the first 128 seats one
    competitor and 30 another -- so the clone fitted a blend weighted by who ranked
    highest rather than by who was distinct."""

    from agents.dqn_market_free_v3d.training.ladder import stratified_indices

    provenance = (
        [_prov("alice", 150_000 - i, i) for i in range(40)]
        + [_prov("bob", 140_000 - i, 40 + i) for i in range(30)]
        + [_prov("carol", 130_000 - i, 70 + i) for i in range(5)]
        + [_prov("dave", 120_000, 75)]
    )
    picked = stratified_indices(provenance, 8)

    assert len(picked) == 8
    names = [provenance[i]["name"] for i in picked]
    # Every distinct player is reached before anyone contributes a second seat.
    assert set(names[:4]) == {"alice", "bob", "carol", "dave"}
    # Depth 1 is a, b, c -- dave has only the one seat and drops out -- then depth 2
    # takes alice again, so a player with more seats does end up with more of them
    # once the short ones are exhausted. Head-of-order selection returns eight alices.
    assert [names.count(n) for n in ("alice", "bob", "carol", "dave")] == [3, 2, 2, 1]


def test_stratified_selection_takes_each_players_best_season_first():
    from agents.dqn_market_free_v3d.training.ladder import stratified_indices

    provenance = [
        _prov("alice", 100_000, 0),
        _prov("alice", 150_000, 1),
        _prov("bob", 120_000, 2),
        _prov("bob", 90_000, 3),
    ]
    picked = stratified_indices(provenance, 2)
    assert [provenance[i]["final_money"] for i in picked] == [150_000.0, 120_000.0]


def test_stratified_selection_falls_back_to_depth_when_players_run_out():
    """A `limit` above the player count has to keep going, not stop short."""

    from agents.dqn_market_free_v3d.training.ladder import stratified_indices

    provenance = [_prov("alice", 100, 0), _prov("alice", 90, 1), _prov("bob", 80, 2)]
    assert len(stratified_indices(provenance, 3)) == 3
    # And a limit past the pack returns everything exactly once.
    picked = stratified_indices(provenance, 99)
    assert sorted(picked) == [0, 1, 2]


def test_pack_stats_accepts_an_index_list():
    """Stratified selection pins non-contiguous records, so the stats the update
    reports have to be summed over the records actually held."""

    import json
    from pathlib import Path

    from agents.dqn_market_free_v3d.training.ladder import pack_stats

    directory = Path(
        "artifacts/dqn_market_free_v3d/demonstrations/ladder-001"
    )
    if not (directory / "pack.json").is_file():
        pytest.skip("ladder-001 pack not present")
    payload = json.loads((directory / "pack.json").read_text())
    total = len(payload["provenance"])
    head = pack_stats(directory, 2)
    picked = pack_stats(directory, [0, 1])
    assert head.decisions == picked.decisions
    # A different pair sums to something else, so the argument is really read.
    other = pack_stats(directory, [0, total - 1])
    assert other.decisions != head.decisions or total <= 2


def test_an_idle_checkpoint_is_never_imported():
    """`market-free-top25-v1` regression. A policy that never trades keeps its whole
    stake, so it scores the stake *exactly* -- 3,000 -- and outranks every sibling that
    traded and lost. The floor was set to 3,000 and compared with `>=`, so the one
    candidate the floor existed to refuse was the one it admitted, and the fork landed
    in position 0 of all five slots."""

    from agents.dqn_market_free_v3d.training.members import admit_foreign

    idle = _foreign(3_000.0)
    idle["active"] = False
    own = [_own(130.25)]

    # Measured-inert is refused however high it scores: money alone selects for silence.
    assert admit_foreign(own, [idle], min_score=3_000.0, margin=0.25, max_adopt=1) == []
    rich_but_idle = _foreign(50_000.0)
    rich_but_idle["active"] = False
    assert admit_foreign(
        own, [rich_but_idle], min_score=6_000.0, margin=0.25, max_adopt=1
    ) == []


def test_the_floor_is_strict_so_the_stake_itself_is_refused():
    from agents.dqn_market_free_v3d.training.members import admit_foreign

    own = [_own(130.25)]
    at_floor = _foreign(6_000.0)
    assert admit_foreign(own, [at_floor], min_score=6_000.0, margin=0.25, max_adopt=1) == []
    above = _foreign(6_000.01)
    assert len(
        admit_foreign(own, [above], min_score=6_000.0, margin=0.25, max_adopt=1)
    ) == 1


def test_an_unmeasured_checkpoint_is_not_imported_on_absent_evidence():
    """`_rank_key` lets `active is None` ride with the scored tier rather than demoting
    a member over one failed arena. Importing across slots is the bigger step and takes
    the opposite default."""

    from agents.dqn_market_free_v3d.training.members import admit_foreign

    unmeasured = _foreign(50_000.0)
    unmeasured["active"] = None
    assert admit_foreign(
        [_own(130.25)], [unmeasured], min_score=6_000.0, margin=0.25, max_adopt=1
    ) == []


def test_the_driver_floor_clears_the_stake():
    """3,000 is the stake and therefore the idle score; no floor may sit on it.

    Cross-slot reseed is off at `LEARNERS = 1` -- there are no siblings to reach
    across -- so the floor that matters is the one on the mirror pool. Both are
    checked so that re-enabling either cannot quietly reintroduce the stake.
    """

    import train_market_free_v3d as driver

    config = driver.FREE_CONFIG
    if config.get("cross_slot_reseed"):
        assert config["cross_slot_reseed_min_score"] > 3_000.0
    else:
        assert "cross_slot_reseed_min_score" not in config
    # A checkpoint that cannot beat holding its cash must never become a teacher.
    assert config["self_play_min_score"] >= 3_000.0


# --------------------------------------------------------------------------
# Block G: own standing crops and their maturity. The fork exists for this.
#
# v3c could not tell "I invested and have a harvest coming" from "I am broke
# with nothing": for the ~250 turns the Kaggle ladder's median seat spends at
# 20-50 coins, both read money~0 with a crop-agnostic `planted / 100.0` and a
# supply projection that switches on 1-3 days before ripeness.
# --------------------------------------------------------------------------


def _crop_state(standing_crops=(), day=4, money=3000.0):
    """A built vector for a fixed board, with `standing_crops` planted on day 0.

    `standing_crops` is a sequence of `(crop, planted_day)`.
    """

    from agents.dqn_market_free_v3d.market_state import market_state_features

    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    for index, (crop, planted_day) in enumerate(standing_crops):
        tiles[index // 5][index % 5] = {
            "kind": "PLANT",
            "crop": crop,
            "planted_day": planted_day,
            "yield_units": 0,
        }
    farm = {
        "money": money,
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
        "tiles": tiles,
        "hands": [],
        "farmer": [4, 4],
    }
    private = {"shed": {}, "seeds": {}, "inventories": [{}]}
    obs = {
        "player": 0,
        "step": day * 24,
        "day": day,
        "hour": 6,
        "farms": [farm, {**farm, "tiles": [[None] * 10 for _ in range(10)]}],
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
        "private": private,
    }
    field = {"farmer": ["PASS"], "hands": []}
    roles = herd.field_roles(obs, farm, private)
    return market_state_features(obs, farm, private, roles, field)


def _crop_block(state):
    start = INTRA_TURN_SLICE.start - CROP_MATURITY_DIM
    return state[start : INTRA_TURN_SLICE.start]


def test_standing_crops_are_distinguishable_from_an_empty_farm_at_equal_money():
    """The aliasing this fork removes. Both farms hold the same cash; one has twenty
    melons in the ground and the other has nothing, and under v3c the only column that
    separated them was a crop-agnostic tile count."""

    invested = _crop_state([("MELON", 0)] * 20, day=4, money=20.0)
    broke = _crop_state([], day=4, money=20.0)

    assert invested.shape == (MARKET_STATE_DIM,) == (296,)
    assert not np.array_equal(_crop_block(invested), _crop_block(broke))
    # An empty farm reads no crop, no maturity, and a full season of waiting.
    empty = _crop_block(broke)
    assert_array_equal(empty[: len(donor.CROPS)], np.zeros(len(donor.CROPS), dtype=np.float32))
    assert np.all(empty[2 * len(donor.CROPS) : 3 * len(donor.CROPS)] == 1.0), (
        "nothing planted must read a full season away, never 'harvest imminent'"
    )
    assert empty[-1] == 0.0


def test_maturity_climbs_and_the_wait_shrinks_as_a_crop_ages():
    """The column that carries the 250-turn investment window."""

    crops = list(donor.CROPS)
    melon = crops.index("MELON")
    rule = donor.CROPS["MELON"]

    seen = []
    for day in (0, 3, 6, 9, 12):
        block = _crop_block(_crop_state([("MELON", 0)] * 5, day=day))
        maturity = block[len(crops) + melon]
        wait = block[2 * len(crops) + melon]
        seen.append((maturity, wait))

    maturities = [item[0] for item in seen]
    waits = [item[1] for item in seen]
    assert maturities == sorted(maturities), "maturity must not go backwards"
    assert maturities[0] < maturities[-1]
    assert waits == sorted(waits, reverse=True), "the wait must shrink as the crop ages"
    # Capped at ripe, and the wait bottoms out at zero rather than going negative.
    assert maturities[-1] == pytest.approx(1.0)
    assert waits[-1] == 0.0
    assert rule["ripe"] == 10 and rule["first"] == 10


def test_the_crop_block_separates_crops_and_prices_what_is_standing():
    crops = list(donor.CROPS)
    wheat, melon = crops.index("WHEAT"), crops.index("MELON")

    block = _crop_block(_crop_state([("WHEAT", 0)] * 5 + [("MELON", 0)] * 10, day=1))
    # Counts land on the right crop, normalised the way the opponent block is.
    assert block[wheat] == pytest.approx(5 / 25.0)
    assert block[melon] == pytest.approx(10 / 25.0)
    assert block[crops.index("CARROT")] == 0.0
    # Wheat yields on day 2, melon on day 10, so at day 1 the waits differ.
    assert block[2 * len(crops) + wheat] < block[2 * len(crops) + melon]
    # The value scalar prices the standing crop and is comparable to the money column.
    assert block[-1] > 0.0
    assert _crop_block(_crop_state([], day=1))[-1] == 0.0


def test_the_crop_block_is_constant_across_a_turns_sub_decisions():
    """The precondition for sitting ahead of block E.

    `main.py` reuses the turn-opening encode and overwrites only `INTRA_TURN_SLICE`,
    so every column outside that span must be invariant under a queued order. Asserted
    on the DROP case, which is the one `_encode_case` shows *does* move columns
    elsewhere in the vector -- so this is a real discrimination and not a vacuous pass.
    `PendingTurn.view` moves money, shed and market inventory; it never touches
    `farm["tiles"]`, which is all block G reads.
    """

    from agents.dqn_market_free_v3d.market_state import market_state_features
    from agents.dqn_market_free_v3d.order_space import PendingTurn

    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    for index in range(5):
        tiles[0][index] = {
            "kind": "PLANT", "crop": "MELON", "planted_day": 0, "yield_units": 0,
        }
    farm = {
        "money": 3000.0, "unlocked_quadrants": ["NW"], "hires_today": 0,
        "tiles": tiles, "hands": [], "farmer": [4, 4],
    }
    private = {"shed": {"WHEAT": 3}, "seeds": {}, "inventories": [{"WHEAT": 12}]}
    obs = {
        "player": 0, "step": 96, "day": 4, "hour": 6,
        "farms": [farm, farm], "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []}, "private": private,
    }
    field = {"farmer": ["DROP"], "hands": []}
    roles = herd.field_roles(obs, farm, private)
    pending = PendingTurn(obs, farm, private, field, None, plantable={}, roles=roles)

    opening = market_state_features(obs, farm, private, roles, field, turn=None)
    obs_view, farm_view, private_view = pending.view(obs, farm, private)
    fresh = market_state_features(
        obs_view, farm_view, private_view, roles, field, turn=pending
    )

    moved = np.flatnonzero(opening != fresh)
    assert moved.size, "the DROP case must actually move columns somewhere"
    block = range(INTRA_TURN_SLICE.start - CROP_MATURITY_DIM, INTRA_TURN_SLICE.start)
    assert not [index for index in moved if index in block], (
        "block G must not move when a turn queues work"
    )
    assert_array_equal(_crop_block(opening), _crop_block(fresh))
    # And it is non-trivial: the melons are actually in the vector.
    assert _crop_block(opening)[list(donor.CROPS).index("MELON")] > 0.0


def test_the_pack_builder_can_hold_one_player():
    """`build_demo_pack(players=...)` is the control for the clone's blend problem.

    A pack spanning many competitors fits an *average* of them, and where two take
    different actions in similar states no single action clears the large-margin hinge --
    which is why `mean_behavior_clone_loss` floors at the margin however evenly the seats
    are picked. Pinned on the built pack rather than on the filter in isolation, because
    the thing that matters is that the sidecar really does hold one name.
    """

    import json
    from pathlib import Path

    directory = Path("artifacts/dqn_market_free_v3d/demonstrations/kawashigi-296")
    if not (directory / "pack.json").is_file():
        pytest.skip("single-player pack not built")
    payload = json.loads((directory / "pack.json").read_text())
    names = {item["name"] for item in payload["provenance"]}
    assert len(names) == 1, f"a single-player pack holds one name, got {names}"
    assert payload["players"] == list(names)
    # Above that player's own median, so the clone sees their stronger half.
    assert min(item["final_money"] for item in payload["provenance"]) >= 89_000
    # And it holds more seats than get pinned, so the whole pinned set is one policy.
    assert payload["records"] >= 128


def test_the_two_arms_differ_only_in_the_pack():
    """`market-free-solo-v1` is an A/B against `market-free-crop-v1`. If anything else
    drifts apart the comparison stops meaning anything."""

    import train_market_free_v3d as blended
    import train_market_free_v3d_solo as solo

    assert blended.SESSION_ID != solo.SESSION_ID
    assert blended.SLOT_PREFIX != solo.SLOT_PREFIX
    differing = {
        key
        for key in set(blended.FREE_CONFIG) | set(solo.FREE_CONFIG)
        if blended.FREE_CONFIG.get(key) != solo.FREE_CONFIG.get(key)
    }
    assert differing == {"demonstration_pack"}, differing
    assert blended.LEARNERS == solo.LEARNERS
    assert blended.MANIFEST == solo.MANIFEST


# --------------------------------------------------------------------------
# Hire cap, animal potential, prune hysteresis
# --------------------------------------------------------------------------


def test_hire_rungs_above_the_fields_demand_are_shut():
    """`labor.hire_cap` bounds the HIRE ladder on the v2 path; v1's is untouched.

    An empty board with no seed justifies `MIN_HANDS + HIRE_SLACK` hands, not the
    donor's floor of 10 -- that floor is the 18,300-coin wage bill `labor.py`
    measures. `hires_today` already counts against it.
    """

    from agents.dqn_market_free_v3d import labor

    pending, (obs, farm, private, field) = _pending(money=40_000)
    cap = labor.hire_cap(obs, farm, private, herd.field_roles(obs, farm, private))
    assert labor.MIN_HANDS <= cap < 10
    assert pending.hire_cap == cap
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    open_hands = [
        ACTIONS[index].quantity for index in np.flatnonzero(mask) if ACTIONS[index].op == "HIRE"
    ]
    assert open_hands and max(open_hands) == cap

    pending.hires_today = cap - 1
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    open_hands = [
        ACTIONS[index].quantity for index in np.flatnonzero(mask) if ACTIONS[index].op == "HIRE"
    ]
    assert open_hands == [1]

    # The v1 branch keeps affordability alone.
    mask_v1 = feasible_actions(obs, farm, private, field, None)
    assert max(
        ACTIONS[index].quantity for index in np.flatnonzero(mask_v1) if ACTIONS[index].op == "HIRE"
    ) == int(donor.MAX_HANDS)


def test_the_animal_columns_are_where_the_potential_reads_them():
    """`OWNED_ANIMAL_SLICE` / `HOMELESS_ANIMAL_SLICE` pinned against the builder.

    Three of a species in the shed with nothing to stand on raise the owned column
    and the homeless column by the same 0.3 and nothing else the potential reads,
    so the placed basis is zero; one on a structure raises owned alone.
    """

    from agents.dqn_market_free_v3d import market_state
    from agents.dqn_market_free_v3d.market_state import (
        HOMELESS_ANIMAL_SLICE,
        OWNED_ANIMAL_SLICE,
        PLACED_ANIMAL_BASIS,
    )

    obs, farm, private, field = _mask_state(shed={})
    base = market_state.market_state_features(obs, farm, private, herd.field_roles(obs, farm, private), field)
    for index, animal in enumerate(market_state.ANIMALS):
        obs2, farm2, private2, field2 = _mask_state(shed={animal: 3})
        vec = market_state.market_state_features(obs2, farm2, private2, herd.field_roles(obs2, farm2, private2), field2)
        delta = vec - base
        assert delta[OWNED_ANIMAL_SLICE][index] == pytest.approx(0.3)
        assert delta[HOMELESS_ANIMAL_SLICE][index] == pytest.approx(0.3)
        assert float(delta @ PLACED_ANIMAL_BASIS) == pytest.approx(0.0, abs=1e-3)

    structure = str(donor.ANIMALS["COW"]["structure"])
    obs3, farm3, private3 = _herd_state(
        tiles={(1, 1): {"kind": structure, "animal": "COW", "yield_units": 0}}
    )
    vec = market_state.market_state_features(obs3, farm3, private3, herd.field_roles(obs3, farm3, private3), field)
    assert float((vec - base) @ PLACED_ANIMAL_BASIS) == pytest.approx(
        float(donor.ANIMALS["COW"]["cost"]), rel=1e-3
    )


def test_the_animal_potential_moves_the_charge_and_writes_off_at_the_end():
    from agents.dqn_market_free_v3d.market_state import MARKET_STATE_DIM, PLACED_ANIMAL_BASIS
    from agents.dqn_market_free_v3d.training.reward import MONEY_SCALE, episode_rewards

    cost = float(donor.ANIMALS["COW"]["cost"])
    unit = cost / MONEY_SCALE
    # `money[t]` is cash as turn t opens: bought on turn 0, on the board from turn 1.
    money = [3000.0, 3000.0 - cost, 3000.0 - cost, 3000.0 - cost]
    final = 3000.0 - cost
    states = np.zeros((4, MARKET_STATE_DIM), dtype=np.float32)
    cow = np.flatnonzero(PLACED_ANIMAL_BASIS > 0)[list(donor.ANIMALS).index("COW")]
    states[1:, cow] = 0.1
    gamma = 0.999
    target, raw, _ = episode_rewards(money, final, gamma, turn_state=states)
    assert raw.tolist() == pytest.approx([-unit, 0.0, 0.0, 0.0])
    # The purchase turn is answered by the placement credit a turn on ...
    assert target[0] == pytest.approx(-unit + gamma * unit, abs=1e-6)
    # ... holding costs only the discount drag ...
    assert target[1] == pytest.approx((gamma - 1.0) * unit, abs=1e-6)
    assert target[2] == pytest.approx((gamma - 1.0) * unit, abs=1e-6)
    # ... and the write-off lands on the last turn, where the season ends -- next to
    # the ruin charge, because buying a cow from the 3,000 stake ends the season under
    # it. The two are independent terms on the same step.
    from agents.dqn_market_free_v3d.training.reward import ruin_penalty

    charge = ruin_penalty(final)
    assert charge < 0.0
    assert target[3] == pytest.approx(-unit + charge, abs=1e-6)
    # Without states the label is the bare cash delta plus that same charge.
    bare, _, _ = episode_rewards(money, final, gamma)
    assert bare[:-1].tolist() == pytest.approx(raw[:-1].tolist())
    assert float(bare[-1]) == pytest.approx(float(raw[-1]) + charge, abs=1e-6)


def test_an_incumbent_lineage_keeps_its_seat_inside_the_margin():
    from agents.dqn_market_free_v3d.training.members import rank_and_prune

    def cand(cid, lineage, score, active=True, floor=0):
        return {
            "checkpoint_id": cid,
            "lineage_id": lineage,
            "score": score,
            "active": active,
            "batch_floor": floor,
            "inference_path": None,
            "trainer_path": None,
        }

    # Rival leads by 10%: inside a 25% margin, both incumbents stay, no fork.
    candidates = [cand("a1", "A", 1000.0), cand("b1", "B", 900.0), cand("a2", "A", 1100.0), cand("b2", "B", 850.0)]
    kept = rank_and_prune(
        candidates, 2, model_id="m", generation=3, cutoff=60,
        incumbents={0: "B", 1: "A"}, margin=0.25,
    )
    assert [entry.lineage_id for entry in kept] == ["B", "A"]
    assert all(entry.forked_from is None for entry in kept)
    assert kept[0].checkpoint_id == "b1" and kept[1].checkpoint_id == "a2"

    # Rival leads by 40%: B is cut, A's second copy forks into position 0.
    candidates = [cand("a1", "A", 1000.0), cand("b1", "B", 700.0), cand("a2", "A", 1100.0), cand("b2", "B", 650.0)]
    kept = rank_and_prune(
        candidates, 2, model_id="m", generation=3, cutoff=60,
        incumbents={0: "B", 1: "A"}, margin=0.25,
    )
    assert kept[1].lineage_id == "A" and kept[1].forked_from is None
    assert kept[0].forked_from == "A" and kept[0].batch_floor == 60

    # An inert incumbent loses to an active rival regardless of money.
    candidates = [cand("a1", "A", 1000.0), cand("b1", "B", 2000.0, active=False)]
    kept = rank_and_prune(
        candidates, 1, model_id="m", generation=3, cutoff=60, incumbents={0: "B"}, margin=0.25,
    )
    assert kept[0].lineage_id == "A"

    # No margin is the old pure top-N.
    candidates = [cand("a1", "A", 1000.0), cand("b1", "B", 900.0)]
    kept = rank_and_prune(candidates, 1, model_id="m", generation=3, cutoff=60, incumbents={0: "B"})
    assert kept[0].lineage_id == "A"
