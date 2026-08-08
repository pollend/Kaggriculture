"""Correctness guards for the DQN job-scoring agent.

These pin the invariants that would otherwise fail silently: train/inference
feature skew, a shaping term that is not actually potential-based, a replay
sampler that reconstructs something other than what the agent saw, and a policy
that stops being deterministic at epsilon=0.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.dqn_job_scorer import features as F
from agents.dqn_job_scorer.jobs import MAX_JOBS, enumerate_jobs
from agents.dqn_job_scorer.main import JobScorerAgent
from agents.dqn_job_scorer.policy import QNetwork, select
from agents.dqn_job_scorer.training import network as net_module
from agents.dqn_job_scorer.training.collect import (
    EPISODE_ARRAY_FIELDS,
    MONEY_SCALE,
    WIN_BONUS,
    Recorder,
    collect_episode,
    load_episode_batch,
    potential,
    save_episode_batch,
)
from agents.dqn_job_scorer.training.replay import ReplayBuffer
from kaggriculture_lab.identity import AgentManifest
from kaggriculture_lab.model_updates import record_training_batch, update_model_in_store
from kaggriculture_lab.storage import ExperimentStore
from sim.runner import run_match

GAMMA = 0.997


@pytest.fixture(scope="module")
def episode():
    record = collect_episode(
        QNetwork.random(seed=1).layers,
        opponent_spec="starter",
        seed=3,
        epsilon=0.3,
        gamma=GAMMA,
    )
    assert record is not None, "collection returned nothing -- the learner errored"
    return record


# --- train / inference parity ------------------------------------------------


def test_torch_and_numpy_networks_agree():
    """The single most important guard: skew here silently ruins every run."""
    numpy_net = QNetwork.random(seed=7)
    torch_module = net_module.from_numpy(numpy_net)

    rng = np.random.default_rng(0)
    features = rng.standard_normal((37, F.FEATURE_DIM)).astype(np.float32)

    with torch.no_grad():
        torch_q = torch_module(torch.from_numpy(features)).squeeze(1).numpy()

    np.testing.assert_allclose(numpy_net(features), torch_q, rtol=0, atol=1e-5)


def test_export_round_trips():
    original = QNetwork.random(seed=11)
    restored = net_module.to_numpy(net_module.from_numpy(original))
    for (w1, b1), (w2, b2) in zip(original.layers, restored.layers):
        np.testing.assert_array_equal(w1, w2)
        np.testing.assert_array_equal(b1, b2)


def test_episode_batch_round_trips_without_pickle(episode, tmp_path):
    path = save_episode_batch([episode], tmp_path / "batch.npz")
    restored = load_episode_batch(path)
    assert len(restored) == 1
    for field_name in EPISODE_ARRAY_FIELDS:
        np.testing.assert_array_equal(
            getattr(restored[0], field_name), getattr(episode, field_name)
        )


def test_persistent_dqn_updater_resumes_from_eight_batches(
    episode, tmp_path, monkeypatch
):
    from agents.dqn_job_scorer.training import updater as updater_module

    manifest_path = Path("agents/dqn_job_scorer/agent.toml")
    manifest = AgentManifest.load(manifest_path)
    artifact = save_episode_batch([episode], tmp_path / "rollout.npz")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    config = {
        **(manifest.models[0].config or {}),
        "gradient_steps_per_batch": 1,
        "batch_size": 8,
        "buffer_episodes": 16,
        "warmup_episodes": 1,
        "jobs": 1,
        "eval_seeds": 1,
    }
    unresolved_evaluation = {
        "opponent": "test",
        "pairs": 1,
        "win_rate": 0.5,
        "margin": 0.0,
        "margin_ci": [-1.0, 1.0],
        "significant": False,
        "errors": {},
    }
    monkeypatch.setattr(
        updater_module,
        "_arena_evaluate",
        lambda _weights, opponent, *_args, **_kwargs: {
            **unresolved_evaluation,
            "margin": 10.0 if opponent.startswith(updater_module.SELF_SPEC) else 0.0,
            "margin_ci": [1.0, 20.0]
            if opponent.startswith(updater_module.SELF_SPEC)
            else [-1.0, 1.0],
        },
    )

    with ExperimentStore(tmp_path / "dqn.duckdb") as store:
        store.register_family(manifest, manifest_path)
        store.connection.execute(
            "UPDATE training_models SET config_json = ? WHERE model_id = ?",
            [json.dumps(config), "dqn_job_scorer.q_network"],
        )
        for index in range(8):
            record_training_batch(
                store,
                "dqn_job_scorer.q_network",
                source_kind="dqn_rollout",
                source_id=f"rollout-{index}",
                sample_count=1,
                artifact_path=artifact,
                artifact_sha256=digest,
                metadata={"transitions": int(episode.unit_mask.sum())},
            )
        report = update_model_in_store(
            store, "dqn_job_scorer.q_network", artifacts=tmp_path
        )
        assert report.status == "promoted"
        checkpoint = store.connection.execute(
            """
            SELECT trainer_artifact_path, inference_artifact_path
            FROM model_training_checkpoints WHERE checkpoint_id = ?
            """,
            [report.checkpoint_id],
        ).fetchone()
        assert Path(checkpoint[0]).exists()
        assert Path(checkpoint[1]).exists()


def test_feature_dimensions_are_pinned():
    assert F.FEATURE_DIM == F.GLOBAL_DIM + F.JOB_STATIC_DIM + F.JOB_DISTANCE_DIM
    numpy_net = QNetwork.zeros()
    assert numpy_net.layers[0][0].shape[0] == F.FEATURE_DIM


# --- reward shaping ----------------------------------------------------------


def test_shaping_is_potential_based(episode):
    """Sum of shaped rewards minus raw must telescope to gamma*Phi(s_T) - Phi(s_0).

    If it does not, the shaping is changing the optimal policy rather than just
    the learning speed, and any result would be about the wrong objective.
    """
    phi = episode.potential
    shaping = GAMMA * phi[1:] - phi[:-1]
    terminal_bonus = float(episode.reward[-1] - episode.raw_reward[-1] - shaping[-1])

    assert terminal_bonus == pytest.approx(
        min((-WIN_BONUS, 0.0, WIN_BONUS), key=lambda value: abs(value - terminal_bonus)),
        abs=1e-5,
    )
    np.testing.assert_allclose(
        episode.reward[:-1], episode.raw_reward[:-1] + shaping[:-1], atol=1e-5
    )
    assert float(episode.reward[-1]) == pytest.approx(
        float(episode.raw_reward[-1] + shaping[-1] + terminal_bonus), abs=1e-5
    )


def test_raw_reward_matches_money_trajectory(episode):
    expected = (episode.money[1:] - episode.money[:-1]) / MONEY_SCALE
    np.testing.assert_allclose(episode.raw_reward, expected, atol=1e-5)


def test_potential_counts_unsold_value_but_not_cash():
    """Cash belongs to the base reward; Phi contains only liquidatable crops."""
    farm = {"money": 1000.0, "tiles": [[{"kind": "PLANT", "crop": "WHEAT", "yield_units": 4}]]}
    private = {"shed": {"CARROT": 3}, "inventories": [{"WHEAT": 2}]}
    obs = {"market": {"prices": {"WHEAT": 25, "CARROT": 35}}}
    assert potential(farm, private, obs) == pytest.approx((3 * 35 + 2 * 25 + 4 * 25) / MONEY_SCALE)


def test_terminal_potential_is_zero_and_draw_bonus_is_neutral():
    recorder = Recorder()
    recorder.record(
        global_feats=np.zeros(F.GLOBAL_DIM, dtype=np.float32),
        job_static=np.zeros((1, F.JOB_STATIC_DIM), dtype=np.float32),
        job_targets=np.zeros((1, 2), dtype=np.int8),
        positions=[(0, 0)],
        chosen=[0],
        committed=[-1],
        available=[np.array([True])],
        money=1000.0,
        obs={"market": {"prices": {"WHEAT": 25}}},
        farm={
            "money": 1000.0,
            "tiles": [[{"kind": "PLANT", "crop": "WHEAT", "yield_units": 4}]],
        },
        private={"shed": {}, "inventories": [{}]},
        truncated=0,
    )
    record = recorder.finish(final_money=1000.0, outcome=0.5, gamma=GAMMA)

    assert record.potential[-1] == 0.0
    assert record.raw_reward[-1] == 0.0
    assert record.reward[-1] == pytest.approx(-record.potential[0])


# --- replay fidelity ---------------------------------------------------------


def test_replay_reconstructs_the_features_the_agent_saw(episode):
    """The memory saving rests on this equivalence, so it is asserted directly."""
    buffer = ReplayBuffer(capacity=4, gamma=GAMMA, n_step=3, seed=0)
    buffer.add([episode])

    for t in (0, 5, episode.steps // 2, episode.steps - 1):
        for unit in np.flatnonzero(episode.unit_mask[t]):
            rebuilt, available = buffer._features_at(episode, t, int(unit))

            count = int(episode.job_mask[t].sum())
            position = (int(episode.unit_pos[t, unit, 0]), int(episode.unit_pos[t, unit, 1]))
            expected = F.assemble(
                episode.global_feats[t],
                episode.job_static[t, :count].astype(np.float32),
                F.distance_features(
                    episode.job_targets[t, :count], position, int(episode.committed[t, unit])
                ),
            )
            np.testing.assert_array_equal(rebuilt, expected)
            np.testing.assert_array_equal(
                available, episode.available[t, unit, :count]
            )
            assert available[int(episode.chosen[t, unit])]


def test_sampled_batch_shapes_and_terminal_discount(episode):
    buffer = ReplayBuffer(capacity=4, gamma=GAMMA, n_step=3, seed=1)
    buffer.add([episode])
    batch = buffer.sample(64)

    assert batch.features.shape == (64, F.FEATURE_DIM)
    assert batch.next_features.shape[0] == 64
    assert batch.next_features.shape[2] == F.FEATURE_DIM
    assert batch.next_mask.shape == batch.next_features.shape[:2]
    # A terminal sample must not bootstrap, and must expose no live candidates.
    terminal = batch.discount == 0.0
    assert not batch.next_mask[terminal].any()
    assert np.all(batch.discount[~terminal] > 0)


def test_masked_max_ignores_padding():
    """Padding must be -inf, not zero, or a pad row wins when all Q are negative."""
    q_values = torch.tensor([[-5.0, -2.0, 0.0]])
    mask = torch.tensor([[True, True, False]])
    assert float(net_module.masked_max(q_values, mask)) == pytest.approx(-2.0)
    assert int(net_module.masked_argmax(q_values, mask)) == 1


# --- job enumeration ---------------------------------------------------------


def test_pass_is_always_available_and_first():
    farm = {"tiles": [[None]], "farmer": [0, 0], "money": 0}
    jobs, truncated = enumerate_jobs(farm, {}, 0, [(0, 0)], [{}])
    assert jobs[0].kind == "PASS"
    assert truncated == 0


def test_enumeration_caps_and_reports_truncation():
    tiles = [[None] * 10 for _ in range(10)]
    farm = {"tiles": tiles, "farmer": [0, 0]}
    private = {"seeds": {crop: 5 for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")}}

    jobs, truncated = enumerate_jobs(farm, private, 0, [(0, 0)], [{}])
    assert len(jobs) == MAX_JOBS
    # 5 crops x 100 empty tiles + PASS, minus what survived the cap.
    assert truncated == 501 - MAX_JOBS


def test_drop_jobs_stay_with_their_carrier():
    tiles = [[None] * 10 for _ in range(10)]
    farm = {"tiles": tiles, "farmer": [0, 0], "hands": [[1, 1]]}
    jobs, _ = enumerate_jobs(farm, {}, 0, [(0, 0), (1, 1)], [{}, {"WHEAT": 3}])
    drops = [job for job in jobs if job.kind == "DROP"]
    assert len(drops) == 1
    assert drops[0].owner == 1


def test_fertilize_jobs_require_and_follow_the_carrier():
    plant = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": True,
        "yield_units": 0,
        "fertilized_until_day": -1,
    }
    farm = {"tiles": [[plant]], "farmer": [0, 0], "hands": [[0, 0]]}

    shed_only = {"shed": {"FERTILIZER": 2}}
    jobs, _ = enumerate_jobs(farm, shed_only, 1, [(0, 0), (0, 0)], [{}, {}])
    assert not [job for job in jobs if job.kind == "FERTILIZE"]

    jobs, _ = enumerate_jobs(
        farm, shed_only, 1, [(0, 0), (0, 0)], [{}, {"FERTILIZER": 1}]
    )
    fertilize = [job for job in jobs if job.kind == "FERTILIZE"]
    assert len(fertilize) == 1
    assert fertilize[0].owner == 1


# --- policy behaviour --------------------------------------------------------


def test_commitment_flag_marks_only_the_committed_job():
    targets = np.array([[0, 0], [3, 4], [1, 1]], dtype=np.int16)
    plain = F.distance_features(targets, (0, 0))
    assert plain[:, 2].sum() == 0.0

    marked = F.distance_features(targets, (0, 0), committed=1)
    np.testing.assert_array_equal(marked[:, 2], [0.0, 1.0, 0.0])
    # Distance columns must be untouched by the commitment flag.
    np.testing.assert_array_equal(marked[:, :2], plain[:, :2])


def test_recorder_captures_the_exact_committed_action_mask():
    class Capture:
        def reset(self):
            pass

        def record(self, **values):
            self.values = values

    capture = Capture()
    agent = JobScorerAgent(QNetwork.random(seed=3), recorder=capture)
    tile = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 0,
        "yield_units": 0,
        "max_lifespan_step": -1,
        "fertilized_until_day": -1,
    }
    agent._day = 1
    agent._commitments[0] = (("WATER", (0, 0), "WHEAT", None), 1)
    obs = {
        "player": 0,
        "step": 1,
        "day": 1,
        "hour": 1,
        "farms": [
            {
                "money": 3000.0,
                "tiles": [[tile]],
                "farmer": [0, 0],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
            {
                "money": 3000.0,
                "tiles": [[None]],
                "farmer": [0, 0],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
        ],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }

    agent(obs)
    mask = capture.values["available"][0]
    chosen = capture.values["chosen"][0]
    assert mask.sum() == 1
    assert mask[chosen]


def test_units_stick_to_a_job_instead_of_oscillating():
    """The failure this fixes: re-picking every turn made units walk back and
    forth between two targets and never arrive, so plants died unwatered."""
    agent = JobScorerAgent(QNetwork.random(seed=3))

    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][0] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "watered_today": False,
        "consecutive_unwatered": 0, "yield_units": 1, "max_lifespan_step": -1,
        "fertilized_until_day": -1,
    }
    tiles[9][9] = dict(tiles[0][0])

    def observation(position, step):
        return {
            "player": 0, "step": step, "day": 1, "hour": step % 24,
            "farms": [
                {"money": 3000.0, "tiles": tiles, "farmer": list(position), "hands": [],
                 "unlocked_quadrants": ["NW", "NE", "SW", "SE"], "hires_today": 0},
                {"money": 3000.0, "tiles": [[None]], "farmer": [0, 0], "hands": [],
                 "unlocked_quadrants": ["NW"], "hires_today": 0},
            ],
            "market": {"inventory": {}, "prices": {}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        }

    position = [4, 4]
    moves = []
    for step in range(1, 12):
        action = agent(observation(position, step))["farmer"]
        moves.append(action[0])
        delta = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}.get(action[0])
        if delta is None:
            break
        position = [position[0] + delta[0], position[1] + delta[1]]

    # A committed walk never reverses: no NORTH after SOUTH, no EAST after WEST.
    assert not ({"NORTH", "SOUTH"} <= set(moves))
    assert not ({"EAST", "WEST"} <= set(moves))


def test_commitment_clears_at_day_boundary():
    """Hands are re-hired daily, so unit indices stop meaning the same worker."""
    agent = JobScorerAgent(QNetwork.random(seed=4))
    agent._commitments[0] = (("WATER", (9, 9), "WHEAT", None), 3)
    agent._day = 1

    obs = {
        "player": 0, "step": 24, "day": 2, "hour": 0,
        "farms": [
            {"money": 3000.0, "tiles": [[None]], "farmer": [0, 0], "hands": [],
             "unlocked_quadrants": ["NW"], "hires_today": 0},
            {"money": 3000.0, "tiles": [[None]], "farmer": [0, 0], "hands": [],
             "unlocked_quadrants": ["NW"], "hires_today": 0},
        ],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }
    agent(obs)
    assert 0 not in agent._commitments


def test_selection_breaks_ties_to_lowest_index():
    q_values = np.zeros(5, dtype=np.float32)
    assert select(q_values) == 0
    q_values[3] = 1.0
    assert select(q_values) == 3


def test_selection_respects_the_allowed_mask():
    q_values = np.array([0.0, 9.0, 1.0], dtype=np.float32)
    allowed = np.array([True, False, True])
    assert select(q_values, allowed) == 2


def test_epsilon_zero_play_is_deterministic(tmp_path):
    """The arena's reproducibility checks depend on this."""
    weights = tmp_path / "w.npz"
    QNetwork.random(seed=5).save(weights)
    spec = f"agents/dqn_job_scorer/main.py?weights={weights}"

    first = run_match((spec, "starter"), seed=4)
    second = run_match((spec, "starter"), seed=4)
    assert not first.failed
    assert [s.reward for s in first.seats] == [s.reward for s in second.seats]


def test_agent_emits_only_legal_actions(tmp_path):
    weights = tmp_path / "w.npz"
    QNetwork.random(seed=9).save(weights)
    match = run_match((f"agents/dqn_job_scorer/main.py?weights={weights}", "starter"), seed=2, keep_replay=True)
    assert not match.failed

    for state in match.replay["steps"]:
        action = state[0]["action"]
        assert set(action) == {"farmer", "hands", "market"}
        assert isinstance(action["farmer"], list) and action["farmer"]
        assert len(action["market"]) <= 10
        for unit_action in [action["farmer"], *action["hands"]]:
            assert isinstance(unit_action[0], str)


def test_inference_stays_far_under_act_timeout():
    """actTimeout is 1s. Measure the worst case: full job set, many units."""
    agent = JobScorerAgent(QNetwork.random(seed=2))
    tiles = [[None] * 10 for _ in range(10)]
    for y in range(5):
        for x in range(5):
            tiles[y][x] = {
                "kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "watered_today": False,
                "consecutive_unwatered": 0, "yield_units": 2, "max_lifespan_step": -1,
                "fertilized_until_day": -1,
            }
    obs = {
        "player": 0, "step": 100, "day": 4, "hour": 4,
        "farms": [
            {"money": 5000.0, "tiles": tiles, "farmer": [4, 4],
             "hands": [[i % 10, i // 10] for i in range(12)],
             "unlocked_quadrants": ["NW", "NE", "SW", "SE"], "hires_today": 12},
            {"money": 5000.0, "tiles": tiles, "farmer": [4, 4], "hands": [],
             "unlocked_quadrants": ["NW"], "hires_today": 0},
        ],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {"WHEAT": 20, "MELON": 20}, "inventories": [{} for _ in range(13)]},
    }

    agent(obs)  # warm up numpy
    started = time.perf_counter()
    for _ in range(20):
        agent(obs)
    worst = (time.perf_counter() - started) / 20

    assert worst < 0.05, f"{worst * 1000:.1f}ms per turn is too close to the 1s actTimeout"


def test_submission_agent_never_imports_torch():
    """agents/ must stay torch-free; the submission ships numpy weights only."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import agents.dqn_job_scorer.main; "
         "assert 'torch' not in sys.modules, 'torch leaked into the submission agent'"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


# --- post-episode evaluation hook --------------------------------------------


class _FakeReader:
    """Duck-typed stand-in for EpisodeReader; the hook contract is structural."""

    def __init__(self, history):
        self._history = history

    def summary(self):
        return {
            "final_money": 21000.0,
            "money_difference": 2000.0,
            "win": 1.0,
            "metrics": {
                "productive_action_percentage": 0.15,
                "movement_percentage": 0.62,
                "plants_lost_inferred": 4,
            },
        }

    def history(self, revision_id, limit=50):
        return self._history[:limit]


def _episodes(margins):
    return [{"failed": False, "money_difference": m, "win": 1.0 if m > 0 else 0.0} for m in margins]


def _context():
    from kaggriculture_lab.hooks import EpisodeHookContext

    return EpisodeHookContext(
        session_id="s", generation_id="g", pairing_id="p", match_id="m",
        revision_id="rev1", opponent_revision_id="rev2", seat=0,
    )


def test_evaluator_reports_this_episode_and_the_window():
    from agents.dqn_job_scorer.evaluation import WINDOW, evaluate_episode

    result = evaluate_episode(_context(), _FakeReader(_episodes([2000.0] * (WINDOW * 2))))
    assert result["schema_version"] == 1
    assert result["final_money"] == 21000.0
    assert result["episodes_seen"] == WINDOW * 2
    assert result["should_reevaluate"] is False
    assert result["reevaluate_reason"] == "stable"


def test_evaluator_flags_a_collapse():
    """The dqn02 signature: recent games far worse than the older ones."""
    from agents.dqn_job_scorer.evaluation import WINDOW, evaluate_episode

    margins = [-9000.0] * WINDOW + [2000.0] * WINDOW  # newest first: collapsed, then healthy
    result = evaluate_episode(_context(), _FakeReader(_episodes(margins)))

    assert result["should_reevaluate"] is True
    assert result["margin_regression"] == pytest.approx(11000.0)
    assert "fell" in result["reevaluate_reason"]


def test_evaluator_will_not_fire_on_thin_history():
    """A handful of games cannot distinguish a collapse from variance."""
    from agents.dqn_job_scorer.evaluation import evaluate_episode

    result = evaluate_episode(_context(), _FakeReader(_episodes([-9000.0, 2000.0])))
    assert result["should_reevaluate"] is False
    assert result["reevaluate_reason"] == "insufficient history"


def test_evaluator_ignores_failed_episodes():
    from agents.dqn_job_scorer.evaluation import WINDOW, evaluate_episode

    history = _episodes([2000.0] * (WINDOW * 2))
    history.insert(0, {"failed": True, "money_difference": -99999.0, "win": 0.0})
    result = evaluate_episode(_context(), _FakeReader(history))
    assert result["should_reevaluate"] is False


def test_episode_reader_rejects_mutating_and_multi_statement_sql(tmp_path):
    """A reporting hook must not be able to write to the league it reports on."""
    from kaggriculture_lab.hooks import EpisodeReader

    with ExperimentStore(tmp_path / "readonly.duckdb") as store:
        reader = EpisodeReader(connection=store.connection, match_id="m", seat=0)
        for statement in (
            "DELETE FROM episode_players",
            "UPDATE runs SET status = 'x'",
            "DROP TABLE runs",
            "SELECT 1; DELETE FROM episode_players",
        ):
            with pytest.raises(ValueError, match="only permits SELECT/WITH"):
                reader.fetchall(statement)


def test_truncation_keeps_maintenance_over_planting():
    """Only PLANT multiplies, so it must be what overflows the cap.

    Ranking candidates purely by distance let a nearby PLANT evict a distant
    plant that dies tonight without water -- a far worse trade.
    """
    from agents.dqn_job_scorer.jobs import MAX_JOBS, enumerate_jobs

    tiles = [[None] * 10 for _ in range(10)]
    # One thirsty plant in the far corner, about to become a weed.
    tiles[9][9] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "watered_today": False,
        "consecutive_unwatered": 1, "yield_units": 0, "max_lifespan_step": -1,
        "fertilized_until_day": -1,
    }
    farm = {"tiles": tiles, "farmer": [0, 0]}
    private = {"seeds": {crop: 9 for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")}}

    jobs, truncated = enumerate_jobs(farm, private, 1, [(0, 0)], [{}])

    assert len(jobs) == MAX_JOBS
    assert truncated > 0, "this scenario is supposed to overflow the cap"

    water = [job for job in jobs if job.kind == "WATER"]
    assert water and water[0].target == (9, 9), "the dying plant was evicted by planting jobs"
    assert all(job.kind == "PLANT" for job in jobs if job.kind not in {"PASS", "WATER"})


def test_evaluator_runs_against_the_real_duckdb_schema(tmp_path):
    """The fake-reader tests pin the logic; this pins the SQL.

    `EpisodeReader.history` joins `episode_players` by `strategy_id`, and a
    column rename or a wrong filter would only ever surface in a live league run.
    """
    from kaggriculture_lab.hooks import EpisodeHookContext, EpisodeReader
    from kaggriculture_lab.storage import ExperimentStore
    from sim.runner import MatchResult, SeatResult

    def match(match_id: str, mine: float, theirs: float) -> MatchResult:
        return MatchResult(
            seed=1,
            seats=[
                SeatResult("dqn", 0, mine, "DONE", False, None, None, 1, 0.0, 0.0, 0.0, mine,
                           {"productive_action_percentage": 0.2, "movement_percentage": 0.6,
                            "plants_lost_inferred": 3}),
                SeatResult("heur", 1, theirs, "DONE", False, None, None, 1, 0.0, 0.0, 0.0, theirs, {}),
            ],
            steps=1,
            wallclock=0.1,
            replay={"steps": [[{"observation": {"step": 0}}]]},
        )

    with ExperimentStore(tmp_path / "league.duckdb") as store:
        # 30 prior games: the older half strong, the recent half collapsed.
        for index in range(30):
            mine = 21_000.0 if index < 15 else 8_000.0
            store.add_match(f"m{index:03d}", "run", "league:t", ("rev1", "rev2"), ("rev1", "rev2"),
                            match(f"m{index:03d}", mine, 19_000.0))
        store.add_match("current", "run", "league:t", ("rev1", "rev2"), ("rev1", "rev2"),
                        match("current", 8_000.0, 19_000.0))

        reader = EpisodeReader(store.connection, "current", 0)
        history = reader.history("rev1", limit=80)
        assert len(history) == 30, "history must exclude the current match"
        assert all(row["match_id"] != "current" for row in history)

        context = EpisodeHookContext("s", "g", "p", "current", "rev1", "rev2", 0)
        result = evaluate_episode_hook(context, reader)

    assert result["episodes_seen"] == 30
    assert result["should_reevaluate"] is True, result["reevaluate_reason"]


def evaluate_episode_hook(context, reader):
    from agents.dqn_job_scorer.main import evaluate_episode

    return evaluate_episode(context, reader)


def test_evaluator_reports_only_metrics_the_league_actually_records():
    """Guards against reporting a key the replay cannot produce.

    `job_truncations` is counted inside the agent's job enumerator, so it never
    reaches `episode_metrics`, which derives everything from the stored replay.
    Reading it here would emit a constant zero that looks like a measurement.
    """
    from agents.dqn_job_scorer.evaluation import evaluate_episode
    from kaggriculture_lab.episode_metrics import episode_metrics
    from sim.runner import run_match

    match = run_match(("agents/dqn_job_scorer/main.py", "starter"), seed=0, analyzer=episode_metrics)
    available = set(match.seats[0].metrics)

    result = evaluate_episode(_context(), _FakeReader(_episodes([2000.0] * 30)))
    derived = {"productive_action_percentage", "movement_percentage", "plants_lost_inferred"}

    assert derived <= available, f"evaluator reads keys the league never records: {derived - available}"
    assert "job_truncations" not in result


# --- checkpoint promotion ----------------------------------------------------


def _promotions(evals, rule):
    """Replay a promotion rule over (round, margin, ci) triples."""
    best, promoted = None, []
    for rnd, margin, ci in evals:
        if rule(margin, ci, best):
            best = margin
            promoted.append(rnd)
    return promoted


# Real evaluations from run dqn03, 24 seeds each.
DQN03_EVALS = [
    (10, -10120.0, (-10603.0, -9637.0)),
    (20, -13143.6, (-14110.2, -12176.9)),
    (30, -4249.1, (-4848.6, -3649.7)),
    (40, -1027.5, (-1864.8, -190.3)),
    (50, -26616.1, (-30120.1, -23112.1)),
    (60, 2401.6, (1780.4, 3022.9)),
    (70, 2466.3, (1437.2, 3495.3)),
]


def test_promotion_requires_a_resolved_improvement():
    """A bare `>` promotes noise.

    Round 70 beat round 60 by +64 on 24 seeds and was promoted; a later 60-seed
    head-to-head had round 60 winning 41-19. Requiring the challenger's lower
    confidence bound to clear the incumbent's point estimate drops exactly that
    promotion and keeps every real one.
    """
    raw = _promotions(DQN03_EVALS, lambda margin, ci, best: best is None or margin > best)
    resolved = _promotions(DQN03_EVALS, lambda margin, ci, best: best is None or ci[0] > best)

    assert raw == [10, 30, 40, 60, 70]
    assert resolved == [10, 30, 40, 60]


def test_promotion_rejects_collapses():
    """The failure that motivated best-not-last at all."""
    resolved = _promotions(DQN03_EVALS, lambda margin, ci, best: best is None or ci[0] > best)
    assert 50 not in resolved  # -26,616
    assert 20 not in resolved  # -13,144


# --- replay retention --------------------------------------------------------


def test_buffer_keeps_a_sample_of_old_episodes(episode):
    """Pure FIFO forgets everything older than one turnover.

    dqn03's buffer held 200 episodes at 16 per round, so after ~13 rounds the
    network trained only on its own current policy -- flat loss, wildly swinging
    evaluation. The reservoir segment keeps old states reachable at fixed cost.
    """
    from agents.dqn_job_scorer.training.replay import ReplayBuffer

    buffer = ReplayBuffer(capacity=20, gamma=GAMMA, n_step=3, seed=0, reservoir_fraction=0.5)

    tagged = []
    for index in range(200):
        clone = copy.copy(episode)
        clone.tag = index  # attribute for provenance only; not used by sampling
        tagged.append(clone)
        buffer.add([clone])

    assert len(buffer.episodes) == 20
    assert len(buffer.recent) == 10
    assert len(buffer.reservoir) == 10

    tags = sorted(item.tag for item in buffer.episodes)
    assert tags[-10:] == list(range(190, 200)), "recent segment must be strictly the newest"
    # The reservoir samples the whole evicted stream, so it must reach well back.
    assert min(tags) < 100, f"reservoir retained nothing early: {tags}"


def test_buffer_without_reservoir_is_plain_fifo():
    """reservoir_fraction=0 must reproduce the old behaviour exactly."""
    from agents.dqn_job_scorer.training.replay import ReplayBuffer

    buffer = ReplayBuffer(capacity=5, gamma=GAMMA, n_step=3, seed=0, reservoir_fraction=0.0)
    assert buffer.recent_capacity == 5
    assert buffer.reservoir_capacity == 0


def test_reservoir_sampling_has_no_recency_bias():
    """Algorithm R gives every evicted episode the same retention probability.

    Asserting that *every* episode appears would be wrong: with a 10-of-90
    retention chance, roughly 1% of items miss any 40 trials by luck. The
    property that matters is that early episodes are retained as often as late
    ones -- a recency-biased buffer is the thing being fixed.
    """
    from agents.dqn_job_scorer.training.replay import ReplayBuffer

    counts = [0] * 100
    for seed in range(120):
        buffer = ReplayBuffer(capacity=20, gamma=GAMMA, n_step=3, seed=seed, reservoir_fraction=0.5)
        for index in range(100):
            item = types.SimpleNamespace(tag=index)
            buffer.recent.append(item)
            if len(buffer.recent) > buffer.recent_capacity:
                buffer._offer_to_reservoir(buffer.recent.pop(0))
        for item in buffer.reservoir:
            counts[item.tag] += 1

    evicted = counts[:90]  # the last 10 never leave the recent segment
    assert sum(evicted) > 0

    third = len(evicted) // 3
    oldest, newest = sum(evicted[:third]), sum(evicted[-third:])
    ratio = oldest / max(1, newest)
    assert 0.6 < ratio < 1.7, f"retention favours one end: oldest={oldest} newest={newest}"


def test_self_play_share_matches_the_configured_ratio():
    """The mix used to be an accident of list length.

    `[spec, *pool]` with a 5-snapshot pool made round-robin collection play 5 of
    every 6 episodes against past selves while evaluation stayed on the
    benchmark. dqn04 fell from +2,269 to -22,154 on the round that switched on.
    """
    from agents.dqn_job_scorer.training.loop import Config, opponents_for

    snapshots = [pathlib.Path(f"runs/x/snapshots/round{n:04d}.npz") for n in range(10, 60, 10)]

    for ratio, tolerance in ((0.25, 0.08), (0.5, 0.1)):
        config = Config(rounds=100, self_play_after=0.5, self_play_pool=5, self_play_ratio=ratio)
        opponents = opponents_for(60, config, snapshots)
        # collect_batch assigns opponents round-robin, so list share is episode share.
        share = sum(1 for item in opponents if "weights=" in item) / len(opponents)
        assert abs(share - ratio) < tolerance, f"ratio {ratio} gave share {share:.2f}"

    disabled = Config(rounds=100, self_play_after=0.5, self_play_pool=5, self_play_ratio=0.0)
    assert opponents_for(60, disabled, snapshots) == [Config().curriculum[-1][1]]


def test_self_play_is_off_before_its_threshold():
    from agents.dqn_job_scorer.training.loop import Config, opponents_for

    config = Config(rounds=100, self_play_after=0.5, self_play_pool=5)
    snapshots = [pathlib.Path("runs/x/snapshots/round0010.npz")]
    assert all("weights=" not in item for item in opponents_for(40, config, snapshots))


def test_duel_decides_promotion_not_benchmark_margin(tmp_path):
    """Benchmark margin alone picks the wrong checkpoint.

    dqn04-r100 and dqn03-r60 scored +2,558 and +2,545 against the heuristic over
    80 seeds -- indistinguishable, both 78-2 -- yet r60 beat r100 65-15 head to
    head. Equal margin against a common opponent is not equal strength.
    """
    from agents.dqn_job_scorer.training.loop import evaluate_head_to_head

    challenger = tmp_path / "challenger.npz"
    incumbent = tmp_path / "incumbent.npz"
    QNetwork.random(seed=1).save(challenger)
    QNetwork.random(seed=2).save(incumbent)

    duel = evaluate_head_to_head(challenger, incumbent, seeds=4, jobs=2)

    assert duel["pairs"] == 4
    assert duel["wins"] + duel["losses"] <= 4
    assert set(duel) == {"pairs", "wins", "losses", "margin", "margin_ci", "significant"}
    # The duel must be symmetric in the sense that swapping the pair negates it.
    reverse = evaluate_head_to_head(incumbent, challenger, seeds=4, jobs=2)
    assert reverse["margin"] == pytest.approx(-duel["margin"], abs=1e-6)
