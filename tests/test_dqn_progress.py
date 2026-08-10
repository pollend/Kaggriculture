from pathlib import Path
from types import SimpleNamespace

from agents.dqn_job_scorer.train import DQNTrainer
from agents.dqn_job_scorer.training.progress import DQNProgressEvaluator


class RecordingWriter:
    def __init__(self):
        self.calls = []

    def run_evaluation_round(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "pairs": len(kwargs["matchups"]),
            "margin": 12.5,
            "win_rate": 0.75,
        }


def _context(tmp_path: Path, *, scope: str = "round", round_index: int = 0):
    incumbent = tmp_path / "incumbent.npz"
    incumbent.write_bytes(b"joint-policy")
    return SimpleNamespace(
        model_id="dqn_job_scorer.adaptive_hybrid_q_v2@fixed",
        family_id="dqn_job_scorer",
        scope=scope,
        session_id="mixed-session",
        generation_id="mixed-session:g0003",
        generation_index=3,
        round_index=round_index,
        active_checkpoint=None,
        config={
            "incumbent_weights": str(incumbent),
            "progress_observation_seeds": 2,
            "progress_opponents": ["starter", "crop.py", "planner.py"],
            "progress_seed_base": 1_500_000_000,
            "jobs": 2,
            "episode_steps": 720,
        },
        artifacts=tmp_path,
    )


def test_progress_observation_records_two_seeds_for_three_opponents(tmp_path):
    writer = RecordingWriter()

    summary = DQNProgressEvaluator().observe(
        _context(tmp_path), writer, phase="start"
    )

    assert summary["pairs"] == 6
    assert len(writer.calls) == 1
    call = writer.calls[0]
    assert call["pairing_kind"] == "dqn_progress"
    assert call["request_id"].endswith(":g0003:start")
    assert call["jobs"] == 2
    assert call["episode_steps"] == 720
    assert len(call["matchups"]) == 6
    assert len({seed for _, _, seed in call["matchups"]}) == 6
    assert {right.agent_spec for _, right, _ in call["matchups"]} == {
        "starter",
        "crop.py",
        "planner.py",
    }
    assert all(left.role == "progress_learner" for left, _, _ in call["matchups"])


def test_dqn_trainer_observes_only_generation_boundaries(tmp_path):
    trainer = DQNTrainer()
    events = []
    trainer._progress = SimpleNamespace(
        observe=lambda context, writer, phase: events.append(("observe", phase))
    )
    trainer._collector = SimpleNamespace(
        collect=lambda context, reader, writer: events.append(
            ("collect", context.round_index)
        )
    )

    trainer.on_round(_context(tmp_path, round_index=0), None, None)
    trainer.on_round(_context(tmp_path, round_index=1), None, None)
    trainer.on_round(
        _context(tmp_path, scope="generation", round_index=-1), None, None
    )

    assert events == [
        ("observe", "start"),
        ("collect", 0),
        ("collect", 1),
        ("observe", "end"),
    ]
