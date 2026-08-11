"""Small, durable evaluation rounds for observing DQN progression."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlencode

from kaggriculture_lab.model_trainers import EvaluationPolicy

SELF_SPEC = "agents/dqn_job_scorer/main.py"
SELF_MANIFEST = "agents/dqn_job_scorer/agent.toml"


def _policy_identity(prefix: str, spec: str) -> str:
    return f"{prefix}:{hashlib.sha256(spec.encode()).hexdigest()[:20]}"


def _opponent_policy(
    spec: str,
    *,
    role: str = "progress_fixed_opponent",
    prefix: str = "dqn-progress-opponent",
) -> EvaluationPolicy:
    """Identity for whatever the learner was put up against.

    An opponent arrives as a bare spec and can be any of the three things the
    league knows how to name: a builtin, a manifest-backed agent, or a plain
    `main.py` with no manifest beside it.
    """
    if spec in {"starter", "pass", "random"}:
        return EvaluationPolicy(f"builtin:{spec}", spec, role=role)
    plain = Path(spec.partition("?")[0])
    manifest = plain.parent / "agent.toml"
    return EvaluationPolicy(
        _policy_identity(prefix, spec),
        spec,
        str(manifest) if plain.name == "main.py" and manifest.is_file() else None,
        role,
    )


def _learner_policy(context) -> EvaluationPolicy | None:
    """The policy to observe, or None when there is not yet a file to observe.

    A randomly initialized run has no incumbent and no checkpoint until its
    first promotion: the learner exists only inside the trainer. Falling back to
    the packaged weights here would record observations for a policy that is not
    the learner, so the phase is skipped instead.
    """

    if context.active_checkpoint is not None:
        path = Path(context.active_checkpoint.inference_artifact_path)
        identity = context.active_checkpoint.checkpoint_id
    else:
        configured = context.config.get("incumbent_weights")
        if not configured:
            return None
        path = Path(str(configured))
        if not path.is_file():
            raise FileNotFoundError(f"DQN progress checkpoint does not exist: {path}")
        identity = hashlib.sha256(path.read_bytes()).hexdigest()[:24]
    spec = f"{SELF_SPEC}?{urlencode({'weights': str(path)})}"
    return EvaluationPolicy(
        f"dqn-progress-policy:{identity}",
        spec,
        SELF_MANIFEST,
        "progress_learner",
    )


class DQNProgressEvaluator:
    """Record two small replayable observations around each generation."""

    def observe(self, context, writer, *, phase: str) -> dict | None:
        count = int(context.config.get("progress_observation_seeds", 0))
        if count < 1:
            return None
        if phase not in {"start", "end"}:
            raise ValueError(f"unsupported DQN progress phase: {phase}")
        opponents = list(
            dict.fromkeys(
                str(item)
                for item in context.config.get(
                    "progress_opponents",
                    context.config.get("benchmarks", []),
                )
            )
        )
        if not opponents:
            return None

        learner = _learner_policy(context)
        if learner is None:
            return None
        base = int(context.config.get("progress_seed_base", 1_500_000_000))
        phase_offset = 0 if phase == "start" else 500
        generation_offset = context.generation_index * 10_000
        matchups = []
        for opponent_index, opponent in enumerate(opponents):
            fixed = _opponent_policy(opponent)
            for seed_index in range(count):
                seed = (
                    base
                    + generation_offset
                    + phase_offset
                    + opponent_index * count
                    + seed_index
                )
                matchups.append((learner, fixed, seed))

        summary = writer.run_evaluation_round(
            context=context,
            request_id=(
                f"dqn-progress:{context.model_id}:"
                f"g{context.generation_index:04d}:{phase}"
            ),
            pairing_kind="dqn_progress",
            matchups=matchups,
            jobs=int(context.config.get("jobs", 1)),
            episode_steps=int(context.config.get("episode_steps", 720)),
        )
        print(
            f"[dqn-progress] generation {context.generation_index} {phase}: "
            f"{summary['pairs']} pairings, margin={float(summary['margin']):+.1f}, "
            f"win_rate={float(summary['win_rate']):.1%}",
            flush=True,
        )
        return summary
