"""Run one kaggriculture episode and report what actually happened.

Two properties of the environment make the naive `env.run(...)` /
`env.steps[-1]` reading actively misleading, and this module exists to correct
for both:

1. A crashed agent still finishes the episode with a plausible score, and the
   *final* step's status reads `DONE`. The `ERROR` status only appears on the
   step after the exception and is not carried to the end. So statuses must be
   scanned over every step, not read off the last one.
2. `debug=True` lets agent exceptions propagate instead of recording them, so
   the arena runs with `debug=False` and reads tracebacks out of `env.logs`.

Note `env.logs` is indexed by env step, which lags the observation `step` an
agent saw by 2 -- an exception raised on obs step 300 lands at `logs[302]`.
Only the relative ordering matters here, so logs are scanned wholesale.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kaggle_environments import make

ENV_NAME = "kaggriculture"

# Everything else (ERROR, INVALID, TIMEOUT, ...) means the agent failed rather
# than merely lost.
OK_STATUSES = frozenset({"ACTIVE", "DONE", "INACTIVE"})


@dataclass
class SeatResult:
    """Outcome for one seat of one episode."""

    spec: str
    seat: int
    reward: float | None
    status: str            # worst status seen across the episode
    failed: bool
    fail_step: int | None  # env step where the bad status first appeared
    traceback: str | None
    calls: int
    time_max: float
    time_p99: float
    time_mean: float
    final_money: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchResult:
    """Outcome of one episode, from the perspective of seat order as given."""

    seed: int
    seats: list[SeatResult]
    steps: int
    wallclock: float
    replay: Any | None = field(default=None, repr=False)
    training_artifacts: list[dict[str, Any] | None] = field(default_factory=list, repr=False)

    @property
    def failed(self) -> bool:
        return any(s.failed for s in self.seats)

    @property
    def margin(self) -> float | None:
        """Seat 0 reward minus seat 1 reward. None if either seat failed."""
        if self.failed or any(s.reward is None for s in self.seats):
            return None
        return self.seats[0].reward - self.seats[1].reward


def _scan_statuses(steps: list, seat: int) -> tuple[str, int | None]:
    """Return the worst status this seat ever held and where it first appeared.

    The last step is not authoritative -- the interpreter reports DONE there
    even for a seat that errored hundreds of steps earlier.
    """
    for index, step in enumerate(steps):
        status = step[seat].status
        if status not in OK_STATUSES:
            return status, index
    return steps[-1][seat].status, None


def _seat_logs(logs: list, seat: int) -> tuple[list[float], str | None]:
    """Collect per-call durations and the first traceback for one seat."""
    durations: list[float] = []
    traceback_text: str | None = None
    for entry in logs:
        if seat >= len(entry):
            continue
        record = entry[seat]
        if not record:  # {} once the seat stops being called
            continue
        if "duration" in record:
            durations.append(record["duration"])
        if traceback_text is None and record.get("stderr"):
            traceback_text = record["stderr"]
    return durations, traceback_text


def run_match(
    specs: tuple[str, str],
    seed: int,
    config: dict[str, Any] | None = None,
    keep_replay: bool = False,
    env_name: str = ENV_NAME,
    analyzer: Callable[[list, int], dict[str, Any]] | None = None,
) -> MatchResult:
    """Play one episode of `specs[0]` (seat 0) against `specs[1]` (seat 1).

    `analyzer` optionally extracts game-specific per-seat metrics from the
    finished episode. It is injected rather than built in so this module stays
    a generic match runner -- it knows about seats, statuses, rewards and
    timings, and nothing about crops. See
    `kaggriculture_lab.episode_metrics.episode_metrics` for the implementation
    the league passes in.
    """
    from sim import registry

    agents = [registry.resolve(spec) for spec in specs]
    configuration = {**(config or {}), "seed": seed}

    started = time.perf_counter()
    env = make(env_name, configuration=configuration, debug=False)
    env.run(agents)
    wallclock = time.perf_counter() - started

    training_artifacts: list[dict[str, Any] | None] = []
    for seat, resolved in enumerate(agents):
        finalize = getattr(resolved, "finalize_episode", None)
        training_artifacts.append(
            None if finalize is None else finalize(env.steps, seat)
        )

    seats: list[SeatResult] = []
    for seat, spec in enumerate(specs):
        status, fail_step = _scan_statuses(env.steps, seat)
        durations, traceback_text = _seat_logs(env.logs, seat)
        failed = status not in OK_STATUSES
        analysis = {} if analyzer is None else analyzer(env.steps, seat)
        seats.append(
            SeatResult(
                spec=spec,
                seat=seat,
                reward=env.steps[-1][seat].reward,
                status=status,
                failed=failed,
                fail_step=fail_step,
                traceback=traceback_text,
                calls=len(durations),
                time_max=max(durations, default=0.0),
                time_p99=_percentile(durations, 0.99),
                time_mean=statistics.fmean(durations) if durations else 0.0,
                final_money=analysis.get("final_money"),
                metrics=analysis,
            )
        )

    return MatchResult(
        seed=seed,
        seats=seats,
        steps=len(env.steps),
        wallclock=wallclock,
        replay=env.toJSON() if keep_replay else None,
        training_artifacts=training_artifacts,
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))
    return ordered[index]
