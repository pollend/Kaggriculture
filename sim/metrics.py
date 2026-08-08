"""Aggregate seat-swapped match results into a verdict.

Every seed is played twice, once in each seat orientation, so each seed yields
a *paired* observation of the same underlying episode conditions. Aggregating
on the paired margin rather than on two independent reward means is what makes
a few dozen seeds enough to resolve a real difference.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from sim.runner import MatchResult

# Two-sided normal quantile at 95%.
Z95 = 1.959963985


@dataclass
class Summary:
    """Head-to-head verdict for agent A against agent B."""

    a: str
    b: str
    pairs: int                 # seeds where both orientations completed cleanly
    wins: int
    losses: int
    draws: int
    win_rate: float
    win_rate_lo: float
    win_rate_hi: float
    margin_mean: float         # mean of (A money - B money), averaged over both seats
    margin_se: float
    margin_ci: tuple[float, float]
    sign_p: float              # two-sided sign test on per-seed margins
    a_money_mean: float
    b_money_mean: float
    a_money_median: float
    b_money_median: float
    errors: dict[str, int] = field(default_factory=dict)
    dropped_seeds: int = 0
    time_max: dict[str, float] = field(default_factory=dict)
    time_p99: dict[str, float] = field(default_factory=dict)
    act_timeout: float = 1.0
    wallclock: float = 0.0

    @property
    def significant(self) -> bool:
        """True when the paired-margin 95% interval excludes zero."""
        lo, hi = self.margin_ci
        return lo > 0 or hi < 0

    @property
    def seeds_needed(self) -> int | None:
        """Seeds required to resolve the observed margin at 95%, or None.

        Returns None when the effect is already resolved or when the observed
        margin is zero (nothing to resolve).
        """
        if self.significant or self.pairs == 0 or self.margin_mean == 0:
            return None
        sd = self.margin_se * math.sqrt(self.pairs)
        if sd == 0:
            return None
        return math.ceil((Z95 * sd / abs(self.margin_mean)) ** 2)


def wilson(successes: float, total: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval -- well behaved at 0%, 100%, and small n."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - spread), min(1.0, center + spread))


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def sign_test(margins: list[float]) -> float:
    """Two-sided sign test p-value on non-zero margins.

    Distribution-free, so it does not care that money is heavy-tailed -- a
    useful cross-check on the interval, which does assume roughly normal means.
    """
    non_zero = [m for m in margins if m != 0]
    n = len(non_zero)
    if n == 0:
        return 1.0
    positive = sum(1 for m in non_zero if m > 0)
    k = min(positive, n - positive)
    tail = sum(math.exp(_log_comb(n, i) - n * math.log(2)) for i in range(k + 1))
    return min(1.0, 2 * tail)


def summarize(
    a: str,
    b: str,
    results: dict[int, tuple[MatchResult, MatchResult]],
    act_timeout: float = 1.0,
) -> Summary:
    """Fold {seed: (match_a_first, match_b_first)} into a Summary.

    A seed contributes only if *both* orientations completed without an agent
    error; partial seeds would bias the pairing toward whichever seat happened
    to survive.
    """
    margins: list[float] = []
    a_money: list[float] = []
    b_money: list[float] = []
    errors: dict[str, int] = {}
    time_max: dict[str, float] = {a: 0.0, b: 0.0}
    time_p99: dict[str, float] = {a: 0.0, b: 0.0}
    wallclock = 0.0
    dropped = 0

    for forward, reverse in results.values():
        wallclock += forward.wallclock + reverse.wallclock
        for match in (forward, reverse):
            for seat in match.seats:
                time_max[seat.spec] = max(time_max.get(seat.spec, 0.0), seat.time_max)
                time_p99[seat.spec] = max(time_p99.get(seat.spec, 0.0), seat.time_p99)
                if seat.failed:
                    errors[seat.spec] = errors.get(seat.spec, 0) + 1

        if forward.failed or reverse.failed:
            dropped += 1
            continue

        # forward: A is seat 0. reverse: A is seat 1, so negate.
        a_forward, b_forward = forward.seats[0].reward, forward.seats[1].reward
        b_reverse, a_reverse = reverse.seats[0].reward, reverse.seats[1].reward

        margins.append(((a_forward - b_forward) + (a_reverse - b_reverse)) / 2)
        a_money.extend([a_forward, a_reverse])
        b_money.extend([b_forward, b_reverse])

    pairs = len(margins)
    wins = sum(1 for m in margins if m > 0)
    losses = sum(1 for m in margins if m < 0)
    draws = pairs - wins - losses

    win_rate = wins / pairs if pairs else 0.0
    # Draws count as half a win so the rate stays centred at 50% under symmetry.
    lo, hi = wilson(wins + draws / 2, pairs)

    margin_mean = statistics.fmean(margins) if margins else 0.0
    margin_se = (statistics.stdev(margins) / math.sqrt(pairs)) if pairs > 1 else 0.0

    return Summary(
        a=a,
        b=b,
        pairs=pairs,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=win_rate,
        win_rate_lo=lo,
        win_rate_hi=hi,
        margin_mean=margin_mean,
        margin_se=margin_se,
        margin_ci=(margin_mean - Z95 * margin_se, margin_mean + Z95 * margin_se),
        sign_p=sign_test(margins),
        a_money_mean=statistics.fmean(a_money) if a_money else 0.0,
        b_money_mean=statistics.fmean(b_money) if b_money else 0.0,
        a_money_median=statistics.median(a_money) if a_money else 0.0,
        b_money_median=statistics.median(b_money) if b_money else 0.0,
        errors=errors,
        dropped_seeds=dropped,
        time_max=time_max,
        time_p99=time_p99,
        act_timeout=act_timeout,
        wallclock=wallclock,
    )


def render(summary: Summary) -> str:
    """Human-readable block for one head-to-head."""
    s = summary
    lines = [
        f"{s.a}  vs  {s.b}",
        f"  paired seeds : {s.pairs}" + (f"  ({s.dropped_seeds} dropped for agent errors)" if s.dropped_seeds else ""),
        f"  record       : {s.wins}W {s.losses}L {s.draws}D",
        f"  win rate     : {s.win_rate:6.1%}   95% CI [{s.win_rate_lo:.1%}, {s.win_rate_hi:.1%}]",
        f"  margin       : {s.margin_mean:+10,.0f}   95% CI [{s.margin_ci[0]:+,.0f}, {s.margin_ci[1]:+,.0f}]   sign p={s.sign_p:.4f}",
        f"  money        : {s.a} {s.a_money_mean:,.0f} mean / {s.a_money_median:,.0f} median"
        f"   |   {s.b} {s.b_money_mean:,.0f} / {s.b_money_median:,.0f}",
    ]

    verdict = "RESOLVED" if s.significant else "NOT RESOLVED"
    line = f"  verdict      : {verdict}"
    if not s.significant:
        needed = s.seeds_needed
        line += f" -- need ~{needed} seeds for this margin" if needed else " -- no measurable difference"
    lines.append(line)

    for spec in (s.a, s.b):
        budget = s.time_max.get(spec, 0.0) / s.act_timeout if s.act_timeout else 0.0
        flag = "  <-- NEAR TIMEOUT" if budget > 0.5 else ""
        lines.append(
            f"  timing       : {spec} max {s.time_max.get(spec, 0.0) * 1000:.1f}ms"
            f" p99 {s.time_p99.get(spec, 0.0) * 1000:.1f}ms"
            f" ({budget:.0%} of actTimeout){flag}"
        )

    if s.errors:
        for spec, count in s.errors.items():
            lines.append(f"  ERRORS       : {spec} failed in {count} episode(s)")

    return "\n".join(lines)
