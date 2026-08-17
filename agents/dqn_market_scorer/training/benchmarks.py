"""Which fixed opponents a release is measured against.

Its own module because both `updater.py` and `progress.py` need it and the
updater already imports the progress helpers -- putting it in either one makes
that import cycle. Nothing here imports torch, so the progress evaluator does
not pull the trainer in behind it.
"""

from __future__ import annotations

from typing import Any

DONOR_BENCHMARK = "agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py"


def configured_benchmarks(config: dict[str, Any]) -> list[str]:
    """Return the fixed release suite, promotion baseline first.

    Index 0 is the baseline the release gate reads; the rest are non-regression
    diagnostics. There is deliberately no separate singular `benchmark` key --
    one config value naming the baseline and a list whose first entry is also
    the baseline can disagree, and every reader would then have to decide which
    one wins.

    The default is the unmodified donor itself, so the paired match measures the
    complete Q-guided residual against its source policy.
    """

    configured = config.get("benchmarks")
    if configured:
        return list(dict.fromkeys(str(item) for item in configured))
    return [DONOR_BENCHMARK]


def promotion_benchmark(config: dict[str, Any]) -> str:
    """The single opponent the release gate is measured against."""

    return configured_benchmarks(config)[0]
