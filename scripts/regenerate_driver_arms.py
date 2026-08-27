#!/usr/bin/env python3
"""Rebuild every experiment arm from `train_market_free_v3d.py`.

Arms are generated, not hand-maintained. Keeping them in sync by editing each file
failed three times in one session -- a stale roster, a stale opponent mix, and a
stale gate config -- so the blended driver is the single source and each arm is a
named list of string substitutions applied to it. Edit the blended driver, run this.

Every substitution must match exactly once or the arm fails loudly rather than
silently diverging. That check is the whole point of the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "train_market_free_v3d.py"

ARMS = {
    # The single-player demonstration A/B. Differs only in the pinned pack.
    "train_market_free_v3d_solo.py": {
        "banner": '''"""Single-player demonstration arm. GENERATED -- do not edit.

Rebuild with `python scripts/regenerate_driver_arms.py`. The A/B against the blended
driver is the pinned pack and nothing else.

`demonstration_episodes` is left at the blended arm's value on purpose. The packs are
different sizes -- `ladder-top25-296` holds 525 seats from 98 players, `kawashigi-296`
holds 144 from one -- and `ladder.stratified_indices` returns everything a pack has
when the limit exceeds it, so this arm pins 144 against the other's 525. That
asymmetry is the experiment: one player deep against the ladder's top quarter broad.
"""
''',
        "subs": [
            (
                '"demonstration_pack": str(ARTIFACTS / "demonstrations" / "ladder-top25-296")',
                '"demonstration_pack": str(ARTIFACTS / "demonstrations" / "kawashigi-296")',
            ),
            ('SESSION_ID = "market-free-ruin-v2"', 'SESSION_ID = "market-free-ruinpack-v1"'),
            ('SLOT_PREFIX = "ruin2"', 'SLOT_PREFIX = "ruinpack"'),
        ],
    },
    # The open-gate arm. Promotion still decides what is released; it no longer
    # decides what is trained, and it no longer rejects on trading rate.
    "train_market_free_v3d_open.py": {
        "banner": '''"""Open-gate arm. GENERATED -- do not edit.

Rebuild with `python scripts/regenerate_driver_arms.py`.

`market-free-ruin-v1` froze at batch 60. It promoted a 27,987 spike, and then updates
80/100/120/140 were all rejected and all resumed from the identical checkpoint -- four
updates of 960 gradient steps each, trained and discarded, with batch 140 scoring
24,969 and missing the bar by 219 coins. That is verbatim the `terminal-selfplay-ladder-v1`
failure the blended driver's own `lineage_source` comment describes: *"Promotion should
decide what is released; it must not decide what is trained, or a gate that says no
freezes the learner it is grading."*

The snapback that should have caught it is disconnected: the promoted branch of
`updater.lineage_checkpoint` builds its `LineageCheckpoint` without `rejection_streak`,
so the field sits at its default of 0 and `resume_reset_after_rejections` can never
fire. That is a bug to fix, not a setting -- this arm sidesteps it rather than relying
on it.

Three gates come off, all in the same direction: stop the grader steering the learner.
"""
''',
        "subs": [
            # 1. The freeze itself. Training follows the lineage, not the release.
            ('"lineage_source": "promoted",', '"lineage_source": "latest",'),
            # 2. The activity floor rejected three of six gates on this run, including
            #    a 19,793 and a 10,018, while the historical relationship it encodes
            #    (money rises with trading rate) has inverted here -- money is at a
            #    repo record while greedy trading sits at 0.15-0.30.
            (
                '"viability_min_orders_per_decision": 0.20,',
                '"viability_min_orders_per_decision": 0.02,',
            ),
            # 3. The ratchet bar is the best of the last 25 releases, so one lucky
            #    100-seed draw becomes a wall for the rest of the run. 3 keeps a
            #    non-regression check without letting a spike freeze the arm.
            ('"promotion_reference_window": 25,', '"promotion_reference_window": 3,'),
            ('SESSION_ID = "market-free-ruin-v2"', 'SESSION_ID = "market-free-open-v1"'),
            ('SLOT_PREFIX = "ruin2"', 'SLOT_PREFIX = "open"'),
        ],
    },
}


def build(target: str, spec: dict, source: str) -> int:
    generated = source
    for old, new in spec["subs"]:
        if generated.count(old) != 1:
            print(
                f"{target}: expected exactly one occurrence of {old!r}, "
                f"found {generated.count(old)}",
                file=sys.stderr,
            )
            return 1
        generated = generated.replace(old, new)
    body = generated[generated.index("\nfrom __future__ import annotations") :]
    (ROOT / target).write_text("#!/usr/bin/env python3\n" + spec["banner"] + body)
    print(f"regenerated {target}")
    return 0


def main() -> int:
    source = SOURCE.read_text()
    return max(build(target, spec, source) for target, spec in ARMS.items())


if __name__ == "__main__":
    raise SystemExit(main())
