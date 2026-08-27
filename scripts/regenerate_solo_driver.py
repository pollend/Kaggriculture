#!/usr/bin/env python3
"""Rebuild `train_market_free_v3d_solo.py` from `train_market_free_v3d.py`.

The two drivers are an A/B whose only intended difference is the pinned
demonstration pack, which `tests/test_dqn_market_free_v3d.py::
test_the_two_arms_differ_only_in_the_pack` enforces on `FREE_CONFIG`. Keeping that
true by editing both files by hand failed twice in one session -- once leaving the
solo arm on a stale roster, once on a stale opponent mix -- so the solo arm is
generated instead. Edit the blended driver, run this, commit both.

Three substitutions, and deliberately no more: any other divergence between the arms
is a bug in the experiment rather than a setting.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "train_market_free_v3d.py"
TARGET = ROOT / "train_market_free_v3d_solo.py"

SUBSTITUTIONS = [
    (
        '"demonstration_pack": str(ARTIFACTS / "demonstrations" / "ladder-top25-296")',
        '"demonstration_pack": str(ARTIFACTS / "demonstrations" / "kawashigi-296")',
    ),
    ('SESSION_ID = "market-free-ruin-v1"', 'SESSION_ID = "market-free-ruinpack-v1"'),
    ('SLOT_PREFIX = "ruin"', 'SLOT_PREFIX = "ruinpack"'),
]

BANNER = '''#!/usr/bin/env python3
"""Single-player demonstration arm. GENERATED -- do not edit.

Rebuild with `python scripts/regenerate_solo_driver.py` after editing
`train_market_free_v3d.py`. The A/B is the pinned pack and nothing else.

`demonstration_episodes` is left at the blended arm's value on purpose. The packs are
different sizes -- `ladder-top25-296` holds 525 seats from 98 players, `kawashigi-296`
holds 144 from one -- and `ladder.stratified_indices` returns everything a pack has
when the limit exceeds it, so this arm pins 144 against the other's 525. That
asymmetry is the experiment: one player deep against the ladder's top quarter broad.
"""
'''


def main() -> int:
    source = SOURCE.read_text()
    generated = source
    for old, new in SUBSTITUTIONS:
        if generated.count(old) != 1:
            print(
                f"expected exactly one occurrence of {old!r} in {SOURCE.name}, "
                f"found {generated.count(old)}",
                file=sys.stderr,
            )
            return 1
        generated = generated.replace(old, new)

    body = generated[generated.index("\nfrom __future__ import annotations") :]
    TARGET.write_text(BANNER + body)
    print(f"regenerated {TARGET.name} from {SOURCE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
