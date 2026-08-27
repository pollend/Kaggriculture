"""How many hands the field can actually use, from the jobs it has.

`donor._target_hands` is pilkwang's rule for pilkwang's farm: it floors at 10 hands
whenever any role is active and the day is 27 or earlier, on the assumption that a
board full of cows and melons is coming. That floor is what the market head saw as
a state feature and what the `HIRE` rows let it act on, and on a farm that never
got the cows it is a wage bill without a crew: measured on `market-free-solo-v1`
generation 10, the learner spent 18,300 coins an episode on hire -- 9 to 16 hands
a day at the fibonacci prefix, 609 coins at 13 -- against 41,000 gross sales, and
its hands passed on 55-66% of their turns. Ladder winners spend 13,700 on the same
head count and sell 172,000, because their hands have work.

`hire_demand` is the donor's own demand term with the floor removed: the jobs due
now plus the risk load, at seven jobs a hand. `hire_cap` adds a small slack so a
turn that buys seed can also hire the hands to plant it -- the job list is read at
the turn's opening board, before this turn's purchases are in the shed -- and is
what `order_space.feasible_actions` shuts the `HIRE` rungs above.

This is a policy cap, not a feasibility one, and `feasible_actions`' preamble says
such caps do not belong there. It goes in anyway, for the reason the reserve floor
already does: a cash charge with no term for an idle hand cannot teach the
difference between a hand that harvests and one that stands in the field, and
with 24-turn n-step credit the harvest a hand would have brought is outside the
window the wage is charged in.
"""

from __future__ import annotations

import math

from agents.dqn_market_free_v3d import donor

#: Jobs one hand clears in a day, the donor's own divisor in `_target_hands`.
JOBS_PER_HAND = 7

#: Hands a turn may hire beyond what the opening board's jobs justify. Two, not
#: zero: the job list is read before this turn's seed purchases exist, and a plant
#: job needs the seed in the shed, so the hands to plant a turn's own purchase would
#: otherwise be one turn late -- and a hand hired late has lost part of its shift.
HIRE_SLACK = 2

#: The donor's own lower bound on a crew, kept as the floor here so a farm between
#: jobs can still water and dig.
MIN_HANDS = 4


def hire_demand(obs: dict, farm: dict, private: dict, roles: dict) -> int:
    """Hands the field's due jobs and at-risk stock justify, without the floor."""

    day = int(obs.get("day", 0) or 0)
    try:
        summary = donor._survey(farm, private, roles, day)
        due_jobs = len(donor._field_jobs(obs, farm, private, roles, liquidation=False))
    except Exception:
        # `feasible_actions` never raises; a malformed board justifies nothing.
        return MIN_HANDS
    risk_load = 2 * (summary["at_risk_animals"] + summary["at_risk_crops"])
    demand = int(math.ceil((due_jobs + risk_load) / float(JOBS_PER_HAND)))
    return max(MIN_HANDS, min(int(donor.MAX_HANDS), demand))


def hire_cap(obs: dict, farm: dict, private: dict, roles: dict) -> int:
    """Most hands a turn may end the day with: demand plus the planting slack."""

    return min(int(donor.MAX_HANDS), hire_demand(obs, farm, private, roles) + HIRE_SLACK)
