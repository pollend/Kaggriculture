"""Reward and its scales for the free-market learner.

Local rather than shared with `dqn_market_scorer`: the scorer's market layer
decides land and cash reserves in code, so sharing the constants would mean
changing the scorer's reward to fix this one.

The reward is the episode's own outcome -- the tanh of the log growth on its own
stake -- paid twice: a lump on the terminal step and a per-step annuity of
`(1 - gamma) * outcome`. The two sum to a flat `outcome` in Q at every timestep,
so a state on a winning trajectory is valued above the same-looking state on a
losing one no matter when it was seen.

Three per-turn charges sit alongside it, ordered deliberately:

    accepted order    0
    NOOP             -IDLE_PENALTY
    refused order    -REJECTED_ORDER_PENALTY

so trading beats sitting out and sitting out beats emitting junk.

Two of the remaining terms are potential-based -- `LAND_POTENTIAL_WEIGHT` and
`LOT_POTENTIAL_WEIGHT` -- so they are policy-invariant by construction and can only
move *when* credit arrives, never which policy wins. Everything else is deliberately
not: the three charges above, `LAND_PURCHASE_BONUS`, and the two pairs that re-inject
the money delta this stream dropped -- `SEED_WASTE_WEIGHT` with `SEED_GAIN_WEIGHT`
for a seed lot, `ANIMAL_WASTE_WEIGHT` with `ANIMAL_GAIN_WEIGHT` for a bought animal,
each a lot's realised loss and its realised profit paid back onto the turn the
purchase was made.

The animal pair has no potential beside it, where the seed pair has
`LOT_POTENTIAL_WEIGHT`. That is not an omission. An animal's cost basis is a function
of `(owned, day)`, and `owned` moves only on a filled `BUY_ANIMAL` while `day` moves
on the clock, so such a potential would be action-independent on every other turn --
a per-state constant contributing nothing to `argmax_a Q(s, a)`, and on the buy turn
still an invariant term that cancels out of every difference. Only the two legs below
can make the policy buy an animal or decline to.

Reading the sizes: an invariant term cancels out of `Q(s, a) - Q(s, a')` entirely,
so it never offsets a non-invariant one no matter what the immediate rewards do on a
given index. Every constant below states its strength in the second sense, because
that is the one the policy ranks on.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from agents.dqn_market_free_v3 import donor
from agents.dqn_market_free_v3.market_state import order_refused
from agents.dqn_market_free_v3 import lots

#: Reward units per coin, sized so a whole episode's return lands in single
#: digits. `SmoothL1Loss` has a beta of 1.0, so a larger scale puts every TD error
#: in the linear regime where a 5-unit and a 50-unit error give the same gradient.
#:
#: Scaled rather than clipped. Per-step clipping to [-1, 1] is the usual DQN answer
#: and is wrong here: the magnitude of a fill *is* the signal.
MONEY_SCALE = 20_000.0

#: Natural-log units of growth at which the outcome term reaches tanh(1) = 0.76.
#:
#: Growth on the seat's **own stake**, not the margin against the opponent, and
#: dropping the opponent is what lets one constant do a job two could not. A log
#: ratio is scale-free, so unlike `tanh(margin / S)` it has no saturation point to
#: place between the mirror's margins in the hundreds and a scripted field's in the
#: hundred thousands:
#:
#:     final money  3,000   6,000   12,000   40,000   142,000   500,000
#:     outcome       0.00    0.14     0.27     0.48      0.65      0.77
#:
#:   * Opponent-independent, so mirror, scripted opponent and benchmark all score
#:     the same way.
#:   * The quantity the promotion gate already reads -- training on margin while
#:     gating on money is what put the previous constant in an impossible position.
#:   * Sized on headroom, not on a measurement: every earlier value was sized on the
#:     distribution the run happened to produce and was saturated by the next
#:     improvement.
#:
#: The cost is at the bottom: break-even against a fully idle season is 2,092 coins.
#: If `trading_rate` climbs while money falls, re-read this ratio first, not
#: `IDLE_PENALTY`.
OUTCOME_LOG_SCALE = 5.0

#: Weight on the opponent-relative leg of the outcome. Held at 0.0, landed inert --
#: a test asserts the stream is bit-identical to the absolute one.
#:
#: "The market is shared, so train on the margin" is the intuition that produced
#: every failure `OUTCOME_LOG_SCALE` records, and it is wrong for a precise reason:
#: **the opponent's effect on us is already in our own money.** One market, and
#: every unit either seat sells depresses the price for both. What absolute money
#: cannot see is the reverse -- denial, dumping at a loss to crash a quote before
#: their stock lands -- and that is the entire content of this leg.
#:
#: Denial is out of reach here and measurably absent: `corr(final_money,
#: opponent_money)` is 0.006, so nothing cancels and a difference costs a factor of
#: eight in per-episode signal-to-noise for a behaviour the policy cannot perform.
#:
#: Turn it on when that correlation departs from zero. 0.1 is the value to try -- a
#: tiebreaker at +/-0.1 against the absolute leg's +/-1.0. The mirror is gated off
#: at the caller regardless of this value: there the opponent is a checkpoint of the
#: same slot, so its purse carries this policy's own variance and nothing else.
OPPONENT_WEIGHT = 0.0

#: Multiplier on the per-step outcome annuity.
#:
#: The terminal lump lands on one turn of ~720 and the buffer samples 24-turn
#: windows, so only ~3% of transitions ever see it. Paying `(1 - gamma) * outcome`
#: on every step as well makes the outcome's contribution to Q flat at `outcome`:
#:
#:     sum_{k>=t} gamma^(k-t) * (1-gamma) * outcome  +  gamma^(T-1-t) * outcome
#:       = outcome * (1 - gamma^(T-t) + gamma^(T-1-t))  ~=  outcome
#:
#: Scaled by `(1 - gamma)` rather than `1 / T`. An even spread leaves Q decaying
#: from ~0.71 * outcome to nothing, which is *less* late-game signal than the
#: terminal lump alone, and it fails to vanish at gamma = 1 where nothing needs
#: fixing.
OUTCOME_ANNUITY_WEIGHT = 1.0

#: Charged on a turn that emitted an order the simulator then refused.
#:
#: Every action scores the same on the turn it happens -- the stream pays only the
#: annuity -- so an untrained value function sees a flat plateau and picks off it
#: arbitrarily. This separates "interacted with the world" from "did not" without
#: expressing a view on *which* trade is good: an impossible order falls below
#: NOOP, a real purchase rises above it, and the choice among real purchases is
#: untouched. Not policy-invariant, so it is kept small.
REJECTED_ORDER_PENALTY = 0.001

#: Charged on every turn the policy emits no order at all.
#:
#: Without it nothing discourages sitting still: `REJECTED_ORDER_PENALTY` needs an
#: emission and `INSOLVENCY_PENALTY` needs a spent wallet, so a NOOP turn paid
#: neither and idling was weakly dominant. That is the NOOP attractor this family
#: has collapsed into repeatedly, and everything opposing it used to live outside
#: the reward, so the machinery was fighting the objective instead of stating it.
#:
#: Three constraints on the shape and the size:
#:
#:   * Decays linearly to zero at the deadline, so a fully idle season pays ~0.072.
#:     An idle turn on day 1 forfeits a season of compounding; one on day 29
#:     forfeits almost nothing.
#:   * **Predictable from the state.** `day / total_days` is already a feature, so
#:     the agent can see this turn's charge. A streak or cumulative-count curve
#:     would vary it on something the state does not carry -- a new feature, a
#:     changed `MARKET_STATE_DIM`, and every checkpoint invalidated.
#:   * Sized against the *outcome term's* range, not against the other charges.
#:     Getting that backwards is what the previous value did, making a trade down to
#:     1,898 coins score better than sitting on 3,000.
#:
#: Strictly below `REJECTED_ORDER_PENALTY` at t=0, so a policy that cannot afford
#: anything is not pushed into emitting junk to escape a charge.
IDLE_PENALTY = 0.0002

#: Coins below which the farm has no runway left. The cheapest thing worth buying
#: is a seed and the cheapest hand is one coin, so under a couple of hundred coins
#: `feasible_actions` has already closed almost every column.
INSOLVENCY_FLOOR = 250.0

#: Charged on every turn the farm spends below the floor.
#:
#: Bankruptcy is an *opportunity* cost: the wallet cannot go negative, so spending
#: the last of the stake reads as one small debit and everything it forecloses
#: arrives only as the absence of later fills. Recovering that through the bootstrap
#: means crossing up to 720 turns in 24-turn hops through a buffer where nearly
#: every state is the dead absorbing one. This charges for the state directly.
#:
#: Not policy-invariant -- it is meant to change which policy is optimal -- and it is
#: **not small**: a season spent below the floor past the grace costs about 1.39,
#: against an `outcome` that tops out at 0.77 for a 167x season. That is deliberate
#: and pinned by `test_the_insolvency_charge_is_the_scale_of_the_outcome_not_of_a_fill`
#: -- going broke has to cost about what losing costs, or the network optimises the
#: floor instead of the money. What keeps it honest is not its size but its shape: it
#: states a *floor* and nothing else, so it cannot rank two solvent policies against
#: each other.
INSOLVENCY_PENALTY = 0.002

#: Consecutive turns below the floor before the charge starts billing.
#:
#: The charge above prices a *permanent* absorbing state. Without a grace it also
#: prices the wallet dipping under the floor between a purchase and the sale that
#: funds it, which inverts the constant: `d(outcome)/d(log money)` is at most 1/5, so
#: doubling a season is worth `tanh(ln 2 / 5)` = +0.138 while the ~120 low-cash turns
#: it takes to buy the inputs cost -0.240. Doubling your money netted **-0.10**.
#:
#: (An earlier revision put +0.24 on that line and concluded the two cancelled
#: exactly. That figure is `ln(2) / 3`, left from `OUTCOME_LOG_SCALE = 3.0`; at 5.0
#: the charge outweighs the doubling by 1.7x, which sharpens this argument rather
#: than softening it. Break-even is 69 below-floor turns, so the grace is covering a
#: real gap and does not have much room to spare.)
#:
#: 24 turns, one in-game day: long enough that a purchase and its funding sale are
#: one event, short enough that a farm with no runway is still billed for ~696 turns
#: of a 720-turn season.
#:
#: A streak rather than a cap on the episode total. A cap prices every dip and then
#: stops counting; this prices no dip and all of the absorbing state, which is what
#: the constant was bought for. 0 or 1 restores the per-turn charge exactly.
INSOLVENCY_GRACE_TURNS = 24

#: Weight on the unlocked-land potential.
#:
#:     PHI(s) = w * extra_quadrants(s) * max(0, TOTAL_DAYS - day(s)) / TOTAL_DAYS
#:
#: **Policy-invariant by construction**, which is the only reason it is allowed to
#: exist -- everything above deliberately changes which policy is optimal. Ng,
#: Harada & Russell: `F(s, s') = gamma * PHI(s') - PHI(s)` leaves the optimum
#: unchanged for any `PHI`, so this can only change *when* credit arrives.
#:
#: What it is for: a plot bought on day 5 pays back through harvests from day 15,
#: which reaches the purchase decision only through the terminal outcome, while the
#: 1,000-coin cost lands immediately. The policy learns the cost and not the return.
#: If land really is bad, invariance means this term cannot overturn that.
#:
#: Proportional to remaining usage, which also makes `PHI` vanish at the terminal
#: state for free -- the condition the telescoping needs. 0.08 is sized against the
#: outcome term, so one quadrant taken at the start is worth about what a 25% season
#: is worth. Sizing against `LAND_PRICES` was the alternative and is worse: it makes
#: the payout largest exactly where the purchase is most dangerous.
#:
#: The old objection to potential shaping here was that it marked the shed to a
#: quote the policy could pump. `PHI` reads a monotone integer bought from a fixed
#: price ladder and never sold, times an exogenous clock. There is no cycle to pump.
#:
#: 0.0 removes the term, as does `quadrants=None`.
LAND_POTENTIAL_WEIGHT = 0.08


#: Paid once on the turn a quadrant actually appears, on top of the potential above.
#:
#: **Zero since v43, and the reason is that there is no longer a decision to pay
#: for.** This is the one land term that is *not* potential-based, and that was the
#: whole point of it: an invariant term can move when credit arrives and can never
#: make an argmax pick land, so this existed to tilt the argmax. `BUY_LAND` has left
#: the head -- `order_space.land_score` decides it and `choose_order` never reads
#: the row -- so the tilt now lands on a transition the policy does not choose. What
#: it would still do is bias Q on the states *around* a purchase, in a direction
#: nothing can act on.
#:
#: v33 measured it working at 0.02 and 0.035, and that measurement is not retracted:
#: it stopped land being unlearned, holding a 0.13-0.16 buy rate for 70 batches
#: where v25-v27 all decayed to 0.005-0.01, with 14.2 of 25 tiles worked per
#: purchase against 8.7. The term is retired because its job moved, not because it
#: failed at it.
#:
#: Kept as a constant rather than deleted: it is the natural revert if
#: `LAND_BUY_SET_POINT` goes back to `None` and land returns to the argmax. The old
#: sizing was 0.035, read as half a fully idle season's charge, scaled by price so
#: the 2,000 and 4,000 quadrants are not overpaid -- the bonus is this constant
#: times `LAND_PRICES[0] / price`. `shaped` never carries the money delta, so it is
#: unopposed on the purchase turn.
LAND_PURCHASE_BONUS = 0.0


#: Weight on the cost basis a seat has committed to growing something.
#:
#:     PHI(s) = w * (held cost basis in coins) / MONEY_SCALE
#:
#: counting a purchased seed at its price, a standing plant at the price of the seed
#: that became it, and a harvested unit at that price over the crop's `max_yield`.
#:
#: What it is for: nothing else distinguishes a purchase from a NOOP on the turn it
#: happens, and the only term with an opinion about whether the seed was worth
#: buying is the terminal outcome -- 48 turns away for wheat, 240 for the melon
#: `planting.py` optimises for, against `n_step = 24`. This puts a credit on the buy
#: turn itself.
#:
#: Three things to keep right if this is ever re-derived:
#:
#:   * **Potential-based.** The shaping added to return-to-go from any state is
#:     exactly `-PHI(s)`, a constant of the state, so Q ordering is preserved
#:     everywhere -- and by the same token this term can never *oppose* a
#:     non-invariant one, because `-PHI(s)` cancels out of every `Q(s, a) - Q(s, a')`.
#:     The visible consequence is that a **sale** carries the largest negative
#:     immediate reward in the file: six melons leaving the shed is `-80 /
#:     MONEY_SCALE`, 20x a NOOP, and the proceeds that repay it are not in the
#:     stream. Invariance is what makes that safe, exactly for the Q being regressed
#:     on and not for any single 24-turn window read on its own.
#:     **Do not go back to a reallocation.** The version this replaced
#:     moved a lot's proceeds from the sale turn to the buy turn. The episode total
#:     was preserved, which is what made it look safe, but a preserved total is not
#:     a preserved ordering: at the sale the debit was entirely in the future so
#:     return-to-go fell by the full amount, while at the buy it was `D` turns away
#:     so return-to-go rose by only `coins * (1 - gamma**D)` -- 0.214 at `D = 240`.
#:     It deterred selling five times harder than it encouraged buying.
#:   * **Cost basis, never market price.** Every price is a fixed constant out of
#:     `donor.CROPS` and nothing in `PHI` reads `obs.market`, so there is no quote
#:     to pump. Telescoping forbids it anyway.
#:   * **No equivalent for `HIRE`.** Hands clear every night, so a hire's whole
#:     working life fits inside one n-step window and there is no distance to close.
#:     Labour is also a shared input across every live lot, so a per-hire basis
#:     would be an invented allocation rather than a reconstruction.
#:
#: Sized so one seed is visible against the charges it competes with: an 80-coin
#: melon seed moves `PHI` by 0.004 on the purchase turn, 20x `IDLE_PENALTY`, and a
#: fully planted melon quadrant costs `(1 - gamma) * PHI` a turn to sit on -- 4e-6 per
#: seed at `gamma = 0.999`. A real holding cost, far below the idle charge, and the
#: correct sign; `SEED_WASTE_WEIGHT` and `SEED_GAIN_WEIGHT` below exist because 4e-6
#: a turn is too diffuse to reach a 24-turn window, and because an invariant term
#: cannot express a preference about a purchase in the first place.
#:
#: The failure mode is that it does nothing; invariance forbids inflation and pumps.
#: Watch `seed_holding_turns` falling while `seed_margin` holds. `seed_spend`
#: climbing with `unrecovered_seed_share` is the policy buying more rather than
#: earlier, and `LAND_PURCHASE_BONUS` is the constant that would cause that.
#:
#: 0.0 removes it, as does any of the three traces being absent.
LOT_POTENTIAL_WEIGHT = 1.0


#: Weight on a seed lot's unrecovered cost, charged back onto its purchase turn.
#:
#:     shaped[buy_turn] -= w * max(0, lot.cost - lot.proceeds) / MONEY_SCALE
#:
#: over `lots.LotLedger.unrecovered()`, seed lots only. `SEED_GAIN_WEIGHT` is the
#: other half of the pair and the two are meant to be read together.
#:
#: **Not policy-invariant, and that is the point** -- same footing as
#: `LAND_PURCHASE_BONUS`. What it fixes: `shaped` never carries the money delta, so a
#: purchase is *unopposed*. The only thing landing on a buy turn is
#: `LOT_POTENTIAL_WEIGHT`'s credit, `+gamma * cost / MONEY_SCALE`, and nothing charges
#: for the coins. The potential does claw a wasted seed back, but as a
#: `(1 - gamma) * PHI` drip of 4e-6 a turn plus a terminal lump -- neither inside
#: `n_step = 24` against a 720-turn season. Buying seed and never selling it was
#: locally the cheapest action available.
#:
#: **The deterrent is the full `cost / MONEY_SCALE`.** This was previously documented
#: as an anchor rather than a tuning, on the grounds that the potential pays
#: `+gamma * cost` on the same index so a wasted lot nets `-(1 - gamma) * cost`, about
#: -4e-6. That is the immediate-reward reading of an invariant term, and it is not
#: what the policy ranks on: a potential contributes exactly `-PHI(s)` to `Q(s, a)`
#: for every `a`, so it cancels out of `Q(s, buy) - Q(s, noop)` and cannot offset a
#: term that does not. The same docstring said as much two paragraphs up -- the
#: potential's clawback is too diffuse to reach `n_step` -- and both cannot be true.
#: Read the size off the coins instead:
#:
#:     lot                     charge   = idle turns   break-even P(waste) vs NOOP
#:     1 wheat seed   (10)     0.0005            2.5                          40%
#:     1 melon seed   (80)     0.0040             20                           5%
#:     10 melon seeds (800)    0.0400            200                         0.5%
#:
#: 0.04 on one transition is 55% of a whole idle season. So `w = 1.0` *is* a tuning,
#: and the one it makes is "a wasted coin costs a coin" -- defensible, but only
#: alongside `SEED_GAIN_WEIGHT`. On its own this leg is a pure downside tax at those
#: sizes, which is a NOOP attractor and not, as the anchor argument concluded,
#: incapable of being one.
#:
#: **This is not the reallocation the `LOT_POTENTIAL_WEIGHT` note bans.** That version
#: moved a lot's *proceeds* from the sale turn to the buy turn, so the debit was full
#: strength at the sale and discounted by `1 - gamma**D` at the buy, deterring selling
#: five times harder than it encouraged buying. This adds nothing at the sale turn at
#: all; there is no counterpart to invert.
#:
#: **It does not make the outcome term redundant.** The flows do not sum to final
#: money -- nothing anywhere carries HIRE wages, land outlay or shed overflow, and the
#: wheat a herd eats is billed to the wheat's own lot rather than the animal's.
#: `outcome` is also compressive and
#: scale-free where summed coins are linear, and its annuity is the only thing making
#: the contribution to Q flat across timesteps. The signs need both: without `outcome`
#: a fully wasted buy still beats NOOP by the `IDLE_PENALTY` it avoided.
#:
#: Known bias, deliberate: a lot wasted because labour ran out on day 20 charges the
#: day-3 buy. The buy state carries cash, land and hand features, so "do not buy what
#: you cannot work" is learnable, but the blame does land upstream of the mistake.
#: More generally the charge is a fact about the rest of the episode paid at a turn
#: whose state cannot see it, so two identical buy states score differently and the
#: network can only fit the mean. That is the intent -- it is a Monte Carlo estimate
#: of "did this seed pay off" delivered where the decision was made -- at the cost of
#: target variance on buy turns.
#:
#: **0.35 rather than 1.0, and the two legs are deliberately not equal.** The
#: symmetry argument above -- each lot clamped to +/- its own cost, so the pair
#: cannot exceed `seed_spend` either way -- is exact per lot and does not survive
#: aggregation, because `lots.match` is FIFO over fungible units. FIFO hands each
#: sale to the oldest lot with room, so on a season whose seeds returned 2.57x the
#: proceeds concentrate in the lots that happen to be early: their credit is clipped
#: at `+cost` while every later lot is charged the full `-cost`. The sign of the
#: season's total then follows an accounting convention rather than the policy.
#:
#: Measured over 20 stored episodes, per episode, `gain_weight = 1.0` throughout:
#:
#:     waste   mean     worst    corr(outcome)
#:      1.00   -0.086   -0.788      +0.588
#:      0.50   -0.027   -0.348      +0.619
#:      0.35   -0.010   -0.215      +0.630
#:      0.25   +0.002   -0.127      +0.617
#:
#: At 1.0 the pair was a spend-proportional tax averaging -0.086 against an
#: `outcome` whose whole range is [-0.92, +0.42] on the same sample -- a NOOP
#: attractor by exactly the argument two paragraphs up, and one episode's -0.788
#: exceeded the outcome term outright. 0.35 is where the mean is neutral, the tail
#: sits inside the outcome's range instead of past it, and the correlation with
#: `outcome` is at its maximum, so this is a de-biasing rather than a weakening.
#:
#: Pro-rata allocation was the other candidate and is worse: it removes more of the
#: bias (mean -0.016) but drops `corr(outcome)` to +0.222 and pays a *positive*
#: credit to a season that ended on 56 coins. FIFO keeps the per-lot resolution;
#: this constant pays for the bias it comes with.
#:
#: Watch `seed_waste_cost` against `seed_gain_credit` per batch -- the coins each
#: leg actually moved, which `unrecovered_seed_share` cannot show because it is an
#: aggregate ratio while the reward bills per lot. If `trading_rate` collapses,
#: lower this before touching `SEED_GAIN_WEIGHT`, and do not touch `IDLE_PENALTY`.
#: 0.0 removes the term, as does `ops=None`.
SEED_WASTE_WEIGHT = 0.35


#: Weight on a seed lot's realised profit, credited back onto its purchase turn.
#:
#:     shaped[buy_turn] += w * min(lot.cost, max(0, lot.proceeds - lot.cost)) / MONEY_SCALE
#:
#: over `lots.LotLedger.surplus()`, seed lots only. The mirror of
#: `SEED_WASTE_WEIGHT` in every respect: not policy-invariant, paid on the buy turn,
#: gated on the same `ops` trace, inert at 0.0.
#:
#: **Why it has to exist.** `unrecovered()` clamps at zero per lot, so the charge
#: alone scores a lot sold at 1.1x cost and one sold at 6x cost *identically*. Profit
#: then reaches the buy decision only through the terminal `outcome`, up to 720 turns
#: away against `n_step = 24` -- the same distance `SEED_WASTE_WEIGHT` exists to
#: close, approached from the other side and left open. The consequence is not
#: neutral, it is a specific wrong preference: with only the loss leg the
#: risk-adjusted local optimum is the cheapest seed the policy is confident it can
#: recover, so wheat at 10 coins and 48 turns dominates the melon at 80 coins and 240
#: that `planting.py` optimises for -- their expected credit is identical at zero and
#: only their expected charge differs.
#:
#: **Capped at the lot's own cost**, in `surplus()`, which is what lets `w = 1.0`
#: stand. Bounded both ways, the pair sums to at most `+/- seed_spend / MONEY_SCALE`
#: over a season -- 0.25 on a 5,000-coin seed programme, the same order as the
#: outcome term rather than several times it. Uncapped, a melon lot sold at six times
#: cost would pay five times what wasting it charges, the season's seed P&L would run
#: to several units, and a linear-coin term would quietly replace the compressive
#: `outcome` as the objective. That is the failure the cap is bought against, and it
#: is why this is not simply `seed_margin` paid back.
#:
#: **It is still not the reallocation `LOT_POTENTIAL_WEIGHT` bans**, for the same
#: reason the charge is not: nothing lands on the sale turn, so there is no
#: full-strength debit there for a discounted credit at the buy to lose against.
#:
#: **Held at 1.0 while `SEED_WASTE_WEIGHT` came down to 0.35.** The two are still a
#: pair and are still meant to be read together; what changed is that "the pair is
#: symmetric" was only ever true per lot. See `SEED_WASTE_WEIGHT` for the
#: measurement -- FIFO concentrates a profitable season's proceeds in whichever lots
#: are early, this leg clips their credit at `+cost`, and the charge on every lot
#: behind them does not clip. The credit leg is the one being clipped, so it is the
#: charge that moves.
#:
#: Watch `seed_gain_credit` rising while `seed_waste_cost` falls. Both climbing
#: together is volume, not skill; `seed_margin` is the check, being the uncapped
#: version of the same quantity.
SEED_GAIN_WEIGHT = 1.0


#: Weight on a bought animal's unrecovered cost, charged onto its purchase turn.
#:
#:     shaped[buy_turn] -= w * max(0, lot.cost - lot.proceeds) / MONEY_SCALE
#:
#: over `lots.LotLedger.animal_unrecovered()`. `ANIMAL_GAIN_WEIGHT` is the other half
#: and the two are meant to be read together, exactly as the seed pair is.
#:
#: **Why animals need their own pair.** The distance is worse than the seed one, not
#: better: a cow bought on day 3 first milks on day 11, roughly 200 turns, and then
#: sells for the rest of the season -- against `n_step = 24`. Nothing was opposing the
#: outlay at all before this. `shaped` never carries the money delta, `lots.py` opened
#: no lot for an animal, and `_UNIT_BASIS` is zero for MILK and WOOL, so a 400-coin
#: cow was free on the turn it was bought and its milk was invisible on every turn
#: after. `donor.TARGET_HERD` is 15, so that is up to 6,500 coins the stream had no
#: opinion about.
#:
#: What it catches that no other term does:
#:
#:   * an animal bought and never placed -- it sits in the shed, produces nothing, and
#:     consumes shed slots the produce needed;
#:   * an animal bought past its deadline. A cow bought after day 21 and a sheep after
#:     day 23 cannot yield once before the season ends, and `order_space` has no day
#:     gate on `BUY_ANIMAL` -- only shed room and cash -- so the action is offered.
#:
#: Both land as a lot opened with no capacity, matched against nothing, charged in
#: full on the buy turn.
#:
#: **0.25, against the seed leg's 0.35.** Read the sizes off the coins:
#:
#:     lot                  charge   = idle turns   % of a fully idle season
#:     1 cow   (400)        0.0050            25                        6.9%
#:     1 sheep (500)        0.0063            31                        8.7%
#:     herd of 15, all wasted             0.075..0.094
#:
#: 0.0050 is one melon seed's charge at `w = 1.0`, a size this file already sanctions
#: as visible but not dominant, and eight times below the 0.04 the `SEED_WASTE_WEIGHT`
#: docstring identifies as a NOOP attractor. The all-wasted tail sits inside
#: `outcome`'s measured range of [-0.92, +0.42] rather than past it, which is the test
#: that constant was de-biased against.
#:
#: **The ratio to `ANIMAL_GAIN_WEIGHT` is the argument, not the levels.** Break-even
#: probability of waste against NOOP is `w_waste / (w_waste + w_gain)` = 0.25 / 0.95 =
#: 0.263, and the seed pair's is 0.35 / 1.35 = 0.259. The two families state the same
#: prior about how often a purchase has to work out to be worth making; that is why
#: this is a port of a tuning rather than a new one.
#:
#: **The credit leg leads for the same reason it does on seeds, and worse.** FIFO hands
#: each sale to the oldest lot with room, and every COW lot shares one `queues["MILK"]`
#: where five crops split over five queues, so on a profitable herd the proceeds
#: concentrate in whichever cows were bought first: their credit clips at `+cost` while
#: every cow behind them takes the full charge. Concentration is the reason, not the
#: cap -- an animal's gross multiple over its basis is 4-5x where a melon's is 19x, so
#: the cap actually binds *less* here.
#:
#: These two constants have not been through the 20-episode mean/worst/`corr(outcome)`
#: sweep the seed pair was chosen by. They ship on the break-even argument; run that
#: sweep before treating them as measured. Watch `animal_waste_cost` against
#: `animal_gain_credit` per batch. 0.0 removes the term, as does an absent animal trace.
ANIMAL_WASTE_WEIGHT = 0.25


#: Weight on a bought animal's realised profit, credited onto its purchase turn.
#:
#:     shaped[buy_turn] += w * min(lot.cost, max(0, lot.proceeds - lot.cost)) / MONEY_SCALE
#:
#: over `lots.LotLedger.animal_surplus()`. The mirror of `ANIMAL_WASTE_WEIGHT` in every
#: respect, capped at the lot's own cost so the pair is bounded by the season's herd
#: spend either way: 0.0140 for a cow, 0.0175 for a sheep, +0.21 to +0.26 over a herd
#: of 15 that all paid off.
#:
#: **Why it cannot be dropped.** `animal_unrecovered()` clamps at zero per lot, so the
#: charge alone scores a cow that broke even and a cow that returned four times its
#: cost identically. With only the loss leg the local optimum is to buy nothing:
#: expected credit is zero for every animal and only the expected charge differs. That
#: is the same wrong preference the seed pair documents, and here it points at the
#: cheaper animal and at no animal at all.
#:
#: **The cap binds nearly always, which is the point.** A cow that gets placed early
#: grosses on the order of 1,600 coins of milk against 400 spent, so the surplus clips
#: at `+cost` for essentially any animal that produced at all. The pair is therefore
#: close to binary -- a full credit if the animal worked, a full charge if it did not
#: -- which is exactly the signal wanted for a decision whose payoff is 200 turns away.
#: Uncapped it would be a linear-coin term several times the size of `outcome`, and
#: `outcome` would stop being the objective.
#:
#: Watch `animal_gain_credit` rising while `animal_waste_cost` falls. Both climbing is
#: herd size, not skill; `animal_margin` is the check, being the uncapped version.
ANIMAL_GAIN_WEIGHT = 0.70


#: Coins of committed cost per unit held, by `lots.STOCK_ITEMS` index.
#:
#: A harvested unit carries `seed / max_yield`, so a maximal harvest conserves `PHI`
#: and a poor one writes the difference off on the turn it lands.
#:
#: **Still zero for EGG, MILK, WOOL and FERTILIZER, now for a different reason.**
#: `lots.py` does match animals, so the basis is no longer unknown -- it is a cow's
#: 400 coins over what the season can still milk out of it. The conserving version of
#: the crop rule would put `cost / remaining_events` on each unit and decrement the
#: animal's own basis as it produces, and that is the exact analogue. It is not
#: usable: it needs the count of *placed* animals, and an animal in the shed is not a
#: feature (`market_state` covers `PRODUCTS` only), so `PHI` would be a function of
#: something the network cannot see. A potential stays invariant either way, but
#: fitting `Q - PHI` where `PHI` is unrepresentable is strictly worse than fitting `Q`.
#: The ledger may use the exact per-tile trace because it is offline accounting; a
#: potential may use only what the state vector carries. Do not unify the two.
_UNIT_BASIS: tuple[float, ...] = tuple(
    float(donor.CROPS[item]["seed"]) / max(1, int(donor.CROPS[item]["max_yield"]))
    if item in donor.CROPS
    else 0.0
    for item in lots.STOCK_ITEMS
)

#: Coins of committed cost per seed held or plant standing, by `CROP_ITEMS` index.
_SEED_BASIS: tuple[float, ...] = tuple(
    float(donor.CROPS[crop]["seed"]) for crop in lots.CROP_ITEMS
)


def _held(
    trace: Sequence,
    width: int,
    span: int,
    fill: float = 0.0,
    columns: int | None = None,
) -> np.ndarray:
    """One per-turn trace widened to `money_array`'s length, holding its last value.

    Padded rather than truncated when the trace is short: holding the last known
    state is the reading that cannot invent a purchase, and trailing entries are
    pinned to the terminal zero anyway.

    `span` is passed in rather than derived per trace because a group is padded in
    lockstep -- `quadrants` and `days` share one span, and so do the three lot
    traces, so a short member cannot slide the others' indexing.
    """

    shape = (width,) if columns is None else (width, columns)
    padded = np.full(shape, float(fill), dtype=np.float64)
    if span:
        padded[:span] = np.asarray(trace[:span], dtype=np.float64)
        padded[span:] = padded[span - 1]
    return padded


def _telescope(potential: np.ndarray, gamma: float) -> np.ndarray:
    """`F(s, s') = gamma * PHI(s') - PHI(s)` over a trace one entry longer than `raw`.

    One function because a sign slip here is silent: the stream still trains, just on
    a different objective than the constant claims.
    """

    return np.float32(gamma) * potential[1:] - potential[:-1]


def land_potential(
    quadrants: Sequence[int],
    days: Sequence[int],
    weight: float = LAND_POTENTIAL_WEIGHT,
    total_days: int = donor.TOTAL_DAYS,
) -> np.ndarray:
    """`PHI` over one episode's land trace, terminal entry forced to zero.

    Returned one entry longer than the reward stream, indexed like `money_array`,
    so the shaping is a plain difference over it.

    The terminal zero is set rather than derived: `PHI` reaches zero on its own only
    if the last recorded day is `total_days`, and a trace ending early would leave a
    residual lump the telescoping never repays.
    """

    extra = np.maximum(0.0, np.asarray(quadrants, dtype=np.float64) - 1.0)
    remaining = np.clip(
        1.0 - np.asarray(days, dtype=np.float64) / max(1, int(total_days)), 0.0, 1.0
    )
    potential = float(weight) * extra * remaining
    potential[-1] = 0.0
    return potential


def lot_potential(
    seeds: Sequence[Sequence[int]],
    standing: Sequence[Sequence[int]],
    stock: Sequence[Sequence[int]],
    weight: float = LOT_POTENTIAL_WEIGHT,
) -> np.ndarray:
    """`PHI` over one episode's held-cost-basis trace, terminal entry forced to zero.

    All three traces arrive already padded to `money_array`'s width, so neither
    potential term invents its own indexing convention.

    Here the terminal zero agrees with the simulator rather than closing a residual:
    unsold stock, standing crops and unplanted seed all liquidate to nothing at the
    last step, where the reward is `farm["money"]` and only that.
    """

    seed_basis = np.asarray(_SEED_BASIS, dtype=np.float64)
    unit_basis = np.asarray(_UNIT_BASIS, dtype=np.float64)
    held = (
        np.asarray(seeds, dtype=np.float64) @ seed_basis
        + np.asarray(standing, dtype=np.float64) @ seed_basis
        + np.asarray(stock, dtype=np.float64) @ unit_basis
    )
    potential = float(weight) * held / MONEY_SCALE
    potential[-1] = 0.0
    return potential


def episode_rewards(
    money: list[float],
    final_money: float,
    opponent_money: float,
    gamma: float,
    emitted: Sequence[bool] | None = None,
    #: What each turn's order list was quoted at, from `PendingTurn.expected_delta`.
    #: Optional: without it the refusal test falls back to v1's zero-delta rule, which
    #: is what a demonstration read out of a replay can offer.
    expected: Sequence[float] | None = None,
    opponent_weight: float = OPPONENT_WEIGHT,
    rejected_penalty: float = REJECTED_ORDER_PENALTY,
    idle_penalty: float = IDLE_PENALTY,
    insolvency_floor: float = INSOLVENCY_FLOOR,
    insolvency_penalty: float = INSOLVENCY_PENALTY,
    insolvency_grace: int = INSOLVENCY_GRACE_TURNS,
    quadrants: Sequence[int] | None = None,
    days: Sequence[int] | None = None,
    land_weight: float = LAND_POTENTIAL_WEIGHT,
    land_bonus: float = LAND_PURCHASE_BONUS,
    seeds: Sequence[Sequence[int]] | None = None,
    standing: Sequence[Sequence[int]] | None = None,
    stock: Sequence[Sequence[int]] | None = None,
    lot_weight: float = LOT_POTENTIAL_WEIGHT,
    ops: Sequence[str | None] | None = None,
    #: Per-turn market orders as `(op, item, quantity, cash)`. A v2 turn can hold a
    #: buy and a sale at once, and both legs below are priced off `lots.match`'s
    #: per-turn attribution -- which without this splits one net wallet move between
    #: them and bills a lot that came back. See `lots.match`.
    events: Sequence[Sequence[tuple[str, str | None, int, float]]] | None = None,
    waste_weight: float = SEED_WASTE_WEIGHT,
    gain_weight: float = SEED_GAIN_WEIGHT,
    owned: Sequence[Sequence[int]] | None = None,
    produce_room: Sequence[Sequence[int]] | None = None,
    animal_waste_weight: float = ANIMAL_WASTE_WEIGHT,
    animal_gain_weight: float = ANIMAL_GAIN_WEIGHT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn one episode's money trace into `(shaped, raw, money)`.

    Shared by live rollouts and by demonstrations read out of stored replays, so a
    cloned episode and a self-collected one are the same kind of sample and cannot
    drift apart.

    `emitted[t]` says whether the policy put an order on the wire that turn; a turn
    that emitted one and moved no money had it refused, since every accepted order
    moves at least a coin.

    The optional traces come in three all-or-nothing groups -- `quadrants`/`days`,
    `seeds`/`standing`/`stock`, and `owned`/`produce_room`. With any member absent the
    stream is bit-identical to the one without that term, so a caller that cannot
    recover a trace stays on the old stream rather than silently scoring a different
    quantity.

    The animal group needs `days` as well, since an animal's capacity is a deadline
    rather than a fixed yield, and it rides on the lot group's padding: `owned` is what
    `lots.match` opens an animal lot off, and `produce_room` is the ceiling it
    reconciles against. Unlike the seed legs it is not gated on `ops` -- an animal
    purchase is the one outlay a wallet diff cannot mistake, because `owned` rising is
    the count actually delivered.

    `ops[t]` is the market op the policy chose on turn `t` -- `ACTIONS[chosen].op` --
    and gates `SEED_WASTE_WEIGHT` and `SEED_GAIN_WEIGHT` on top of the lot group. It
    is required rather than optional for those terms because `lots.match`'s
    `ops=None` fallback infers the op from the sign of the wallet move, which opens
    phantom lots on HIRE, BUY_LAND and BUY_ANIMAL turns, and a phantom lot here would
    be a phantom charge. Supplying the sequence is enough to disarm that: `lots.match`
    reads a short sequence, or a None entry inside one, as "no purchase on that turn"
    rather than as a per-turn licence to guess.

    `opponent_weight` is a parameter rather than a straight read of the constant
    because the mirror has to be gated off and only the caller knows who played.

    `raw` -- the money delta -- is returned although the stream no longer carries
    it, so money diagnostics keep meaning what they meant.
    """

    money_array = np.asarray([*money, final_money], dtype=np.float32)

    raw = (money_array[1:] - money_array[:-1]) / MONEY_SCALE
    shaped = np.zeros_like(raw)
    # Read off the trace rather than declared, so a `startingMoney` override moves
    # the reference with it instead of scoring a season against a float it never had.
    stake = max(0.0, float(money_array[0]))
    outcome = float(
        np.tanh(np.log((max(0.0, final_money) + 1.0) / (stake + 1.0)) / OUTCOME_LOG_SCALE)
    )
    if opponent_weight:
        # Folded into `outcome` before it is paid rather than added to `shaped`
        # separately: the lump and the annuity together are what make the
        # contribution to Q flat, and a second term paid any other way would not.
        outcome += float(opponent_weight) * float(
            np.tanh(
                np.log((max(0.0, final_money) + 1.0) / (max(0.0, opponent_money) + 1.0))
                / OUTCOME_LOG_SCALE
            )
        )
    shaped[-1] += outcome
    shaped += np.float32(OUTCOME_ANNUITY_WEIGHT * (1.0 - gamma) * outcome)

    if emitted is not None:
        tried = np.zeros(raw.shape[0], dtype=bool)
        # `emitted` can be shorter than the reward stream -- the agent skips a
        # malformed observation -- and an unknown turn must not be charged for
        # silence it was never observed to have.
        known = np.zeros(raw.shape[0], dtype=bool)
        width = min(len(emitted), tried.shape[0])
        tried[:width] = np.asarray(emitted[:width], dtype=bool)
        known[:width] = True
        if rejected_penalty:
            # `order_refused` lives in `market_state` because the state encodes the
            # same fact, so a feature saying "refused" and a reward charging for it
            # cannot disagree.
            quoted = None
            if expected is not None:
                quoted = np.zeros(raw.shape[0], dtype=np.float64)
                span = min(len(expected), quoted.shape[0])
                quoted[:span] = np.asarray(expected[:span], dtype=np.float64)
            shaped[order_refused(tried, raw, quoted)] -= float(rejected_penalty)
        if idle_penalty:
            # `turns` is the recorded length, so a truncated trace decays over what
            # it actually played -- the only thing it can know.
            turns = raw.shape[0]
            falloff = 1.0 - np.arange(turns, dtype=np.float32) / max(1, turns)
            charge = np.float32(idle_penalty) * falloff
            idle_turns = known & ~tried
            shaped[idle_turns] -= charge[idle_turns]

    if insolvency_penalty:
        # Read off the *resulting* wallet, so the turn that spends the last of the
        # stake is the first one counting toward a streak.
        below = money_array[1:] < float(insolvency_floor)
        grace = max(1, int(insolvency_grace))
        if grace > 1:
            # Consecutive below-floor turns ending at each index, without a python
            # loop. `reset` carries each solvent turn's own index and -1 elsewhere,
            # so the running maximum is the last solvent turn at or before `i`.
            index = np.arange(below.shape[0])
            last_solvent = np.maximum.accumulate(np.where(~below, index, -1))
            below = (index - last_solvent) >= grace
        shaped[below] -= float(insolvency_penalty)

    if (land_weight or land_bonus) and quadrants is not None and days is not None:
        width = money_array.shape[0]
        span = min(width, len(quadrants), len(days))
        # A fresh farm carries exactly the NW quadrant, so an absent trace reads as
        # no land bought rather than as a spurious purchase.
        held_quadrants = _held(quadrants, width, span, fill=1.0)
        held_days = _held(days, width, span)
        if land_weight:
            shaped += _telescope(
                land_potential(held_quadrants, held_days, land_weight), gamma
            )

        if land_bonus:
            # Keyed off the quadrant count rising, not off a `BUY_LAND` emission:
            # the simulator silently no-ops a purchase it will not sell, so paying
            # on the emission would reward asking rather than buying.
            bought = np.diff(held_quadrants)
            for index in np.flatnonzero(bought > 0):
                # `held_quadrants[index]` is the count *before* this step, so the
                # extras already owned are that minus the always-unlocked NW.
                for step in range(int(bought[index])):
                    already = int(held_quadrants[index]) - 1 + step
                    if not 0 <= already < len(donor.LAND_PRICES):
                        continue
                    price = float(donor.LAND_PRICES[already])
                    shaped[index] += np.float32(
                        float(land_bonus) * float(donor.LAND_PRICES[0]) / price
                    )

    wants_pnl = bool(waste_weight or gain_weight) and ops is not None
    # The animal legs need `ops` too, although nothing about an animal purchase reads
    # it: without a supplied sequence `lots.match` falls back to inferring a purchase
    # from the sign of the wallet, and a milk harvest landing on the BUY_ANIMAL turn
    # itself is "stock up, money down" -- a phantom bought-MILK lot sitting in the very
    # queue the cow's lot was opened in, absorbing the sales it is owed.
    wants_animal_pnl = (
        bool(animal_waste_weight or animal_gain_weight)
        and ops is not None
        and owned is not None
        and produce_room is not None
        and days is not None
    )
    if (
        (lot_weight or wants_pnl or wants_animal_pnl)
        and seeds is not None
        and standing is not None
        and stock is not None
    ):
        # All three traces or none, and padded exactly as the land trace is.
        width = money_array.shape[0]
        span = min(width, len(seeds), len(standing), len(stock))
        crops = len(lots.CROP_ITEMS)
        held_seeds = _held(seeds, width, span, columns=crops)
        held_standing = _held(standing, width, span, columns=crops)
        held_stock = _held(stock, width, span, columns=len(lots.STOCK_ITEMS))
        if lot_weight:
            shaped += _telescope(
                lot_potential(held_seeds, held_standing, held_stock, lot_weight), gamma
            )
        if wants_pnl or wants_animal_pnl:
            # The ledger indexes `buy_turn` the way `raw` does -- both sample before
            # the turn's action -- so no leg needs an offset.
            #
            # One ledger for all four legs, because a charge and its credit have to be
            # the same matching: a lot the FIFO attributed one way for the loss and
            # another for the profit would let a policy collect on both. The animal
            # group rides along or is absent entirely; absent, `match` behaves exactly
            # as it did before animals were matched.
            held_owned = held_room = held_animal_days = None
            if wants_animal_pnl:
                animals = len(lots.ANIMAL_ITEMS)
                # Its own span rather than the lot group's: `_held` pads from index
                # zero and holds the last value, so a shorter trace cannot slide the
                # others' indexing, and a group that padded on someone else's span
                # would invent holdings it never recorded.
                animal_span = min(width, len(owned), len(produce_room), len(days))
                held_owned = _held(owned, width, animal_span, columns=animals)
                held_room = _held(produce_room, width, animal_span, columns=animals)
                held_animal_days = _held(days, width, animal_span)
            ledger = lots.match(
                held_seeds,
                held_stock,
                money_array,
                ops,
                held_standing,
                owned=held_owned,
                produce_room=held_room,
                events=events,
                days=held_animal_days,
            )
            legs: list[tuple[float, int, list[tuple[int, float]]]] = []
            if wants_pnl:
                legs.append((float(waste_weight), -1, ledger.unrecovered()))
                legs.append((float(gain_weight), +1, ledger.surplus()))
            if wants_animal_pnl:
                legs.append((float(animal_waste_weight), -1, ledger.animal_unrecovered()))
                legs.append((float(animal_gain_weight), +1, ledger.animal_surplus()))
            for weight, sign, entries in legs:
                if not weight:
                    continue
                for buy_turn, coins in entries:
                    if 0 <= buy_turn < shaped.shape[0]:
                        shaped[buy_turn] += np.float32(
                            sign * weight * coins / MONEY_SCALE
                        )

    return shaped.astype(np.float32), raw.astype(np.float32), money_array
