"""Reward and its scales for the free-market learner.

Local rather than imported from `agents/dqn_market_scorer/training/collect.py`,
which is where this started: the scorer's market layer decides land and cash
reserves in code, so sharing the constants would have meant changing the
scorer's reward to fix this one.

The reward is the episode's own outcome, the tanh of the log growth on its own
stake. It is paid twice: as a lump on the terminal step, and as a per-step
annuity of `(1 - gamma) * outcome` on every step. The two sum to a flat
`outcome` in Q at every timestep, so a state on a winning trajectory is valued
above the same-looking state on a losing one no matter when in the season it was
seen.

Three small per-turn charges sit alongside it: an order the simulator refused, a
turn that emitted nothing at all, and a wallet with no runway left. None is
policy-invariant -- that is the point, and with the money delta gone they are the
only terms that separate two actions at the immediate-reward level.

They are ordered deliberately: an accepted order pays nothing, NOOP pays
`IDLE_PENALTY`, and a refused order pays the larger `REJECTED_ORDER_PENALTY`. So
trading beats sitting out, and sitting out beats emitting junk.

There is a second, opponent-relative leg of the same shape, landed inert at
`OPPONENT_WEIGHT = 0.0`. Its constant carries the argument for why the shared
market does *not* imply training on the margin -- the opponent's effect on us is
already inside our own purse, and only the reverse direction is missing -- and
the measurement that says the reverse direction is worth nothing yet.

The money delta and potential-based shaping were removed at v13, having been
switched off since v4. `Phi` existed only to cancel the cash leg of a purchase,
so the two had to go together: dropping the delta alone would price a buy at
`+Phi` and a sale at `-Phi`, and dropping `Phi` alone makes every purchase a
pure loss and every sale a pure gain -- the churn that retired it. Measured over
300 episodes of the last run that used shaping: 219 `BUY_PRODUCT` and 204 `SELL`
per episode, wheat and fertilizer 94% of the sale volume, against 13 wheat
actually planted.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from agents.dqn_market_free import donor
from agents.dqn_market_free.market_state import order_refused
from agents.dqn_market_free import lots

#: Reward units per coin.
#:
#: Sized so a whole episode's return lands in single digits. At the scorer's
#: 1,000 a donor-scale season is a return of about 100, and `SmoothL1Loss` has a
#: beta of 1.0 -- so every TD error above one coin-thousandth sat in the linear
#: regime, where a 5-unit error and a 50-unit error produce exactly the same
#: gradient magnitude. Essentially all of training happened there, which threw
#: away the relative size of the error.
#:
#: Scaled rather than clipped. Per-step reward clipping to [-1, 1] is the usual
#: DQN answer and is wrong here: the magnitude of a fill *is* the signal.
MONEY_SCALE = 20_000.0

#: Natural-log units of growth at which the outcome term reaches tanh(1) = 0.76.
#:
#: The outcome is the season's growth on **its own stake**, not the margin
#: against the opponent, and dropping the opponent is what lets one constant do
#: the job two could not.
#:
#: `tanh(margin / S)` had to discriminate two distributions at once and no `S`
#: does. Against the mirror the margins are hundreds -- both seats run the same
#: policy under the same frozen donor field -- so a large `S` vanishes; against a
#: scripted field they are tens or hundreds of thousands, so a small one
#: saturates. `market-free-population-v25` ran at `S = 10_000` with
#: `self_play_ratio = 0.65` and measured the consequence in both directions at
#: once: `league_pairings.mean_margin` was exactly 0.000 in all eleven
#: generations, so two of every three rollouts carried *no outcome term at all*
#: and the only live rewards in them were `IDLE_PENALTY` and
#: `INSOLVENCY_PENALTY` -- whose joint optimum is to emit an order every turn and
#: stay a coin above the floor. The policy learned exactly that: `trading_rate`
#: 0.00 to 0.78 while benchmark money fell to 98 from a 3,000 stake.
#:
#: The remaining third, against a bankrupt scripted opponent, has margin equal to
#: its own money -- and there `tanh(m / 10_000)` is 0.76 at 10k, 0.96 at 20k and
#: 0.999 at 40k. pilkwang finishes a season on about 142,000. Every policy from
#: 20k upward was therefore reward-identical, which is where this family has
#: stopped twice: `explore-v7` plateaued at 11-12k and `falloff-v18` at 5-6k,
#: both sitting on the flat part of the curve. The roadmap read that plateau as a
#: stuck trajectory and built a ten-slot population for it; a saturated objective
#: explains the same series without one.
#:
#: A log ratio is scale-free, so it does not have a saturation point to place:
#:
#:     final money  3,000   6,000   12,000   40,000   142,000   500,000
#:     outcome       0.00    0.14     0.27     0.48      0.65      0.77
#:
#: Monotone from the stake to the benchmark, with the steepest region exactly
#: where this agent lives. It is also *opponent-independent*, so the mirror, the
#: scripted opponent and the benchmark all score the same way and the
#: bimodal-by-source target variance diagnosed under `outcome_v4` -- half of every
#: minibatch labelled a flat +1.0 or -1.0 on states the network cannot tell apart
#: -- has no mechanism left to recur.
#:
#: And it is the number the promotion gate already reads. `benchmark_ratchet`
#: promotes on the candidate's own money against `benchmark_release_min_money`;
#: training on margin while gating on money is the mismatch that put
#: `TERMINAL_MARGIN_SCALE` in an impossible position to begin with.
#:
#: At 5.0 the tanh(1) = 0.76 point is a 167x season, or 500,000 coins from the
#: 3,000 stake -- deliberately well past pilkwang's ~142,000, so beating the
#: benchmark is a point on the slope rather than the end of it. Sized on the
#: headroom rather than on any measurement, because every previous value of this
#: constant was sized on the distribution the run happened to be producing at the
#: time and was saturated by the next one that improved.
#:
#: The cost is at the bottom, and it is the number to watch. Trading a 3,000
#: stake down to 1,898 now scores -0.091 against a fully idle season's -0.072
#: (see `IDLE_PENALTY`), so idling still beats churning, but by 0.019 rather than
#: by the 0.080 a scale of 3.0 would have given. The break-even is 2,092 coins:
#: lose more than about a third of the stake and a whole season of silence would
#: have scored better. That is the intended bar and it is deliberately not
#: tighter -- but if `trading_rate` climbs while money falls, this ratio is the
#: first thing to re-read, not `IDLE_PENALTY`.
OUTCOME_LOG_SCALE = 5.0

#: Weight on the opponent-relative leg of the outcome. **Held at 0.0.**
#:
#: The mechanism is landed inert, the same way `priority_alpha` was: the leg, its
#: caller gate and its mirror detection are all in place and a test asserts the
#: stream is bit-identical to the absolute one, so if this constant is ever moved
#: the only thing that changes is the number.
#:
#: It is 0.0 rather than 0.1 because the argument for a relative term is weaker
#: than it looks, and the reason is worth stating precisely, since "the market is
#: shared, so train on the margin" is the intuition that produced
#: `TERMINAL_MARGIN_SCALE` and every failure `OUTCOME_LOG_SCALE` records.
#:
#: **The opponent's effect on us is already in our own money.** One market object
#: is shared by both seats and every unit either sells depresses the price for
#: both. When the opponent gluts wheat, our revenue falls -- and the absolute
#: outcome sees that in full, because it reads our purse after the fact. Playing
#: against a productive opponent is already a harder episode that scores lower.
#: Nothing needs to be added for the policy to care.
#:
#: What absolute money cannot see is the mirror image: *our* effect on *them*.
#: Dumping sixty strawberries at a loss to crash the quote before their two
#: hundred reach the market reads as a pure loss to us and a large gain to a
#: margin. That -- denial -- is the entire content of the relative term, and it is
#: the only thing it buys.
#:
#: Denial is out of reach for this agent and measurably absent from the runs. On
#: `episode_players` for `league:market-free-logmoney-v26:g0011`:
#:
#:                        mean     sd    mean/sd
#:     own final money   2,715  1,176      2.31
#:     money difference    535  1,868      0.29
#:
#: with `corr(final_money, opponent_money) = 0.006`. Two readings, and they point
#: the same way. First, at that correlation neither seat is moving the market
#: enough to register in the other's outcome, so the coupling denial exploits is
#: not yet operating. Second, a difference is a control variate and a control
#: variate only pays when the subtracted term correlates with the kept one: at
#: 0.006 nothing cancels, `Var(own - opp)` is very nearly `Var(own) + Var(opp)`,
#: and the per-episode signal-to-noise ratio falls by a factor of eight. That is
#: `OUTCOME_LOG_SCALE`'s failure quantified from the other side -- the margin was
#: not merely mis-scaled, it was a noisier measurement of a smaller quantity.
#:
#: And the strategy needs inventory to dump. This family finishes a season on
#: 1,928 coins from a 3,000 stake and plants about thirteen wheat; the $1 floor is
#: absorbing, so gluting caps out early even when it is affordable. A term that
#: costs variance now to reward a behaviour the policy cannot yet perform is the
#: trade this file exists to refuse.
#:
#: Turn it on when the agent produces at scale and the correlation above departs
#: from zero -- that is the measurement that says the coupling has started to
#: matter. 0.1 is the value to try, sized as a tiebreaker rather than a driver: it
#: spans +/-0.1 against the absolute leg's +/-1.0, so two policies finishing on
#: equal money are separated by what they did to the opponent while a policy
#: finishing on more money is not overturned. Watch `mean_abs_q` against the
#: reward bound; if it climbs past it, halve this rather than re-reading
#: `OUTCOME_LOG_SCALE`.
#:
#: A log ratio against the opponent's purse rather than a difference, and at the
#: same `OUTCOME_LOG_SCALE`, so the two legs cannot drift apart in shape. Scale-
#: free is the property `tanh(margin / S)` lacked and the reason no `S` served
#: both the mirror's hundreds and the scripted field's hundred thousands. It is
#: also exactly 0 at parity, which makes the mirror's degenerate case harmless
#: rather than merely small.
#:
#: The mirror is gated off at the caller regardless of this value, because there
#: the leg is noise by construction: the opponent is a recent checkpoint of the
#: same slot, so its purse carries this policy's own variance and nothing else.
#: `market-free-population-v25` is what that gate is for -- at
#: `self_play_ratio = 0.65` two of every three rollouts would otherwise carry a
#: zero-mean term worth only what it adds in noise.
OPPONENT_WEIGHT = 0.0

#: Multiplier on the per-step outcome annuity.
#:
#: The terminal margin term lands on one turn of about 720, and the buffer samples
#: 24-turn windows, so only ~3% of transitions ever see it as a reward. The rest
#: wait on a bootstrap chain up to 720 links deep to find out whether the episode
#: they came from made money. Paying `(1 - gamma) * outcome` on every step as well
#: as the terminal lump makes the outcome's contribution to Q flat at `outcome`
#: for every t:
#:
#:     sum_{k>=t} gamma^(k-t) * (1-gamma) * outcome  +  gamma^(T-1-t) * outcome
#:       = outcome * (1 - gamma^(T-t) + gamma^(T-1-t))  ~=  outcome
#:
#: Scaled by `(1 - gamma)` rather than the more obvious `1 / T`. An even
#: `outcome / T` spread leaves Q at about 0.71 * outcome on the first turn decaying
#: to nothing on the last, which is *less* late-game outcome signal than the
#: terminal lump alone already provides. It also means the annuity vanishes exactly
#: at gamma = 1, where an undiscounted lump already reaches every step at full
#: weight and there is nothing to fix.
OUTCOME_ANNUITY_WEIGHT = 1.0

#: Charged on a turn that emitted an order the simulator then refused.
#:
#: The immediate reward is identical for every action on a turn -- the stream
#: pays only the outcome annuity -- so NOOP, BUY_SEED and an impossible order all
#: score the same on the turn they happen, and differ only many turns later. An
#: untrained value function sees a flat plateau and picks off it arbitrarily. Measured: one run settled on `SELL WOOL 16` for 57% of its
#: decisions while owning no sheep, with 98.4% of everything it emitted refused.
#:
#: This charges for the refusal alone, so it separates "interacted with the
#: world" from "did not" without expressing any view on *which* trade is good --
#: an impossible order falls below NOOP, a real purchase rises above it, and the
#: choice among real purchases is left exactly where it was. It is not
#: policy-invariant, so it is kept small.
REJECTED_ORDER_PENALTY = 0.001

#: Charged on every turn the policy emits no order at all.
#:
#: Without it nothing in the stream discourages sitting still, and the terms that
#: exist actively reward it. `REJECTED_ORDER_PENALTY` requires an emission and
#: `INSOLVENCY_PENALTY` requires a spent wallet, so a NOOP turn scored exactly
#: zero and paid neither -- while a policy that keeps its 3,000 stake untouched
#: also beats a bankrupting opponent on the terminal margin. Idling was weakly
#: dominant at the immediate-reward level, which is the NOOP attractor this
#: family has collapsed into repeatedly: `market-free-noisy-v6` sat at exactly
#: 3,000 money with `trading_rate` 0.0 for twelve updates and never promoted.
#:
#: Everything opposing that lived outside the reward -- the gate's activity
#: floor, the idle burst, the flip-to-order controller -- so the machinery was
#: fighting the objective instead of expressing it. This states it in the
#: objective.
#:
#: Charged at this rate on turn 0 and decaying linearly to zero at the deadline,
#: so the *mean* charge is half of it and a fully idle season pays about 0.072.
#:
#: The falloff is economic rather than cosmetic. An idle turn on day 1 forfeits a
#: season of compounding; an idle turn on day 29 forfeits almost nothing, because
#: there is no time left to plant, grow and sell whatever the money would have
#: bought. A flat charge prices those the same and is wrong about the late one.
#:
#: It is also predictable from the state, which is the constraint that ruled out
#: the alternatives. `day / total_days` is already encoded (`market_state`), so
#: the agent can see exactly what this turn's charge will be. A streak-based or
#: cumulative-count curve would vary the charge on something the state does not
#: carry -- the same drift `order_refused` exists in one place to prevent -- and
#: fixing that means a new state feature, a changed `MARKET_STATE_DIM`, and every
#: checkpoint invalidated.
#:
#: The size is set against the *outcome term's* range, not against the other
#: charges, and getting that wrong is what the previous value did. At 0.0005 flat
#: a fully idle season cost 0.36 while the old `tanh(margin / 10_000)` only
#: spanned 0.19 to 0.29 between 1,898 and 3,000 coins -- so trading yourself
#: *down* to 1,898 scored 0.257 better than sitting on 3,000, and the policy
#: correctly learned to trade on 73-99% of turns and lose money. Measured on
#: `market-free-hot-v16`: mean money fell 3,542 to 1,898 while `mean_abs_q` rose
#: 0.034 to 0.133 -- a Q function converging cleanly on a bad objective.
#:
#: At 0.0002 with the falloff the season total is 0.072. Under the absolute
#: outcome the same 3,000-to-1,898 slide costs 0.091, so the ordering still
#: holds -- narrowly, and `OUTCOME_LOG_SCALE` states the break-even exactly --
#: but it now holds for the stronger reason: the outcome is no longer zero in the
#: mirror, so this charge is not the only term with an opinion
#: there. `market-free-population-v25` is what that
#: correction is for: at `self_play_ratio = 0.65` against a margin of exactly
#: 0.000, two of every three rollouts had nothing in the stream *but* this charge
#: and the insolvency one, and the policy went to `trading_rate` 0.78 on 98 coins.
#: The per-turn ordering still holds at every point in the season:
#:
#:     accepted order    0
#:     NOOP             -0.0002 decaying to 0
#:     refused order    -0.001
#:
#: Strictly below `REJECTED_ORDER_PENALTY` at t=0, so NOOP stays *above* a refused
#: order -- the ordering `order_space.feasible_actions` relies on, since a policy
#: that cannot afford anything must not be pushed into emitting junk to escape a
#: charge. Strictly above zero early, so an accepted trade beats sitting out.
IDLE_PENALTY = 0.0002

#: Coins below which the farm has no runway left.
#:
#: The cheapest thing worth buying is a seed, and the cheapest hand is one coin,
#: so a farm under a couple of hundred coins is not choosing between actions --
#: `feasible_actions` has already closed almost every column and left it NOOP.
INSOLVENCY_FLOOR = 250.0

#: Charged on every turn the farm spends below the floor.
#:
#: Bankruptcy is an *opportunity* cost: the money delta is bounded below by a
#: wallet that cannot go negative, so spending the last of the stake reads as a
#: one-off -0.15 at `MONEY_SCALE` and everything it forecloses arrives only as
#: the absence of later fills. Recovering that through the bootstrap means
#: propagating value back across up to 720 turns in 24-turn hops, through a
#: buffer where nearly every state is the dead absorbing one -- measured on
#: `market-free-simple-v2`, the policy went broke on turn four and then held
#: 715 turns of NOOP, so that is what the network trained on.
#:
#: This charges for the state directly instead. At 0.002 a whole episode spent
#: broke costs about -1.4, the same order as the outcome term, while a farm that
#: dips below the floor for a few turns between a purchase and a sale pays almost
#: nothing. Like `REJECTED_ORDER_PENALTY` it is not policy-invariant -- it is
#: meant to change which policy is optimal -- so it is kept small and states a
#: floor rather than a preference about trades.
INSOLVENCY_PENALTY = 0.002

#: Consecutive turns below the floor before the charge starts billing.
#:
#: The charge above was bought to price a *permanent* absorbing state -- v2 spent
#: the stake on `HIRE 13` by turn four and then held 715 turns of NOOP, and that
#: dead state was ~99% of the buffer. It was never meant to price the wallet
#: dipping under the floor between a purchase and the sale that pays for it, and
#: it turned out to be doing exactly that.
#:
#: Measured on `market-free-logmoney-v26` at `cutoff_batch_seq = 360`, decomposing
#: `own_mean_return` = -0.193 against a mean final money of 3,038 on a 3,000 stake:
#:
#:     outcome (money growth)   +0.004   2%
#:     IDLE_PENALTY             -0.044  22%
#:     REJECTED_ORDER_PENALTY   -0.00006
#:     INSOLVENCY_PENALTY       -0.153  78%   (~75 turns/episode below the floor)
#:
#: The term the whole reward exists to express was 45x smaller than the charges,
#: and the run had spent 36 updates optimising the charges: of the +1.92 return
#: improvement from batch 50 to 360, +1.46 came from escaping this one and only
#: +0.46 from making money. Read the other way, `d(outcome)/d(log money)` is 1/5,
#: so *doubling* a season is worth +0.24 of return while the ~120 low-cash turns
#: it takes to buy the inputs cost 0.002 * 120 = -0.24. Doubling your money netted
#: exactly zero, which is why `own_orders_per_decision` sat at 0.38 and
#: `own_solvent_fraction` was *falling*.
#:
#: 24 turns, one in-game day: long enough that a purchase and the sale that funds
#: it are the same event, short enough that a farm with no runway is still billed
#: for ~696 turns of a 720-turn season and the v2 pathology stays priced.
#:
#: A streak rather than a cap on the episode total, which was the alternative. A
#: cap prices every dip and then stops counting; this prices no dip and all of the
#: absorbing state, and the absorbing state is the thing the constant was bought
#: for. What it deliberately does *not* fix: a genuine bust still costs about
#: -1.35 against a p90 season's +0.27, so the objective stays asymmetric against
#: risk -- that is the outcome term's own shape (a bust reads -0.59 before any
#: charge at all) and `OUTCOME_LOG_SCALE` is the constant to re-read if that
#: asymmetry is still binding once the dips are free.
#:
#: 0 or 1 restores the per-turn charge exactly.
INSOLVENCY_GRACE_TURNS = 24

#: Weight on the unlocked-land potential.
#:
#: The one term in this file that is **policy-invariant by construction**, and
#: the only reason it is allowed to exist. Everything above deliberately changes
#: which policy is optimal; this cannot. Ng, Harada & Russell: a shaping of the
#: form `F(s, s') = gamma * PHI(s') - PHI(s)` leaves the optimal policy unchanged
#: for any `PHI`, so this can only change *when* credit arrives, never what the
#: agent should ultimately do.
#:
#: What it is for. Buying a plot is a real multiplier and the agent cannot see
#: it. `donor._plan_roles` skips quadrants with no zone and zones come from
#: `farm["unlocked_quadrants"]`, so a purchase creates `("CROP", crop)` roles,
#: raises `donor._seed_needs`, and `planting.py` turns held seed into plants. But
#: a plot bought on day 5 pays back through harvests from day 15, which reaches
#: the purchase decision only through the terminal outcome -- discounted
#: `0.999^720 = 0.49`, in 24-turn hops -- while the 1,000-coin cost lands in the
#: money trace immediately. So the policy learns the cost and not the return.
#:
#: Measured on `market-free-streak-v27`, generations 2-6, split on whether the
#: seat ever bought land:
#:
#:                          bought    did not
#:     seats                   792      4,548
#:     plants                 38.0       36.9
#:     plants lost            22.6       15.1
#:     median final money        2        400
#:
#: and the seat share buying land ran 3.6 -> 14 -> 21.5 -> 22.9 -> 12.3 -> 0 -> 1
#: percent across generations 2 to 8, peaking exactly at the money trough. The
#: agent tried land, executed it badly, and unlearned it. That is a credit
#: assignment failure, not a verdict on land -- and if it *is* a verdict on land,
#: policy invariance means this term will not overturn it.
#:
#:     PHI(s) = w * extra_quadrants(s) * max(0, TOTAL_DAYS - day(s)) / TOTAL_DAYS
#:
#: **Proportional to remaining usage**, which is the whole shape: a plot taken on
#: day 5 has 25 days of harvests left and one taken on day 28 has two. It also
#: makes `PHI` vanish at the terminal state for free, which is the condition the
#: telescoping needs -- so the requirement and the intuition are the same thing
#: here rather than two constraints to reconcile.
#:
#: 0.08 sized against the *outcome term*, not against `LAND_PRICES`. The outcome
#: is `tanh(log(final / stake) / 5)` paid at about 1.72x once the terminal lump
#: and the annuity are both counted, so a 25% season is worth
#: `tanh(log(1.25) / 5) * 1.72 = 0.077`. One quadrant taken at the start is
#: therefore worth about what a 25% season is worth, decaying from there:
#:
#:     bought on day    0      5      9     15     24
#:     one-time      +0.080 +0.067 +0.056 +0.040 +0.016
#:
#: Sizing it against the land price was the alternative and is worse: it makes
#: the payout largest exactly where the purchase is most dangerous.
#:
#: **The v13 objection does not transfer, and this is the paragraph to read
#: before removing it again.** Potential-based shaping was retired from this file
#: once because it "marked the whole shed to a quote the agent moved itself by
#: buying" -- the potential read market prices the policy could pump, so the
#: telescoping was over a quantity the agent controlled. `PHI` here reads a
#: monotone integer bought from a fixed price ladder and never sold, times an
#: exogenous clock. There is no cycle to pump.
#:
#: Both halves are already state features -- `market_state.py` carries
#: `len(unlocked_quadrants) / 4` and `(TOTAL_DAYS - day) / TOTAL_DAYS` -- so the
#: network can represent this exactly rather than having to infer it.
#:
#: 0.0 removes the term. `quadrants=None` does too, and is what a caller with no
#: land trace gets.
LAND_POTENTIAL_WEIGHT = 0.08


#: Paid once on the turn a quadrant actually appears, on top of the potential
#: above. The two are different *kinds* of term and the distinction is the whole
#: reason this exists.
#:
#: `LAND_POTENTIAL_WEIGHT` is potential-based: `gamma * PHI(s') - PHI(s)`
#: telescopes to zero over an episode, which is what makes it provably
#: policy-invariant. That is a guarantee about what it cannot do. It can move
#: *when* credit arrives; it can never change which policy is optimal. Across six
#: sessions it has never moved the land rate, and it was never going to.
#:
#: This one is not potential-based and will change the optimum. It is a nudge
#: toward exercising a row the policy has learned to avoid, and it is only
#: reachable from states `order_space._land_is_survivable` has already cleared --
#: mid-season, seed in hand, hiring covered -- so the states it can push toward
#: are ones the farm can work.
#:
#: **The magnitude was justified against a cost term that does not exist, and the
#: correction is recorded here rather than quietly applied.** This block used to
#: read "`MONEY_SCALE` is 20,000 and the raw per-turn reward is
#: `money_delta / MONEY_SCALE`, so the first quadrant's 1,000 coins already scores
#: -0.05 on the purchase turn", and priced itself as repaying a fraction of that
#: outlay. `episode_rewards` builds `shaped` from `np.zeros_like(raw)` and never
#: adds `raw` to it; `collect.py` stores `raw` as `raw_reward` for diagnostics and
#: `replay.py` trains on `reward`, which is `shaped`. `IDLE_PENALTY` states the
#: consequence in its own ordering table -- an accepted order scores 0 on the turn
#: it happens. **There is no -0.05.** The bonus is unopposed on the purchase turn
#: and always was, so the "fraction repaid" column below is what the sizing
#: intended, not what the stream does:
#:
#:     bonus    intended reading                 what it actually is
#:     0.05+    100% of the outlay repaid        3.5 idle turns' charge, unopposed
#:     0.035     70%                             half a fully idle season (0.072)
#:     0.02      40%                             a quarter of one
#:     0.00       0%                             the measured status quo
#:
#: Held at 0.035 deliberately rather than re-sized to the true scale. The term is
#: doing what it was bought for whatever its stated justification: v31 held a
#: 0.13-0.16 buy rate for 70 batches where every earlier session decayed to
#: 0.005-0.01, v33 went 0.043 to 0.35 with `land_tiles_planted_per_purchase`
#: *rising* 12.3 to 16.3, and rollout money went 1,636 to 6,500 across the same
#: span. A constant that is empirically working is not cut on the strength of a
#: corrected comment; it is cut when `land_tiles_planted_per_purchase` falls while
#: the rate climbs, which is still the check below.
#:
#: The v26 failure the old text cited stands and is worth keeping in view:
#: generations 2-5 ran 18% of seats buying land at a mean of 488 coins, the worst
#: stretch of that session.
#:
#: Raised from 0.02 on the v32 evidence. 0.02 did move the row -- it is the first
#: constant in this file to stop the land rate collapsing, holding 0.13-0.16 for
#: 70 batches on `market-free-widecap-v31` where every earlier session spiked and
#: unlearned it back to 0.005-0.01. What it did not do is move the *rate* on a run
#: that started with it: `market-free-melon-v32` reached only 3.2% of episodes
#: buying against widecap's 14.3%, and its buyers took one quadrant and stopped
#: (1.15 per buying episode against 1.62). The gate is not what is binding --
#: purchases land on day 10 with 1,965 coins, well clear of `_land_is_survivable`
#: -- so the row is being declined on value, which is the thing this constant
#: prices. 0.035 is the largest step that still leaves the purchase 30% out of
#: pocket; the ceiling is `MONEY_SCALE`-derived and unchanged.
#:
#: Scaled by price, so the 2,000 and 4,000 quadrants are not overpaid relative to
#: what they cost: the bonus is this constant times `LAND_PRICES[0] / price`.
#:
#: The check that catches it going wrong is `land_tiles_planted_per_purchase`
#: against `buy_land_rate`, both per batch. A rate that climbs while tiles stay
#: near zero is the agent taking the bonus and abandoning the field -- cut this,
#: do not widen the gate.
#:
#: 0.0 removes the term, as does `quadrants=None`.
LAND_PURCHASE_BONUS = 0.035


#: Weight on the cost basis a seat has committed to growing something.
#:
#:     PHI(s) = w * (held cost basis in coins) / MONEY_SCALE
#:
#: where the basis counts a purchased seed at its price, a standing plant at the
#: price of the seed that became it, and a harvested unit at that price divided
#: by the crop's `max_yield`.
#:
#: What it is for. Nothing else in this stream distinguishes a purchase from a
#: NOOP on the turn it happens -- `shaped` never carries the money delta, so the
#: ordering is the three lines at `IDLE_PENALTY` and a BUY_SEED scores exactly 0.
#: The only term with an opinion about whether that seed was worth buying is the
#: terminal outcome, and the distance to it is a crop cycle: 48 turns for wheat,
#: 240 for the melon `planting.py` optimises for, against `n_step = 24`. So the
#: answer arrives two to ten bootstraps later, attenuated by the value function's
#: error at every hop, to a decision that lasted one turn. This puts a credit on
#: the buy turn itself.
#:
#: **Potential-based, and that is the whole design.** `gamma * PHI(s') - PHI(s)`
#: telescopes, so the shaping added to the return-to-go from any state is exactly
#: `-PHI(s)` -- a constant of the state, identical across every action available
#: there. Q ordering is preserved at every state and the optimal policy cannot
#: move. It is the same guarantee `LAND_POTENTIAL_WEIGHT` is bought for.
#:
#: **The version this replaced was a reallocation, and it failed on exactly that
#: point.** `LOT_CREDIT_WEIGHT` used `lots.py` to move a lot's realised proceeds
#: from the sale turn back to the buy turn, paying `+coins` at one and `-coins`
#: at the other. The episode total was preserved, which is what made it look
#: safe, but a preserved *total* is not a preserved *ordering*: at the sale state
#: the debit is entirely in the future, so return-to-go fell by the full amount,
#: while at the buy state the credit was immediate and the debit `D` turns away,
#: so return-to-go rose by only `coins * (1 - gamma**D)` -- `0.214` of it at
#: `D = 240`, `gamma = 0.999`. The term deterred selling about five times harder
#: than it encouraged buying. In magnitude, a 24-unit melon dump at 250 coins
#: charged `-0.03` on one turn against `IDLE_PENALTY` 0.0002,
#: `REJECTED_ORDER_PENALTY` 0.001, `INSOLVENCY_PENALTY` 0.002 and an outcome
#: annuity capped at 0.001 per step: the largest single-turn term in the
#: non-terminal stream by an order of magnitude, carried by all 24 transitions in
#: the n-step window before every sale. It moved the credit assignment problem
#: from buying to selling rather than removing it. The potential form pays the
#: same credit on the same turn and cannot do that.
#:
#: **Cost basis, never market price, and the v13 objection is why.** Potential
#: shaping was retired from this file once for marking the shed to a quote the
#: policy moved itself by buying. Every price here is a fixed constant out of
#: `donor.CROPS`; nothing in `PHI` reads `obs.market`. Telescoping makes a pump
#: impossible in any case -- a buy-and-sell round trip returns `PHI` to where it
#: started -- but the constants mean it is not even a question about magnitudes.
#:
#: Sized so one seed is visible against the charges it competes with. A melon
#: seed at 80 coins moves `PHI` by `w * 0.004`; at `w = 1.0` that is 4.0 in units
#: of `IDLE_PENALTY` and twice `INSOLVENCY_PENALTY`, on the turn of the purchase.
#: A fully planted 25-tile melon quadrant holds `PHI = 0.1`, so the `(gamma - 1)`
#: leg charges 0.0001 a turn for sitting on it -- a real holding cost, below the
#: idle charge, and the correct sign.
#:
#: The failure mode is not inflation and not a pump; invariance forbids both. It
#: is that the term does nothing, which is what `LAND_POTENTIAL_WEIGHT` has done
#: across six sessions. What to watch is `seed_holding_turns` falling while
#: `seed_margin` holds. `seed_spend` climbing with `unrecovered_seed_share` is
#: the policy buying more rather than earlier, and this term is not what would be
#: causing it -- `LAND_PURCHASE_BONUS` is the constant in this file that can.
#:
#: 0.0 removes it, as does any of the three traces being absent.
#:
#: **Why this stops at seeds and there is no equivalent for `HIRE`.** The obvious
#: next step is a labour term, on the reading that hiring is unrewarded and
#: therefore underdone. Four measurements say otherwise.
#:
#: There is no distance to close. The whole argument above is that a seed's payoff
#: is 48 to 240 turns from its purchase against `n_step = 24`. A hand has no such
#: gap: the simulator clears `farm["hands"]` and resets `hires_today` every night,
#: so a hire is a one-day lease whose entire working life fits inside a single
#: n-step window. It is cheap and self-limiting too -- `_hire_cost` is
#: `fib(n_already_today)` at `FARM_HAND_COST_MULT = 1`, so the day's first hands
#: cost 1, 1, 2, 3, 5 and the ladder only bites past the tenth.
#:
#: Labour is already saturated, not scarce. Seats reach ~390 hires a season
#: against `donor._LABOR_GATE_INSIDE_CAP = 13` a day, which is the ceiling. Median
#: money by hires per episode, learner families over v33 and v31:
#:
#:     hires      v33 median   v33 idle    v31 median   v31 idle
#:     0-49          2,680       0.36         1,350       0.39
#:     100-149       2,950       0.50         2,875       0.57
#:     200-249       3,814       0.50         2,802       0.65
#:     250-299       4,230       0.55         2,797       0.64
#:     300+          2,902       0.51         2,634       0.61
#:
#: Money rises with hiring up to the cap and turns over past it, so the outcome
#: term is already pricing the marginal hire about right. And `idle_percentage`
#: climbs monotonically with hires -- the hands being bought are increasingly not
#: dispatched, so the binding constraint is the field scheduler's ability to *use*
#: labour, not the policy's willingness to buy it. The lever for that is
#: `_LABOR_GATE_INSIDE_CAP` or the scheduler, not a term in this file.
#:
#: There is nothing to attribute. `lots.py` can carry a cost basis because a seed
#: is an identifiable unit that becomes identifiable produce. Labour is a shared
#: input across every lot alive that day, so a per-hire basis would be an
#: allocation invented for the purpose rather than a reconstruction -- and a
#: potential over a stock that is zeroed every night is just its own cost restated
#: on a one-day cycle.
#:
#: And the failure mode is on the record. `train_market_free.py` keeps v2 spending
#: the whole 3,000 stake on `HIRE 13` inside four turns of a 720-turn episode and
#: then holding NOOP, finishing at 0.7 coins. Over-hiring is this family's original
#: bankruptcy attractor, and the money delta is not in `shaped`, so nothing here
#: charges for a hire -- exactly the unopposed shape `LAND_PURCHASE_BONUS` turned
#: out to have.
#:
#: What would reopen it: `idle_percentage` *falling* while hires sit at the cap.
#: That would mean the scheduler had learned to use the hands and labour had
#: become the binding constraint.
LOT_POTENTIAL_WEIGHT = 1.0


#: Coins of committed cost per unit held, by `lots.STOCK_ITEMS` index.
#:
#: A seed and the plant it becomes both carry the full seed price; a harvested
#: unit carries `seed / max_yield`, so a maximal harvest conserves `PHI` and a
#: poor one writes the difference off on the turn it lands. That write-off is
#: honest and, being potential-based, cannot change which policy is optimal.
#:
#: Zero for EGG, MILK, WOOL and FERTILIZER. Animal produce has a cost basis --
#: the bird or the beast -- but it belongs to a purchase this file would have to
#: model separately, and `lots.py` excludes animals for the same reason. Bought
#: WHEAT gets the seed-derived basis rather than what was paid for it, which is
#: wrong in the small and harmless in the whole: telescoping returns `PHI` to
#: where it started over any round trip regardless of the constant used.
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


def land_potential(
    quadrants: Sequence[int],
    days: Sequence[int],
    weight: float = LAND_POTENTIAL_WEIGHT,
    total_days: int = donor.TOTAL_DAYS,
) -> np.ndarray:
    """`PHI` over one episode's land trace, terminal entry forced to zero.

    Returned one entry longer than the reward stream, indexed like `money_array`
    in `episode_rewards`, so the shaping is a plain difference over it.

    The terminal zero is set rather than derived. `PHI` reaches zero on its own
    only if the last recorded day is `total_days`, and a replay that ends on day
    29 -- or a truncated trace -- would otherwise leave a residual lump on the
    final turn that the telescoping never repays. That residual is exactly the
    thing policy invariance is bought to avoid, so it is closed here rather than
    assumed away.
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

    All three traces come in already padded to `money_array`'s width, the way
    `land_potential` takes its two, so the shaping is a plain difference over the
    result and neither term invents its own indexing convention.

    The terminal zero is not merely the condition the telescoping needs -- it is
    what the simulator does. Unsold stock, standing crops and unplanted seed all
    liquidate to nothing at step 718, where the reward is `farm["money"]` and
    only that (`kaggriculture.py:945-951`). `PHI` genuinely *is* zero at the
    terminal state, so forcing it agrees with the environment rather than closing
    a residual the way `land_potential` has to.
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn one episode's money trace into a reward stream.

    Returns `(shaped, raw, money)`. Shared by live rollouts and by demonstrations
    read out of stored replays, so a cloned episode and a self-collected one are
    the same kind of sample and can sit in one buffer -- and cannot drift apart.

    The stream is the season's log growth on its own stake, spread as an
    annuity, minus the refusal and insolvency charges. It took a `mode` and a
    potential trace until v13: the
    `shaped` arm paid the money delta plus potential-based shaping, and it was
    retired because the shaping marked the whole shed to a quote the agent moved
    itself by buying. Measured over 300 episodes of the last run that used it:
    219 `BUY_PRODUCT` and 204 `SELL` per episode, wheat and fertilizer 94% of the
    sale volume -- the only two goods `BUY_PRODUCT` can buy back -- against 13
    wheat actually planted.

    `emitted[t]` says whether the policy put an order on the wire that turn. A
    turn that emitted one and moved no money had it refused: every order the
    simulator accepts moves cash by at least a coin, since prices floor at 1 and
    quantities start at 1.

    `quadrants[t]` and `days[t]` carry the unlocked-quadrant count and the game
    day on the same index; `seeds[t]`, `standing[t]` and `stock[t]` carry what
    was held, counted over `lots.CROP_ITEMS` and `lots.STOCK_ITEMS`. They add the
    two policy-invariant terms in the file -- see `LAND_POTENTIAL_WEIGHT` and
    `LOT_POTENTIAL_WEIGHT`. Each group is all-or-nothing: with any member absent
    the stream is bit-identical to the one without that term, which is what lets
    a caller that cannot recover the trace stay on the old stream rather than
    silently score a different quantity.

    All three of the other terms are deliberately not policy-invariant: the
    outcome growth,
    the refused-order charge, and the insolvency charge. Each is documented at
    its constant.

    `opponent_money` is read only through `opponent_weight`, which defaults to
    `OPPONENT_WEIGHT` and is 0.0 there -- so today it is not read at all, and the
    stream is bit-identical to the absolute-only one this replaced. The outcome
    became absolute at `OUTCOME_LOG_SCALE` and that is still the trunk; the
    relative leg is a tiebreaker whose argument and measurements live at its
    constant.

    `opponent_weight` is a parameter rather than a straight read of the constant
    because the mirror has to be gated off and only the caller knows who played:
    `collect_episode` passes 0.0 against this agent's own entry point. Defaulting
    it to the constant means the two callers -- `training/collect.py` on a live
    rollout and `training/clone.py` on a stored replay -- cannot diverge by
    forgetting it, only by disagreeing on the gate, which is one decision in one
    place each.

    `raw` -- the money delta -- is returned although the stream no longer carries
    it, so the money diagnostics an update logs keep meaning what they meant and
    `order_refused` keeps reading a real cash trace.
    """

    money_array = np.asarray([*money, final_money], dtype=np.float32)

    raw = (money_array[1:] - money_array[:-1]) / MONEY_SCALE
    # `raw` is still computed although the stream no longer carries it:
    # `order_refused` reads it to tell a refused order from an accepted one, and
    # the update logs its money diagnostics off it.
    shaped = np.zeros_like(raw)
    # The stake is read off the trace rather than declared as a constant, so a
    # `startingMoney` override moves the reference with it. Anything else would
    # score a season against a float it never had -- the same class of drift
    # `market_state` avoids by normalising prices against the *effective* base.
    stake = max(0.0, float(money_array[0]))
    # tanh of a log ratio rather than of a linear one. Both bound the term to
    # (-1, 1) so one blowout episode cannot dominate a batch, but the linear form
    # has to pick a coin scale, and no scale separates the mirror's hundreds from
    # the benchmark's hundred thousands. The ratio is scale-free: doubling the
    # stake is the same reward wherever it happens, and doing nothing is exactly
    # 0.0. See `OUTCOME_LOG_SCALE`.
    outcome = float(
        np.tanh(np.log((max(0.0, final_money) + 1.0) / (stake + 1.0)) / OUTCOME_LOG_SCALE)
    )
    if opponent_weight:
        # The same shape against the opponent's purse instead of the stake, so
        # the two legs share `OUTCOME_LOG_SCALE` and cannot drift apart. Folded
        # into `outcome` before it is paid rather than added to `shaped`
        # separately: the terminal lump and the annuity below are what make the
        # contribution to Q flat across the season, and a second term paid any
        # other way would not have that property.
        outcome += float(opponent_weight) * float(
            np.tanh(
                np.log((max(0.0, final_money) + 1.0) / (max(0.0, opponent_money) + 1.0))
                / OUTCOME_LOG_SCALE
            )
        )
    shaped[-1] += outcome
    # And again as an annuity on every step, so the outcome reaches the ~97% of
    # sampled transitions that never contain the terminal turn.
    shaped += np.float32(OUTCOME_ANNUITY_WEIGHT * (1.0 - gamma) * outcome)

    if emitted is not None:
        tried = np.zeros(raw.shape[0], dtype=bool)
        # Which turns the trace actually covers. `emitted` can be shorter than
        # the reward stream -- the agent skips a malformed observation -- and the
        # distinction is load-bearing for the idle charge below: an unknown turn
        # must not be charged for silence it was never observed to have.
        known = np.zeros(raw.shape[0], dtype=bool)
        width = min(len(emitted), tried.shape[0])
        tried[:width] = np.asarray(emitted[:width], dtype=bool)
        known[:width] = True
        if rejected_penalty:
            # `order_refused` lives in `market_state` because the state encodes
            # the same fact -- a refusal streak the policy can see. One
            # definition, so a feature saying "refused" and a reward charging for
            # it cannot disagree.
            shaped[order_refused(tried, raw)] -= float(rejected_penalty)
        if idle_penalty:
            # Linear falloff across the season. `turns` is the recorded length,
            # which for a complete episode is the season the state's
            # `day / total_days` also measures against; a truncated trace decays
            # over what it actually played, which is the only thing it can know.
            turns = raw.shape[0]
            falloff = 1.0 - np.arange(turns, dtype=np.float32) / max(1, turns)
            charge = np.float32(idle_penalty) * falloff
            idle_turns = known & ~tried
            shaped[idle_turns] -= charge[idle_turns]

    if insolvency_penalty:
        # Read off the *resulting* wallet, so the turn that spends the last of
        # the stake is the first one that counts toward a streak.
        below = money_array[1:] < float(insolvency_floor)
        grace = max(1, int(insolvency_grace))
        if grace > 1:
            # Consecutive below-floor turns ending at each index, without a
            # python loop over 720 turns per episode times 1,200 episodes per
            # buffer fill. `reset` carries each solvent turn's own index and -1
            # everywhere else, so the running maximum is the position of the last
            # solvent turn at or before `i` and the difference is the run length.
            # A solvent turn resets to itself and scores 0, so `run >= grace`
            # already implies `below` for any grace of at least one.
            index = np.arange(below.shape[0])
            last_solvent = np.maximum.accumulate(np.where(~below, index, -1))
            below = (index - last_solvent) >= grace
        shaped[below] -= float(insolvency_penalty)

    if (land_weight or land_bonus) and quadrants is not None and days is not None:
        # `gamma * PHI(s') - PHI(s)`, the whole of it. Indexed off `money_array`,
        # which already carries one entry per reward step plus the terminal, so
        # the difference lines up with `shaped` without a second convention.
        #
        # Padded rather than truncated when the trace is short: the agent skips a
        # turn on a malformed observation and `money` skips with it, but a caller
        # that hands over fewer land samples than money samples has a real
        # mismatch, and holding the last known state is the reading that cannot
        # invent a purchase. Trailing entries pinned to the terminal zero anyway.
        width = money_array.shape[0]
        held_quadrants = np.ones(width, dtype=np.float64)
        held_days = np.zeros(width, dtype=np.float64)
        span = min(width, len(quadrants), len(days))
        held_quadrants[:span] = np.asarray(quadrants[:span], dtype=np.float64)
        held_days[:span] = np.asarray(days[:span], dtype=np.float64)
        if span < width:
            held_quadrants[span:] = held_quadrants[span - 1] if span else 1.0
            held_days[span:] = held_days[span - 1] if span else 0.0
        if land_weight:
            potential = land_potential(held_quadrants, held_days, land_weight)
            shaped += np.float32(gamma) * potential[1:] - potential[:-1]

        if land_bonus:
            # Keyed off the quadrant count rising, not off a `BUY_LAND` emission:
            # the simulator silently no-ops a purchase it will not sell, and a
            # refused order costs nothing and gains nothing, so paying for it
            # would reward asking rather than buying. Same rule the replay
            # instrumentation in `diagnostics.py` uses.
            #
            # Scaled by price so the ladder stays proportionate: the 1,000 plot
            # pays the full constant, the 2,000 half of it, the 4,000 a quarter.
            bought = np.diff(held_quadrants)
            for index in np.flatnonzero(bought > 0):
                # `held_quadrants[index]` is the count *before* this step, so the
                # extras already owned are that minus the always-unlocked NW.
                for step in range(int(bought[index])):
                    owned = int(held_quadrants[index]) - 1 + step
                    if not 0 <= owned < len(donor.LAND_PRICES):
                        continue
                    price = float(donor.LAND_PRICES[owned])
                    shaped[index] += np.float32(
                        float(land_bonus) * float(donor.LAND_PRICES[0]) / price
                    )

    if lot_weight and seeds is not None and standing is not None and stock is not None:
        # All three traces or none, the same contract the land pair uses: a
        # caller that cannot recover them stays on the stream it had rather than
        # silently scoring a different quantity.
        #
        # Padded exactly as the land trace is, and for the same reason -- the
        # agent skips a turn on a malformed observation and `money` skips with
        # it, so holding the last known basis is the reading that cannot invent a
        # purchase. Trailing entries are pinned to the terminal zero anyway.
        width = money_array.shape[0]
        held_seeds = np.zeros((width, len(lots.CROP_ITEMS)), dtype=np.float64)
        held_standing = np.zeros((width, len(lots.CROP_ITEMS)), dtype=np.float64)
        held_stock = np.zeros((width, len(lots.STOCK_ITEMS)), dtype=np.float64)
        span = min(width, len(seeds), len(standing), len(stock))
        held_seeds[:span] = np.asarray(seeds[:span], dtype=np.float64)
        held_standing[:span] = np.asarray(standing[:span], dtype=np.float64)
        held_stock[:span] = np.asarray(stock[:span], dtype=np.float64)
        if span < width and span:
            held_seeds[span:] = held_seeds[span - 1]
            held_standing[span:] = held_standing[span - 1]
            held_stock[span:] = held_stock[span - 1]
        potential = lot_potential(held_seeds, held_standing, held_stock, lot_weight)
        shaped += np.float32(gamma) * potential[1:] - potential[:-1]

    return shaped.astype(np.float32), raw.astype(np.float32), money_array
