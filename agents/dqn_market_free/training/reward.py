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

from agents.dqn_market_free.market_state import order_refused

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

    All three terms are deliberately not policy-invariant: the outcome growth,
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
        # the stake is the first one charged.
        shaped[money_array[1:] < float(insolvency_floor)] -= float(insolvency_penalty)

    return shaped.astype(np.float32), raw.astype(np.float32), money_array
