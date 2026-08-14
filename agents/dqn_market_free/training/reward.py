"""Reward and its scales for the free-market learner.

Local rather than imported from `agents/dqn_market_scorer/training/collect.py`,
which is where this started: the scorer's market layer decides land and cash
reserves in code, so sharing the constants would have meant changing the
scorer's reward to fix this one.

The reward is the episode's own outcome, the tanh of the final margin. It is
paid twice: as a lump on the terminal step, and as a per-step annuity of
`(1 - gamma) * outcome` on every step. The two sum to a flat `outcome` in Q at
every timestep, so a state on a winning trajectory is valued above the
same-looking state on a losing one no matter when in the season it was seen.

Three small per-turn charges sit alongside it: an order the simulator refused, a
turn that emitted nothing at all, and a wallet with no runway left. None is
policy-invariant -- that is the point, and with the money delta gone they are the
only terms that separate two actions at the immediate-reward level.

They are ordered deliberately: an accepted order pays nothing, NOOP pays
`IDLE_PENALTY`, and a refused order pays the larger `REJECTED_ORDER_PENALTY`. So
trading beats sitting out, and sitting out beats emitting junk.

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

#: Coins of final margin at which the terminal term reaches tanh(1) = 0.76.
#:
#: Sized to the *self-play* margin distribution, which is the one this term has
#: to discriminate. Against the scripted field it never did: measured across 640
#: recorded episodes the margin ran from -21,681 to -210,076, so the original
#: 20,000 with a hard clip returned exactly -1.0 on every one of them, and 100,000
#: only moved the saturation point -- by the last recorded generation every
#: episode margin was about -128,000, giving a constant -0.86. A constant on the
#: terminal reward shifts every Q value uniformly and can never change an argmax,
#: so the one term tying training to what the promotion gate measures was
#: contributing nothing either way.
#:
#: Against its own mirror the margins are hundreds to low thousands, where
#: `tanh(margin / 100_000)` is about 0.01 -- a vanishing term rather than a
#: saturated one, which is no better. At 10,000 a 2,000-coin mirror margin maps
#: to 0.20 and discriminates.
#:
#: The anchor games stay saturated at this scale, deliberately. Their job in the
#: mix is the absolute money signal, which the per-step delta already carries;
#: the margin term is there for the mirror.
TERMINAL_MARGIN_SCALE = 10_000.0

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
#: a fully idle season cost 0.36 while `tanh(margin / TERMINAL_MARGIN_SCALE)` only
#: spans 0.19 to 0.29 between 1,898 and 3,000 coins -- so trading yourself *down*
#: to 1,898 scored 0.257 better than sitting on 3,000, and the policy correctly
#: learned to trade on 73-99% of turns and lose money. Measured on
#: `market-free-hot-v16`: mean money fell 3,542 to 1,898 while `mean_abs_q` rose
#: 0.034 to 0.133 -- a Q function converging cleanly on a bad objective.
#:
#: At 0.0002 with the falloff the season total is 0.072, comfortably under the
#: 0.104 outcome gap that mistake turned on, so losing money to stay busy is never
#: worth it. The per-turn ordering still holds at every point in the season:
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
    rejected_penalty: float = REJECTED_ORDER_PENALTY,
    idle_penalty: float = IDLE_PENALTY,
    insolvency_floor: float = INSOLVENCY_FLOOR,
    insolvency_penalty: float = INSOLVENCY_PENALTY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn one episode's money trace into a reward stream.

    Returns `(shaped, raw, money)`. Shared by live rollouts and by demonstrations
    read out of stored replays, so a cloned episode and a self-collected one are
    the same kind of sample and can sit in one buffer -- and cannot drift apart.

    The stream is the terminal margin, spread as an annuity, minus the refusal
    and insolvency charges. It took a `mode` and a potential trace until v13: the
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

    All three terms are deliberately not policy-invariant: the outcome margin,
    the refused-order charge, and the insolvency charge. Each is documented at
    its constant.

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
    # tanh rather than a hard clip. Both bound the term to (-1, 1) so one blowout
    # episode cannot dominate a batch, but a clip is flat outside its range --
    # and every margin this agent has produced so far was outside it, which made
    # the term a constant. tanh keeps discriminating everywhere and still
    # compresses the tail.
    outcome = float(np.tanh((final_money - opponent_money) / TERMINAL_MARGIN_SCALE))
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
