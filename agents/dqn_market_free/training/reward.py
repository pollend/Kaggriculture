"""Reward, potential shaping, and their scales for the free-market learner.

Local rather than imported from `agents/dqn_market_scorer/training/collect.py`,
which is where this started. The scorer's market layer decides land and cash
reserves in code, so its potential function never had to account for them; here
the network decides, and two of the three numbers below have to differ from the
scorer's as a result. Sharing them would have meant changing the scorer's reward
to fix this one.

The reward is the money delta plus potential-based shaping:

    r_t = (money_{t+1} - money_t) / MONEY_SCALE + gamma * Phi(s_{t+1}) - Phi(s_t)

Shaping of this form is policy-invariant as long as the terminal potential is
forced to zero (Ng, Harada & Russell 1999), which `episode_rewards` does. It
cannot change which policy is optimal -- only how quickly the network finds it --
so `Phi` is free to price anything that is worth something, and it should.

On top of that sits the episode's own outcome, the tanh of the final margin. It
is paid twice: as a lump on the terminal step, and as a per-step annuity of
`(1 - gamma) * outcome` on every step of the episode. The two sum to a flat
`outcome` in Q at every timestep, so a state on a winning trajectory is valued
above the same-looking state on a losing one no matter when in the season it was
seen. Unlike the shaping, this is not policy-invariant -- that is the point.

Two small per-turn charges sit alongside it, also not policy-invariant, for the
two states the shaping cannot price because it prices everything at zero: an
order the simulator refused, and a wallet with no runway left.

`REWARD_MODES` selects whether the first two terms are present at all. The
`outcome_only` mode drops the money delta and the shaping together, leaving the
episode outcome and the two charges -- see the constant for what that buys and
what it costs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from agents.dqn_market_free import donor
from agents.dqn_market_free.market_state import order_refused

PRODUCTS: tuple[str, ...] = tuple(donor.PRODUCTS)

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

#: How `episode_rewards` builds the stream.
#:
#: ``shaped`` is the reward this file was written for: money delta, potential
#: shaping, outcome, penalties.
#:
#: ``outcome_only`` drops the money delta *and* the potential shaping, leaving
#: the outcome annuity plus the two penalties. The pair goes together -- `Phi`
#: exists only to cancel the cash leg of a purchase (see `_land_value`), so
#: dropping the delta and keeping `Phi` would price a buy at `+Phi` and a sale at
#: `-Phi`, which is the wrong sign on both. Dropping `Phi` and keeping the delta
#: is the opposite mistake: every purchase becomes a pure loss and every sale a
#: pure gain, which is the churn this mode exists to stop.
#:
#: What it costs, stated plainly because the tests cannot: the immediate reward
#: becomes identical for every action on a turn, so `REJECTED_ORDER_PENALTY` and
#: `INSOLVENCY_PENALTY` are the *only* terms separating any two actions at the
#: immediate-reward level. Everything else has to arrive through the bootstrap
#: from one scalar per episode. And `tanh` saturates: measured margins against
#: the scripted anchors run -26,000 to -134,000, which is a flat -1.0 at
#: `TERMINAL_MARGIN_SCALE`, so those episodes carry a constant and only the
#: self-play mirror games discriminate. The answer is to stop playing the
#: saturated sources rather than to rescale them -- `market-free-mirror-v5` runs
#: no scripted opponent and no demonstrations for exactly this reason; see
#: `docs/dqn_market_free_roadmap.md`.
REWARD_MODES: tuple[str, ...] = ("shaped", "outcome_only")


def _price(prices: dict, item: str) -> float:
    return float(prices.get(item, donor.MARKET[item][0]) or donor.MARKET[item][0])


def _land_value(farm: dict) -> float:
    """What the farm has spent unlocking quadrants.

    Without this, `BUY_LAND` is the one action whose cost the shaping never
    returns: 1,000 to 4,000 coins of immediate negative reward whose payoff
    arrives only as extra planting several days later. Every other capital
    purchase nets to about zero on the turn it happens -- seeds and animals leave
    cash and enter `Phi` at the price paid -- so land was uniquely punished and
    the network's cheapest response is to stay in the starting quadrant, on a
    quarter of the board.

    Priced at cost. Land cannot be sold, so this is not liquidation value; it is
    there to stop the shaping from lying about a purchase being a pure loss.
    """

    extra = max(0, len(farm.get("unlocked_quadrants", ["NW"]) or ["NW"]) - 1)
    return float(sum(donor.LAND_PRICES[:extra]))


def potential(farm: dict, private: dict, obs: Any) -> float:
    """Phi(s): everything the farm owns that cash was spent on, scaled.

    Cash is deliberately absent because the base reward already records its
    delta; counting it twice would double the weight of every fill.
    """

    prices = (obs.get("market", {}) or {}).get("prices", {}) or {}

    value = _land_value(farm)

    shed = private.get("shed", {}) or {}
    for item in PRODUCTS:
        value += int(shed.get(item, 0) or 0) * _price(prices, item)
    seeds = private.get("seeds", {}) or {}
    for crop, rule in donor.CROPS.items():
        value += int(seeds.get(crop, 0) or 0) * float(rule["seed"])
    for animal, rule in donor.ANIMALS.items():
        value += int(shed.get(animal, 0) or 0) * float(rule["cost"])
    for slot in private.get("inventories", []) or []:
        for item in PRODUCTS:
            value += int((slot or {}).get(item, 0) or 0) * _price(prices, item)
        for animal, rule in donor.ANIMALS.items():
            value += int((slot or {}).get(animal, 0) or 0) * float(rule["cost"])

    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                crop = str(tile.get("crop", ""))
                if crop in donor.CROPS:
                    value += int(tile.get("yield_units", 0) or 0) * _price(prices, crop)
            animal = tile.get("animal")
            if animal in donor.ANIMALS:
                product = str(donor.ANIMALS[str(animal)]["product"])
                value += int(tile.get("yield_units", 0) or 0) * _price(prices, product)
                value += int(bool(tile.get("fertilizer_available"))) * _price(prices, "FERTILIZER")
                # A live animal keeps producing; half a unit of its product is a
                # cheap stand-in for that stream and keeps culling unattractive.
                value += 0.5 * _price(prices, product)
    return value / MONEY_SCALE


#: Charged on a turn that emitted an order the simulator then refused.
#:
#: The shaping deliberately leaves every sensible purchase at exactly zero
#: immediate reward -- cash out, `Phi` in at the price paid -- so NOOP, BUY_SEED
#: and an impossible order all score 0 on the turn they happen, and only differ
#: many turns later. An untrained value function sees a flat plateau and picks
#: off it arbitrarily. Measured: one run settled on `SELL WOOL 16` for 57% of its
#: decisions while owning no sheep, with 98.4% of everything it emitted refused.
#:
#: This charges for the refusal alone, so it separates "interacted with the
#: world" from "did not" without expressing any view on *which* trade is good --
#: an impossible order falls below NOOP, a real purchase rises above it, and the
#: choice among real purchases is left exactly where it was. It is not
#: policy-invariant, unlike the potential shaping, so it is kept small.
REJECTED_ORDER_PENALTY = 0.001

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
#: nothing. Like `REJECTED_ORDER_PENALTY` and unlike the potential shaping, it is
#: not policy-invariant -- it is meant to change which policy is optimal -- so it
#: is kept small and states a floor rather than a preference about trades.
INSOLVENCY_PENALTY = 0.002


def episode_rewards(
    money: list[float],
    potentials: list[float],
    final_money: float,
    opponent_money: float,
    gamma: float,
    emitted: Sequence[bool] | None = None,
    rejected_penalty: float = REJECTED_ORDER_PENALTY,
    insolvency_floor: float = INSOLVENCY_FLOOR,
    insolvency_penalty: float = INSOLVENCY_PENALTY,
    mode: str = "shaped",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn one episode's money and potential traces into a reward stream.

    Returns `(shaped, raw, phi)`. Shared by live rollouts and by demonstrations
    read out of stored replays, so a cloned episode and a self-collected one are
    the same kind of sample and can sit in one buffer -- and cannot drift apart.

    `emitted[t]` says whether the policy put an order on the wire that turn. A
    turn that emitted one and moved no money had it refused: every order the
    simulator accepts moves cash by at least a coin, since prices floor at 1 and
    quantities start at 1.

    On top of the shaping sit three terms that are deliberately not
    policy-invariant: the outcome margin, the refused-order charge, and the
    insolvency charge. Each is documented at its constant.

    `mode` selects whether the money delta and the potential shaping are in the
    stream at all; see `REWARD_MODES`. `raw` is returned either way, so the
    money-delta diagnostics an update logs mean the same thing in both modes and
    `order_refused` keeps reading a real cash trace.
    """

    if mode not in REWARD_MODES:
        raise ValueError(f"unknown reward mode {mode!r}, expected one of {REWARD_MODES}")

    money_array = np.asarray([*money, final_money], dtype=np.float32)
    # The environment ends in an absorbing state. Forcing Phi(s_T)=0 is what
    # makes the shaping policy-invariant, and it also stops shaping from
    # assigning terminal value to inventory that scores nothing.
    phi = np.asarray([*potentials, 0.0], dtype=np.float32)

    raw = (money_array[1:] - money_array[:-1]) / MONEY_SCALE
    # Always computed: `order_refused` reads it to tell a refused order from an
    # accepted one, and the update logs money diagnostics off it, so both have to
    # keep working when the stream itself no longer carries the delta.
    if mode == "outcome_only":
        shaped = np.zeros_like(raw)
    else:
        shaped = raw + gamma * phi[1:] - phi[:-1]
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

    if emitted is not None and rejected_penalty:
        tried = np.zeros(raw.shape[0], dtype=bool)
        width = min(len(emitted), tried.shape[0])
        tried[:width] = np.asarray(emitted[:width], dtype=bool)
        # `order_refused` lives in `market_state` because the state encodes the
        # same fact -- a refusal streak the policy can see. One definition, so a
        # feature saying "refused" and a reward charging for it cannot disagree.
        shaped[order_refused(tried, raw)] -= float(rejected_penalty)

    if insolvency_penalty:
        # Read off the *resulting* wallet, so the turn that spends the last of
        # the stake is the first one charged.
        shaped[money_array[1:] < float(insolvency_floor)] -= float(insolvency_penalty)

    return shaped.astype(np.float32), raw.astype(np.float32), money_array
