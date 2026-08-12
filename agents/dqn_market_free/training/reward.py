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
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from agents.dqn_market_free import donor

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
#: Sized from the observed margin range, because the previous 20,000 with a hard
#: clip was informationless: measured across 640 recorded episodes the margin ran
#: from -21,681 to -210,076, so `clip(margin / 20_000, -1, 1)` returned exactly
#: -1.0 on every one of them. A constant on the terminal reward shifts every Q
#: value uniformly and can never change an argmax, so the one term tying training
#: to what the promotion gate measures was contributing nothing.
#:
#: At 100,000 that same range maps to -0.21 .. -0.97, which discriminates, and a
#: run that closes on the donor sees the term shrink toward zero and cross into
#: positive as it should.
TERMINAL_MARGIN_SCALE = 100_000.0


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


def episode_rewards(
    money: list[float],
    potentials: list[float],
    final_money: float,
    opponent_money: float,
    gamma: float,
    emitted: Sequence[bool] | None = None,
    rejected_penalty: float = REJECTED_ORDER_PENALTY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn one episode's money and potential traces into a reward stream.

    Returns `(shaped, raw, phi)`. Shared by live rollouts and by demonstrations
    read out of stored replays, so a cloned episode and a self-collected one are
    the same kind of sample and can sit in one buffer -- and cannot drift apart.

    `emitted[t]` says whether the policy put an order on the wire that turn. A
    turn that emitted one and moved no money had it refused: every order the
    simulator accepts moves cash by at least a coin, since prices floor at 1 and
    quantities start at 1.
    """

    money_array = np.asarray([*money, final_money], dtype=np.float32)
    # The environment ends in an absorbing state. Forcing Phi(s_T)=0 is what
    # makes the shaping policy-invariant, and it also stops shaping from
    # assigning terminal value to inventory that scores nothing.
    phi = np.asarray([*potentials, 0.0], dtype=np.float32)

    raw = (money_array[1:] - money_array[:-1]) / MONEY_SCALE
    shaped = raw + gamma * phi[1:] - phi[:-1]
    # tanh rather than a hard clip. Both bound the term to (-1, 1) so one blowout
    # episode cannot dominate a batch, but a clip is flat outside its range --
    # and every margin this agent has produced so far was outside it, which made
    # the term a constant. tanh keeps discriminating everywhere and still
    # compresses the tail.
    shaped[-1] += float(np.tanh((final_money - opponent_money) / TERMINAL_MARGIN_SCALE))

    if emitted is not None and rejected_penalty:
        tried = np.zeros(raw.shape[0], dtype=bool)
        width = min(len(emitted), tried.shape[0])
        tried[:width] = np.asarray(emitted[:width], dtype=bool)
        shaped[tried & (raw == 0.0)] -= float(rejected_penalty)

    return shaped.astype(np.float32), raw.astype(np.float32), money_array
