"""Batch Glicko-2 ratings for one generation-sized rating period."""

from __future__ import annotations

import math
from dataclasses import dataclass

GLICKO_SCALE = 173.7178


@dataclass(frozen=True, slots=True)
class GlickoRating:
    rating: float = 1500.0
    deviation: float = 350.0
    volatility: float = 0.06
    games: int = 0


@dataclass(frozen=True, slots=True)
class RatingGame:
    opponent: GlickoRating
    score: float


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _expected(mu: float, opponent_mu: float, opponent_phi: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(opponent_phi) * (mu - opponent_mu)))


def update_glicko2(
    current: GlickoRating,
    games: list[RatingGame],
    *,
    tau: float = 0.5,
    epsilon: float = 1e-6,
) -> GlickoRating:
    """Update one player from a complete, frozen rating period."""

    mu = (current.rating - 1500.0) / GLICKO_SCALE
    phi = current.deviation / GLICKO_SCALE
    if not games:
        new_phi = math.sqrt(phi * phi + current.volatility * current.volatility)
        return GlickoRating(
            current.rating,
            min(350.0, new_phi * GLICKO_SCALE),
            current.volatility,
            current.games,
        )

    converted = [
        (
            (game.opponent.rating - 1500.0) / GLICKO_SCALE,
            game.opponent.deviation / GLICKO_SCALE,
            game.score,
        )
        for game in games
    ]
    variance_inverse = sum(
        _g(opponent_phi) ** 2
        * _expected(mu, opponent_mu, opponent_phi)
        * (1.0 - _expected(mu, opponent_mu, opponent_phi))
        for opponent_mu, opponent_phi, _ in converted
    )
    variance = 1.0 / variance_inverse
    delta = variance * sum(
        _g(opponent_phi) * (score - _expected(mu, opponent_mu, opponent_phi))
        for opponent_mu, opponent_phi, score in converted
    )

    alpha = math.log(current.volatility * current.volatility)

    def objective(value: float) -> float:
        exp_value = math.exp(value)
        numerator = exp_value * (delta * delta - phi * phi - variance - exp_value)
        denominator = 2.0 * (phi * phi + variance + exp_value) ** 2
        return numerator / denominator - (value - alpha) / (tau * tau)

    lower = alpha
    if delta * delta > phi * phi + variance:
        upper = math.log(delta * delta - phi * phi - variance)
    else:
        step = 1
        upper = alpha - step * tau
        while objective(upper) < 0:
            step += 1
            upper = alpha - step * tau

    f_lower = objective(lower)
    f_upper = objective(upper)
    while abs(upper - lower) > epsilon:
        candidate = lower + (lower - upper) * f_lower / (f_upper - f_lower)
        f_candidate = objective(candidate)
        if f_candidate * f_upper <= 0:
            lower = upper
            f_lower = f_upper
        else:
            f_lower /= 2.0
        upper = candidate
        f_upper = f_candidate

    volatility = math.exp(lower / 2.0)
    pre_rating_phi = math.sqrt(phi * phi + volatility * volatility)
    new_phi = 1.0 / math.sqrt(1.0 / (pre_rating_phi * pre_rating_phi) + 1.0 / variance)
    new_mu = mu + new_phi * new_phi * sum(
        _g(opponent_phi) * (score - _expected(mu, opponent_mu, opponent_phi))
        for opponent_mu, opponent_phi, score in converted
    )
    return GlickoRating(
        rating=1500.0 + GLICKO_SCALE * new_mu,
        deviation=GLICKO_SCALE * new_phi,
        volatility=volatility,
        games=current.games + len(games),
    )

