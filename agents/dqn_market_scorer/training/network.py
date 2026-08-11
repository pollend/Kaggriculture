"""Torch mirror of `agents/dqn_market_scorer/policy.py`.

The two must produce identical outputs for identical weights;
`tests/test_dqn_market_scorer.py` pins that. Layer sizes and activation order
are read from the numpy module so they cannot be changed in one place only.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from agents.dqn_market_scorer.market_policy import MARKET_FEATURE_DIM
from agents.dqn_market_scorer.policy import HIDDEN, MarketQNetwork


def build(feature_dim: int = MARKET_FEATURE_DIM, hidden: tuple[int, ...] = HIDDEN) -> nn.Sequential:
    sizes = (feature_dim, *hidden)
    layers: list[nn.Module] = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers.append(nn.Linear(sizes[-1], 1))
    return nn.Sequential(*layers)


def to_numpy(module: nn.Sequential) -> MarketQNetwork:
    """Export to the inference network. Torch stores weights transposed."""
    layers = [
        (layer.weight.detach().cpu().numpy().T.copy(), layer.bias.detach().cpu().numpy().copy())
        for layer in module
        if isinstance(layer, nn.Linear)
    ]
    return MarketQNetwork(layers)


def from_numpy(network: MarketQNetwork, module: nn.Sequential | None = None) -> nn.Sequential:
    module = module or build()
    linears = [layer for layer in module if isinstance(layer, nn.Linear)]
    with torch.no_grad():
        for layer, (weight, bias) in zip(linears, network.layers):
            layer.weight.copy_(torch.from_numpy(np.ascontiguousarray(weight.T)))
            layer.bias.copy_(torch.from_numpy(bias))
    return module


def masked_max(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max over candidates, ignoring padding.

    Padded slots are pushed to -inf rather than zeroed: with zeroing, a padded
    row would win any state where every real Q is negative, quietly biasing the
    bootstrap upward.
    """
    return q_values.masked_fill(~mask, float("-inf")).max(dim=1).values


def masked_argmax(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return q_values.masked_fill(~mask, float("-inf")).argmax(dim=1)
