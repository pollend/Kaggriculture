"""Torch mirror of `agents/dqn_market_free/policy.py`.

The two must produce identical outputs for identical weights; the parity test in
`tests/test_dqn_market_free.py` pins that. Layer sizes and activation order are
read from the numpy module so they cannot be changed in one place only.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from agents.dqn_market_free.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free.order_space import ACTION_COUNT
from agents.dqn_market_free.policy import HIDDEN, OrderQNetwork


def build(
    state_dim: int = STATE_DIM,
    hidden: tuple[int, ...] = HIDDEN,
    actions: int = ACTION_COUNT,
) -> nn.Sequential:
    sizes = (state_dim, *hidden)
    layers: list[nn.Module] = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers.append(nn.Linear(sizes[-1], actions))
    return nn.Sequential(*layers)


def to_numpy(module: nn.Sequential) -> OrderQNetwork:
    """Export to the inference network. Torch stores weights transposed."""

    layers = [
        (layer.weight.detach().cpu().numpy().T.copy(), layer.bias.detach().cpu().numpy().copy())
        for layer in module
        if isinstance(layer, nn.Linear)
    ]
    return OrderQNetwork(layers)


def from_numpy(network: OrderQNetwork, module: nn.Sequential | None = None) -> nn.Sequential:
    module = module or build()
    linears = [layer for layer in module if isinstance(layer, nn.Linear)]
    with torch.no_grad():
        for layer, (weight, bias) in zip(linears, network.layers):
            layer.weight.copy_(torch.from_numpy(np.ascontiguousarray(weight.T)))
            layer.bias.copy_(torch.from_numpy(bias))
    return module


def has_inert_hidden_trunk(network: OrderQNetwork) -> bool:
    """True when every hidden weight is exactly zero.

    Such a network scores every action identically, so the argmax ties to index
    0 and the agent emits nothing at all. It is also ungradable: an all-zero MLP
    has no gradient outside its output bias, so a run that reaches this state can
    never leave it.
    """

    hidden = network.layers[:-1]
    return bool(hidden) and all(not weight.any() for weight, _ in hidden)
