"""Torch mirror of `agents/dqn_market_free/policy.py`.

The two must produce identical outputs for identical weights; the parity test in
`tests/test_dqn_market_free.py` pins that. Layer sizes and activation order are
read from the numpy module so they cannot be changed in one place only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agents.dqn_market_free.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free.order_space import ACTION_COUNT
from agents.dqn_market_free.policy import HIDDEN, OrderQNetwork

NOISY_LAYERS = ("none", "head", "all")


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer (Fortunato et al., 2018).

    Deliberately **not** a subclass of `nn.Linear`, for the exporter's sake: a
    subclass passes an `isinstance(layer, nn.Linear)` filter, so whichever tensor
    `layer.weight` happened to name would be exported. Name anything but the
    trained mean and the run ships a randomly initialized policy that
    `has_inert_hidden_trunk` waves through -- the gate then rejects the candidate
    and every metric points at the policy instead of the export. `_exportable`
    dispatches on this class positively instead.
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma0 = float(sigma0)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        # Not persistent: the target network is synced with
        # `load_state_dict(online.state_dict())`, so a persistent epsilon would copy
        # the online network's *current* noise into the target and correlate the two
        # sources until the next sync -- silently, and invisibly in every metric.
        self.register_buffer("epsilon_in", torch.zeros(in_features), persistent=False)
        self.register_buffer("epsilon_out", torch.zeros(out_features), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma0 * bound)
        nn.init.constant_(self.bias_sigma, self.sigma0 * bound)

    @staticmethod
    def _factorized(size: int, device, generator: torch.Generator | None) -> torch.Tensor:
        draw = torch.randn(size, device=device, generator=generator)
        return draw.sign() * draw.abs().sqrt()

    def reset_noise(
        self,
        generator: torch.Generator | None = None,
        scale: float | Sequence[float] | torch.Tensor = 1.0,
    ) -> None:
        """Draw a fresh epsilon pair, optionally scaled.

        Rebinds the buffers rather than filling them in place: the TD loop resamples
        *between* forward passes, and mutating a tensor autograd saved for backward
        corrupts the gradient or raises.

        `scale` multiplies the *draw*, never `sigma`. The perturbation is bilinear in
        the two epsilons, so folding the factor into one scales the whole term
        exactly -- and sigma stays a trained parameter, so a scaled forward cannot
        leak into the checkpoint or compound across calls.

        A per-output vector is also accepted, applying only to a layer whose width it
        matches: directed exploration, widening one action's row. The numpy mirror in
        `policy.py` takes the same argument under the same rule.
        """

        device = self.weight_mu.device
        self.epsilon_in = self._factorized(self.in_features, device, generator)
        out = self._factorized(self.out_features, device, generator)
        if isinstance(scale, (int, float)):
            self.epsilon_out = out if scale == 1.0 else out * float(scale)
            return
        factor = torch.as_tensor(scale, dtype=out.dtype, device=device)
        self.epsilon_out = out * factor if factor.shape == out.shape else out

    def zero_noise(self) -> None:
        self.epsilon_in = torch.zeros(self.in_features, device=self.weight_mu.device)
        self.epsilon_out = torch.zeros(self.out_features, device=self.weight_mu.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight_mu, self.bias_mu)
        if not bool(self.epsilon_out.any()):
            return out
        # The factored identity, not a materialized `mu + sigma * outer(...)`.
        # Both are algebraically the same; only this one associates in the same
        # order as the numpy mirror, which is what keeps the parity test's 1e-5
        # tolerance honest in float32. It also avoids allocating a full weight
        # matrix per forward.
        f_in, f_out = self.epsilon_in, self.epsilon_out
        return out + F.linear(x * f_in, self.weight_sigma) * f_out + self.bias_sigma * f_out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, sigma0={self.sigma0}"


def reset_noise(
    module: nn.Module,
    generator: torch.Generator | None = None,
    scale: float = 1.0,
) -> None:
    for layer in module.modules():
        if isinstance(layer, NoisyLinear):
            layer.reset_noise(generator, scale)


def set_noise(module: nn.Module, enabled: bool) -> None:
    """Enable or silence every noisy layer.

    This agent's stand-in for `.train()` / `.eval()`, which nothing in `agents/`
    calls. Two explicit, greppable call sites beat a mode flag that can be
    forgotten in silence.
    """

    for layer in module.modules():
        if isinstance(layer, NoisyLinear):
            if enabled:
                layer.reset_noise()
            else:
                layer.zero_noise()


def build(
    state_dim: int = STATE_DIM,
    hidden: tuple[int, ...] = HIDDEN,
    actions: int = ACTION_COUNT,
    *,
    noisy: str = "none",
    sigma0: float = 0.5,
) -> nn.Sequential:
    """`noisy` defaults to "none", so every existing caller builds today's module.

    "head" puts noise on the action head alone, which is the recommended first
    setting: it leaves `has_inert_hidden_trunk` -- the only structural guard
    against the NOOP collapse -- correct as written, and keeps the trunk shared
    with the behaviour-clone cross-entropy head deterministic.
    """

    if noisy not in NOISY_LAYERS:
        raise ValueError(f"unknown noisy setting {noisy!r}; expected one of {NOISY_LAYERS}")
    sizes = (state_dim, *hidden)

    def linear(a: int, b: int, *, head: bool) -> nn.Module:
        if noisy == "all" or (noisy == "head" and head):
            return NoisyLinear(a, b, sigma0=sigma0)
        return nn.Linear(a, b)

    layers: list[nn.Module] = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [linear(a, b, head=False), nn.ReLU()]
    layers.append(linear(sizes[-1], actions, head=True))
    return nn.Sequential(*layers)


def _exportable(module: nn.Sequential) -> list[nn.Module]:
    """Every parameter-bearing layer, in order, or a hard failure.

    Raises rather than skips, which is the point. A filter that silently ignores an
    unrecognised layer produces a *short* network: the gate inspects `layers[:-1]`,
    sees a nonzero weight and waves it through, and the first evidence is a shape
    error inside an arena worker an update later. An unrecognised parameterised
    layer means this exporter is out of date and nothing downstream can detect that.
    Activations carry no parameters and are the one thing passed over.
    """

    exportable: list[nn.Module] = []
    for layer in module:
        if isinstance(layer, (nn.Linear, NoisyLinear)):
            exportable.append(layer)
        elif any(True for _ in layer.parameters(recurse=True)):
            raise TypeError(
                f"{type(layer).__name__} carries parameters that "
                f"{__name__}.to_numpy cannot export"
            )
    return exportable


def _out(tensor: torch.Tensor) -> np.ndarray:
    """Torch stores weights transposed relative to the inference network."""

    array = tensor.detach().cpu().numpy()
    return array.T.copy() if array.ndim == 2 else array.copy()


def to_numpy(module: nn.Sequential) -> OrderQNetwork:
    """Export to the inference network, carrying sigma where a layer has it."""

    layers: list[tuple[np.ndarray, np.ndarray]] = []
    sigma: list[tuple[np.ndarray, np.ndarray] | None] = []
    for layer in _exportable(module):
        if isinstance(layer, NoisyLinear):
            layers.append((_out(layer.weight_mu), _out(layer.bias_mu)))
            sigma.append((_out(layer.weight_sigma), _out(layer.bias_sigma)))
        else:
            layers.append((_out(layer.weight), _out(layer.bias)))
            sigma.append(None)
    return OrderQNetwork(layers, sigma)


def from_numpy(network: OrderQNetwork, module: nn.Sequential | None = None) -> nn.Sequential:
    """Load mu, and sigma only where the checkpoint carries it.

    A checkpoint written before NoisyNet has no sigma, and leaving torch's own
    sigma at its `sigma0` initialization there is both the backward-compatible
    resume path and the cold start. It has to work that way: sigma = 0 is a
    stationary point of the loss -- `dL/dsigma = L'(W) * E[eps] = 0` for
    zero-mean eps -- so a sigma that arrives as zero could never grow, exactly as
    an all-zero MLP can never break symmetry. Sigma is therefore born here, in
    torch, and never in the numpy cold start.
    """

    module = module or build()
    linears = _exportable(module)
    # `zip` alone would truncate in silence: a two-layer checkpoint loaded into a
    # three-layer module leaves the last layer at torch's initialization, and the
    # run trains from a partially random start without a word.
    if len(linears) != len(network.layers):
        raise ValueError(
            f"module has {len(linears)} parameterised layers, "
            f"checkpoint has {len(network.layers)}"
        )

    def load(destination: torch.Tensor, source: np.ndarray) -> None:
        array = np.ascontiguousarray(source.T) if source.ndim == 2 else source
        destination.copy_(torch.from_numpy(array))

    with torch.no_grad():
        for layer, (weight, bias), pair in zip(linears, network.layers, network.sigma):
            if isinstance(layer, NoisyLinear):
                load(layer.weight_mu, weight)
                load(layer.bias_mu, bias)
                if pair is not None:
                    load(layer.weight_sigma, pair[0])
                    load(layer.bias_sigma, pair[1])
            else:
                if pair is not None:
                    raise ValueError(
                        "checkpoint carries sigma for a layer the module builds as "
                        "deterministic; the collector and the updater disagree about "
                        "`noisy_layers`"
                    )
                load(layer.weight, weight)
                load(layer.bias, bias)
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
