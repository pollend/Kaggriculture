"""Torch mirror of `agents/dqn_market_free_v3c/policy.py`.

The two must produce identical outputs for identical weights; the parity test in
`tests/test_dqn_market_free_v3c.py` pins that. Layer sizes and activation order are
read from the numpy module so they cannot be changed in one place only.

v3 adds a recurrence, and with it two structural changes to this file. The module
is no longer an `nn.Sequential` -- it cannot be, because the GRU ticks once a turn
and the head runs once a sub-decision -- so `_exportable`'s isinstance walk is no
longer a proof that every parameter was exported, and `_decompose` replaces it
with parameter-identity accounting. And the GRU is deliberately never noisy, which
is what makes the TD step's `reset_noise` ordering legal; see `OrderQModule`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agents.dqn_market_free_v3c.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free_v3c.order_space import ACTION_COUNT
from agents.dqn_market_free_v3c.policy import (
    HIDDEN,
    HIDDEN_SIZE,
    GruCell,
    OrderQNetwork,
)

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


def build_head(
    head_in: int,
    hidden: tuple[int, ...] = HIDDEN,
    actions: int = ACTION_COUNT,
    *,
    noisy: str = "none",
    sigma0: float = 0.5,
) -> nn.Sequential:
    """The action head alone, over an already-concatenated input.

    "head" puts noise on the action head's last layer alone, which is the
    recommended first setting: it leaves `has_inert_hidden_trunk` -- the only
    structural guard against the NOOP collapse -- correct as written, and keeps the
    trunk shared with the behaviour-clone cross-entropy head deterministic.
    """

    if noisy not in NOISY_LAYERS:
        raise ValueError(f"unknown noisy setting {noisy!r}; expected one of {NOISY_LAYERS}")
    sizes = (head_in, *hidden)

    def linear(a: int, b: int, *, head: bool) -> nn.Module:
        if noisy == "all" or (noisy == "head" and head):
            return NoisyLinear(a, b, sigma0=sigma0)
        return nn.Linear(a, b)

    layers: list[nn.Module] = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [linear(a, b, head=False), nn.ReLU()]
    layers.append(linear(sizes[-1], actions, head=True))
    return nn.Sequential(*layers)


class OrderQModule(nn.Module):
    """A GRU at turn rate, and v2's action head at sub-decision rate.

    **Holds no hidden state.** Every entry point takes it as an argument and
    returns it. The replay buffer owns the stored state, and a module that also
    cached one would give two sources of truth that disagree only under burn-in --
    silently, and only on resumed runs.

    **The GRU is never noisy, at any `noisy` setting**, and that fact is
    load-bearing well outside this class. v2's TD step is ordered so the no-grad
    bootstrap runs first, because `reset_noise` rebinds the epsilon buffers and a
    resample landing between a forward and its backward corrupts the gradient
    (`training/updater.py`). v3 appears to break that rule: the *graded* GRU
    rollout happens before two `reset_noise` calls. It is legal only because
    nothing in the recurrent graph references an epsilon buffer. Put a
    `NoisyLinear` in the cell and the ordering silently becomes wrong.

    `nn.GRUCell`, not `nn.GRU`. Three reasons, in order of how much they cost to
    get wrong: `nn.GRU` may dispatch to a fused cuDNN kernel whose arithmetic the
    numpy mirror cannot match to 1e-5; it packs its tensors into `_flat_weights`
    rather than the four named parameters `to_numpy` exports; and it cannot express
    the per-step validity gate `recur` needs for a left-padded chunk.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_size: int = HIDDEN_SIZE,
        hidden: tuple[int, ...] = HIDDEN,
        actions: int = ACTION_COUNT,
        *,
        noisy: str = "none",
        sigma0: float = 0.5,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.cell = nn.GRUCell(self.state_dim, self.hidden_size, bias=True)
        self.head = build_head(
            self.state_dim + self.hidden_size,
            hidden,
            actions,
            noisy=noisy,
            sigma0=sigma0,
        )

    def initial_hidden(self, batch: int, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(
            batch,
            self.hidden_size,
            device=self.cell.weight_hh.device if device is None else device,
            dtype=dtype,
        )

    def tick(self, turn_features: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """`[B, S]`, `[B, H]` -> `[B, H]`. One *turn*, never one sub-decision."""

        return self.cell(turn_features, hidden)

    def recur(
        self,
        turn_input: torch.Tensor,
        hidden: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`[B, L, S]`, `[B, H]` -> `[B, L, H]`, gated so an invalid step is a no-op.

        `valid` is False on the left pad of a chunk that starts before turn 0 and on
        the tail past an episode's end. Where it is False the state is carried
        forward unchanged.

        Gating rather than feeding the padded step zeros, and the difference is not
        cosmetic: a GRU fed a zero input from a zero state settles on
        `(1 - z) * tanh(b_in + b_hn)`, not on zero, so a left-padded chunk fed zeros
        would begin its burn-in from a bias artefact that grows with every pad step.
        This is the one place in the sequence machinery where the obvious
        implementation is silently wrong.
        """

        out = []
        for step in range(turn_input.shape[1]):
            advanced = self.cell(turn_input[:, step], hidden)
            if valid is None:
                hidden = advanced
            else:
                hidden = torch.where(valid[:, step].unsqueeze(1), advanced, hidden)
            out.append(hidden)
        return torch.stack(out, dim=1)

    def head_forward(self, states: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """`[..., S]`, `[..., H]` -> `[..., A]`. Never advances the recurrence."""

        hidden = hidden.expand(*states.shape[:-1], self.hidden_size)
        return self.head(torch.cat((states, hidden), dim=-1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """`[N, S + H]` -> `[N, A]`: the head over an already-concatenated input.

        Kept as `forward` so the behaviour-clone step and the flip-rate probe, which
        take a module and a feature matrix, need no knowledge of the recurrence.
        """

        return self.head(features)


def build(
    state_dim: int = STATE_DIM,
    hidden: tuple[int, ...] = HIDDEN,
    actions: int = ACTION_COUNT,
    *,
    hidden_size: int = HIDDEN_SIZE,
    noisy: str = "none",
    sigma0: float = 0.5,
) -> OrderQModule:
    """`noisy` defaults to "none", so every existing caller builds today's module."""

    return OrderQModule(
        state_dim,
        hidden_size,
        hidden,
        actions,
        noisy=noisy,
        sigma0=sigma0,
    )


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


def _decompose(module: OrderQModule) -> tuple[nn.GRUCell, list[nn.Module]]:
    """`(cell, head layers in order)`, or a hard failure.

    Same contract as `_exportable` and for the same reason -- a filter that
    silently ignores an unrecognised parameterised layer produces a *short*
    network, `has_inert_hidden_trunk` sees a nonzero head weight and waves it
    through, and the first evidence is a shape error inside an arena worker an
    update later.

    What changed in v3 is how the contract is *enforced*. The module is no longer a
    flat `Sequential`, so "walk the children and recognise each one" is no longer a
    proof of completeness -- a sub-module hung off `OrderQModule` and forgotten
    here would simply never be visited. Instead every parameter tensor in the
    module has to be **claimed** by a layer this exporter knows how to write, by
    identity. Anything unclaimed raises.
    """

    if not isinstance(module, OrderQModule):
        raise TypeError(f"{type(module).__name__} is not the exportable module type")
    cell = module.cell
    if not isinstance(cell, nn.GRUCell):
        raise TypeError(f"{type(cell).__name__} is not a GRUCell and cannot be exported")
    if cell.bias_ih is None or cell.bias_hh is None:
        raise TypeError("a bias-free GRUCell has no numpy mirror in this schema")
    head = _exportable(module.head)
    if not head:
        raise TypeError("module has no exportable head layers")
    expected = module.state_dim + module.hidden_size
    observed = getattr(head[0], "in_features", None)
    if observed != expected:
        raise ValueError(
            f"head takes {observed} inputs but the concatenation is "
            f"{module.state_dim} + {module.hidden_size} = {expected}; the module is "
            "mis-wired and `policy.load` would accept the export"
        )
    claimed = {id(p) for layer in (cell, *head) for p in layer.parameters(recurse=True)}
    stray = [name for name, p in module.named_parameters() if id(p) not in claimed]
    if stray:
        raise TypeError(f"{__name__}.to_numpy cannot export parameters {stray}")
    return cell, head


def to_numpy(module: OrderQModule) -> OrderQNetwork:
    """Export to the inference network, carrying sigma where a layer has it."""

    cell, head = _decompose(module)
    gru = GruCell(
        # `_out` transposes, so torch's `[3H, in]` becomes `[in, 3H]` with the
        # (r, z, n) row blocks landing as column blocks in the same order --
        # which is what lets `GruCell.step` slice thirds with no permute.
        _out(cell.weight_ih),
        _out(cell.weight_hh),
        _out(cell.bias_ih),
        _out(cell.bias_hh),
    )
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    sigma: list[tuple[np.ndarray, np.ndarray] | None] = []
    for layer in head:
        if isinstance(layer, NoisyLinear):
            layers.append((_out(layer.weight_mu), _out(layer.bias_mu)))
            sigma.append((_out(layer.weight_sigma), _out(layer.bias_sigma)))
        else:
            layers.append((_out(layer.weight), _out(layer.bias)))
            sigma.append(None)
    return OrderQNetwork(layers, sigma, gru=gru)


def from_numpy(network: OrderQNetwork, module: OrderQModule | None = None) -> OrderQModule:
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
    cell, linears = _decompose(module)
    if network.gru is None:
        # Same reasoning as the layer-count check below. Leaving torch's own
        # randomly initialised GRU in place would train from a half-random start
        # without a word, and a checkpoint carrying no recurrence is a pre-v3 file
        # rather than a v3 one with a missing piece.
        raise ValueError("checkpoint carries no GRU; it cannot seed a recurrent module")
    if (
        network.gru.hidden_size != module.hidden_size
        or network.gru.input_size != module.state_dim
    ):
        raise ValueError(
            f"checkpoint gru is {network.gru.input_size}->{network.gru.hidden_size}, "
            f"module is {module.state_dim}->{module.hidden_size}"
        )
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
        load(cell.weight_ih, network.gru.wx)
        load(cell.weight_hh, network.gru.wh)
        load(cell.bias_ih, network.gru.bx)
        load(cell.bias_hh, network.gru.bh)
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
    """True when every *head* trunk weight is exactly zero.

    Such a network scores every action identically, so the argmax ties to index
    0 and the agent emits nothing at all. It is also ungradable: with the trunk at
    zero, `a = relu(0) = 0` and `relu'(0) == 0`, which closes every path back to
    every parameter -- the head's and, in v3, the GRU's too. A run that reaches
    this state can never leave it.

    Unchanged from v2 despite the recurrence, and correct only because
    `network.layers` is the head and nothing else. If the GRU were an entry in that
    list, `layers[:-1]` would include it and a `quiet_start` -- random GRU, zeroed
    head trunk -- would be waved through by the one structural guard that exists to
    catch exactly that shape. See `policy.GruCell`.
    """

    hidden = network.layers[:-1]
    return bool(hidden) and all(not weight.any() for weight, _ in hidden)


def has_inert_recurrence(network: OrderQNetwork) -> bool:
    """True when the GRU is missing, or present with every weight exactly zero.

    This is *not* the NOOP collapse, and it must not be treated as one. Such a
    network trades perfectly well -- the head reads the raw decision state
    directly -- and trains normally. It is the other silent failure: v3 shipped, v2
    delivered, `h` pinned at zero for the whole episode, 128 of the head's 408
    input columns dead, and no metric saying so.

    Deliberately not raised on by `updater._reject_inert_network`, unlike
    `has_inert_hidden_trunk`. An all-zero GRU is recoverable: at zero weights
    `r = z = sigmoid(0) = 0.5` and `tanh'(0) = 1`, so `dh/dn = 1 - z = 0.5` and the
    gradient path is live -- the opposite of an all-zero ReLU trunk. And a hard
    reject would take down `OrderQNetwork.zeros`, the inference fallback, which is
    legitimately all-zero. Report it; do not fail on it.
    """

    if network.gru is None:
        return True
    return not network.gru.wx.any() and not network.gru.wh.any()
