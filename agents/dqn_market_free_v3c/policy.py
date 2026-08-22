"""Numpy inference for the free-market recurrent Q-network.

A GRU that ticks once per *turn*, and an action head that scores
`concat(state, hidden)` once per sub-decision -- state first, which is the order
every implementation here and in `training/network.py` actually concatenates in. Torch-free for the same reason
`agents/dqn_market_scorer/policy.py` is -- the submission bundle may not depend on
torch -- but it carries its own shape, and `load` rejects a checkpoint whose input
or output width disagrees with the current schema rather than silently reading it
as a shape match.

`training/network.py` is the torch mirror and the two must agree to the last place
float32 carries for identical weights -- `_sigmoid`'s "to within a ulp" is the
honest bound rather than literal bit equality, since the exponent clip and the
contiguous-versus-transposed-view matmul each change the rounding. The GRU raises
the stakes on that: an MLP that disagrees is
wrong once per decision, a recurrence that disagrees compounds over 720 ticks. The
association order in `GruCell.step` is therefore not a style choice -- see its
docstring for the two ways to get it silently wrong.

Arithmetic per turn, for the cost argument that used to live here and was quoting
a stale number (~53k, which matched a single hidden layer and predated the second
one): the GRU is `3 * (280*128 + 128*128)` = 156k multiply-accumulates once a
turn, and the head is `408*128 + 128*128 + 128*199` = 94k per sub-decision against
v2's 75k. A worst-case eleven-decision turn is a third more than v2 and a typical
two-decision turn is about twice, both of which are tens of milliseconds across a
720-turn episode. Nothing here is chunked and per-turn decisions stay affordable.

The real cost is the encoder, not these matmuls -- see `main.py`, which reuses the
turn-opening encode for the turn's first sub-decision so that v3 pays for exactly
as many `market_state_features` calls as v2 did on the common single-decision turn.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from agents.dqn_market_free_v3c.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free_v3c.order_space import ACTION_COUNT

#: The action head's trunk, unchanged from v2 -- deliberately, so that every
#: sizing intuition, `has_inert_hidden_trunk`, and `reseed.perturb`'s head arm
#: port across the fork without re-derivation. Only its *input* width changed.
HIDDEN = (128, 128)

#: Width of the recurrent state.
HIDDEN_SIZE = 128

#: A GRU's gate count. The three blocks are (reset, update, candidate), in
#: `torch.nn.GRUCell`'s order, and every weight here is laid out as three
#: contiguous column blocks in that order so a slice recovers them with no permute.
GATES = 3

#: What the head actually takes: the decision state with the turn's latent
#: concatenated on. 360 against v3's 357.
HEAD_IN = STATE_DIM + HIDDEN_SIZE

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.npz"

_LAYER_KEY = re.compile(r"w\d+")

#: Exponent ceiling for the sigmoid below.
_EXP_CEILING = np.float32(80.0)

Pair = tuple[np.ndarray, np.ndarray]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid in float32, with the saturated leg made finite.

    `1 / (1 + exp(-x))` overflows to inf below about x = -88 in float32, with a
    RuntimeWarning and a nan the moment it meets a zero. Clipping the exponent at
    80 leaves `exp` finite and the result at 1.8e-35 -- torch's answer to within a
    ulp on every input that was not already saturated -- and it is branchless, so a
    [B, L, 3H] gate block is one pass rather than two fancy-indexed ones.

    Deliberately **not** `0.5 * (tanh(0.5 * x) + 1)`. Algebraically identical,
    rounds differently, and it destroys the relative precision of `1 - z` -- which
    is the term the whole recurrence is multiplied by, so the error would compound
    rather than cancel.
    """

    x = np.asarray(x, dtype=np.float32)
    return np.float32(1.0) / (np.float32(1.0) + np.exp(np.minimum(-x, _EXP_CEILING)))


class GruCell:
    """Numpy mirror of `torch.nn.GRUCell`, in this module's `x @ W` layout.

    Kept **outside** `OrderQNetwork.layers`, and that is load-bearing rather than
    tidy. Three things slice `layers` and all three would be wrong with a GRU in
    it:

      * `training/network.has_inert_hidden_trunk` reads `layers[:-1]` and is the
        only structural guard against the NOOP collapse. With the GRU in the list,
        a `quiet_start` -- random GRU, zeroed head trunk -- would be waved through
        by the very check that exists to catch it.
      * `training/reseed.perturb` derives its redraw bound as
        `1/sqrt(weight.shape[0])`, which is wrong for every GRU tensor: torch uses
        one `stdv` keyed on the *hidden* size for all four, including `wx`, whose
        fan-in is the state width.
      * `sample_noise` draws one epsilon pair per entry, and the recurrence is
        never noisy -- see `training/network.OrderQModule` for why that matters to
        the whole ordering of the TD step.
    """

    __slots__ = ("wx", "wh", "bx", "bh")

    def __init__(self, wx, wh, bx, bh):
        self.wx = np.ascontiguousarray(wx, dtype=np.float32)  # [in, 3H]
        self.wh = np.ascontiguousarray(wh, dtype=np.float32)  # [H, 3H]
        self.bx = np.ascontiguousarray(bx, dtype=np.float32)  # [3H]
        self.bh = np.ascontiguousarray(bh, dtype=np.float32)  # [3H]
        if self.wx.ndim != 2 or self.wh.ndim != 2:
            raise ValueError("gru weights must be 2-D")
        size = self.wh.shape[0]
        if self.wh.shape != (size, GATES * size):
            raise ValueError(f"gru wh must be (H, 3H); got {self.wh.shape}")
        if self.wx.shape[1] != GATES * size:
            raise ValueError(f"gru wx must be (in, 3H) for H={size}; got {self.wx.shape}")
        if self.bx.shape != (GATES * size,) or self.bh.shape != (GATES * size,):
            raise ValueError(f"gru biases must be (3H,) for H={size}")

    @property
    def hidden_size(self) -> int:
        return self.wh.shape[0]

    @property
    def input_size(self) -> int:
        return self.wx.shape[0]

    def zeros(self, *batch: int) -> np.ndarray:
        return np.zeros((*batch, self.hidden_size), dtype=np.float32)

    def step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """One tick. `x` [..., in] and `h` [..., H] -> [..., H].

        `torch.nn.GRUCell`'s form exactly:

            r  = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
            z  = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
            n  = tanh(W_in x + b_in + r * (W_hn h + b_hn))
            h' = (1 - z) * n + z * h

        Two silent parity breaks are foreclosed here on purpose, because both
        produce a network that trains perfectly well and disagrees with its own
        exported checkpoint forever.

        First, `b_hn` is added to the hidden projection **before** the reset gate
        multiplies it, which is why the whole `h @ wh + bh` product is formed and
        then sliced rather than sliced and then biased. The wrong association
        differs by `(1 - r) * b_hn` -- zero only while `b_hn` is zero, that is,
        never after the first optimizer step.

        Second, `z` is the *keep* gate: it multiplies `h`, not `n`. Swapping the
        legs gives a perfectly good GRU that is a different function.

        `training/network._out` transposes on export, mapping torch's row-block i
        to column-block i, so slicing the last axis into thirds recovers
        (r, z, n) in torch's order with no permute.
        """

        x = np.asarray(x, dtype=np.float32)
        h = np.asarray(h, dtype=np.float32)
        gx = x @ self.wx + self.bx
        gh = h @ self.wh + self.bh  # b_hn folded in *here*, before the reset gate
        size = self.hidden_size
        r = _sigmoid(gx[..., :size] + gh[..., :size])
        z = _sigmoid(gx[..., size : 2 * size] + gh[..., size : 2 * size])
        n = np.tanh(gx[..., 2 * size :] + r * gh[..., 2 * size :])
        return (np.float32(1.0) - z) * n + z * h


def _factorized(rng: np.random.Generator, size: int) -> np.ndarray:
    """`f(x) = sign(x) * sqrt(|x|)` over a standard normal draw.

    float32 explicitly: `standard_normal` defaults to float64, and computing the
    noise in double while the weights are single widens the gap against the torch
    mirror by roughly the square root of the layer width -- straight through the
    parity test's 1e-5 budget.
    """

    draw = rng.standard_normal(size, dtype=np.float32)
    return np.sign(draw) * np.sqrt(np.abs(draw))


class OrderQNetwork:
    """A per-turn GRU and an action head over `concat(state, hidden)`.

    `layers` is the mean (mu) weight of every **head** layer -- the recurrence is
    `gru`, deliberately not an entry in that list; see `GruCell`. `sigma` carries
    the NoisyNet standard deviations, one slot per head layer and `None` where the
    layer is deterministic, so a checkpoint written before NoisyNet loads as an
    ordinary network and scores identically.

    Two kinds of state live on this object and they are kept apart on purpose.

    Noise is drawn by `sample_noise` and held until replaced, rather than passed to
    `__call__`. That keeps `order_space.choose_order` unaware it exists, and makes
    evaluation determinism structural: the arena, the gates and the Kaggle entry
    point all reach this class through `main.agent`, which never calls it.

    The recurrent hidden state is held the same way, for the same reason and with
    the same payoff -- `choose_order`'s signature is unchanged from v2, and
    `network(state)` at `order_space.py`'s single call site still means "score this
    state". `main.py` owns its lifecycle: `reset_hidden` at `step == 0`, `tick`
    once a turn.

    What makes that safe rather than merely convenient is that the stateful path
    and the pure path do not share a name. `tick`/`__call__` are single-sample and
    read `_hidden`; `head_forward`/`forward_sequence` are batched, take the hidden
    as an argument, and neither read nor write it. Each rejects the other's input
    rank outright, so a batched caller cannot silently consume the episode's state
    and a stateful caller cannot silently score against someone else's.
    """

    def __init__(
        self,
        layers: list[Pair],
        sigma: list[Pair | None] | None = None,
        gru: GruCell | None = None,
    ):
        self.gru = gru
        #: The recurrence's state, or None when no episode is in flight. Never
        #: implicitly zero: a network scored before `reset_hidden` would otherwise
        #: run a whole episode against zeros, which reads on every metric exactly
        #: like a trained policy that has learned to ignore its memory. There is no
        #: way to tell those two apart after the fact, so `__call__` raises instead.
        self._hidden: np.ndarray | None = None
        self.layers = [(w.astype(np.float32), b.astype(np.float32)) for w, b in layers]
        self.sigma: list[Pair | None] = (
            [
                None if pair is None else (pair[0].astype(np.float32), pair[1].astype(np.float32))
                for pair in sigma
            ]
            if sigma is not None
            else [None] * len(self.layers)
        )
        if len(self.sigma) != len(self.layers):
            raise ValueError("sigma must carry one slot per layer")
        self._noise: list[Pair | None] = [None] * len(self.layers)

    @property
    def is_noisy(self) -> bool:
        return any(pair is not None for pair in self.sigma)

    @property
    def hidden_size(self) -> int:
        return 0 if self.gru is None else self.gru.hidden_size

    # -- recurrent state, owned by `main.MarketFreeAgent` ----------------------

    def reset_hidden(self) -> None:
        """Start of episode. Called at `step == 0` *and* from the agent's `__init__`.

        Both, not just the step test: `main.agent` keeps a module-level `_AGENT`
        that survives across episodes inside one interpreter, so a network built
        once and never reset would carry the previous episode's memory into the
        next one.
        """

        self._hidden = None if self.gru is None else self.gru.zeros()

    def tick(self, turn_features: np.ndarray) -> np.ndarray:
        """Advance the recurrence by one *turn*, and return the new hidden state.

        Once per turn, never once per sub-decision. The turn is the unit reward is
        paid on and the unit `n_step` counts, and ticking per decision would make
        the sequence length a function of how chatty the policy currently is -- the
        same trap `replay.sample`'s `order_step` search was written to avoid.
        """

        if self.gru is None:
            return np.zeros(0, dtype=np.float32)
        if self._hidden is None:
            raise RuntimeError("tick() before reset_hidden(): no episode is in flight")
        x = np.asarray(turn_features, dtype=np.float32)
        if x.shape != (self.gru.input_size,):
            raise ValueError(
                f"tick() is the stateful single-sample path; got {x.shape}, "
                f"expected ({self.gru.input_size},)"
            )
        self._hidden = self.gru.step(x, self._hidden)
        return self._hidden

    @property
    def hidden(self) -> np.ndarray | None:
        """The state the next `__call__` will score against. Read-only by convention."""

        return self._hidden

    def sample_noise(
        self,
        rng: np.random.Generator,
        scale: float | np.ndarray = 1.0,
    ) -> None:
        """Draw one factorized epsilon pair per noisy layer, held until replaced.

        Factorized rather than independent: `p + q` draws per layer instead of
        `p * q`, which at this width is 314 against 23,994.

        `scale` multiplies the draw, never `sigma`. `_linear` is bilinear in the two
        epsilons, so a factor on either scales the perturbation exactly; keeping it
        off `sigma` means it cannot compound across calls or reach `save`.

        `scale` may also be a *vector*, one entry per output unit -- directed
        exploration, widening one action's row. It applies only to layers whose
        output width it matches, so the same vector is correct whether the run
        perturbs the head alone or the whole trunk.
        """

        factor = np.asarray(scale, dtype=np.float32)
        self._noise = [
            None
            if pair is None
            else (
                _factorized(rng, weight.shape[0]),
                _factorized(rng, weight.shape[1])
                * (factor if factor.ndim == 0 or factor.shape == (weight.shape[1],) else 1.0),
            )
            for (weight, _bias), pair in zip(self.layers, self.sigma)
        ]

    def clear_noise(self) -> None:
        """Back to mu only. The evaluation and submission mode."""

        self._noise = [None] * len(self.layers)

    @classmethod
    def zeros(
        cls,
        state_dim: int = STATE_DIM,
        hidden: tuple[int, ...] = HIDDEN,
        actions: int = ACTION_COUNT,
        hidden_size: int = HIDDEN_SIZE,
    ) -> OrderQNetwork:
        """All-zero network. Inference fallback only, never a training start.

        Scores every action identically, so the argmax falls to NOOP at index 0 --
        survivable, and a defined behaviour for a missing weights file. The GRU is
        all-zero too, which keeps the fallback silent for the whole episode rather
        than only its first turn: at zero weights and `h_0 = 0`,
        `h' = (1 - 0.5) * tanh(0) + 0.5 * 0` is zero again, forever.

        **Never initialize training with this.** An all-zero head has zero gradient
        everywhere but its output bias, since both hidden activations are zero for
        any input and `relu'(0) == 0` closes the path back to the biases -- and
        with the head dead, no gradient reaches the recurrence either. The symmetry
        never breaks and the run stays at NOOP forever while reporting a healthy
        loss. Use `quiet_start`.
        """

        sizes = (state_dim + hidden_size, *hidden, actions)
        gru = GruCell(
            np.zeros((state_dim, GATES * hidden_size), dtype=np.float32),
            np.zeros((hidden_size, GATES * hidden_size), dtype=np.float32),
            np.zeros(GATES * hidden_size, dtype=np.float32),
            np.zeros(GATES * hidden_size, dtype=np.float32),
        )
        return cls(
            [
                (np.zeros((a, b), dtype=np.float32), np.zeros(b, dtype=np.float32))
                for a, b in zip(sizes, sizes[1:])
            ],
            gru=gru,
        )

    @staticmethod
    def _random_gru(
        rng: np.random.Generator,
        state_dim: int = STATE_DIM,
        hidden_size: int = HIDDEN_SIZE,
    ) -> GruCell:
        """A GRU at `nn.GRUCell`'s own initialisation.

        `U(-1/sqrt(H), +1/sqrt(H))` for all four tensors, `wx` included -- note the
        bound is keyed on the *hidden* size even though `wx`'s fan-in is
        `state_dim`. That is torch's rule (`RNNCellBase.reset_parameters` uses one
        `stdv` for every parameter), and matching it is what lets
        `training/reseed.perturb` redraw a GRU entry at the scale it was born at.
        """

        bound = float(1.0 / np.sqrt(hidden_size))
        draw = lambda shape: rng.uniform(-bound, bound, shape).astype(np.float32)  # noqa: E731
        return GruCell(
            draw((state_dim, GATES * hidden_size)),
            draw((hidden_size, GATES * hidden_size)),
            draw(GATES * hidden_size),
            draw(GATES * hidden_size),
        )

    @classmethod
    def quiet_start(
        cls,
        seed: int = 0,
        state_dim: int = STATE_DIM,
        hidden: tuple[int, ...] = HIDDEN,
        actions: int = ACTION_COUNT,
        hidden_size: int = HIDDEN_SIZE,
    ) -> OrderQNetwork:
        """Random GRU and head trunk, zero output layer: NOOP behaviour, live gradients.

        Zeroed, and nothing else is: the head's last layer, `w_out` and `b_out`.
        Random: the four GRU tensors at `nn.GRUCell`'s own bound, and the head
        trunk at `N(0, 1/sqrt(fan_in))` exactly as v2 drew it.

        Every action scores exactly 0.0 -- `Q = a @ 0 + 0` -- so the argmax ties to
        index 0 and the greedy policy is silent, the same cold start `zeros` gives.

        **It trains, and the recurrence does not enter the argument.**
        `dL/dw_out` is proportional to `a = relu(concat(s_1, h_1) @ w0 + b0)`, and
        `a` is non-zero because `concat(s_1, h_1)` contains `s_1`, the raw
        280-column decision state, whatever `h_1` happens to be. The claim
        therefore survives even the degenerate `h_1 = 0` an all-zero GRU would
        produce. With `w0` Gaussian, about half its 128 rows fire on any generic
        input. This unconditionality is a second reason the head takes the full
        state rather than the latent alone: under a shared encoder the cold-start
        gradient argument would have to route through the GRU and would depend on
        how the GRU happened to be initialised.

        On step 1 everything upstream of `w_out` -- the head trunk and all four GRU
        tensors -- has exactly zero gradient, because `dL/da = delta @ w_out.T = 0`.
        From step 2 `w_out` is non-zero, `dL/dh` flows through `w0`'s first H rows,
        and truncated BPTT carries it back across the burn-in window. That is v2's
        two-phase story unchanged, and the GRU adds no new stationary point to it:
        at zero weights `r = z = 0.5` and `tanh'(0) = 1`, so `dh/dn = 1 - z = 0.5`
        stays open -- the opposite of an all-zero ReLU trunk, which is why `zeros`
        is untrainable and this is not.

        Two consequences worth stating. On update 1 the burn-in is pure overhead.
        And any "every parameter moved" test has to run at least two optimizer
        steps.
        """

        rng = np.random.default_rng(seed)
        gru = cls._random_gru(rng, state_dim, hidden_size)
        sizes = (state_dim + hidden_size, *hidden)
        layers = [
            (
                rng.standard_normal((a, b)).astype(np.float32) / np.sqrt(a),
                rng.standard_normal(b).astype(np.float32) * 0.01,
            )
            for a, b in zip(sizes, sizes[1:])
        ]
        layers.append(
            (
                np.zeros((sizes[-1], actions), dtype=np.float32),
                np.zeros(actions, dtype=np.float32),
            )
        )
        return cls(layers, gru=gru)

    @classmethod
    def random(
        cls,
        seed: int = 0,
        state_dim: int = STATE_DIM,
        hidden: tuple[int, ...] = HIDDEN,
        actions: int = ACTION_COUNT,
        sigma0: float = 0.0,
        noisy_layers: str = "head",
        hidden_size: int = HIDDEN_SIZE,
    ) -> OrderQNetwork:
        """`sigma0 > 0` mints a noisy network directly; the default is deterministic.

        Only a test constructor. Training never starts a noisy network from here:
        sigma is born in torch (`training/network.py`) and reaches numpy through
        `to_numpy`, because sigma = 0 is a stationary point of the loss and a
        network exported with it could never grow sigma back.

        The GRU is drawn but never given sigma, at any `sigma0`. The recurrence is
        deterministic by design -- see `training/network.OrderQModule`, where that
        fact is what makes the TD step's `reset_noise` ordering legal.
        """

        rng = np.random.default_rng(seed)
        gru = cls._random_gru(rng, state_dim, hidden_size)
        sizes = (state_dim + hidden_size, *hidden, actions)
        layers = [
            (
                rng.standard_normal((a, b)).astype(np.float32) / np.sqrt(a),
                rng.standard_normal(b).astype(np.float32) * 0.01,
            )
            for a, b in zip(sizes, sizes[1:])
        ]
        if sigma0 <= 0.0:
            return cls(layers, gru=gru)
        last = len(layers) - 1
        sigma: list[Pair | None] = [
            (
                np.full(weight.shape, sigma0 / np.sqrt(weight.shape[0]), dtype=np.float32),
                np.full(bias.shape, sigma0 / np.sqrt(weight.shape[0]), dtype=np.float32),
            )
            if noisy_layers == "all" or index == last
            else None
            for index, (weight, bias) in enumerate(layers)
        ]
        return cls(layers, sigma, gru=gru)

    @classmethod
    def load(cls, path: Path | str = WEIGHTS_PATH) -> OrderQNetwork:
        with np.load(str(path), allow_pickle=False) as data:
            # Anchored, not `startswith("w")`. The sigma keys are named `sw{i}`
            # precisely so they cannot inflate this count, but a loose prefix
            # test is one badly-chosen key away from making `data[f"w{count-1}"]`
            # raise KeyError -- which `main.load_network` swallows, silently
            # falling back to `zeros()`. Challenger and incumbent then play as
            # one policy and every paired margin comes out exactly 0.0.
            count = sum(1 for key in data.files if _LAYER_KEY.fullmatch(key))
            layers = [(data[f"w{index}"], data[f"b{index}"]) for index in range(count)]
            sigma: list[Pair | None] = []
            for index, (weight, bias) in enumerate(layers):
                names = (f"sw{index}", f"sb{index}")
                present = [name in data.files for name in names]
                if any(present) and not all(present):
                    raise ValueError(f"checkpoint layer {index} carries half a sigma pair")
                if not all(present):
                    sigma.append(None)
                    continue
                sigma_w, sigma_b = data[names[0]], data[names[1]]
                if sigma_w.shape != weight.shape or sigma_b.shape != bias.shape:
                    raise ValueError(
                        f"checkpoint layer {index} sigma shape {sigma_w.shape} disagrees "
                        f"with mu shape {weight.shape}"
                    )
                sigma.append((sigma_w, sigma_b))

            # The recurrence, as four named tensors rather than another `w{i}`
            # entry -- see `GruCell` for why it stays out of `layers`. None of
            # these keys can `fullmatch` `_LAYER_KEY`, so none can inflate the
            # head's layer count above; `tests/test_dqn_market_free_v3c.py` asserts
            # that directly rather than leaving it to inspection.
            names = ("gru_wx", "gru_wh", "gru_bx", "gru_bh")
            present = [name in data.files for name in names]
            if any(present) and not all(present):
                raise ValueError("checkpoint carries a partial GRU")
            if not all(present):
                # Rejected by *name*, deliberately, rather than left to the width
                # test below. A v2 checkpoint's head is 259 wide and this one's
                # is `280 + 128 = 408`; those differ today, but a `HIDDEN_SIZE` of
                # 21 would make `238 + 21` collide with a v2 file and load it as a
                # network whose recurrence is permanently absent, and every metric
                # would report a healthy policy that has no memory. The v3 fork hit
                # the same coincidence exactly (`229 + 30 == 259`).
                raise ValueError(
                    "checkpoint carries no GRU: this is a pre-v3 file, and v3 is a "
                    "cold start rather than a resume"
                )
            gru = GruCell(*(data[name] for name in names))
            if "hidden_size" in data.files and int(data["hidden_size"][0]) != gru.hidden_size:
                raise ValueError("checkpoint hidden_size disagrees with its own gru weights")
            # v2 wrote this key and never read it back. Read it -- inside the
            # `with`, because `NpzFile` closes its archive on exit and every
            # `data[...]` after that raises an AttributeError rather than a
            # KeyError, which `main.load_network` does not catch.
            if "state_dim" in data.files and int(data["state_dim"][0]) != STATE_DIM:
                raise ValueError(
                    f"checkpoint declares state_dim {int(data['state_dim'][0])}, "
                    f"schema is {STATE_DIM}"
                )
            network = cls(layers, sigma, gru=gru)
        if not network.layers:
            raise ValueError("checkpoint contains no layers")
        if gru.input_size != STATE_DIM:
            raise ValueError(
                f"checkpoint gru takes {gru.input_size} inputs, schema is {STATE_DIM}"
            )
        observed_head = network.layers[0][0].shape[0]
        observed_actions = network.layers[-1][0].shape[1]
        expected_head = STATE_DIM + gru.hidden_size
        if observed_head != expected_head or observed_actions != ACTION_COUNT:
            raise ValueError(
                f"checkpoint shape {observed_head}->{observed_actions} does not match "
                f"schema {STATE_DIM} + {gru.hidden_size} = {expected_head}->{ACTION_COUNT}"
            )
        return network

    def save(self, path: Path | str = WEIGHTS_PATH) -> None:
        """Write mu always, sigma only where a layer carries it.

        A deterministic network therefore writes exactly the key set this agent
        wrote before NoisyNet existed, so runs that leave `noisy` off churn no
        checkpoints -- and an inference bundle built before NoisyNet, which knows
        only `w{i}`/`b{i}`, reads a noisy checkpoint as its own mu-only policy
        rather than failing.
        """

        payload: dict[str, np.ndarray] = {
            "state_dim": np.asarray([STATE_DIM], dtype=np.int32),
            "action_count": np.asarray([ACTION_COUNT], dtype=np.int32),
            "hidden_size": np.asarray([self.hidden_size], dtype=np.int32),
        }
        if self.gru is not None:
            payload["gru_wx"] = self.gru.wx
            payload["gru_wh"] = self.gru.wh
            payload["gru_bx"] = self.gru.bx
            payload["gru_bh"] = self.gru.bh
        for index, (weight, bias) in enumerate(self.layers):
            payload[f"w{index}"] = weight
            payload[f"b{index}"] = bias
            pair = self.sigma[index]
            if pair is not None:
                payload[f"sw{index}"], payload[f"sb{index}"] = pair
        np.savez_compressed(str(path), **payload)

    def _linear(self, index: int, x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        out = x @ weight + bias
        sigma = self.sigma[index]
        noise = self._noise[index]
        if sigma is None or noise is None:
            return out
        sigma_w, sigma_b = sigma
        f_in, f_out = noise
        # Deliberately the factored form, and deliberately this association
        # order: it is what `training/network.py` computes, and materializing
        # `mu + sigma * outer(f_out, f_in)` on one side only would round
        # differently in float32 and eat the parity test's tolerance.
        return out + ((x * f_in) @ sigma_w) * f_out + sigma_b * f_out

    def _head(self, activation: np.ndarray) -> np.ndarray:
        """The MLP, over an already-concatenated `[..., HEAD_IN]` input."""

        last = len(self.layers) - 1
        for index, (weight, bias) in enumerate(self.layers):
            activation = self._linear(index, activation, weight, bias)
            if index != last:
                activation = np.maximum(activation, 0.0)
        return activation

    def head_forward(self, states: np.ndarray, hidden: np.ndarray | None) -> np.ndarray:
        """`(..., STATE_DIM)` and `(..., H)` -> `(..., ACTION_COUNT)`. Pure.

        `hidden` broadcasts over the leading axes, which is what lets a turn's
        several sub-decisions share one latent without materialising a copy per
        decision.

        Reads and writes nothing on `self` except the weights and the noise draw,
        so training, the parity tests and the flip-rate probe all reach the head
        through here and cannot disturb an episode in flight.
        """

        states = np.asarray(states, dtype=np.float32)
        if self.gru is None:
            return self._head(states)
        hidden = np.asarray(hidden, dtype=np.float32)
        hidden = np.broadcast_to(hidden, (*states.shape[:-1], self.gru.hidden_size))
        return self._head(np.concatenate((states, hidden), axis=-1))

    def forward_sequence(
        self,
        turn_features: np.ndarray,
        decision_states: np.ndarray,
        hidden: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """`[B, L, S]`, `[B, L, K, S]`, `[B, H]` -> `[B, L, K, A]`, `[B, H]`. Pure.

        A Python loop over the same `GruCell.step` the stateful path calls, rather
        than a fused scan. That is the point: the two can then be compared with
        *exact* equality instead of a tolerance, and any difference at all means
        one of them has grown its own arithmetic -- which is how a submission and
        the training runs that produced it come to disagree while every parity
        tolerance still passes.
        """

        turns = np.asarray(turn_features, dtype=np.float32)
        decisions = np.asarray(decision_states, dtype=np.float32)
        h = np.asarray(hidden, dtype=np.float32)
        out = np.empty((*decisions.shape[:-1], ACTION_COUNT), dtype=np.float32)
        for step in range(turns.shape[1]):
            h = self.gru.step(turns[:, step], h)
            out[:, step] = self.head_forward(decisions[:, step], h[:, None, :])
        return out, h

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Q for ONE sub-decision, against the hidden state `tick` last left.

        Two guards, and they are what stop the convenience of holding the state on
        the network becoming a bug.

        A `_hidden` of None raises. A network loaded and scored without an episode
        reset would otherwise score against zeros for 720 turns, which reads on
        every metric exactly like a trained policy that has learned to ignore its
        memory -- and there is no way to tell those apart after the fact.

        Only a 1-D state is accepted, so a batched caller cannot reach this path at
        all. Batched work goes through `head_forward` / `forward_sequence`, which
        take the hidden as an argument and neither read nor write `_hidden`.
        Neither path can silently consume the other's state, because they do not
        share a name.
        """

        x = np.asarray(state, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError(
                "__call__ is the stateful single-decision path; use "
                "head_forward(states, hidden) for a batch"
            )
        if self.gru is not None and self._hidden is None:
            raise RuntimeError("scored before reset_hidden(): the GRU was never initialised")
        return self.head_forward(x, self._hidden)
