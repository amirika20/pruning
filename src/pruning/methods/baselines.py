"""The two reference baselines every structured-pruning table needs.

`random` is the control and `magnitude` is the standard cheap heuristic (Li et
al., "Pruning Filters for Efficient ConvNets", ICLR 2017, rank filters by their
weight norm and drop the smallest). They live in one module because they share
all their plumbing and differ only in how they order the units.

WHY BOTH ARE WORTH REGISTERING RATHER THAN ASSUMED

`random` fixes the scale. On an over-provisioned network a surprising fraction
of units can go with no measurable loss, so a capacity number means nothing
without the floor. It is also the null the similarity analysis' removed-pair
`lift` is defined against.

`magnitude` is the thing to beat, and it has a subtlety that makes it more
interesting than a baseline. ReLU is positively homogeneous, so
(w, b, c) -> (t w, t b, c / t) for t > 0 leaves the network's FUNCTION
completely unchanged while scaling ||w|| by t. Ranking by ||w|| is therefore not
a property of the unit at all -- it is a property of the arbitrary gauge the
trained weights happen to sit in, set by initialization and weight decay. The
same holds for ||c||. Only the product is invariant:

    a_i = ||w_i|| * ||c_i|| = alpha_i ||c_i||        (the unit's MASS)

so `norm='mass'` is the well-posed version of magnitude pruning, and it is
exactly MASH with the similarity term deleted -- keep the heaviest units, merge
nothing, repair nothing. Both are provided (`norm='w'`, `norm='c'`, `norm='l1'`
are the naive variants) because the GAP BETWEEN THEM is itself a measurement: if
the naive norms do materially worse, that is direct evidence for the gauge
argument; if they match, the trained networks happen to sit in a near-canonical
gauge, which is equally worth knowing.

Norms are computed on BN-FOLDED weights, since that is the filter the ReLU
actually sees. Li et al. rank raw conv weights, which differs by the per-channel
BN scale -- folding is the eval-mode-exact version of the same idea.

REPAIR. Both default to `repair='none'`, so they stay in the data-free tier
their table column claims. Passing `repair='kernel'`/`'empirical'` re-solves the
surviving columns by least squares on the same calibration budget as the other
methods. That is worth running: with a global repair the surviving SPAN is what
sets capacity, so `random` plus repair is the sharpest available test of whether
a method's selection is doing anything a same-sized arbitrary subset could not.

Run `python -c "from src.pruning.methods.baselines import _selftest; _selftest()"`
for the numerical self-tests.
"""

from __future__ import annotations

import numpy as np
import torch

from src.models.registry import PrunableModel
from src.pruning.methods.mash import (
    TINY, extract_units, repair_deletion, reshape_outgoing)
from src.pruning.registry import (
    PruneContext, PruneDecision, PruningMethod, register_pruning_method)

NORMS = ("mass", "w", "c", "l1")
REPAIRS = ("none", "kernel", "empirical", "bias_only")


class _Baseline(PruningMethod):
    """Budget plumbing shared by both: width, exclusions, optional repair."""

    def __init__(self, n_remove: int = 1, fraction: float | None = None,
                 repair: str = "none", n_calib: int = 128,
                 max_rows: int = 20000):
        if fraction is not None and not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        if repair not in REPAIRS:
            raise ValueError(f"repair must be one of {REPAIRS}, got {repair!r}")
        self.n_remove = int(n_remove)
        self.fraction = None if fraction is None else float(fraction)
        self.repair = repair
        self.n_calib = int(n_calib)
        self.max_rows = int(max_rows)

    # -- subclasses provide an ORDER; smallest goes first ------------------

    def _order(self, model: PrunableModel, layer_idx: int,
               ctx: PruneContext) -> np.ndarray:
        raise NotImplementedError

    def select(self, model: PrunableModel, layer_idx: int,
               ctx: PruneContext) -> list[int] | PruneDecision:
        H = model.prunable_layer(layer_idx).weight.shape[0]
        blocked = set(int(i) for i in ctx.already_selected)
        budget = (self.n_remove if self.fraction is None
                  else int(round(self.fraction * (H - 1))))
        budget = min(budget, max(H - 1 - len(blocked), 0))
        if budget <= 0:
            return []

        order = [int(i) for i in self._order(model, layer_idx, ctx)
                 if int(i) not in blocked]
        removed = order[:budget]
        if not removed or self.repair == "none":
            return sorted(removed)

        C_new, const = repair_deletion(
            model, layer_idx, ctx.train_inputs[: self.n_calib], removed,
            repair=self.repair, max_rows=self.max_rows)
        dec = PruneDecision(remove=sorted(removed))
        if C_new is not None:
            dec.new_outgoing = torch.from_numpy(
                reshape_outgoing(model, layer_idx, C_new))
        if const is not None:
            dec.bias_delta = torch.from_numpy(const)
        return dec


@register_pruning_method("random")
class RandomPruning(_Baseline):
    """Remove a uniformly random subset -- the control.

    The draw is a deterministic function of (seed, layer_idx), so the same seed
    reproduces the same removal set across runs and across methods, which is
    what makes the comparison of WHICH units get removed meaningful.

    params: n_remove OR fraction, seed, repair, n_calib, max_rows
    """

    def __init__(self, seed: int = 0, **kw):
        super().__init__(**kw)
        self.seed = int(seed)

    def _order(self, model, layer_idx, ctx) -> np.ndarray:
        H = model.prunable_layer(layer_idx).weight.shape[0]
        return np.random.default_rng(self.seed * 1000 + layer_idx).permutation(H)


@register_pruning_method("magnitude")
class MagnitudePruning(_Baseline):
    """Remove the units with the smallest norm.

    params: n_remove OR fraction, norm, repair, n_calib, max_rows

      norm='mass'  alpha_i ||c_i||  -- gauge-invariant, the well-posed version
      norm='w'     alpha_i          -- the naive standard; NOT gauge-invariant
      norm='c'     ||c_i||          -- likewise
      norm='l1'    sum |w_eff|      -- Li et al.'s filter criterion
    """

    def __init__(self, norm: str = "mass", **kw):
        super().__init__(**kw)
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
        self.norm = norm

    def scores(self, model: PrunableModel, layer_idx: int) -> np.ndarray:
        """Per-unit importance; larger = keep. BatchNorm is folded in."""
        units, ok = extract_units(model, layer_idx)
        if self.norm == "mass":
            v = units.mass
        elif self.norm == "w":
            v = units.alpha
        elif self.norm == "c":
            v = np.linalg.norm(units.C, axis=1)
        else:                                    # l1 of the effective filter
            v = np.abs(units.alpha[:, None] * units.u).sum(axis=1)
        # a unit with no hyperplane contributes a constant, not a direction;
        # rank it last so it is removed first
        return np.where(ok, v, -np.inf)

    def _order(self, model, layer_idx, ctx) -> np.ndarray:
        return np.argsort(self.scores(model, layer_idx), kind="stable")


# ── self-tests ───────────────────────────────────────────────────────────────

def _selftest() -> None:  # pragma: no cover
    import copy

    import torch.nn as nn
    from torch.utils.data import TensorDataset

    from src.config import PruningConfig
    from src.models.cnn import CNN
    from src.models.mlp import MLP
    from src.pruning.registry import build_pruning_method
    from src.pruning.surgery import prune_model

    rng = np.random.default_rng(0)
    d, H = 12, 16
    torch.manual_seed(0)
    net = MLP(input_dim=d, output_dim=3, hidden_sizes=[H, 8]).eval()
    X = torch.from_numpy(rng.normal(size=(128, d))).float()

    class _B:
        train_ds = TensorDataset(X, torch.zeros(len(X), dtype=torch.long))

    bundle = _B()
    ctx = PruneContext(train_inputs=X, bundle=bundle, device=torch.device("cpu"))

    def run(kind, **params):
        class _M:
            pass
        _M.kind, _M.params = kind, params
        return prune_model(net, PruningConfig(methods=[_M()]), bundle,
                           torch.device("cpu"))

    # 1. random is reproducible from the seed, and different seeds differ
    a = build_pruning_method("random", n_remove=5, seed=1).select(net, 0, ctx)
    b = build_pruning_method("random", n_remove=5, seed=1).select(net, 0, ctx)
    c = build_pruning_method("random", n_remove=5, seed=2).select(net, 0, ctx)
    assert a == b, "same seed must give the same removal set"
    assert a != c, "different seeds must give different removal sets"
    assert len(a) == 5 and len(set(a)) == 5

    # 2. already_selected is respected
    ctx2 = PruneContext(train_inputs=X, bundle=bundle,
                        device=torch.device("cpu"), already_selected={0, 1, 2})
    got = build_pruning_method("random", n_remove=5, seed=1).select(net, 0, ctx2)
    assert not ({0, 1, 2} & set(got)), f"re-selected an excluded unit: {got}"

    # 3. magnitude removes the smallest first. The planted unit is shrunk on
    # BOTH sides -- incoming and outgoing -- so it is genuinely small under
    # every norm rather than only the incoming ones. (Shrinking one side alone
    # would be a gauge change, which is what test 4 is for.)
    small = copy.deepcopy(net)
    with torch.no_grad():
        small.net[0].weight[7] *= 1e-6
        small.net[0].bias[7] *= 1e-6
        small.net[2].weight[:, 7] *= 1e-6
    for norm in NORMS:
        m = build_pruning_method("magnitude", n_remove=1, norm=norm)
        assert m.select(small, 0, ctx) == [7], f"{norm}: tiny unit not first"

    # 4. THE GAUGE TEST. Rescale one unit by (t w, t b, c / t): the network's
    # function is unchanged, so an importance score worth the name must be
    # unchanged too. `mass` is; `w`, `c` and `l1` are not.
    scaled = copy.deepcopy(net)
    t = 40.0
    with torch.no_grad():
        scaled.net[0].weight[3] *= t
        scaled.net[0].bias[3] *= t
        scaled.net[2].weight[:, 3] /= t
    with torch.no_grad():
        assert (scaled(X) - net(X)).abs().max() < 1e-3, "rescale changed the function"
    ranks = {}
    for norm in NORMS:
        m = build_pruning_method("magnitude", n_remove=1, norm=norm)
        r0 = list(m._order(net, 0, ctx))
        r1 = list(m._order(scaled, 0, ctx))
        ranks[norm] = (r0.index(3), r1.index(3))
    assert ranks["mass"][0] == ranks["mass"][1], \
        f"mass must be gauge invariant, unit 3 moved {ranks['mass']}"
    assert ranks["w"][0] != ranks["w"][1], \
        "the ||w|| baseline should have been perturbed by a pure gauge change"

    # 5. repair helps, and repair='none' keeps it a plain index list
    with torch.no_grad():
        ref = net(X).double()
    errs = {}
    for rep in ("none", "bias_only", "empirical", "kernel"):
        pruned, _ = run("magnitude", fraction=0.5, norm="mass", repair=rep)
        with torch.no_grad():
            errs[rep] = float((pruned(X).double() - ref).abs().max())
    assert errs["kernel"] < errs["none"], errs
    assert errs["empirical"] < errs["none"], errs

    # 6. conv + BatchNorm, both methods, with and without repair
    torch.manual_seed(0)
    cnet = CNN(hidden_sizes=[8, 10], input_channels=1, output_dim=3).eval()
    Xc = torch.randn(6, 1, 12, 12)

    class _CB:
        train_ds = TensorDataset(Xc, torch.zeros(len(Xc), dtype=torch.long))

    for kind, params in [("random", dict(fraction=0.25, seed=3)),
                         ("magnitude", dict(fraction=0.25, norm="mass")),
                         ("magnitude", dict(fraction=0.25, norm="mass",
                                            repair="empirical"))]:
        class _M:
            pass
        _M.kind, _M.params = kind, params
        out, rep = prune_model(cnet, PruningConfig(methods=[_M()]), _CB(),
                               torch.device("cpu"))
        with torch.no_grad():
            out(Xc)
        w = [out.prunable_layer(i).weight.shape[0]
             for i in range(out.n_prunable_layers())]
        assert w == [6, 8], f"{kind}{params}: unexpected widths {w}"
        assert not out.prunable_bn(0).training, "pruned BN left in train mode"
    # conv must refuse the closed-form repair for the documented reason
    try:
        repair_deletion(cnet, 0, Xc, [0, 1], repair="kernel")
    except NotImplementedError as exc:
        assert "empirical" in str(exc)
    else:
        raise AssertionError("conv kernel repair should be refused")

    # 7. parameter validation
    for kind, bad in [("magnitude", {"norm": "nope"}),
                      ("random", {"repair": "nope"}),
                      ("random", {"fraction": 1.5})]:
        try:
            build_pruning_method(kind, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kind} {bad}")

    print("baselines.py self-tests passed:")
    print("  random reproducible from its seed, seeds differ, exclusions honored")
    print("  magnitude removes the smallest unit first under all four norms")
    print(f"  GAUGE: a function-preserving rescale leaves mass rank fixed "
          f"{ranks['mass']} but moves ||w|| rank {ranks['w']}")
    print(f"  repair helps: none {errs['none']:.3e} -> empirical "
          f"{errs['empirical']:.3e}, kernel {errs['kernel']:.3e}")
    print("  conv+BN for both methods, BN mode preserved, kernel repair refused")
    print("  parameter validation")


if __name__ == "__main__":
    _selftest()
