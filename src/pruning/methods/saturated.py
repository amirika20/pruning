"""Saturated-unit pruning: remove units that are provably dead or provably affine.

A ReLU unit is SATURATED on a region when its pre-activation never changes sign
there. Two cases, and they are not symmetric:

    DEAD (never active)   sigma(z_i) == 0 on the region, so the unit's whole
                          contribution is zero. Removing it is EXACTLY free --
                          nothing to repair, no approximation, no budget spent.

    ALWAYS-ON             sigma(z_i) == z_i on the region, so the unit is AFFINE
                          there. Its constant part folds into the consumer's
                          bias exactly; its linear part is generally NOT in the
                          span of the surviving ReLUs, so removing it needs a
                          least-squares repair and is only approximately free.

That asymmetry is the whole design: dead units are a free lunch and are taken
unconditionally, always-on units are a trade and are taken with a repair.

DETECTION (`criterion=`). Both tests read the pre-activation
z_i = w_i^eff . x + b_i^eff, with any paired BatchNorm folded in exactly.

    interval   sound over the measured region. We take the layer's INPUT box
               [lo, hi] and bound
                   z_hi = w+ . hi + w- . lo + b,  z_lo = w+ . lo + w- . hi + b
               so z_hi < 0 certifies dead and z_lo > 0 certifies always-on for
               EVERY point of that box, not just the sampled ones. Measuring the
               box AT THE LAYER beats propagating it from the input box by
               interval arithmetic, which loses all tightness with depth.

    margin     Gaussian tail on the pre-activation moments: E[z] + kappa std(z)
               < 0 for dead, E[z] - kappa std(z) > 0 for always-on. Weaker as a
               guarantee, but it needs only (mean, std) per unit and works on
               conv channels. At kappa = 4 it tracks strict deadness closely on
               trained MLPs (47/56/61 flagged against 42/52/55 truly dead across
               three seeds), the extras being units active on a handful of
               points whose removal costs ~1e-5 in output.

    empirical  strict support over the sampled inputs: max_n z <= 0 for dead,
               min_n z > 0 for always-on. No model and no margin -- exactly the
               "never fires on any training input" test, which is what the
               retired `silent` method computed. Weakest guarantee of the three
               (it says nothing about unsampled inputs) but it is the tightest
               on the sample, so it is the one to use when reproducing an
               empirical saturation count.

SAMPLE SIZE IS NOT ONE NUMBER. Sample MOMENTS concentrate; sample SUPPORT does
not. So `n_calib` (moments, repair Grams) can stay at ~128, while `n_box` -- the
sample the region and the pre-activation statistics are measured from -- should
be as large as available, which is the default. Measuring the box from only 128
points shrinks it, and a too-small box "certifies" units that fire just outside
it: on trained MLPs that inflated the false-dead count by 5-10x.

ZERO-ENERGY GUARD (always applied to the dead test). A unit whose contribution
carries no energy is removable whether or not its pre-activation has a sign:

    E||y_i||^2 = a_i^2 E[sigma(t_i)^2],   a_i = alpha_i ||c_i||  (its mass)

Units below `energy_tol` times the layer's total contribution energy are treated
as dead. This is what catches WEIGHT-COLLAPSED units -- w ~ 0 and b ~ 0, so
z ~ 0 and both sign tests are undecided (t = 0/0) -- which weight decay produces
in deep layers and which a margin test alone silently keeps.

REPAIR of the always-on removals (`repair=`). Writing R for the removed
always-on set and S for the survivors, we solve

    min_X || sum_{i in R} v_i phi_i - sum_{k in S} X_k phi_k - x_0 * 1 ||^2

over the survivors' effective-weight adjustments X and a constant x_0, i.e. the
normal equations with the CONSTANT FUNCTION adjoined to the dictionary. The
constant coefficient is then exact bias compensation, folded into the consumer's
bias, and it comes out of the same solve rather than being applied afterwards.
`kernel` evaluates the Grams in closed form under N(mu, Sigma) from the same
calibration moments; `empirical` uses sample averages; `bias_only` keeps the
constant term and drops the linear projection; `none` deletes outright.

SCOPE. Dead removal works on Linear and Conv2d (detection only needs the
pre-activation, and there is nothing to repair). Always-on removal is Linear
only, since re-solving the consumer's columns needs the activation Grams.
`criterion='interval'` is Linear only for the same reason the box is: conv units
live on im2col patches.

EXPECT NOTHING ON BATCHNORM NETWORKS. BatchNorm actively suppresses saturation
-- it standardizes the pre-activation, so no channel sits entirely on one side
of zero. An empty selection on a BN network is the correct answer, not a bug;
saturation reappears when BN is removed (deep blocks went ~29% strictly dead in
the study's no-BN ablation).

Run `python -c "from src.pruning.methods.saturated import _selftest; _selftest()"`
for the numerical self-tests.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.registry import PrunableModel
from src.pruning.registry import (
    PruneContext, PruneDecision, PruningMethod, register_pruning_method)
# MASH owns the canonical unit extraction (BN folding, the unit-gain gauge) and
# the rectified-Gaussian Grams; importing keeps one implementation of each.
from src.pruning.methods.mash import (
    TINY, UnitMoments, _layer_inputs, _relu_moments, extract_units)

CRITERIA = ("interval", "margin", "empirical")
REPAIRS = ("kernel", "empirical", "bias_only", "none")
MODES = ("dead", "always_on", "both")


def _preactivations(model: PrunableModel, layer_idx: int,
                    x: torch.Tensor) -> np.ndarray:
    """Pre-ReLU activations of the prunable layer, [N', H] -- flattened over any
    spatial or token axes, so one row is one (sample, position) pair. Reads the
    paired BatchNorm's output when there is one, which is where the ReLU reads."""
    bn = model.prunable_bn(layer_idx)
    target = bn if bn is not None else model.prunable_layer(layer_idx)
    grabbed: list[torch.Tensor] = []
    h = target.register_forward_hook(lambda m, i, o: grabbed.append(o.detach()))
    try:
        with torch.no_grad():
            model(x)
    finally:
        h.remove()
    z = grabbed[0]
    if z.dim() == 4:                                  # [N, C, H, W] -> [N*H*W, C]
        z = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    return z.reshape(-1, z.shape[-1]).double().cpu().numpy()


def interval_bounds(W: np.ndarray, b: np.ndarray, lo: np.ndarray,
                    hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sound elementwise bounds on z = W x + b for every x in the box [lo, hi]."""
    Wp, Wn = np.clip(W, 0.0, None), np.clip(W, None, 0.0)
    return Wp @ lo + Wn @ hi + b, Wp @ hi + Wn @ lo + b


def classify(z: np.ndarray, criterion: str = "margin", kappa: float = 4.0,
             W: np.ndarray | None = None, b: np.ndarray | None = None,
             box: tuple[np.ndarray, np.ndarray] | None = None):
    """(dead, always_on) boolean masks over units.

    `z` are the observed pre-activations [N, H] (used by 'margin' and to report
    activation frequency); `W, b, box` are required by 'interval'.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
    if criterion == "interval":
        if W is None or b is None or box is None:
            raise ValueError("criterion='interval' needs W, b and the input box")
        z_lo, z_hi = interval_bounds(W, b, box[0], box[1])
        return z_hi < 0.0, z_lo > 0.0
    if criterion == "empirical":
        return z.max(axis=0) <= 0.0, z.min(axis=0) > 0.0
    m, s = z.mean(axis=0), z.std(axis=0)
    return m + kappa * s < 0.0, m - kappa * s > 0.0


@register_pruning_method("saturated")
class SaturatedPruning(PruningMethod):
    """Remove provably dead units (free) and provably always-on units (repaired).

    params:
      mode         'dead' | 'always_on' | 'both'         (default 'both')
      criterion    'interval' | 'margin' | 'empirical'   (default 'margin')
      kappa        margin multiplier, criterion='margin' (default 4.0)
      energy_tol   relative contribution energy below which a unit counts as
                   dead; catches weight-collapsed units (default 1e-10)
      repair       'kernel' | 'empirical' | 'bias_only' | 'none', for the
                   always-on removals only                (default 'kernel')
      n_calib      inputs used for the MOMENTS and the repair Grams; these
                   concentrate quickly (default 128)
      n_box        inputs used to measure the region for criterion='interval';
                   min/max does NOT concentrate, so this should be as large as
                   you can afford -- None means every available input (default)
      max_fraction refuse to remove more than this share of a layer, as a guard
                   against a misconfigured criterion emptying it (default 0.9)
    """

    def __init__(self, mode: str = "both", criterion: str = "margin",
                 kappa: float = 4.0, energy_tol: float = 1e-10,
                 repair: str = "kernel", n_calib: int = 128,
                 n_box: int | None = None, max_fraction: float = 0.9):
        for name, val, allowed in (("mode", mode, MODES),
                                   ("criterion", criterion, CRITERIA),
                                   ("repair", repair, REPAIRS)):
            if val not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {val!r}")
        if kappa < 0.0:
            raise ValueError(f"kappa must be >= 0, got {kappa}")
        if not 0.0 <= energy_tol < 1.0:
            raise ValueError(f"energy_tol must be in [0, 1), got {energy_tol}")
        self.mode = mode
        self.criterion = criterion
        self.kappa = float(kappa)
        self.energy_tol = float(energy_tol)
        self.repair = repair
        self.n_calib = int(n_calib)
        self.n_box = None if n_box is None else int(n_box)
        self.max_fraction = float(max_fraction)

    # -- the two tests -----------------------------------------------------

    def _detect(self, model, layer_idx, ctx):
        """(dead, always_on, diagnostics) masks over the layer's units."""
        layer = model.prunable_layer(layer_idx)
        is_linear = isinstance(layer, nn.Linear)
        x = ctx.train_inputs[: self.n_calib]
        # Pre-activation statistics are cheap to widen (one forward pass, no
        # covariance), and the energy guard is a max-like quantity, so use the
        # same wider sample the box uses.
        xz = (ctx.train_inputs if self.n_box is None
              else ctx.train_inputs[: self.n_box])
        z = _preactivations(model, layer_idx, xz)
        H = z.shape[1]

        units = extract_units(model, layer_idx)[0] if is_linear else None

        W = b = box = None
        if self.criterion == "interval":
            if not is_linear:
                raise NotImplementedError(
                    "criterion='interval' needs the layer-input box, and conv "
                    "units live on im2col patches; use criterion='margin' on "
                    "conv layers (it reads the pre-activation directly)")
            W = units.alpha[:, None] * units.u          # w_eff (BN folded)
            b = -units.alpha * units.rho                # b_eff
            # The region is measured from as many inputs as available, NOT from
            # the n_calib used for moments: sample MOMENTS concentrate, sample
            # SUPPORT does not, so a small-sample box is systematically too
            # small and would certify units that fire outside it.
            xb = (ctx.train_inputs if self.n_box is None
                  else ctx.train_inputs[: self.n_box])
            Zb = _layer_inputs(model, layer_idx, xb)
            box = (Zb.min(axis=0), Zb.max(axis=0))
        dead, always = classify(z, self.criterion, self.kappa, W, b, box)

        # zero-energy guard. The unit contributes y_i = c_i sigma(z_i), so its
        # energy is ||c_i||^2 E[sigma(z_i)^2]; z is the RAW pre-activation, which
        # already carries alpha_i. Without an outgoing column (conv), rank by the
        # response energy alone.
        resp2 = (np.maximum(z, 0.0) ** 2).mean(axis=0)
        energy = (np.linalg.norm(units.C, axis=1) ** 2 * resp2 if is_linear
                  else resp2)
        total = float(energy.sum())
        collapsed = energy <= self.energy_tol * max(total, TINY)
        return dead | collapsed, always, {"energy": energy, "collapsed": collapsed,
                                          "act_freq": (z > 0).mean(axis=0)}

    # -- always-on repair --------------------------------------------------

    def _repair_columns(self, model, layer_idx, ctx, keep: np.ndarray,
                        removed: np.ndarray):
        """(new_outgoing [H, m] or None, bias_delta [m] or None).

        Solves the constant-augmented normal equations for the survivors'
        effective-weight adjustments, so the removed units' affine contribution
        is absorbed as far as the surviving span allows and the leftover
        constant lands in the consumer's bias exactly.
        """
        units, ok = extract_units(model, layer_idx)
        V = units.V
        C_new = units.C.copy()
        if len(removed) == 0:
            return None, None

        Z = _layer_inputs(model, layer_idx, ctx.train_inputs[: self.n_calib])
        if self.repair == "empirical":
            Phi = np.maximum(Z @ units.u.T - units.rho[None, :], 0.0)
            N = len(Z)
            G = Phi[:, keep].T @ Phi[:, keep] / N
            g1 = Phi[:, keep].mean(axis=0)
            B = Phi[:, keep].T @ Phi[:, removed] / N
            b1 = Phi[:, removed].mean(axis=0)
        else:
            mu, Sigma = Z.mean(axis=0), np.atleast_2d(np.cov(Z.T))
            mk = UnitMoments(units.u[keep], units.rho[keep], mu, Sigma)
            mr = UnitMoments(units.u[removed], units.rho[removed], mu, Sigma)
            G = mk.gram_with(mk)
            B = mk.gram_with(mr)
            g1 = _relu_moments(mk.m, mk.s)
            b1 = _relu_moments(mr.m, mr.s)

        # adjoin the constant function: [[G, g1], [g1^T, 1]]
        K = len(keep)
        A = np.empty((K + 1, K + 1))
        A[:K, :K] = G
        A[:K, K] = g1
        A[K, :K] = g1
        A[K, K] = 1.0
        rhs = np.empty((K + 1, len(removed)))
        rhs[:K] = B
        rhs[K] = b1
        lam = 1e-8 * max(np.trace(A) / (K + 1), TINY)
        sol = np.linalg.solve(A + lam * np.eye(K + 1), rhs)      # [K+1, |R|]

        X = sol[:K] @ V[removed]                        # [K, m] effective weights
        const = sol[K] @ V[removed]                     # [m] leftover constant
        if self.repair == "bias_only":
            X = np.zeros_like(X)
            const = (b1 @ V[removed])
        safe_alpha = np.where(units.alpha[keep] > TINY, units.alpha[keep], 1.0)
        C_new[keep] = units.C[keep] + X / safe_alpha[:, None]
        return C_new, const

    # -- selection ---------------------------------------------------------

    def select(self, model: PrunableModel, layer_idx: int,
               ctx: PruneContext) -> PruneDecision:
        dead, always, diag = self._detect(model, layer_idx, ctx)
        H = len(dead)
        blocked = np.zeros(H, dtype=bool)
        blocked[list(ctx.already_selected)] = True
        dead &= ~blocked
        always &= ~blocked

        take_dead = self.mode in ("dead", "both")
        take_always = self.mode in ("always_on", "both")
        if take_always and not isinstance(model.prunable_layer(layer_idx), nn.Linear):
            raise NotImplementedError(
                "always-on removal re-solves the consumer's columns, which needs "
                "activation Grams; conv units live on im2col patches. Use "
                "mode='dead' on conv layers -- dead removal needs no repair.")

        rm_dead = np.flatnonzero(dead) if take_dead else np.zeros(0, dtype=int)
        rm_always = np.flatnonzero(always & ~dead) if take_always else np.zeros(0, dtype=int)

        # never empty a layer on the strength of a criterion alone
        cap = int(self.max_fraction * H)
        if len(rm_dead) + len(rm_always) > cap:
            room = max(cap - len(rm_dead), 0)
            order = np.argsort(diag["energy"][rm_always])       # cheapest first
            rm_always = rm_always[order[:room]]
            rm_dead = rm_dead[:cap]

        remove = np.concatenate([rm_dead, rm_always]).astype(int)
        if len(remove) == 0 or len(remove) >= H:
            return PruneDecision(
                remove=sorted(int(i) for i in remove[:H - 1]),
                diagnostics={"role": (["kept"] * H),
                             "act_freq": diag["act_freq"].tolist(),
                             "energy": diag["energy"].tolist(),
                             "_scalars": {"n_dead": 0, "n_always_on": 0,
                                          "criterion": self.criterion}})

        role = np.array(["kept"] * H, dtype=object)
        role[rm_dead] = "dead"
        role[rm_always] = "always_on"
        diag = {
            "role": role.tolist(),
            "act_freq": diag["act_freq"].tolist(),
            "energy": diag["energy"].tolist(),
            "collapsed": diag["collapsed"].tolist(),
            "_scalars": {
                "n_dead": int(len(rm_dead)),
                "n_always_on": int(len(rm_always)),
                "n_collapsed": int(diag["collapsed"].sum()),
                "criterion": self.criterion,
                "kappa": self.kappa,
                "repair": self.repair,
            },
        }
        dec = PruneDecision(remove=sorted(int(i) for i in remove),
                            diagnostics=diag)
        if len(rm_always) and self.repair != "none":
            keep = np.setdiff1d(np.arange(H), remove)
            C_new, const = self._repair_columns(model, layer_idx, ctx, keep,
                                                rm_always)
            if C_new is not None:
                dec.new_outgoing = torch.from_numpy(C_new)
            if const is not None:
                dec.bias_delta = torch.from_numpy(const)
        return dec


# ── self-tests ───────────────────────────────────────────────────────────────

def _selftest() -> None:  # pragma: no cover
    import copy

    from src.pruning.registry import build_pruning_method

    rng = np.random.default_rng(0)

    # 1. interval bounds are sound on a random box
    d, H = 6, 20
    W = rng.normal(size=(H, d)); b = rng.normal(size=H)
    lo, hi = -np.ones(d), np.ones(d)
    z_lo, z_hi = interval_bounds(W, b, lo, hi)
    X = rng.uniform(-1, 1, size=(20000, d))
    Z = X @ W.T + b
    assert (Z.min(axis=0) >= z_lo - 1e-9).all(), "interval lower bound unsound"
    assert (Z.max(axis=0) <= z_hi + 1e-9).all(), "interval upper bound unsound"
    # and tight: the box corners attain them
    assert np.allclose(z_hi, np.clip(W, 0, None) @ hi + np.clip(W, None, 0) @ lo + b)

    # 2. classify: planted dead / always-on units are found by both criteria
    Wp = W.copy(); bp = b.copy()
    bp[0] = -(np.abs(Wp[0]).sum() + 5.0)          # certainly dead on the box
    bp[1] = +(np.abs(Wp[1]).sum() + 5.0)          # certainly always-on
    Zp = X @ Wp.T + bp
    for crit, kw in (("interval", dict(W=Wp, b=bp, box=(lo, hi))),
                     ("margin", {}), ("empirical", {})):
        dead, always = classify(Zp, crit, kappa=4.0, **kw)
        assert dead[0] and not always[0], f"{crit}: planted dead unit missed"
        assert always[1] and not dead[1], f"{crit}: planted always-on unit missed"
        assert not dead[2:].any() or crit == "margin", f"{crit}: false dead call"

    # 3. end-to-end on a tiny MLP with planted units
    from torch.utils.data import TensorDataset

    from src.models.mlp import MLP

    torch.manual_seed(0)
    net = MLP(input_dim=d, output_dim=3, hidden_sizes=[H, 8])
    lin = net.net[0]
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(Wp).float())
        lin.bias.copy_(torch.from_numpy(bp).float())
        # a weight-collapsed unit: w ~ 0, b ~ 0 -> both sign tests undecided
        lin.weight[2] = 0.0
        lin.bias[2] = 0.0
    net.eval()

    Xt = torch.from_numpy(X[:256]).float()

    class _Bundle:
        def __init__(self, x):
            self.train_ds = TensorDataset(x, torch.zeros(len(x), dtype=torch.long))

    bundle = _Bundle(Xt)
    ctx = PruneContext(train_inputs=Xt, bundle=bundle, device=torch.device("cpu"))

    m = build_pruning_method("saturated", mode="dead", criterion="margin",
                             n_calib=256)
    dead_idx = m.select(net, 0, ctx).remove
    assert 0 in dead_idx, f"planted dead unit not selected, got {dead_idx}"
    assert 2 in dead_idx, ("weight-collapsed unit must be caught by the "
                           f"zero-energy guard, got {dead_idx}")
    assert 1 not in dead_idx, "always-on unit must not be called dead"

    # 4. dead removal is EXACTLY free
    with torch.no_grad():
        ref = net(Xt).double()
    from src.config import PruningConfig
    from src.pruning.surgery import prune_model

    class _M:
        kind = "saturated"
        params = {"mode": "dead", "criterion": "margin", "n_calib": 256}

    pruned, _ = prune_model(net, PruningConfig(methods=[_M()]), bundle,
                            torch.device("cpu"))
    with torch.no_grad():
        err = (pruned(Xt).double() - ref).abs().max().item()
    assert err < 1e-6, f"margin dead removal must be ~free, got {err:.3e}"

    # 4b. With the INTERVAL certificate the removal is exact in exact arithmetic:
    # a certified-dead unit contributes identically zero on the whole box. In
    # float32 the residual sits at ~1 ulp because dropping zero terms changes
    # the matmul's summation order, so the claim is checked in float64, where
    # that noise is ~1e-16 rather than ~1e-7.
    class _MI:
        kind = "saturated"
        params = {"mode": "dead", "criterion": "interval", "n_calib": 256,
                  "energy_tol": 0.0}

    net_d = copy.deepcopy(net).double()
    Xd = Xt.double()
    bundle_d = _Bundle(Xd)
    with torch.no_grad():
        ref_d = net_d(Xd)
    p_i, _ = prune_model(net_d, PruningConfig(methods=[_MI()]), bundle_d,
                         torch.device("cpu"))
    with torch.no_grad():
        err_i = (p_i(Xd) - ref_d).abs().max().item()
    assert err_i < 1e-12, \
        f"interval-certified removal must be exact, got {err_i:.3e}"
    assert p_i.prunable_layer(0).out_features < net_d.prunable_layer(0).out_features, \
        "interval criterion removed nothing, so exactness is vacuous"

    # 5. always-on: repair must beat plain deletion
    errs = {}
    for rep in ("none", "bias_only", "kernel", "empirical"):
        class _M2:
            kind = "saturated"
            params = {"mode": "always_on", "criterion": "margin",
                      "n_calib": 256, "repair": rep}
        p, _ = prune_model(net, PruningConfig(methods=[_M2()]), bundle,
                           torch.device("cpu"))
        with torch.no_grad():
            errs[rep] = float((p(Xt).double() - ref).abs().max())
    assert errs["kernel"] < errs["none"], errs
    assert errs["empirical"] < errs["none"], errs
    assert errs["bias_only"] <= errs["none"] * 1.05, errs

    # 5b. criterion='empirical' reproduces the retired `silent` method exactly:
    # dead iff the unit never produces a positive pre-activation on the sample.
    zt = _preactivations(net, 0, Xt)
    ever_active = (zt > 0).any(axis=0)
    dead_e, _ = classify(zt, "empirical")
    assert (dead_e == ~ever_active).all(), \
        "criterion='empirical' must equal 'never fires on any input'"

    # 6. param validation
    for bad in ({"mode": "x"}, {"criterion": "x"}, {"repair": "x"},
                {"kappa": -1.0}, {"energy_tol": 1.5}):
        try:
            build_pruning_method("saturated", **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")

    print("saturated.py self-tests passed:")
    print("  interval bounds sound and attained on the box corners")
    print("  planted dead / always-on units found by both criteria")
    print("  weight-collapsed unit caught by the zero-energy guard")
    print(f"  margin dead removal ~free (max |Delta| = {err:.1e});")
    print(f"  interval-certified dead removal exact in float64 "
          f"(max |Delta| = {err_i:.1e}; float32 residual is reduction-order roundoff)")
    print(f"  always-on repair beats deletion: none {errs['none']:.3e} -> "
          f"kernel {errs['kernel']:.3e}, empirical {errs['empirical']:.3e}")
    print("  parameter validation")


if __name__ == "__main__":
    _selftest()
