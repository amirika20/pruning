"""Iterative anchor-based neuron merging (phase 1: sum-rule fan-out surgery).

Parameterization per neuron i of a hidden Linear layer (orientation sign comes
from w, NOT from the anchor -- the anchor is sign-blind):

    u_i     = w_i / ||w_i||         orientation (unit vector)
    rho_i   = -b_i / ||w_i||        SIGNED offset; anchor beta_i = rho_i * u_i
    alpha_i = ||w_i||               gain
    c_i     = outgoing column [m]   (next Linear's column i)
    a_i     = alpha_i * ||c_i||     merge weight (the error-bound currency)

Merge rule -- covector addition, associative and order-independent:

    S_u += a_i u_i,   S_rho += a_i rho_i,   S_c += alpha_i c_i

realized in the unit-gain gauge:

    ubar = S_u/||S_u||,  rhobar = S_rho/||S_u||,
    row  = ubar, bias = -rhobar, outgoing column = S_c   (sum-rule surgery)

Singletons realize to the exact original function; identical covectors merge
exactly for ARBITRARY outgoing vectors (see self-test).

Pair selection: Ward linkage in box-centered covector space

    qt_i      = [R0 * u_i ; u_i^T x0 - rho_i]
    cost(k,l) = A_k A_l / (A_k + A_l) * ||qt_k - qt_l||^2

with x0 (center) and R0 (radius) of the layer's input box, propagated
data-free through the net by IBP.

Certified per-cluster error bound (Lipschitz, sum rule), against the cluster's
realized unit using the ORIGINAL member parameters:

    sum_i a_i * ( R0 ||u_i - ubar|| + |(u_i - ubar)^T x0 - (rho_i - rhobar)| )

Run `python studies/gram_stability/merge.py` for the numerical self-tests.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.mlp import MLP
from src.models.registry import PrunableModel

TINY = 1e-12
ZERO_NORM = 1e-10


# ── extraction ───────────────────────────────────────────────────────────────

@dataclass
class LayerUnits:
    u: np.ndarray      # [H, d] orientations
    rho: np.ndarray    # [H] signed offsets
    alpha: np.ndarray  # [H] gains
    C: np.ndarray      # [H, m] outgoing columns

    @property
    def a(self) -> np.ndarray:
        return self.alpha * np.linalg.norm(self.C, axis=1)

    def subset(self, idx: np.ndarray) -> "LayerUnits":
        return LayerUnits(self.u[idx], self.rho[idx], self.alpha[idx], self.C[idx])


def extract_units(model: PrunableModel, layer_idx: int) -> tuple[LayerUnits, np.ndarray]:
    """(units, ok): ok marks rows with ||w|| > ZERO_NORM. Zero-norm units have
    no hyperplane (sigma(b) is constant) -- they are excluded from merging and
    carried through frozen."""
    layer = model.prunable_layer(layer_idx)
    if not isinstance(layer, nn.Linear):
        raise TypeError("gram_stability supports Linear layers only")
    W = layer.weight.data.double().cpu().numpy()
    b = layer.bias.data.double().cpu().numpy()
    C = model.outgoing_weights(layer_idx).double().cpu().numpy()  # [H, m]

    alpha = np.linalg.norm(W, axis=1)
    ok = alpha > ZERO_NORM
    safe = np.where(ok, alpha, 1.0)
    return LayerUnits(W / safe[:, None], -b / safe, alpha, C), ok


def input_boxes(model: MLP, x_train: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per prunable layer, the (lo, hi) box of that layer's INPUT, propagated
    data-free by IBP from the input box (per-feature train range)."""
    lo = x_train.min(axis=0).astype(np.float64)
    hi = x_train.max(axis=0).astype(np.float64)
    boxes = []
    for i in range(model.n_prunable_layers()):
        boxes.append((lo, hi))
        W = model.prunable_layer(i).weight.data.double().cpu().numpy()
        b = model.prunable_layer(i).bias.data.double().cpu().numpy()
        Wp, Wn = np.clip(W, 0, None), np.clip(W, None, 0)
        z_lo, z_hi = Wp @ lo + Wn @ hi + b, Wp @ hi + Wn @ lo + b
        lo, hi = np.clip(z_lo, 0, None), np.clip(z_hi, 0, None)
    return boxes


# ── the iterative merge engine ───────────────────────────────────────────────

class IterativeMerge:
    """Greedy Ward-linkage pairwise merging. Every step merges the cheapest
    active pair; state is the raw covector sums, so merging is associative."""

    def __init__(self, units: LayerUnits, lo: np.ndarray, hi: np.ndarray):
        self.orig = units
        self.x0 = (lo + hi) / 2.0
        self.R0 = float(np.linalg.norm((hi - lo) / 2.0))

        a = units.a
        H = len(a)
        self.n_orig = H
        self.members: list[list[int]] = [[i] for i in range(H)]
        self.A = a.copy()
        self.S_u = a[:, None] * units.u                # [H, d]
        self.S_rho = a * units.rho                     # [H]
        self.S_c = units.alpha[:, None] * units.C      # [H, m]
        self.active = np.ones(H, dtype=bool)
        self.bound_terms = np.zeros(H)
        self.cum_ward = 0.0

        self._metric_init()
        self._cost = self._all_costs()

    # -- metric hooks (subclasses override to change pair selection) --------

    def _metric_init(self) -> None:
        self._qt = self._qt_of(np.arange(self.n_orig))  # [H, d+1]

    def _metric_update(self, k: int, l: int) -> None:
        """Refresh metric state after cluster l was merged into k (sums are
        already merged; called before costs are recomputed)."""
        self._qt[k] = self._qt_of(np.array([k]))[0]

    # -- realized units --------------------------------------------------

    def _realized_of(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(ubar [K,d], rhobar [K]) for clusters idx. Degenerate clusters
        (||S_u|| ~ 0, only possible with zero total weight) realize to 0."""
        norms = np.linalg.norm(self.S_u[idx], axis=1)
        safe = np.where(norms > TINY, norms, 1.0)
        ubar = np.where((norms > TINY)[:, None], self.S_u[idx] / safe[:, None], 0.0)
        rhobar = np.where(norms > TINY, self.S_rho[idx] / safe, 0.0)
        return ubar, rhobar

    def _qt_of(self, idx: np.ndarray) -> np.ndarray:
        ubar, rhobar = self._realized_of(idx)
        return np.concatenate(
            [self.R0 * ubar, (ubar @ self.x0 - rhobar)[:, None]], axis=1)

    def realize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(W_rows [K,d], biases [K], outgoing cols [K,m], weights A [K]) of
        the active clusters, in stable index order."""
        idx = np.flatnonzero(self.active)
        ubar, rhobar = self._realized_of(idx)
        return ubar, -rhobar, self.S_c[idx], self.A[idx]

    # -- Ward costs --------------------------------------------------------

    def _pair_costs(self, k: int, idx: np.ndarray) -> np.ndarray:
        d2 = ((self._qt[idx] - self._qt[k]) ** 2).sum(axis=1)
        denom = self.A[idx] + self.A[k]
        return np.where(denom > TINY, self.A[idx] * self.A[k] / np.maximum(denom, TINY), 0.0) * d2

    def _all_costs(self) -> np.ndarray:
        H = self.n_orig
        cost = np.full((H, H), np.inf)
        idx = np.arange(H)
        for k in range(H):
            cost[k, idx] = self._pair_costs(k, idx)
        cost[np.arange(H), np.arange(H)] = np.inf
        return cost

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    # -- one merge step ------------------------------------------------------

    def step(self) -> dict:
        k, l = np.unravel_index(np.argmin(self._cost), self._cost.shape)
        k, l = int(min(k, l)), int(max(k, l))
        ward_cost = float(self._cost[k, l])
        self.cum_ward += ward_cost

        # covector addition (associative)
        self.members[k].extend(self.members[l])
        self.A[k] += self.A[l]
        self.S_u[k] += self.S_u[l]
        self.S_rho[k] += self.S_rho[l]
        self.S_c[k] += self.S_c[l]
        self.active[l] = False
        self._cost[l, :] = np.inf
        self._cost[:, l] = np.inf

        # refresh merged cluster: metric state, costs, certified bound term
        self._metric_update(k, l)
        others = np.flatnonzero(self.active)
        others = others[others != k]
        new_costs = self._pair_costs(k, others)
        self._cost[k, :] = np.inf
        self._cost[:, k] = np.inf
        self._cost[k, others] = new_costs
        self._cost[others, k] = new_costs

        mem = np.array(self.members[k])
        ubar, rhobar = self._realized_of(np.array([k]))
        ubar, rhobar = ubar[0], rhobar[0]
        du = self.orig.u[mem] - ubar
        gap = np.abs(du @ self.x0 - (self.orig.rho[mem] - rhobar))
        term = float((self.orig.a[mem]
                      * (self.R0 * np.linalg.norm(du, axis=1) + gap)).sum())
        self.bound_terms[k] = term
        norm_Su = float(np.linalg.norm(self.S_u[k]))

        return {
            "survivor": k,
            "removed": l,
            "cluster_size": len(self.members[k]),
            "ward_cost": ward_cost,
            "cum_ward": self.cum_ward,
            "bound_cluster": term,
            "bound_total": float(self.bound_terms[self.active].sum()),
            "overshoot": float(self.A[k] / norm_Su) if norm_Su > TINY else np.nan,
        }

    # -- conserved quantities --------------------------------------------------

    def m1_raw(self) -> np.ndarray:
        """Sum of raw covector sums -- conserved EXACTLY by construction."""
        idx = np.flatnonzero(self.active)
        return np.concatenate([self.S_u[idx].sum(axis=0), [self.S_rho[idx].sum()]])

    def m1_realized(self) -> np.ndarray:
        """Sum of A_k * [ubar_k; rhobar_k] -- conserved only up to the
        second-order overshoot A_k/||S_u_k||; its drift measures that."""
        idx = np.flatnonzero(self.active)
        ubar, rhobar = self._realized_of(idx)
        w = self.A[idx][:, None]
        return np.concatenate([(w * ubar).sum(axis=0), [(self.A[idx] * rhobar).sum()]])


# ── self-tests ────────────────────────────────────────────────────────────────

def _selftest() -> None:  # pragma: no cover
    rng = np.random.default_rng(0)
    d, H, m = 10, 12, 4
    W = rng.normal(size=(H, d))
    b = rng.normal(size=H)
    C = rng.normal(size=(H, m))  # arbitrary signs

    # neurons 3,4,5: identical hyperplane, different gains, random outgoing
    for j, s in [(4, 2.5), (5, 0.7)]:
        W[j] = s * W[3]
        b[j] = s * b[3]

    alpha = np.linalg.norm(W, axis=1)
    units = LayerUnits(W / alpha[:, None], -b / alpha, alpha, C)
    lo, hi = -np.ones(d), np.ones(d)

    def f(Wr, br, cols, X):
        return np.maximum(X @ Wr.T + br, 0.0) @ cols

    X = rng.normal(size=(256, d))
    ref = f(W, b, C, X)

    eng = IterativeMerge(units, lo, hi)
    r1 = eng.step()
    r2 = eng.step()
    assert r1["ward_cost"] < 1e-20 and r2["ward_cost"] < 1e-20, "duplicates must cost 0"
    Wr, br, cols, _ = eng.realize()
    assert Wr.shape[0] == H - 2
    err = np.abs(f(Wr, br, cols, X) - ref).max()
    assert err < 1e-9, f"identical-covector merge must be exact, got {err:.2e}"

    # associativity: stepwise sums == direct cluster sums
    a = units.a
    g_direct = (a[[3, 4, 5], None] * units.u[[3, 4, 5]]).sum(axis=0)
    k = [i for i in (3, 4, 5) if eng.active[i]][0]
    assert np.allclose(eng.S_u[k], g_direct, atol=1e-12), "merge must be associative"

    # first-moment conservation over a full random run
    eng2 = IterativeMerge(LayerUnits(*[x.copy() for x in (units.u, units.rho, units.alpha, units.C)]), lo, hi)
    m1_0 = eng2.m1_raw()
    while eng2.n_active > 1:
        eng2.step()
    drift = np.abs(eng2.m1_raw() - m1_0).max() / np.abs(m1_0).max()
    assert drift < 1e-12, f"m1_raw must be conserved, drift {drift:.2e}"

    print("merge.py self-tests passed: exactness, associativity, m1 conservation")


if __name__ == "__main__":
    _selftest()
