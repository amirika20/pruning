"""MASH -- Mass-weighted Aggregation of Structured Hyperplanes.

Our merge-based structured pruning. A ReLU unit h(x) = sigma(w^T x + b) with
w != 0 is written canonically as h(x) = alpha * sigma(u^T x - rho) with

    alpha = ||w||          gain
    u     = w / ||w||      orientation (unit vector)
    rho   = -b / ||w||     SIGNED offset; the boundary is {x : u^T x = rho}
    c                      outgoing column (the consumer's column for the unit)
    v     = alpha * c      effective outgoing weight
    a     = ||v||          the unit's MASS

The network is unchanged by (w, b, c) -> (t w, t b, c / t) for t > 0, so alpha
is not identifiable alone: the gauge-invariant data are the geometric code
q = [u; rho] and the effective outgoing vector v -- hence the mass a = ||v||.
`cylinder` and `delta_f` depend on (q, a) only; `exact_damage` also reads the
outgoing Gram <v_k, v_l>, so it sees whether two fan-outs reinforce or cancel
rather than only how heavy they are. All three are gauge invariant, since v is.

MERGING is addition of mass-weighted covectors. A cluster C carries the triple

    g_C = sum_C a_i u_i,   r_C = sum_C a_i rho_i,   A_C = sum_C a_i (its mass)

plus w_C = sum_C v_i. All four are ADDITIVE, so merging is associative and the
dendrogram does not depend on the order a cluster was assembled. The realized
unit normalizes only at emission:

    ubar = g_C / ||g_C||,   rhobar = r_C / ||g_C||,   eta_C = ||g_C|| / A_C

With `gauge_correct` (default) the emitted column is eta_C * w_C, which makes
the merged unit realize the mass-weighted MEAN pre-activation -- the optimal
affine surrogate. Without it the plain sum w_C is emitted, which overshoots the
slope by 1/eta_C; that variant is kept because the recorded studies used it.

SCORES (`score=`):

    cylinder      d^2 on the covector cylinder, [radius * u ; u^T x0 - rho],
                  evaluated at cluster CENTROIDS. Pre-ReLU and domain-only:
                  needs the input box, never activations.
    delta_f       d^2 = ||phi_k - phi_l||^2 between the units' POST-ReLU
                  unit-gain responses under N(mu, Sigma). Data-light.
    exact_damage  the exact expected squared layer-output error of the merge,
                  including the merged unit's own response and the outgoing
                  Gram <w_k, w_l>. Also data-light, strictly more work than
                  delta_f -- kept for ablation, since the two agree closely in
                  practice (the fan-out term contributes very little).

`cylinder` and `delta_f` are mass-weighted WARD costs: A_k A_l / (A_k + A_l)
times a squared distance between cluster codes. `exact_damage` is not -- it is
an output-space energy carrying no Ward weight and satisfying no Ward identity,
so its pass is Ward-SHAPED, not Ward. It is written in the gain the merge is
actually EMITTED with, psi_C = eta_C phi_C = sigma((g_C^T x - r_C) / A_C) under
`gauge_correct` and the unit-gain phi_C without it, so score and emission always
refer to the same function. Scoring phi_C while emitting eta_C phi_C -- what
this did until the gauge audit -- misprices any merge whose members are not
near-parallel (28% on random units, and the good pairs where eta ~ 1 are
exactly the ones it got right); `gauge_correct=False` reproduces those numbers.

`cylinder` is pre-ReLU by construction, and no metric on the boundary geometry
can know what the ReLU clips -- that needs the measure. This is why the
domain-only tier is a CERTIFICATE tier rather than a capacity tier.

DICTIONARY (`dictionary=`): `merge` emits the new mass-weighted hyperplane;
`medoid` keeps each cluster's heaviest ORIGINAL unit and lets the repair absorb
the rest. Only `merge` synthesizes a hyperplane, so only `merge` requires that
the layer have no paired BatchNorm (see `select`). The engine takes the
dictionary too, because the CERTIFICATE has to be written against the atom the
cluster actually emits: a bound on the distance to the centroid says nothing
about a medoid that was emitted in its place (it is violated by 1.5x on a
three-unit fan whose heaviest member sits at the edge).

REPAIR (`repair=`) of the surviving consumer columns:

    none        DELETE: keep each cluster's medoid with its ORIGINAL column
                and drop the rest. No transfer, no solve -- the selection-only
                control (medoid dictionary only; a merged hyperplane has no
                original column to keep)
    sum         emit the accumulated column; no reference to any measure
    projection  per-cluster rank-1 least squares (each column optimal, but
                clusters stay independent)
    kernel      global least squares, G Chat = B C, with G and B closed-form
                kernel evaluations under N(mu, Sigma) -- no activations
    empirical   the same normal equations with G, B as sample averages over
                the calibration activations

Repair strictly dominates in that order (nested feasible sets), and it is the
largest single effect in the pipeline. Note that once the repair is global
(`kernel`/`empirical`) it depends on the survivors only through the SUBSPACE
they span, so the score and dictionary choices matter much less than they do
under `sum`.

Two entries are registered. `mash` takes a width (`n_remove`); `mash_certified`
is the domain-only tier and instead takes a tolerance, choosing its own width
as the last cut whose certificate stays inside the budget.

Run `python -c "from src.pruning.methods.mash import _selftest; _selftest()"`
for the numerical self-tests.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ndtr, owens_t

from src.models.registry import PrunableModel
from src.pruning.registry import (
    PruneContext, PruneDecision, PruningMethod, forward_chunked,
    register_pruning_method)

_SQRT2PI = np.sqrt(2.0 * np.pi)
_EPS_C = 1e-9        # |corr| beyond 1 - _EPS_C uses the degenerate branches
_NUDGE = 1e-12
TINY = 1e-12
ZERO_NORM = 1e-10

SCORES = ("cylinder", "delta_f", "exact_damage")
DICTIONARIES = ("merge", "medoid")
REPAIRS = ("none", "sum", "projection", "kernel", "empirical")


# ── rectified-Gaussian moments ───────────────────────────────────────────────

def _phi(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(x)) / _SQRT2PI


def _phibar(x: np.ndarray) -> np.ndarray:
    return ndtr(-x)


def _bvn_sf(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """P(X > a, Y > b) for a standard bivariate normal with correlation c,
    via Owen's T (Owen 1956). Vectorized; no quadrature."""
    a, b, c = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (a, b, c)))
    h, k = -a, -b
    cc = np.clip(c, -1.0 + _EPS_C, 1.0 - _EPS_C)
    s = np.sqrt(1.0 - cc * cc)
    hh = np.where(np.abs(h) < _NUDGE, _NUDGE, h)
    kk = np.where(np.abs(k) < _NUDGE, _NUDGE, k)
    beta = np.where(hh * kk < 0.0, 0.5, 0.0)
    general = (0.5 * (ndtr(hh) + ndtr(kk))
               - owens_t(hh, (kk - cc * hh) / (hh * s))
               - owens_t(kk, (hh - cc * kk) / (kk * s)) - beta)
    pos = ndtr(np.minimum(h, k))                            # c -> +1
    neg = np.clip(ndtr(h) + ndtr(k) - 1.0, 0.0, None)       # c -> -1
    out = np.where(c >= 1.0 - _EPS_C, pos,
                   np.where(c <= -1.0 + _EPS_C, neg, general))
    return np.clip(out, 0.0, 1.0)


def relu_self(a: np.ndarray) -> np.ndarray:
    """A(a) = E[(t - a)_+^2] for t ~ N(0, 1)."""
    a = np.asarray(a, dtype=np.float64)
    return (1.0 + a * a) * _phibar(a) - a * _phi(a)


def relu_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """G(a, b, c) = E[(x - a)_+ (y - b)_+] for a standard bivariate normal
    with correlation c. G(0, 0, cos t) is the order-one arc-cosine kernel
    (sin t + (pi - t) cos t) / (2 pi)."""
    a, b, c = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (a, b, c)))
    cc = np.clip(c, -1.0 + _EPS_C, 1.0 - _EPS_C)
    s = np.sqrt(1.0 - cc * cc)
    general = ((cc + a * b) * _bvn_sf(a, b, cc)
               - b * _phi(a) * _phibar((b - cc * a) / s)
               - a * _phi(b) * _phibar((a - cc * b) / s)
               + s * _phi(b) * _phi((a - cc * b) / s))

    m = np.maximum(a, b)                                    # c -> +1
    pos = _phibar(m) * (1.0 + a * b) + _phi(m) * (m - a - b)

    lo, hi = a, -b                                          # c -> -1
    empty = lo >= hi
    lo_s, hi_s = np.where(empty, 0.0, lo), np.where(empty, 0.0, hi)
    I0 = ndtr(hi_s) - ndtr(lo_s)
    I1 = _phi(lo_s) - _phi(hi_s)
    I2 = I0 - (hi_s * _phi(hi_s) - lo_s * _phi(lo_s))
    neg = np.where(empty, 0.0, -I2 + (a - b) * I1 + a * b * I0)

    out = np.where(c >= 1.0 - _EPS_C, pos,
                   np.where(c <= -1.0 + _EPS_C, neg, general))
    return np.clip(out, 0.0, None)


def _relu_moments(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """E[sigma(t)] for t ~ N(m, s^2)."""
    s = np.maximum(s, 1e-300)
    z = -m / s
    return s * (_phi(z) - z * _phibar(z))


def gram(mA, sA, mB, sB, corr) -> np.ndarray:
    """E[sigma(tA) sigma(tB)] on broadcast grids of Gaussian units."""
    sA = np.maximum(sA, 1e-300)
    sB = np.maximum(sB, 1e-300)
    return sA * sB * relu_cross(-mA / sA, -mB / sB, np.clip(corr, -1.0, 1.0))


class UnitMoments:
    """Pre-activation statistics of affine units under z ~ N(mu, Sigma)."""

    def __init__(self, U: np.ndarray, rho: np.ndarray, mu: np.ndarray,
                 Sigma: np.ndarray):
        self.U = U
        self.m = U @ mu - rho
        self._SU = U @ Sigma
        self.s = np.sqrt(np.clip(np.einsum("ij,ij->i", self._SU, U), 0.0, None))

    def corr_with(self, other: "UnitMoments") -> np.ndarray:
        cov = self._SU @ other.U.T
        return cov / np.maximum(np.outer(self.s, other.s), 1e-300)

    def gram_with(self, other: "UnitMoments") -> np.ndarray:
        return gram(self.m[:, None], self.s[:, None], other.m[None, :],
                    other.s[None, :], self.corr_with(other))


# ── units ────────────────────────────────────────────────────────────────────

@dataclass
class Units:
    """Canonical form of one prunable layer's units. BatchNorm, if paired with
    the layer, is already folded into (u, rho, alpha)."""

    u: np.ndarray        # [H, d] orientations
    rho: np.ndarray      # [H] signed offsets
    alpha: np.ndarray    # [H] gains
    C: np.ndarray        # [H, m] outgoing columns

    @property
    def mass(self) -> np.ndarray:
        """a_i = alpha_i ||c_i|| = ||v_i||, gauge-invariant."""
        return self.alpha * np.linalg.norm(self.C, axis=1)

    @property
    def V(self) -> np.ndarray:
        """Effective outgoing weights v_i = alpha_i c_i."""
        return self.alpha[:, None] * self.C

    def subset(self, idx: np.ndarray) -> "Units":
        return Units(self.u[idx], self.rho[idx], self.alpha[idx], self.C[idx])


def extract_units(model: PrunableModel, layer_idx: int) -> tuple[Units, np.ndarray]:
    """(units, ok). `ok` marks rows with ||w|| > 0: a zero-norm unit has no
    hyperplane (sigma(b) is constant), so it is frozen out of merging and
    carried through untouched."""
    layer = model.prunable_layer(layer_idx)
    W = layer.weight.data.double().reshape(layer.weight.shape[0], -1).cpu().numpy()
    b = (layer.bias.data.double().cpu().numpy() if layer.bias is not None
         else np.zeros(W.shape[0]))

    bn = model.prunable_bn(layer_idx)
    if bn is not None:                       # fold exactly, eval-mode BN
        std = np.sqrt(bn.running_var.detach().double().cpu().numpy() + bn.eps)
        scale = bn.weight.detach().double().cpu().numpy() / std
        b = (bn.bias.detach().double().cpu().numpy()
             - scale * bn.running_mean.detach().double().cpu().numpy()
             + scale * b)
        W = scale[:, None] * W

    C = model.outgoing_weights(layer_idx).double().cpu().numpy()
    if C.shape[0] != W.shape[0]:             # patch-major conv consumer
        C = C.reshape(W.shape[0], -1)

    alpha = np.linalg.norm(W, axis=1)
    ok = alpha > ZERO_NORM
    safe = np.where(ok, alpha, 1.0)
    return Units(W / safe[:, None], -b / safe, alpha, C), ok


# ── the greedy engine ────────────────────────────────────────────────────────

class MashEngine:
    """Greedy mass-weighted Ward agglomeration over one layer's units.

    State is the additive quadruple (g, r, A, w) per cluster, so every merge is
    associative and the trajectory is order-free. One pass produces the whole
    dendrogram; a cut at any width is then free, which is what makes a single
    pass serve every target width.
    """

    def __init__(self, units: Units, score: str = "delta_f",
                 x0: np.ndarray | None = None, radius: float | None = None,
                 mu: np.ndarray | None = None, Sigma: np.ndarray | None = None,
                 gauge_correct: bool = True, row_cache: bool = True, measure: str = "gaussian",
                 Z: np.ndarray | None = None, dictionary: str = "merge"):
        if score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}, got {score!r}")
        if dictionary not in DICTIONARIES:
            raise ValueError(f"dictionary must be one of {DICTIONARIES}, "
                             f"got {dictionary!r}")
        if measure not in ("gaussian", "empirical"):
            raise ValueError("measure must be 'gaussian' or 'empirical'")
        if score == "cylinder" and (x0 is None or radius is None):
            raise ValueError("score='cylinder' needs the input box (x0, radius)")
        if score != "cylinder" and measure == "gaussian" and (mu is None or Sigma is None):
            raise ValueError(f"score={score!r} needs calibration moments (mu, Sigma)")
        if score != "cylinder" and measure == "empirical" and Z is None:
            raise ValueError("measure='empirical' needs the layer inputs Z")

        self.orig = units
        self.score = score
        self.gauge_correct = gauge_correct
        # The dictionary is not used to SCORE anything -- it decides which atom
        # the certificate is written against, since that is the one `realize`
        # will emit.
        self.dictionary = dictionary
        # Which gain a cluster's post-ReLU response is expressed in.
        # `exact_damage` claims to be the damage of the emission, so it has to
        # follow the emission: psi_C = eta_C phi_C under gauge correction.
        # `delta_f` is a unit-gain proxy by definition and stays 'realized'.
        self._resp_kind = ("centroid" if score == "exact_damage" and gauge_correct
                           else "realized")
        a = units.mass
        H = len(a)
        self.n_orig = H
        self.members: list[list[int]] = [[i] for i in range(H)]
        self.A = a.copy()                              # cluster mass
        self.g = a[:, None] * units.u                  # [H, d]
        self.r = a * units.rho                         # [H]
        self.w = units.V.copy()                        # [H, m]
        self.active = np.ones(H, dtype=bool)
        self.cum_cost = 0.0

        # certificate state (always maintained; cheap and needed by both tiers)
        self.x0 = np.zeros(units.u.shape[1]) if x0 is None else np.asarray(x0, float)
        self.R = 0.0 if radius is None else float(radius)
        self.cert_terms = np.zeros(H)

        self.measure = measure
        if score == "cylinder":
            self._radius = float(radius)
            # Ward on the cylinder admits the Lance--Williams recursion, so the
            # per-step update is O(K) with no d-dimensional work. Only the
            # initial matrix needs the embedding, and that is one BLAS product
            # rather than H recomputations of the same codes.
            self._lw = True
        elif measure == "empirical":
            # Exact under the empirical measure and free of any d x d
            # covariance: keep the projections Z g^T, which are ADDITIVE under
            # merging exactly as the triples are, so no d-dimensional work
            # happens after initialization.
            self._Z = np.asarray(Z, dtype=np.float64)
            self._Zg = self._Z @ self.g.T                  # [N, H] -> stored .T
            self._Zg = self._Zg.T                          # [H, N], additive
            self._Phi = self._responses(np.arange(H))      # [H, N]
        else:
            self.mu = np.asarray(mu, dtype=np.float64)
            Sig = np.asarray(Sigma, dtype=np.float64)
            self.Sigma = np.diag(Sig) if Sig.ndim == 1 else Sig
            D = self.g @ self.Sigma @ self.g.T
            self._D = 0.5 * (D + D.T)                  # additive
            self._dotmu = self.g @ self.mu             # additive
            if score == "exact_damage":
                self._W = self.w @ self.w.T            # additive
        if score == "exact_damage" and measure == "empirical":
            self._W = self.w @ self.w.T
        self._cost = self._all_costs()
        # Row-minimum cache. Scanning the whole H x H matrix for the global
        # argmin at every step makes a pass O(H^3), which is what put the wide
        # transformer FFNs out of reach (37s at H=4096, ~77min at H=20480).
        # Caching each row's minimum makes the global argmin O(K), and a merge
        # only invalidates the rows whose minimum pointed at the merged pair --
        # so a pass becomes O(H^2) in practice.
        #
        # np.argmin returns the FIRST minimum in row-major order, so taking the
        # smallest row index achieving the smallest row-minimum, then that row's
        # smallest column index, reproduces the flat argmin EXACTLY -- ties
        # included. That matters: a different tie-break is a different method,
        # not a faster one.
        self._row_cache = bool(row_cache)
        if self.n_orig:
            self._row_arg = np.argmin(self._cost, axis=1)
            self._row_min = self._cost[np.arange(self.n_orig), self._row_arg]
        else:
            # An empty layer is reachable, not defensive: see plan().
            self._row_arg = np.zeros(0, dtype=int)
            self._row_min = np.zeros(0)

    # -- codes -------------------------------------------------------------

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    def _norms(self, idx: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.g[idx], axis=1)

    def _resp_den(self, idx: np.ndarray) -> np.ndarray:
        """Denominator of the pre-activation the responses are written in:
        ||g_C|| for the unit-gain phi_C, A_C for the emitted psi_C."""
        return self.A[idx] if self._resp_kind == "centroid" else self._norms(idx)

    def _responses(self, idx: np.ndarray) -> np.ndarray:
        """Post-ReLU responses of clusters `idx` on the calibration rows,
        [len(idx), N], in the gain `_resp_kind` selects. Only defined for
        measure='empirical'."""
        n = self._resp_den(idx)
        safe = np.where(n > TINY, n, 1.0)
        t = (self._Zg[idx] - self.r[idx][:, None]) / safe[:, None]
        return np.maximum(np.where((n > TINY)[:, None], t, 0.0), 0.0)

    def _code(self, idx: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
        """(direction, offset-vs-x0) of clusters `idx`.

        kind='realized'  g/||g||       -- the unit-norm hyperplane emitted
        kind='centroid'  g/A           -- the mass-weighted mean covector; this
                                          is eta_C times the realized one, and
                                          it is the coefficient vector of the
                                          mean pre-activation, hence what the
                                          Ward identity and the gauge-corrected
                                          emission both refer to.
        """
        den = self._norms(idx) if kind == "realized" else self.A[idx]
        safe = np.where(den > TINY, den, 1.0)
        u = np.where((den > TINY)[:, None], self.g[idx] / safe[:, None], 0.0)
        off = np.where(den > TINY, (self.g[idx] @ self.x0 - self.r[idx]) / safe, 0.0)
        return u, off

    def _medoid(self, k: int) -> int:
        """The member `realize` keeps under dictionary='medoid': the heaviest.
        Members are SORTED first, so a tie in mass is broken the same way it is
        there (partition_at hands back ascending members) -- certifying a
        different unit than the one emitted is the whole failure mode here."""
        mem = np.sort(np.asarray(self.members[k], dtype=int))
        return int(mem[np.argmax(self.orig.mass[mem])])

    def emitted_code(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The code of the affine function each cluster actually realizes: the
        mass-weighted centroid under dictionary='merge', the surviving ORIGINAL
        unit under 'medoid'."""
        if self.dictionary == "medoid":
            rep = np.array([self._medoid(int(k)) for k in np.atleast_1d(idx)],
                           dtype=int)
            u = self.orig.u[rep]
            return u, u @ self.x0 - self.orig.rho[rep]
        return self._code(idx, "centroid" if self.gauge_correct else "realized")

    # -- scores ------------------------------------------------------------

    def _ward_weight(self, k: int, idx: np.ndarray) -> np.ndarray:
        den = self.A[idx] + self.A[k]
        return np.where(den > TINY, self.A[idx] * self.A[k] / np.maximum(den, TINY), 0.0)

    def _pair_costs(self, k: int, idx: np.ndarray) -> np.ndarray:
        if len(idx) == 0:
            return np.zeros(0)
        if self.score == "cylinder":
            u, off = self._code(np.concatenate([[k], idx]), "centroid")
            qt = np.concatenate([self._radius * u, off[:, None]], axis=1)
            d2 = ((qt[1:] - qt[0]) ** 2).sum(axis=1)
            return self._ward_weight(k, idx) * d2

        if self.measure == "empirical":
            N = self._Phi.shape[1]
            phik = self._Phi[k]
            if self.score == "delta_f":
                diff = self._Phi[idx] - phik[None, :]
                d2 = (diff * diff).sum(axis=1) / max(N, 1)
                return self._ward_weight(k, idx) * d2
            # exact_damage: the candidate merged unit's own response is needed,
            # and it too follows from the additive projections -- in the same
            # gain as _Phi, so that v_k psi_k + v_l psi_l - (v_k + v_l) psi_c is
            # the difference between the emitted functions.
            gk_ = self._Zg[k][None, :] + self._Zg[idx]
            rc = self.r[k] + self.r[idx]
            nc = (self.A[k] + self.A[idx] if self._resp_kind == "centroid"
                  else np.linalg.norm(self.g[k][None, :] + self.g[idx], axis=1))
            safe = np.where(nc > TINY, nc, 1.0)
            phic = np.maximum((gk_ - rc[:, None]) / safe[:, None], 0.0)
            phil = self._Phi[idx]
            ek = phik[None, :] - phic
            el = phil - phic
            wkk, wll = self._W[k, k], self._W[idx, idx]
            wkl = self._W[k, idx]
            cost = (wkk * (ek * ek).sum(1) + wll * (el * el).sum(1)
                    + 2.0 * wkl * (ek * el).sum(1)) / max(N, 1)
            return np.where(nc > TINY, np.clip(cost, 0.0, None), 0.0)

        # norm(g[k]) rather than _resp_den's norm(g[[k]], axis=1): the two agree
        # only to an ulp, and an ulp is enough to flip a tie in the argmin, so
        # the unit-gain branch keeps the expression its recorded runs used.
        nk = (float(self.A[k]) if self._resp_kind == "centroid"
              else float(np.linalg.norm(self.g[k])))
        nl = self._resp_den(idx)
        Dkk = max(float(self._D[k, k]), 0.0)
        Dll = np.clip(self._D[idx, idx], 0.0, None)
        Dkl = self._D[k, idx]
        SDk, SDl = np.sqrt(Dkk), np.sqrt(Dll)
        # response of a cluster: t = (g^T x - r) / den, with den = ||g|| for the
        # unit-gain phi and A_C for the emitted psi (see _resp_den). Only the
        # SCALE s carries den; the standardized threshold z is free of it, which
        # is why the two gauges share this whole block.
        zk = (self.r[k] - self._dotmu[k]) / max(SDk, 1e-300)
        zl = (self.r[idx] - self._dotmu[idx]) / np.maximum(SDl, 1e-300)
        sk = SDk / max(nk, TINY)
        sl = SDl / np.where(nl > TINY, nl, TINY)
        ckl = np.clip(Dkl / np.maximum(SDk * SDl, 1e-300), -1.0, 1.0)

        Kkk = sk * sk * relu_self(zk)
        Kll = sl * sl * relu_self(zl)
        Kkl = sk * sl * relu_cross(np.full_like(zl, zk), zl, ckl)

        if self.score == "delta_f":
            d2 = np.clip(Kkk + Kll - 2.0 * Kkl, 0.0, None)
            return self._ward_weight(k, idx) * d2

        # exact_damage: bring in the candidate merged unit and the fan-out Gram
        Dcc = np.clip(Dkk + 2.0 * Dkl + Dll, 0.0, None)
        SDc = np.sqrt(Dcc)
        nc = (self.A[k] + self.A[idx] if self._resp_kind == "centroid"
              else np.linalg.norm(self.g[k] + self.g[idx], axis=1))
        zc = (self.r[k] + self.r[idx] - self._dotmu[k] - self._dotmu[idx]) \
            / np.maximum(SDc, 1e-300)
        sc = SDc / np.where(nc > TINY, nc, TINY)
        ckc = np.clip((Dkk + Dkl) / np.maximum(SDk * SDc, 1e-300), -1.0, 1.0)
        clc = np.clip((Dkl + Dll) / np.maximum(SDl * SDc, 1e-300), -1.0, 1.0)
        Kcc = sc * sc * relu_self(zc)
        Kkc = sk * sc * relu_cross(np.full_like(zc, zk), zc, ckc)
        Klc = sl * sc * relu_cross(zl, zc, clc)
        cost = (self._W[k, k] * np.clip(Kkk + Kcc - 2.0 * Kkc, 0.0, None)
                + self._W[idx, idx] * np.clip(Kll + Kcc - 2.0 * Klc, 0.0, None)
                + 2.0 * self._W[k, idx] * (Kkl + Kcc - Kkc - Klc))
        return np.where(nc > TINY, np.clip(cost, 0.0, None), 0.0)

    def _all_costs(self) -> np.ndarray:
        H = self.n_orig
        allidx = np.arange(H)
        if self.score == "cylinder":
            # ||p_i - p_j||^2 from the Gram, so the whole matrix is one matmul
            u, off = self._code(allidx, "centroid")
            P = np.concatenate([self._radius * u, off[:, None]], axis=1)
            sq = (P * P).sum(axis=1)
            d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (P @ P.T), 0.0)
            den = self.A[:, None] + self.A[None, :]
            w = np.where(den > TINY, np.outer(self.A, self.A) / np.maximum(den, TINY), 0.0)
            cost = w * d2
        else:
            cost = np.full((H, H), np.inf)
            for k in range(H):
                cost[k, allidx] = self._pair_costs(k, allidx)
        cost[allidx, allidx] = np.inf
        return cost

    # -- certificate -------------------------------------------------------

    def _cert_term(self, k: int) -> float:
        """sum_i a_i (R ||u_i - uhat|| + |gamma_i - gammahat|) over the members
        of cluster k, against the code it actually emits -- so it follows the
        DICTIONARY, not just the gauge. Summing this over the active clusters
        bounds sup_x ||F_T(x) - F_0(x)|| on the box, for the SUM rule: a global
        repair re-solves the columns and is outside the derivation."""
        mem = np.array(self.members[k])
        uh, oh = self.emitted_code(np.array([k]))
        du = self.orig.u[mem] - uh[0]
        gam = self.orig.u[mem] @ self.x0 - self.orig.rho[mem]
        return float((self.orig.mass[mem]
                      * (self.R * np.linalg.norm(du, axis=1)
                         + np.abs(gam - oh[0]))).sum())

    def certificate(self) -> float:
        return float(self.cert_terms[self.active].sum())

    # -- one merge ---------------------------------------------------------

    def _argmin(self) -> tuple[int, int]:
        """The cheapest active pair, identical to np.argmin over the matrix."""
        if not self._row_cache:
            i, j = np.unravel_index(np.argmin(self._cost), self._cost.shape)
            return int(i), int(j)
        i = int(np.argmin(self._row_min))
        return i, int(self._row_arg[i])

    def step(self) -> dict:
        i0, j0 = self._argmin()
        k, l = int(min(i0, j0)), int(max(i0, j0))
        cost = float(self._cost[k, l])
        self.cum_cost += cost
        # Lance--Williams needs the PRE-merge masses and cost rows
        lw = getattr(self, "_lw", False)
        if lw:
            A_k_old, A_l_old = float(self.A[k]), float(self.A[l])
            row_k_old = self._cost[k].copy()
            row_l_old = self._cost[l].copy()

        self.members[k].extend(self.members[l])
        self.A[k] += self.A[l]
        self.g[k] += self.g[l]
        self.r[k] += self.r[l]
        self.w[k] += self.w[l]
        self.active[l] = False
        self._cost[l, :] = np.inf
        self._cost[:, l] = np.inf

        if self.score != "cylinder" and self.measure == "empirical":
            self._Zg[k] += self._Zg[l]
            self._Phi[k] = self._responses(np.array([k]))[0]
            if self.score == "exact_damage":
                rw = self._W[k, :] + self._W[l, :]
                rw[k] = self._W[k, k] + 2.0 * self._W[k, l] + self._W[l, l]
                self._W[k, :] = rw
                self._W[:, k] = rw
        elif self.score != "cylinder":
            row = self._D[k, :] + self._D[l, :]
            row[k] = self._D[k, k] + 2.0 * self._D[k, l] + self._D[l, l]
            self._D[k, :] = row
            self._D[:, k] = row
            self._dotmu[k] += self._dotmu[l]
            if self.score == "exact_damage":
                rw = self._W[k, :] + self._W[l, :]
                rw[k] = self._W[k, k] + 2.0 * self._W[k, l] + self._W[l, l]
                self._W[k, :] = rw
                self._W[:, k] = rw

        others = np.flatnonzero(self.active)
        others = others[others != k]
        if lw:
            # Ward's Lance--Williams update, in the mass-weighted form:
            #   D(k u l, o) = [(A_k+A_o)D(k,o) + (A_l+A_o)D(l,o) - A_o D(k,l)]
            #                 / (A_k + A_l + A_o)
            A_o = self.A[others]
            new = ((A_k_old + A_o) * row_k_old[others]
                   + (A_l_old + A_o) * row_l_old[others]
                   - A_o * cost) / (A_k_old + A_l_old + A_o)
        else:
            new = self._pair_costs(k, others)
        self._cost[k, :] = np.inf
        self._cost[:, k] = np.inf
        self._cost[k, others] = new
        self._cost[others, k] = new

        if self._row_cache:
            # `l` is gone; `k` changed wholesale, so rescan its row once.
            self._row_min[l] = np.inf
            self._row_arg[l] = l
            self._row_arg[k] = int(np.argmin(self._cost[k]))
            self._row_min[k] = self._cost[k, self._row_arg[k]]
            # For every other active row, the entry to `l` vanished and the
            # entry to `k` changed. A row whose cached minimum pointed at
            # either has lost its reference and must be rescanned; every other
            # row still holds a valid minimum and only needs to be compared
            # against its new cost to `k`. Read old_arg BEFORE writing.
            old_arg = self._row_arg[others]
            stale = (old_arg == k) | (old_arg == l)
            cheap = others[~stale]
            if len(cheap):
                nc = self._cost[cheap, k]
                better = nc < self._row_min[cheap]
                self._row_min[cheap[better]] = nc[better]
                self._row_arg[cheap[better]] = k
            for i in others[stale]:
                a = int(np.argmin(self._cost[i]))
                self._row_arg[i] = a
                self._row_min[i] = self._cost[i, a]

        self.cert_terms[k] = self._cert_term(k)
        nrm = float(np.linalg.norm(self.g[k]))
        return {"survivor": k, "removed": l, "cost": cost,
                "cum_cost": self.cum_cost, "certificate": self.certificate(),
                "cluster_size": len(self.members[k]),
                "eta": nrm / self.A[k] if self.A[k] > TINY else np.nan}

    def dendrogram(self, max_steps: int | None = None) -> list[dict]:
        """Run the full greedy pass (or `max_steps` of it) and return the
        per-step records. Cutting at step T is then a free lookup."""
        recs = []
        budget = self.n_orig - 1 if max_steps is None else max_steps
        while self.n_active > 1 and len(recs) < budget:
            recs.append(self.step())
        return recs

    def mass_total(self) -> float:
        return float(self.A[self.active].sum())


def partition_at(H: int, pairs: list[tuple[int, int]], k: int) -> list[list[int]]:
    """Union-find cut of a merge sequence after `k` merges."""
    parent = list(range(H))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs[:k]:
        parent[find(j)] = find(i)
    clusters: dict[int, list[int]] = {}
    for x in range(H):
        clusters.setdefault(find(x), []).append(x)
    return list(clusters.values())


# ── realization and repair ───────────────────────────────────────────────────

def realize(units: Units, ok: np.ndarray, clusters: list[list[int]],
            dictionary: str = "merge", repair: str = "sum",
            gauge_correct: bool = True, mu: np.ndarray | None = None,
            Sigma: np.ndarray | None = None, Z: np.ndarray | None = None,
            bias_fix: bool = False, ridge: float = 1e-8):
    """(rows, biases, columns, keep_slots, bias_delta) for one layer.

    `rows`/`biases` are the surviving units' own parameters, `columns` their
    consumer columns, `keep_slots` the original index kept for each cluster.
    `Z` (calibration inputs OF THIS LAYER, [N, d]) is required by
    repair='empirical'; (mu, Sigma) by 'projection'/'kernel' and by bias_fix.
    """
    if dictionary not in DICTIONARIES:
        raise ValueError(f"dictionary must be one of {DICTIONARIES}")
    if repair not in REPAIRS:
        raise ValueError(f"repair must be one of {REPAIRS}")
    if repair == "none" and dictionary != "medoid":
        raise ValueError("repair='none' keeps original units, so it needs "
                         "dictionary='medoid'")
    a, V = units.mass, units.V
    rows, biases, cols, keep = [], [], [], []
    # The UNREPAIRED column of each survivor: what it carries under repair=none
    # (medoid: its own effective column) or, for a merged unit that has no
    # original column, the sum rule's emission. The ridge shrinks toward this.
    prior: list[np.ndarray] = []

    for mem in clusters:
        m = np.array(mem)
        if len(m) == 1 and not ok[m[0]]:          # constant unit: carry as-is
            i = int(m[0])
            rows.append(units.u[i]); biases.append(-units.rho[i])
            cols.append(units.C[i]); prior.append(units.C[i]); keep.append(i)
            continue
        w_sum = V[m].sum(axis=0)
        if dictionary == "medoid":
            rep = int(m[np.argmax(a[m])])
            # repair='none': the survivor keeps its own effective column V[rep];
            # every other rule starts from the cluster's accumulated column.
            col = V[rep] if repair == "none" else w_sum
            u_new, rho_new = units.u[rep], units.rho[rep]
            keep.append(rep); prior.append(V[rep])
        else:
            g = (a[m, None] * units.u[m]).sum(axis=0)
            n = float(np.linalg.norm(g))
            A = float(a[m].sum())
            if n < TINY:
                # Total cancellation: the members' covectors sum to zero, so no
                # hyperplane exists -- but the mass-weighted mean pre-activation
                # is still the CONSTANT -r_C/A_C, and the gauge-corrected
                # emission realizes sigma of it. A zero row plus that bias emits
                # exactly the atom the certificate is written against; emitting
                # zero instead silently drops a contribution the bound has
                # already paid for. Without gauge correction the emitted atom is
                # sigma of the unit-gain code, which is 0/0 here, so that
                # variant keeps deleting.
                rho_c = float((a[m] * units.rho[m]).sum()) / max(A, TINY)
                rows.append(np.zeros(units.u.shape[1]))
                biases.append(-rho_c if gauge_correct else 0.0)
                cols.append(w_sum); prior.append(w_sum)
                keep.append(int(m[np.argmax(a[m])]))
                continue
            eta = n / A if A > TINY else 1.0
            u_new = g / n
            rho_new = float((a[m] * units.rho[m]).sum()) / n
            col = eta * w_sum if gauge_correct else w_sum
            keep.append(int(m[np.argmax(a[m])])); prior.append(col)
        rows.append(u_new); biases.append(-rho_new); cols.append(col)

    rows = np.array(rows); biases = np.array(biases); cols = np.array(cols)
    prior_arr = np.array(prior)
    if repair == "none" or (repair == "sum" and not bias_fix):
        return rows, biases, cols, keep, None

    # Grams: sample averages over the calibration inputs, or analytic under
    # N(mu, Sigma). Both estimate the same G, B of the normal equations.
    if repair == "empirical":
        if Z is None:
            raise ValueError("repair='empirical' needs the layer inputs Z")
        # G = Phi_keep^T Phi_keep / N, B = Phi_keep^T (Phi_orig * alpha) / N,
        # Ehat = mean Phi_keep, Eh = mean (Phi_orig * alpha) -- on the GPU.
        G, B, Ehat, Eh = _relu_grams(Z, rows, biases, units.u, -units.rho,
                                     scale_b=units.alpha)
    else:
        if mu is None or Sigma is None:
            raise ValueError(f"repair={repair!r} needs calibration moments")
        S = np.diag(Sigma) if np.ndim(Sigma) == 1 else Sigma
        orig = UnitMoments(units.u, units.rho, mu, S)
        kept = UnitMoments(rows, -biases, mu, S)
        G = kept.gram_with(kept)
        B = kept.gram_with(orig) * units.alpha[None, :]
        Eh = units.alpha * _relu_moments(orig.m, orig.s)
        Ehat = _relu_moments(kept.m, kept.s)

    if repair == "projection":
        for k, mem in enumerate(clusters):
            m = np.array(mem)
            if len(m) == 1 and not ok[m[0]]:
                continue
            if G[k, k] > 1e-30:
                cols[k] = (B[k, m] / G[k, k])[:, None].T @ units.C[m]
    elif repair in ("kernel", "empirical"):
        # Ridge CENTRED ON THE UNREPAIRED COLUMNS: solve for the correction
        # to `prior`, so lam -> 0 is the plain least squares and lam -> inf
        # returns the survivors' own columns (repair=none / the sum rule) --
        # the same limit OSSCAR's damped target and repair_deletion's transfer
        # have. A zero-centred ridge would instead drive every column to zero
        # and make a strong ridge fail for a reason that has nothing to do
        # with regularization. At the 1e-8 operating value the two agree to
        # ~1e-8 relative.
        lam = ridge * max(np.trace(G) / max(len(G), 1), 1e-30)
        target = B @ units.C
        cols = prior_arr + np.linalg.solve(G + lam * np.eye(len(G)),
                                           target - G @ prior_arr)

    delta = (Eh @ units.C - Ehat @ cols) if bias_fix else None
    return rows, biases, cols, keep, delta


def reshape_outgoing(model: PrunableModel, layer_idx: int,
                     C: np.ndarray) -> np.ndarray:
    """Put a [H, ...] column matrix into the layout set_outgoing_weights wants.

    A conv consumer reads each channel through kH*kW taps, so outgoing_weights
    is [H*kk, fan_out] while we carry the flattened block per channel as
    [H, kk*fan_out]. For a Linear consumer this is the identity.
    """
    ow = model.outgoing_weights(layer_idx)
    if ow.shape[0] == C.shape[0]:
        return np.ascontiguousarray(C)
    return np.ascontiguousarray(C.reshape(ow.shape[0], ow.shape[1]))


def fold_constant(model: PrunableModel, layer_idx: int,
                  const: np.ndarray) -> np.ndarray:
    """Reduce a per-COLUMN constant to the consumer's bias shape.

    A unit owns k columns of its consumer (kH*kW across a conv hop, the spatial
    count across a flatten, 1 for a plain Linear), and the constant is solved
    per column. Replacing the unit's response by a constant therefore adds
    sum_t w[t, o] * c to output o, i.e. the per-column vector summed over that
    unit's block -- which is exact for a Linear or a flatten, and the interior
    answer for a padded conv, where border positions see fewer taps.

    Without this the delta is k times too long and `add_outgoing_bias` raises;
    LeNet is what surfaced it, being the first entry whose conv units feed a
    multi-column consumer while `bias_fix`/`bias_only` is in play.
    """
    fan_out = model.outgoing_weights(layer_idx).shape[1]
    if const.shape[0] == fan_out:
        return const
    if const.shape[0] % fan_out:
        raise ValueError(
            f"constant of length {const.shape[0]} is not a whole number of "
            f"consumer columns (fan_out {fan_out}) for layer {layer_idx}")
    return const.reshape(-1, fan_out).sum(axis=0)


def repair_deletion(model: PrunableModel, layer_idx: int, x: torch.Tensor,
                    removed: Sequence[int], repair: str = "kernel",
                    max_rows: int = 20000, ridge: float = 1e-8
                    ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Re-solve the surviving consumer columns after deleting `removed`.

    (new_outgoing in the ORIGINAL gauge [H, ...], bias_delta or None). Least
    squares over the survivors' responses, with the constant function adjoined
    when the consumer can hold one -- so the removed units' contribution is
    absorbed as far as the surviving span allows and the leftover constant is
    exact rather than applied afterwards. This is the shared primitive behind
    every delete-and-repair method here; merging has its own path, since it
    also has to emit a hyperplane.
    """
    if repair == "none" or len(removed) == 0:
        return None, None
    if repair not in ("kernel", "empirical", "bias_only"):
        raise ValueError("repair for deletion must be one of 'kernel', "
                         f"'empirical', 'bias_only', 'none'; got {repair!r}")
    if (isinstance(model.prunable_layer(layer_idx), nn.Conv2d)
            and repair == "kernel"):
        raise NotImplementedError(
            "repair='kernel' evaluates its Grams in closed form under a "
            "Gaussian over the layer's inputs, which on conv means a "
            "patch-space covariance -- expensive and misspecified for patches. "
            "Use repair='empirical'.")

    units, _ = extract_units(model, layer_idx)
    H = len(units.rho)
    removed = np.asarray(sorted(int(i) for i in removed), dtype=int)
    keep = np.setdiff1d(np.arange(H), removed)
    if len(keep) == 0:
        return None, None
    V = units.V
    C_new = units.C.copy()

    Z = _layer_inputs(model, layer_idx, x, max_rows=max_rows)
    if repair in ("empirical", "bias_only"):
        # G = Phi_keep^T Phi_keep / N, B = Phi_keep^T Phi_removed / N and the
        # two column means, formed on the GPU (see _relu_grams).
        G, B, g1, b1 = _relu_grams(Z, units.u[keep], -units.rho[keep],
                                   units.u[removed], -units.rho[removed])
    else:
        mu, Sigma = Z.mean(axis=0), np.atleast_2d(np.cov(Z.T))
        mk = UnitMoments(units.u[keep], units.rho[keep], mu, Sigma)
        mr = UnitMoments(units.u[removed], units.rho[removed], mu, Sigma)
        G, B = mk.gram_with(mk), mk.gram_with(mr)
        g1, b1 = _relu_moments(mk.m, mk.s), _relu_moments(mr.m, mr.s)

    K = len(keep)
    use_const = consumer_has_bias(model, layer_idx)
    n = K + 1 if use_const else K
    A = np.empty((n, n))
    A[:K, :K] = G
    rhs = np.empty((n, len(removed)))
    rhs[:K] = B
    if use_const:
        A[:K, K] = g1
        A[K, :K] = g1
        A[K, K] = 1.0
        rhs[K] = b1
    lam = ridge * max(np.trace(A) / n, TINY)   # relative ridge, as in realize()
    sol = np.linalg.solve(A + lam * np.eye(n), rhs)

    X = sol[:K] @ V[removed]
    const = sol[K] @ V[removed] if use_const else None
    if repair == "bias_only":
        X = np.zeros_like(X)
        const = (b1 @ V[removed]) if use_const else None
    safe_alpha = np.where(units.alpha[keep] > TINY, units.alpha[keep], 1.0)
    C_new[keep] = units.C[keep] + X / safe_alpha[:, None]
    if const is not None:
        const = fold_constant(model, layer_idx, const)
    return C_new, const


# ── the pruning methods ──────────────────────────────────────────────────────

def consumer_has_bias(model: PrunableModel, layer_idx: int) -> bool:
    """Whether the consumer of this layer can absorb a constant.

    Conv consumers in BatchNorm architectures are built with bias=False, so
    there is nowhere to fold a residual mean. A repair that adjoins the constant
    function to its dictionary must know this: solving WITH a constant and then
    failing to apply it is worse than never adjoining it, because the solved
    columns are then optimal for a network that includes an offset which is not
    actually there.
    """
    try:
        module = model.outgoing_module(layer_idx)
    except NotImplementedError:
        return False
    return getattr(module, "bias", None) is not None


def _gram_device() -> torch.device:
    """Where the sample Grams are formed. MASH_GRAM_DEVICE overrides; the
    default is the GPU when there is one, else the CPU (still via torch)."""
    env = os.environ.get("MASH_GRAM_DEVICE")
    if env:
        return torch.device(env)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _relu_grams(Z: np.ndarray, rows_a: np.ndarray, bias_a: np.ndarray,
                rows_b: np.ndarray, bias_b: np.ndarray,
                scale_b: np.ndarray | None = None, chunk: int = 4096
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample-average Grams of two rectified unit sets over the inputs Z.

        Phi_a = relu(Z rows_a^T + bias_a)            [N, K]
        Phi_b = relu(Z rows_b^T + bias_b) * scale_b  [N, H]
        G  = Phi_a^T Phi_a / N        B  = Phi_a^T Phi_b / N
        ea = mean_n Phi_a             eb = mean_n Phi_b

    The SAME expressions the empirical repairs wrote in NumPy, evaluated in
    float64 in torch on `_gram_device()`. This is a speed change only: on
    OPT-350m the two 20000 x 4096 activation matrices and their 4096 x 4096
    Grams took ~10 s per layer in host NumPy on the job's two cores -- 300 of
    the 310 s each width cost, against 7 s for the sum rule -- and the same
    step was 887 s per width at OPT-1.3b. Rows of Z stream through in chunks so
    no N x H matrix is ever resident; only the K x K and K x H results return.
    """
    dev = _gram_device()
    f64 = dict(dtype=torch.float64, device=dev)
    Zt = torch.as_tensor(np.ascontiguousarray(Z), **f64)
    Ra, ba = torch.as_tensor(rows_a, **f64), torch.as_tensor(bias_a, **f64)
    Rb, bb = torch.as_tensor(rows_b, **f64), torch.as_tensor(bias_b, **f64)
    sb = None if scale_b is None else torch.as_tensor(scale_b, **f64)
    K, H = Ra.shape[0], Rb.shape[0]
    G = torch.zeros((K, K), **f64)
    B = torch.zeros((K, H), **f64)
    ea = torch.zeros(K, **f64)
    eb = torch.zeros(H, **f64)
    N = Zt.shape[0]
    with torch.no_grad():
        for z in Zt.split(chunk):
            pa = torch.relu(z @ Ra.T + ba)
            pb = torch.relu(z @ Rb.T + bb)
            if sb is not None:
                pb = pb * sb
            G.addmm_(pa.T, pa)
            B.addmm_(pa.T, pb)
            ea += pa.sum(dim=0)
            eb += pb.sum(dim=0)
    out = (G / N, B / N, ea / N, eb / N)
    return tuple(t.cpu().numpy() for t in out)


def _layer_inputs(model: PrunableModel, layer_idx: int, x: torch.Tensor,
                  max_rows: int = 20000) -> np.ndarray:
    """What the layer's units see, one row per observation, [N, d].

    Linear: one row per sample (or token). Conv: one row per IM2COL PATCH, so a
    filter is a single unit on patch space and d = C_in*kH*kW -- extracted with
    the layer's own kernel/stride/padding/dilation, so the zero padding a filter
    actually sees is included. Patch counts explode (one per spatial position
    per image), so rows are subsampled to `max_rows` with a fixed generator:
    deterministic, and uniform rather than strided so it cannot alias with
    spatial structure.
    """
    layer = model.prunable_layer(layer_idx)
    grabbed: list[torch.Tensor] = []
    h = layer.register_forward_hook(lambda m, inp, out: grabbed.append(inp[0].detach()))
    try:
        # Chunked: one forward of all 128 OPT-1.3b calibration sequences
        # allocated 12 GiB of logits and OOMed a 40GB A100 (see calib_chunk).
        forward_chunked(model, x)
    finally:
        h.remove()
    z = torch.cat(grabbed, dim=0)
    if isinstance(layer, nn.Conv2d):
        # [N, d, L] with d = C_in*kH*kW in channel-major order, matching
        # conv.weight.reshape(C_out, -1)
        patches = F.unfold(z, layer.kernel_size, dilation=layer.dilation,
                           padding=layer.padding, stride=layer.stride)
        z = patches.permute(0, 2, 1).reshape(-1, patches.shape[1])
    else:
        z = z.reshape(-1, z.shape[-1])
    out = z.double().cpu().numpy()
    if len(out) > max_rows:
        rng = np.random.default_rng(12345 + layer_idx)
        out = out[rng.choice(len(out), max_rows, replace=False)]
    return out


@dataclass
class MashPlan:
    """One full greedy pass over a layer, cut-able at any width.

    Holds ONLY what is width-independent AND model-state-independent: the merge
    sequence over the layer's own output units, plus the per-step records. It
    deliberately does NOT cache weights, moments or responses.

    That distinction is the point. Pruning an EARLIER layer removes columns from
    this layer's weight matrix, so this layer's orientations, masses and input
    moments all change -- but its output units, and hence a partition defined
    over them, do not. So the expensive part (the O(H^2) greedy pass and its
    kernel evaluations) is cached, while the realization is recomputed in the
    current model's coordinates at every width. That is the recorded finding
    that intact-model dendrograms suffice and need not be rebuilt on the
    progressively pruned model.
    """

    layer_idx: int
    pairs: list[tuple[int, int]]
    recs: list[dict]
    idx_map: np.ndarray
    n_mergeable: int
    # Mass of each mergeable unit at planning time, so the MEDOID dictionary's
    # removed set (every cluster member but its heaviest) can be reconstructed
    # from the serialized dendrogram alone. Not used by the cut itself.
    mass: np.ndarray | None = None

    @property
    def max_merges(self) -> int:
        return len(self.pairs)

    def merges_for(self, fraction: float) -> int:
        """Merge count for a target fraction of this layer's units."""
        return max(0, min(int(round(fraction * (self.n_mergeable - 1))),
                          self.max_merges))

    @classmethod
    def from_record(cls, rec: dict) -> "MashPlan":
        """Inverse of to_record: rebuild a cut-able plan from dendrogram.json.

        The record stores layer indices; the plan works in sub-indices (positions
        in `mergeable`), so the pairs and the per-step `removed` are mapped back.
        Only the fields a cut needs are restored (cost, certificate) -- the
        engine's other per-step diagnostics are not, and nothing reads them.
        """
        idx = [int(i) for i in rec["mergeable"]]
        pos = {u: i for i, u in enumerate(idx)}
        pairs, recs = [], []
        for st in rec["steps"]:
            s, r = pos[int(st["survivor"])], pos[int(st["removed"])]
            pairs.append((s, r))
            row = {"survivor": s, "removed": r, "cost": float(st.get("cost", float("nan")))}
            if "certificate" in st:
                row["certificate"] = float(st["certificate"])
            recs.append(row)
        mass = rec.get("mass")
        return cls(layer_idx=int(rec["layer"]), pairs=pairs, recs=recs,
                   idx_map=np.asarray(idx, dtype=int), n_mergeable=int(rec["n_mergeable"]),
                   mass=None if mass is None else np.asarray(mass, dtype=float))

    def to_record(self) -> dict:
        """The dendrogram as plain JSON, in the LAYER's own unit indices.

        Every step is (survivor, removed, cost, certificate), so a cut at any
        width -- and any score-vs-oracle or overlap study -- can be re-read from
        the file without re-planning, which at OPT-6.7b is hours per layer.
        `mergeable` is idx_map: the sub-index -> layer-index map the pairs were
        recorded in; `frozen` units (zero-norm rows) never enter the pass.
        """
        idx = [int(i) for i in self.idx_map]
        steps = []
        for (s, r), rec in zip(self.pairs, self.recs):
            row = {"survivor": idx[s], "removed": idx[r],
                   "cost": float(rec.get("cost", float("nan")))}
            if "certificate" in rec:
                row["certificate"] = float(rec["certificate"])
            steps.append(row)
        out = {"layer": int(self.layer_idx), "n_mergeable": int(self.n_mergeable),
               "mergeable": idx, "steps": steps}
        if self.mass is not None:
            out["mass"] = [float(a) for a in self.mass]
        return out


class _MashBase(PruningMethod):
    """Shared plumbing: build the engine, cut it, realize, emit a decision."""

    score = "delta_f"
    dictionary = None          # None = auto: merge where it is safe, else medoid
    repair = None              # None = auto: kernel on Linear, empirical on conv

    def __init__(self, score: str | None = None, dictionary: str | None = None,
                 repair: str | None = None, n_calib: int = 128,
                 gauge_correct: bool = True, bias_fix: bool = False,
                 radius: str = "sup", measure: str | None = None,
                 max_rows: int = 20000, ridge: float = 1e-8):
        if score is not None:
            self.score = score
        if dictionary is not None:
            self.dictionary = dictionary
        if repair is not None:
            self.repair = repair
        for name, val, allowed in (("score", self.score, SCORES),
                                   ("dictionary", self.dictionary, DICTIONARIES),
                                   ("repair", self.repair, REPAIRS)):
            if val is not None and val not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {val!r}")
        if radius not in ("sup", "l2"):
            raise ValueError("radius must be 'sup' or 'l2'")
        if measure is not None and measure not in ("gaussian", "empirical"):
            raise ValueError("measure must be 'gaussian', 'empirical' or None")
        self.n_calib = int(n_calib)
        self.gauge_correct = bool(gauge_correct)
        self.bias_fix = bool(bias_fix)
        self.radius = radius
        # None = empirical (sample Grams over the calibration inputs); 'gaussian'
        # is the closed-form ablation and is FC-only
        self.measure = measure
        self.max_rows = int(max_rows)
        # Ridge on the global repairs' normal equations, RELATIVE to the mean
        # diagonal of the Gram: 1e-8 is a numerical guard (the operating
        # value); OSSCAR damps its Hessian at 1e-2, and `mash_ridge_*` arms
        # test that strength here.
        self.ridge = float(ridge)
        if self.repair == "none" and self.dictionary == "merge":
            raise ValueError("repair='none' needs dictionary='medoid'")

    def plan_key(self) -> dict:
        """Everything the DENDROGRAM depends on, and nothing it does not.

        Two arms with equal keys plan identically, so one arm's dendrogram.json
        serves the other: the repair, the ridge and bias_fix only enter at
        realization. `dictionary` is included because the exact_damage score
        reads the survivor's response through it; `measure`/`dictionary` are
        the CONFIGURED values (None = per-layer auto), which is what makes two
        configs comparable before any layer has been seen.
        """
        return {"kind": "mash", "score": self.score, "dictionary": self.dictionary,
                "measure": self.measure or "empirical",
                "gauge_correct": self.gauge_correct,
                "radius": self.radius, "n_calib": self.n_calib,
                "max_rows": self.max_rows}

    # -- setup ------------------------------------------------------------

    def _prepare(self, model: PrunableModel, layer_idx: int, ctx: PruneContext,
                 for_scoring: bool = True):
        layer = model.prunable_layer(layer_idx)
        is_conv = isinstance(layer, nn.Conv2d)
        if not isinstance(layer, (nn.Linear, nn.Conv2d)):
            raise NotImplementedError(
                f"mash supports nn.Linear and nn.Conv2d prunable layers; "
                f"layer {layer_idx} is a {type(layer).__name__}")

        # Resolve the "auto" defaults against THIS layer, so one config can span
        # a mixed architecture. A merged hyperplane cannot be written back
        # through a paired BatchNorm (see below), and the closed-form repair
        # needs a covariance over the layer's inputs, which on conv is patch
        # space -- so the auto choices are the ones that are both valid and, per
        # the recorded conv results, better there.
        can_merge = (model.prunable_bn(layer_idx) is None
                     and (not is_conv or layer.bias is not None))
        if self.repair == "none":
            self.dictionary = self.dictionary or "medoid"     # nothing else fits
        self.dictionary = self.dictionary or ("merge" if can_merge else "medoid")
        self.repair = self.repair or ("empirical" if is_conv else "kernel")

        if self.dictionary == "merge":
            if model.prunable_bn(layer_idx) is not None:
                raise NotImplementedError(
                    "dictionary='merge' synthesizes a hyperplane no original "
                    "unit realizes. Its parameters are the BN-FOLDED ones, so "
                    "writing them into the layer while the paired BatchNorm "
                    "still applies would double-count the normalization, and "
                    "rewriting the BN's per-channel affine is not something "
                    "PrunableModel exposes. Use dictionary='medoid' on "
                    "BatchNorm layers: survivors keep their original filters "
                    "AND their BN slots, so nothing has to be rewritten -- and "
                    "on BN-trained filters it is the better arm anyway.")
            if is_conv and layer.bias is None:
                raise NotImplementedError(
                    "dictionary='merge' needs somewhere to put the merged "
                    "unit's offset, and this conv has bias=False. Use "
                    "dictionary='medoid'.")

        units, ok = extract_units(model, layer_idx)
        idx_map = np.flatnonzero(ok)
        frozen = np.flatnonzero(~ok)
        sub = units.subset(idx_map)

        # THE SCORE'S MEASURE DEFAULTS TO EMPIRICAL on every layer type (decided
        # 2026-09-10). Conv always did: patch-space Gaussians are misspecified
        # (E10) and the d x d covariance is large. Linear layers used the
        # closed-form rectified-Gaussian moments until the OPT repair tier
        # showed the Gaussian-scored merges recovering nothing under the sum
        # rule while the same recipe worked on conv -- OPT's heavy-tailed,
        # outlier-dominated pre-activations are what the Gaussian model gets
        # wrong. measure='gaussian' is now the explicit ablation
        # (`mash_gaussian_*` arms, FC only).
        measure = self.measure or "empirical"
        if is_conv and self.repair in ("kernel", "projection"):
            raise NotImplementedError(
                f"repair={self.repair!r} evaluates its Grams in closed form "
                "under a Gaussian over the layer's inputs, which on conv means "
                "a patch-space covariance -- expensive and, per the recorded "
                "conv study, misspecified. Use repair='empirical' (the same "
                "normal equations from sample averages) or 'sum'.")

        # Only pay for what this call actually consumes. `for_scoring=False`
        # means the partition is already known (a cached plan is being cut), so
        # the box and the score's moments are not needed -- and the sum rule
        # needs no data at all, which is what makes cutting the certified tier
        # at another width nearly free.
        needs_sigma = ((for_scoring and self.score != "cylinder"
                        and measure == "gaussian")
                       or self.repair in ("kernel", "projection")
                       or self.bias_fix)
        needs_Z = (for_scoring or needs_sigma
                   or self.repair in ("empirical", "projection"))
        Z = mu = Sigma = None
        d = units.u.shape[1]
        x0, R, rad = np.zeros(d), 0.0, 0.0
        if needs_Z:
            Z = _layer_inputs(model, layer_idx, ctx.train_inputs[: self.n_calib],
                              max_rows=self.max_rows)
            mu = Z.mean(axis=0)
            if needs_sigma:
                Sigma = np.atleast_2d(np.cov(Z.T))
            x0 = (Z.min(axis=0) + Z.max(axis=0)) / 2.0
            half = (Z.max(axis=0) - Z.min(axis=0)) / 2.0
            R = float(np.linalg.norm(half))
            rad = R if self.radius == "sup" else R / np.sqrt(d + 2.0)
        return (units, ok, idx_map, frozen, sub, Z, mu, Sigma, x0, R, rad,
                measure)

    @staticmethod
    def _diagnostics(units, H, clusters, keep, recs, idx_map, cert=None):
        """Per-unit bookkeeping: which cluster each unit landed in, whether it
        survived as that cluster's representative or was absorbed, at which
        greedy step and cost, and how much mass it carried."""
        a = units.mass
        cluster = np.full(H, -1)
        role = np.array(["singleton"] * H, dtype=object)
        step = np.full(H, -1)
        cost = np.full(H, np.nan)
        for cid, (mem, slot) in enumerate(zip(clusters, keep)):
            for i in mem:
                cluster[i] = cid
                role[i] = "survivor" if i == slot else "absorbed"
            if len(mem) == 1:
                role[mem[0]] = "singleton"
        # attribute each merge step to the unit it removed (in layer indices)
        for t, r in enumerate(recs):
            rm = int(idx_map[r["removed"]])
            step[rm] = t
            cost[rm] = r["cost"]
        eta = np.full(H, np.nan)
        for cid, mem in enumerate(clusters):
            if len(mem) > 1:
                m = np.array(mem)
                g = (a[m, None] * units.u[m]).sum(axis=0)
                A = float(a[m].sum())
                eta[m] = float(np.linalg.norm(g)) / A if A > TINY else np.nan
        out = {"cluster": cluster.tolist(), "role": role.tolist(),
               "merge_step": step.tolist(), "merge_cost": cost.tolist(),
               "mass": a.tolist(), "eta": eta.tolist()}
        sizes = [len(m) for m in clusters]
        out["_scalars"] = {
            "n_clusters": len(clusters),
            "n_multi_clusters": int(sum(1 for x in sizes if x > 1)),
            "max_cluster_size": int(max(sizes)) if sizes else 0,
            "mass_total": float(a.sum()),
            "mass_absorbed": float(a[[i for i, r in enumerate(role)
                                      if r == "absorbed"]].sum()),
            "cum_cost": float(np.nansum([r["cost"] for r in recs])),
        }
        if cert is not None:
            out["_scalars"]["certificate"] = float(cert)
        return out

    def _emit(self, model, layer_idx, units, ok, idx_map, frozen, clusters_sub,
              Z, mu, Sigma, recs=None, cert=None) -> PruneDecision:
        clusters = [[int(idx_map[i]) for i in cl] for cl in clusters_sub] \
            + [[int(f)] for f in frozen]
        bias_fix = self.bias_fix and consumer_has_bias(model, layer_idx)
        if self.bias_fix and not bias_fix:
            logging.warning(
                f"  layer {layer_idx}: bias_fix requested but the consumer has "
                "no bias term (typical of conv+BatchNorm), so the residual mean "
                "cannot be folded; continuing without it")
        rows, biases, cols, keep, delta = realize(
            units, ok, clusters, dictionary=self.dictionary, repair=self.repair,
            gauge_correct=self.gauge_correct, mu=mu, Sigma=Sigma, Z=Z,
            bias_fix=bias_fix, ridge=self.ridge)

        H, d = units.u.shape
        # `cols` are EFFECTIVE outgoing weights v = alpha * c, so the consumer
        # column depends on what gain the surviving unit ends up with:
        #
        #   merge   we rewrite the layer's rows to the unit-gain hyperplane, so
        #           alpha becomes 1 and the effective weight IS the column.
        #   medoid  the survivor keeps its ORIGINAL row -- and, crucially, its
        #           original BatchNorm slot -- so its gain is still alpha_slot
        #           and the column must be divided by it. Emitting no incoming
        #           weights at all is what makes the medoid dictionary work
        #           under BN without any BN write-back.
        rewrite_incoming = self.dictionary == "merge"
        W_new = units.u.copy()
        b_new = -units.rho.copy()
        C_new = units.C.copy()
        remove: list[int] = []
        for cl, slot, row, bias, col in zip(clusters, keep, rows, biases, cols):
            if rewrite_incoming:
                W_new[slot] = row
                b_new[slot] = bias
                C_new[slot] = col
            else:
                a = units.alpha[slot]
                C_new[slot] = col / (a if a > TINY else 1.0)
            remove.extend(int(i) for i in cl if int(i) != slot)

        # A conv consumer reads each channel through kH*kW taps, so
        # outgoing_weights is [H*kk, fan_out] while we carry [H, kk*fan_out].
        # The internal layout is that flattened block per channel, so a reshape
        # restores what set_outgoing_weights expects.
        C_out = reshape_outgoing(model, layer_idx, C_new)

        dec = PruneDecision(
            remove=sorted(remove),
            new_outgoing=torch.from_numpy(C_out),
            diagnostics=self._diagnostics(units, H, clusters, keep, recs or [],
                                          idx_map, cert))
        if rewrite_incoming:
            layer = model.prunable_layer(layer_idx)
            W_t = torch.from_numpy(W_new).view_as(layer.weight)
            dec.new_incoming = (W_t, torch.from_numpy(b_new))
        if delta is not None:
            dec.bias_delta = torch.from_numpy(
                fold_constant(model, layer_idx, delta))
        return dec


    # -- the anytime interface -------------------------------------------

    def plan(self, model: PrunableModel, layer_idx: int,
             ctx: PruneContext) -> MashPlan:
        """Run the greedy pass ONCE, all the way down. Cutting it afterwards is
        free, which is what lets a single pass serve every target width."""
        (units, ok, idx_map, frozen, sub, Z, mu, Sigma,
         x0, R, rad, measure) = self._prepare(model, layer_idx, ctx)
        if len(idx_map) < 2:
            # Fewer than two mergeable units, so there is no pair to score. This
            # is REACHABLE, not defensive: a unit's alpha = ||w_row|| is taken
            # over the layer's INPUT space, and pruning an earlier layer deletes
            # columns from this one -- so a row whose weight lived in the deleted
            # columns collapses to zero norm and stops being a hyperplane. At an
            # aggressive fraction every row of a narrow layer can go that way.
            # Building an engine on it used to reach np.argmin of an empty
            # matrix and take the whole cell down.
            logging.info(
                f"  layer {layer_idx}: {len(idx_map)} mergeable unit(s) "
                "(earlier pruning collapsed the rest); nothing to plan")
            return MashPlan(layer_idx=layer_idx, pairs=[], recs=[],
                            idx_map=idx_map, n_mergeable=len(idx_map))
        eng = MashEngine(sub, score=self.score, x0=x0, radius=rad, mu=mu,
                         Sigma=Sigma, gauge_correct=self.gauge_correct,
                         measure=measure, Z=Z, dictionary=self.dictionary)
        recs = eng.dendrogram()
        return MashPlan(layer_idx=layer_idx,
                        pairs=[(r["survivor"], r["removed"]) for r in recs],
                        recs=recs, idx_map=idx_map, n_mergeable=len(idx_map),
                        mass=np.asarray(sub.mass, dtype=float).copy())

    def emit_at(self, model: PrunableModel, layer_idx: int, plan: MashPlan,
                n_merges: int, ctx: PruneContext) -> PruneDecision:
        """Realize a cut of `plan` in the CURRENT model's coordinates.

        The partition comes from the cached pass; units, moments and repair are
        recomputed here, because an earlier layer's pruning has changed this
        layer's input space even though its output units are untouched.
        """
        (units, ok, idx_map, frozen, sub, Z, mu, Sigma,
         x0, R, rad, measure) = self._prepare(model, layer_idx, ctx,
                                              for_scoring=False)
        if len(idx_map) != plan.n_mergeable or not np.array_equal(idx_map, plan.idx_map):
            # A unit's norm collapsed to zero since planning, so the index
            # mapping no longer lines up. Re-plan rather than mis-apply it.
            logging.warning(
                f"  layer {layer_idx}: mergeable set changed since planning "
                "(a unit lost its hyperplane); rebuilding the pass")
            plan = self.plan(model, layer_idx, ctx)
        k = max(0, min(int(n_merges), plan.max_merges))
        if k <= 0:
            return PruneDecision(remove=[])
        clusters = partition_at(plan.n_mergeable, plan.pairs, k)
        recs = plan.recs[:k]
        return self._emit(model, layer_idx, units, ok, idx_map, frozen,
                          clusters, Z, mu, Sigma, recs=recs,
                          cert=recs[-1].get("certificate") if recs else None)


@register_pruning_method("mash")
class MASH(_MashBase):
    """Width-driven MASH: merge until `n_remove` units are gone.

    Width is given either absolutely (`n_remove`) or as a fraction of the
    layer (`fraction`), which is what joint equal-fraction pruning of every
    layer needs -- one absolute count cannot express it, since the layers have
    different widths and the count would clamp the narrow ones.

    params: n_remove OR fraction, score, dictionary, repair, n_calib,
            gauge_correct, bias_fix, radius. Defaults are the configuration
            that performs best on fully connected layers: delta_f selection,
            merged dictionary, global kernel repair.
    """

    def __init__(self, n_remove: int = 1, fraction: float | None = None, **kw):
        super().__init__(**kw)
        if fraction is not None and not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        self.n_remove = int(n_remove)
        self.fraction = None if fraction is None else float(fraction)

    def select(self, model: PrunableModel, layer_idx: int,
               ctx: PruneContext) -> PruneDecision:
        plan = self.plan(model, layer_idx, ctx)
        want = (self.n_remove if self.fraction is None
                else plan.merges_for(self.fraction))
        return self.emit_at(model, layer_idx, plan, want, ctx)


@register_pruning_method("mash_certified")
class MASHCertified(_MashBase):
    """Tolerance-driven MASH: the domain-only tier.

    Merges while the certificate stays inside `tol` times the layer's scale,
    so the retained width is chosen by the guarantee rather than set in
    advance. The bound is
        sum_C sum_{i in C} a_i (R ||u_i - uhat_C|| + |gamma_i - gammahat_C|)
    which upper-bounds sup_x ||F_T(x) - F_0(x)|| over the calibration box.
    (uhat_C, gammahat_C) is the code the DICTIONARY emits -- the mass-weighted
    centroid under `merge`, the surviving original unit under `medoid` -- and
    the reported certificate is the ACCEPTED cut's, not the final state of a
    pass that deliberately continues past it. Both of those were wrong before
    the gauge audit, in the direction of a bound that does not hold and a
    number 39x too large respectively. The derivation is the sum rule's, so it
    covers `repair='sum'` only.

    `scale='mass'` normalizes by the layer's total mass sum_i a_i (no
    activations needed, so the whole rule stays domain-only); `scale='output'`
    normalizes by the mean response norm on the calibration inputs, which is
    tighter but reads activations. NOTE that a tolerance is NOT portable across
    datasets -- the certificate's looseness factor is itself data-dependent, so
    calibrate `tol` once per (architecture, dataset).

    params: tol, scale, plus the shared ones. Selection and repair default to
    the certified configuration (cylinder score, sum rule).
    """

    score = "cylinder"
    dictionary = None          # auto: merge where safe, medoid under BatchNorm
    repair = "sum"

    def __init__(self, tol: float = 0.05, scale: str = "mass",
                 max_fraction: float = 1.0, **kw):
        super().__init__(**kw)
        if scale not in ("mass", "output"):
            raise ValueError("scale must be 'mass' or 'output'")
        self.tol = float(tol)
        self.scale = scale
        self.max_fraction = float(max_fraction)

    def select(self, model: PrunableModel, layer_idx: int,
               ctx: PruneContext) -> PruneDecision:
        (units, ok, idx_map, frozen, sub, Z, mu, Sigma,
         x0, R, rad, measure) = self._prepare(model, layer_idx, ctx)
        H = len(idx_map)
        if H <= 1:
            return PruneDecision(remove=[])
        eng = MashEngine(sub, score=self.score, x0=x0, radius=rad, mu=mu,
                         Sigma=Sigma, gauge_correct=self.gauge_correct,
                         measure=measure, Z=Z, dictionary=self.dictionary)
        if self.scale == "mass":
            denom = eng.mass_total()
        else:
            Phi0 = np.maximum(Z @ sub.u.T - sub.rho[None, :], 0.0)
            denom = float(np.linalg.norm(Phi0 @ sub.V, axis=1).mean())
        budget = self.tol * max(denom, 1e-30)

        cap = int(self.max_fraction * (H - 1))
        recs = eng.dendrogram(max_steps=max(cap, 0))
        # Last cut whose certificate is still inside the budget (first
        # crossing). The pass deliberately runs PAST it -- one pass serves every
        # tolerance -- so everything reported here has to be sliced back to the
        # accepted prefix: the engine's final state is the certificate of a cut
        # that was rejected (39x the accepted one on a two-layer MLP), and its
        # cum_cost and per-unit merge steps belong to merges that never
        # happened.
        T, cert_T = 0, 0.0
        for t, rec in enumerate(recs, start=1):
            if rec["certificate"] > budget:
                break
            T, cert_T = t, float(rec["certificate"])
        pairs = [(r["survivor"], r["removed"]) for r in recs[:T]]
        clusters = partition_at(H, pairs, len(pairs))
        return self._emit(model, layer_idx, units, ok, idx_map, frozen,
                          clusters, Z, mu, Sigma, recs=recs[:T], cert=cert_T)


# ── self-tests ───────────────────────────────────────────────────────────────

def _selftest() -> None:  # pragma: no cover
    rng = np.random.default_rng(0)

    # 1. arc-cosine identity and the c = +-1 branches
    for th in [0.0, 0.3, 1.0, np.pi / 2, 2.5, np.pi]:
        want = (np.sin(th) + (np.pi - th) * np.cos(th)) / (2 * np.pi)
        assert abs(float(relu_cross(0.0, 0.0, np.cos(th))) - want) < 1e-12, th
    for a in [-2.0, -0.3, 0.0, 0.7, 3.0]:
        assert abs(float(relu_cross(a, a, 1.0)) - float(relu_self(a))) < 1e-12

    # 2. cross kernel vs Monte Carlo
    n = 1_000_000
    x, e = rng.standard_normal(n), rng.standard_normal(n)
    for a, b, c in [(-0.5, 0.3, 0.6), (1.2, -0.8, -0.4), (2.5, 2.0, 0.3)]:
        y = c * x + np.sqrt(1 - c * c) * e
        prod = np.clip(x - a, 0, None) * np.clip(y - b, 0, None)
        mc, cf = float(prod.mean()), float(relu_cross(a, b, c))
        assert abs(cf - mc) < 5e-3 * max(mc, 1e-3) + 4.0 * np.sqrt(prod.var() / n), \
            (a, b, c, cf, mc)

    d, H, m = 10, 12, 4
    W = rng.normal(size=(H, d)); b = rng.normal(size=H); C = rng.normal(size=(H, m))
    for j, s in [(4, 2.5), (5, 0.7)]:               # 3,4,5 share a hyperplane
        W[j] = s * W[3]; b[j] = s * b[3]
    alpha = np.linalg.norm(W, axis=1)
    units = Units(W / alpha[:, None], -b / alpha, alpha, C)
    ok = np.ones(H, dtype=bool)
    X = rng.normal(size=(512, d))
    mu, Sigma = X.mean(axis=0), np.cov(X.T)
    x0, R = np.zeros(d), float(np.sqrt(d))

    def layer_out(rows, biases, cols, Z):
        return np.maximum(Z @ rows.T + biases, 0.0) @ cols

    ref = layer_out(units.u, -units.rho, units.V, X)

    # 3. mass is gauge invariant
    t = 3.7
    scaled = Units(units.u, units.rho, units.alpha * t, units.C / t)
    assert np.allclose(units.mass, scaled.mass), "mass must be gauge invariant"

    for score in SCORES:
        eng = MashEngine(units, score=score, x0=x0, radius=R, mu=mu, Sigma=Sigma)
        # scores carry different units (exact_damage is a squared output
        # energy), so "free" is judged relative to a typical pair cost
        finite = eng._cost[np.isfinite(eng._cost)]
        scale = float(np.median(finite)) if len(finite) else 1.0
        r1, r2 = eng.step(), eng.step()
        assert max(r1["cost"], r2["cost"]) < 1e-9 * scale, \
            (f"{score}: duplicate hyperplanes must be free, got "
             f"{r1['cost']:.2e}/{r2['cost']:.2e} vs scale {scale:.2e}")
        merged = set(eng.members[r2["survivor"]])
        assert merged == {3, 4, 5}, f"{score}: expected 3,4,5 merged, got {merged}"

        # 4. exactness: merging identical hyperplanes preserves the function
        clusters = [c for c in partition_at(H, [(r["survivor"], r["removed"])
                                                for r in (r1, r2)], 2)]
        rows, biases, cols, keep, _ = realize(units, ok, clusters, repair="sum")
        err = np.abs(layer_out(rows, biases, cols, X) - ref).max()
        assert err < 1e-9, f"{score}: duplicate merge must be exact, got {err:.2e}"
        assert len(rows) == H - 2

        # 5. the additive triple is order-free
        g_direct = (units.mass[[3, 4, 5], None] * units.u[[3, 4, 5]]).sum(axis=0)
        k = r2["survivor"]
        assert np.allclose(eng.g[k], g_direct, atol=1e-12), f"{score}: not associative"

    # 6. mass and the raw covector sum are conserved over a full sweep
    eng = MashEngine(units, score="delta_f", x0=x0, radius=R, mu=mu, Sigma=Sigma)
    m0 = (eng.g[eng.active].sum(axis=0).copy(), eng.mass_total())
    eng.dendrogram()
    assert abs(eng.mass_total() - m0[1]) < 1e-10 * m0[1], "mass must be conserved"
    drift = np.abs(eng.g[eng.active].sum(axis=0) - m0[0]).max() / np.abs(m0[0]).max()
    assert drift < 1e-12, f"covector sum must be conserved, drift {drift:.2e}"

    # 7. the certificate really bounds the realized sup error on the box
    box = x0 + R / np.sqrt(d) * rng.uniform(-1, 1, size=(4000, d))
    ref_box = layer_out(units.u, -units.rho, units.V, box)
    eng = MashEngine(units, score="cylinder", x0=x0, radius=R)
    for _ in range(6):
        eng.step()
    pairs = [(i, j) for i, j in zip(*[[], []])]  # rebuilt below
    eng2 = MashEngine(units, score="cylinder", x0=x0, radius=R)
    recs = eng2.dendrogram(max_steps=6)
    cl = partition_at(H, [(r["survivor"], r["removed"]) for r in recs], 6)
    rows, biases, cols, keep, _ = realize(units, ok, cl, repair="sum")
    realized = np.abs(layer_out(rows, biases, cols, box) - ref_box).sum(axis=1).max()
    cert = recs[-1]["certificate"]
    assert cert >= realized - 1e-9, \
        f"certificate {cert:.4e} must bound realized {realized:.4e}"

    # 7b. The certificate must be written against the atom the DICTIONARY
    # emits. Under 'medoid' the emitted representative is an original unit, and
    # bounding the distance to the centroid instead says nothing about it: on a
    # three-unit fan whose heaviest member sits at the edge, the centroid bound
    # is 1.5x SMALLER than the error the medoid emission actually makes.
    fan = np.array([-0.15, 0.0, 0.15])
    u_fan = np.stack([np.cos(fan), np.sin(fan)], axis=1)
    fan_units = Units(u_fan, np.array([-1.5, -1.0, -0.5]), np.ones(3),
                      np.tile([[1.0, 0.3]], (3, 1)))
    fan_ok = np.ones(3, dtype=bool)
    S2 = rng.normal(size=(60000, 2))
    S2 = 2.0 * S2 / np.linalg.norm(S2, axis=1, keepdims=True)   # the R-sphere
    ref_fan = layer_out(u_fan, -fan_units.rho, fan_units.V, S2)
    for dic in ("merge", "medoid"):
        e_fan = MashEngine(fan_units, score="cylinder", x0=np.zeros(2),
                           radius=2.0, dictionary=dic)
        e_fan.dendrogram()
        rows, biases, cols, keep, _ = realize(fan_units, fan_ok, [[0, 1, 2]],
                                              dictionary=dic, repair="sum")
        real = float(np.linalg.norm(
            layer_out(rows, biases, cols, S2) - ref_fan, axis=1).max())
        assert e_fan.certificate() >= real - 1e-9, \
            (f"dictionary={dic}: certificate {e_fan.certificate():.4f} must "
             f"bound realized {real:.4f}")
        if dic == "medoid":                      # the regression, pinned
            centroid_bound = MashEngine(fan_units, score="cylinder",
                                        x0=np.zeros(2), radius=2.0)
            centroid_bound.dendrogram()
            assert centroid_bound.certificate() < real, \
                "the centroid bound is supposed to be the unsound one here"

    # ... and the engine has to agree with `realize` about WHICH member
    # survives, tie-breaks included, or it certifies a unit nobody emitted
    e_med = MashEngine(units, score="cylinder", x0=x0, radius=R,
                       dictionary="medoid")
    recs = e_med.dendrogram(max_steps=6)
    cl = partition_at(H, [(r["survivor"], r["removed"]) for r in recs], 6)
    rows, biases, cols, keep, _ = realize(units, ok, cl, dictionary="medoid",
                                          repair="sum")
    by_set = {frozenset(c): s for c, s in zip(cl, keep)}
    for kk in np.flatnonzero(e_med.active):
        assert by_set[frozenset(e_med.members[kk])] == e_med._medoid(int(kk)), \
            f"engine medoid disagrees with realize for cluster {kk}"
    realized = np.abs(layer_out(rows, biases, cols, box) - ref_box).sum(axis=1).max()
    assert recs[-1]["certificate"] >= realized - 1e-9, \
        (f"medoid certificate {recs[-1]['certificate']:.4e} must bound "
         f"realized {realized:.4e}")

    # 7c. g_C = 0: the members cancel, so there is no hyperplane -- but the
    # centroid pre-activation is a nonzero CONSTANT, and the gauge-corrected
    # emission has to realize it. Emitting zero dropped a contribution the
    # certificate had already paid for.
    anti = Units(np.array([[1.0, 0.0], [-1.0, 0.0]]), np.array([-1.0, -1.0]),
                 np.ones(2), np.array([[1.0], [1.0]]))
    e_anti = MashEngine(anti, score="cylinder", x0=np.zeros(2), radius=1.0)
    e_anti.step()
    uh, oh = e_anti.emitted_code(np.array([0]))
    rows, biases, cols, keep, _ = realize(anti, np.ones(2, bool), [[0, 1]],
                                          repair="sum")
    assert np.allclose(rows[0], 0.0) and abs(biases[0] - oh[0]) < 1e-12, \
        f"g=0 must emit the constant sigma({oh[0]:.3f}), got bias {biases[0]}"
    S1 = np.linspace(-1.0, 1.0, 2001)[:, None] * np.array([[1.0, 0.0]])
    real = float(np.abs(layer_out(rows, biases, cols, S1)
                        - layer_out(anti.u, -anti.rho, anti.V, S1)).max())
    assert e_anti.certificate() >= real - 1e-9, \
        f"g=0 certificate {e_anti.certificate():.4f} must bound {real:.4f}"

    # 8. repair is ordered: global <= projection <= sum
    recs = MashEngine(units, score="delta_f", x0=x0, radius=R, mu=mu,
                      Sigma=Sigma).dendrogram(max_steps=5)
    cl = partition_at(H, [(r["survivor"], r["removed"]) for r in recs], 5)
    errs = {}
    for rep in ("sum", "projection", "kernel", "empirical"):
        rows, biases, cols, keep, _ = realize(
            units, ok, cl, repair=rep, mu=mu, Sigma=Sigma, Z=X)
        errs[rep] = float(np.linalg.norm(layer_out(rows, biases, cols, X) - ref))
    assert errs["kernel"] <= errs["projection"] + 1e-8, errs
    assert errs["projection"] <= errs["sum"] + 1e-8, errs
    assert errs["empirical"] <= errs["sum"] + 1e-8, errs

    # 8b. exact_damage must equal the damage of the merge it actually scores,
    # under BOTH gauges: it is the one score that claims to be exact, so a
    # response written in a gain the emission does not use is simply a
    # different quantity (it mispriced pairs by up to 28% under the default).
    energy = float((ref * ref).sum(axis=1).mean())
    for gc in (True, False):
        engd = MashEngine(units, score="exact_damage", measure="empirical",
                          Z=X, x0=x0, radius=R, gauge_correct=gc)
        for i, j in [(0, 1), (2, 6), (7, 8), (9, 11), (3, 4)]:
            scored = float(engd._pair_costs(i, np.array([j]))[0])
            cl = [[t] for t in range(H) if t not in (i, j)] + [[i, j]]
            rows, biases, cols, keep, _ = realize(units, ok, cl, repair="sum",
                                                  gauge_correct=gc)
            err = layer_out(rows, biases, cols, X) - ref
            truth = float((err * err).sum(axis=1).mean())
            assert abs(scored - truth) < 1e-9 * energy, \
                (f"exact_damage {scored:.6e} != realized damage {truth:.6e} "
                 f"for pair {(i, j)}, gauge_correct={gc}")

    # ... and the Gaussian branch against a large sample of its own measure
    XG = rng.multivariate_normal(mu, Sigma, size=200_000)
    refG = layer_out(units.u, -units.rho, units.V, XG)
    engg = MashEngine(units, score="exact_damage", x0=x0, radius=R, mu=mu,
                      Sigma=Sigma)
    for i, j in [(0, 1), (7, 8)]:
        scored = float(engg._pair_costs(i, np.array([j]))[0])
        cl = [[t] for t in range(H) if t not in (i, j)] + [[i, j]]
        rows, biases, cols, keep, _ = realize(units, ok, cl, repair="sum")
        pv = ((layer_out(rows, biases, cols, XG) - refG) ** 2).sum(axis=1)
        mc = float(pv.mean())
        tol_mc = 5e-3 * mc + 4.0 * float(np.sqrt(pv.var() / len(pv)))
        assert abs(scored - mc) < tol_mc, \
            (f"gaussian exact_damage {scored:.6e} vs sample {mc:.6e} "
             f"(tol {tol_mc:.2e}) for pair {(i, j)}")

    # 9. registry round-trip and parameter validation
    from src.pruning.registry import build_pruning_method
    for kind in ("mash", "mash_certified"):
        build_pruning_method(kind)
    for bad in ({"score": "nope"}, {"dictionary": "nope"}, {"repair": "nope"}):
        try:
            build_pruning_method("mash", **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")

    # 9b. The Lance--Williams update must agree EXACTLY with recomputing the
    # Ward increment from the centroids -- it is the reason the cylinder score
    # costs O(K) per step instead of O(K d), so a silent drift here would be a
    # silently different method.
    eng_lw = MashEngine(units, score="cylinder", x0=x0, radius=R)
    worst = 0.0
    for _ in range(H - 2):
        eng_lw.step()
        act = np.flatnonzero(eng_lw.active)
        for kk in act:
            oth = act[act != kk]
            if not len(oth):
                continue
            direct = eng_lw._pair_costs(int(kk), oth)
            rel = np.abs(eng_lw._cost[kk, oth] - direct) / np.maximum(np.abs(direct), 1e-12)
            worst = max(worst, float(rel.max()))
    assert worst < 1e-9, f"Lance-Williams disagrees with direct Ward, {worst:.2e}"

    # 9c. The row-minimum cache must reproduce the brute-force argmin's merge
    # sequence EXACTLY, ties included -- a different tie-break would be a
    # different method rather than a faster one. Duplicates and a dead unit are
    # planted above, so the degenerate cases are in scope.
    for score_ in SCORES:
        kw_ = dict(score=score_, x0=x0, radius=R, mu=mu, Sigma=Sigma)
        fast_ = MashEngine(units, row_cache=True, **kw_)
        slow_ = MashEngine(units, row_cache=False, **kw_)
        seq_f = [(r["survivor"], r["removed"], r["cost"]) for r in fast_.dendrogram()]
        seq_s = [(r["survivor"], r["removed"], r["cost"]) for r in slow_.dendrogram()]
        assert [x[:2] for x in seq_f] == [x[:2] for x in seq_s], \
            f"{score_}: row-cache merge sequence differs from brute force"
        assert max(abs(a[2] - b_[2]) for a, b_ in zip(seq_f, seq_s)) == 0.0, \
            f"{score_}: row-cache costs differ from brute force"

    # 9d. A layer can legitimately end up with NO mergeable units: alpha is
    # taken over the layer's input space, so pruning an earlier layer can
    # collapse a row's norm to zero and it stops being a hyperplane. Building an
    # engine on that used to reach np.argmin of an empty matrix; it killed a real
    # benchmark cell (mnist_lenet, cylinder + l2 radius + empirical repair, at
    # an aggressive width) rather than emitting an empty decision.
    for H_ in (0, 1):
        empty = Units(np.zeros((H_, 4)), np.zeros(H_), np.zeros(H_),
                      np.zeros((H_, 3)))
        e_ = MashEngine(empty, score="cylinder", x0=np.zeros(4), radius=1.0)
        assert e_.dendrogram() == [], f"H={H_} should yield no merges"
        assert e_.certificate() == 0.0

    # 9e. The certified tier runs its pass PAST the accepted cut, so every
    # number it reports has to be sliced back to that cut. It used to report the
    # final engine state instead: 39x the accepted certificate on a two-layer
    # MLP, i.e. a tolerance that looked violated by the method enforcing it.
    from src.models.mlp import MLP
    torch.manual_seed(0)
    mnet = MLP(hidden_sizes=[24], input_dim=16, output_dim=3).eval()
    Xm = torch.randn(48, 16)
    mctx = PruneContext(train_inputs=Xm, bundle=None, device=torch.device("cpu"))
    cmeth = build_pruning_method("mash_certified", tol=0.5, n_calib=48)
    dec = cmeth.select(mnet, 0, mctx)
    sc = dec.diagnostics["_scalars"]
    prep = cmeth._prepare(mnet, 0, mctx)
    e_ref = MashEngine(prep[4], score="cylinder", x0=prep[8], radius=prep[10],
                       dictionary=cmeth.dictionary)
    rc = e_ref.dendrogram()
    budget = 0.5 * e_ref.mass_total()
    T, cert_T = 0, 0.0
    for t, r_ in enumerate(rc, start=1):
        if r_["certificate"] > budget:
            break
        T, cert_T = t, float(r_["certificate"])
    assert 0 < T < len(rc), \
        f"the test needs a cut strictly inside the pass, got T={T}/{len(rc)}"
    assert len(dec.remove) == T, (len(dec.remove), T)
    assert abs(sc["certificate"] - cert_T) < 1e-9 * max(cert_T, 1.0), \
        (f"reported certificate {sc['certificate']:.4f} is not the accepted "
         f"cut's {cert_T:.4f} (final state would be {e_ref.certificate():.4f})")
    assert sc["certificate"] <= budget + 1e-9, \
        f"reported certificate {sc['certificate']:.4f} exceeds budget {budget:.4f}"
    cum_T = sum(r_["cost"] for r_ in rc[:T])
    assert abs(sc["cum_cost"] - cum_T) < 1e-9 * max(cum_T, 1.0), \
        f"reported cum_cost {sc['cum_cost']:.4g} is not the cut's {cum_T:.4g}"

    # 10. Conv + BatchNorm end to end. The merged dictionary is refused here on
    # purpose (a folded hyperplane cannot be written back through the BN), the
    # medoid dictionary must work, and zero removals must be bit-exact.
    from torch.utils.data import TensorDataset

    from src.config import PruningConfig
    from src.models.cnn import CNN
    from src.pruning.surgery import prune_model

    torch.manual_seed(0)
    cnet = CNN(hidden_sizes=[8, 12], input_channels=1, output_dim=3).eval()
    Xc = torch.randn(6, 1, 12, 12)

    class _CB:
        train_ds = TensorDataset(Xc, torch.zeros(len(Xc), dtype=torch.long))

    cb = _CB()
    with torch.no_grad():
        cref = cnet(Xc).double()

    def _run(**params):
        class _M:
            kind = "mash"
        _M.params = params
        out, _ = prune_model(cnet, PruningConfig(methods=[_M()]), cb,
                             torch.device("cpu"))
        return out

    zero = _run(n_remove=0, dictionary="medoid")
    with torch.no_grad():
        e0 = float((zero(Xc).double() - cref).abs().max())
    assert e0 == 0.0, f"conv zero-removal must be bit-exact, got {e0:.2e}"

    widths0 = [cnet.prunable_layer(i).weight.shape[0]
               for i in range(cnet.n_prunable_layers())]
    got = _run(fraction=0.25, dictionary="medoid", repair="empirical")
    widths1 = [got.prunable_layer(i).weight.shape[0]
               for i in range(got.n_prunable_layers())]
    assert widths1 == [int(round(0.75 * w)) for w in widths0], (widths0, widths1)
    with torch.no_grad():
        got(Xc)                                   # must still forward
    # BN slots must stay in the mode they were in
    assert not got.prunable_bn(0).training, "pruned BN left in train mode"

    # patch extraction: one row per im2col patch, d = C_in*kH*kW
    Zc = _layer_inputs(cnet, 1, Xc)
    lay1 = cnet.prunable_layer(1)
    assert Zc.shape[1] == lay1.in_channels * int(np.prod(lay1.kernel_size)), Zc.shape
    # the auto default must refuse to merge under BN, and say why
    try:
        _run(fraction=0.25, dictionary="merge")
    except NotImplementedError as exc:
        assert "medoid" in str(exc)
    else:
        raise AssertionError("merge under BatchNorm must be refused")

    print("mash.py self-tests passed:")
    print("  arc-cosine identity, c=+-1 branches, cross kernel vs Monte Carlo")
    print("  mass gauge invariance; duplicate hyperplanes free in all 3 scores")
    print("  duplicate merge exact to 1e-9; additive triples order-free")
    print("  mass and covector sum conserved over a full sweep")
    print("  certificate bounds the realized sup error on the box")
    print("  certificate follows the dictionary (merge AND medoid emissions)")
    print("  g_C = 0 emits its centroid constant, still inside the bound")
    print("  repair ordering global <= projection <= sum (and empirical <= sum)")
    print("  exact_damage == realized merge damage, both gauges (and vs MC)")
    print("  certified tier reports the ACCEPTED cut's certificate/cum_cost")
    print("  registry round-trip for mash / mash_certified + param validation")
    print(f"  Lance-Williams update == direct Ward increment ({worst:.1e})")
    print("  row-minimum cache == brute-force argmin, all 3 scores (exact)")
    print("  a layer with 0 or 1 mergeable units plans and emits empty, not raises")
    print("  conv+BN: patch extraction, medoid path, bit-exact zero removal,")
    print("    BN mode preserved, merge-under-BN refused with a pointer")


if __name__ == "__main__":
    _selftest()
