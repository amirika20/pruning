"""HOPE -- Mobahi & Bartlett, "Hilbert Operator for Progressive Encoding
(HOPE): A Mathematical Framework for Deconstructing Learned Representations in
Deep Networks", Google DeepMind / UC Berkeley, arXiv:2607.21366 (23 Jul 2026).

No official implementation exists (code listed N/A), so this is a port from the
paper's equations. Equation numbers below refer to the arXiv v1 PDF.

WHAT HOPE IS
------------
Each neuron is lifted to a rank-1 Hilbert-Schmidt operator f_i = g_i (x)
w_out,i in H = L2(X, P_X) (x) R^c, where g_i(x) = Psi((w_in,i^eff)^T x + b_i)
is the unit's continuous input landscape and w_out,i its outgoing column
(Sec. 5). Pruning and merging are then one operation -- low-rank projection in
H -- scored by a single distortion cost J.

The inner product factorizes (eq. 74) into a scalar kernel times the outgoing
Gram,

    <f_i, f_j>_H = K(i,j) * <w_out,i, w_out,j>,   K(i,j) = E[Psi(y_i)Psi(y_j)],

so the "capacity" of a neuron is  ||f_i||_H = ||w_out,i||_2 sqrt(K(i,i))
(eq. 75). Positive homogeneity of Psi (PH-1) makes this invariant to the
w_in <-> w_out rescaling gauge, which is HOPE's answer to scale symmetry.

THE DATA-FREE SURROGATE (Sec. 4, App. E.1)
------------------------------------------
P_X is replaced by the maximum-entropy Gaussian consistent with BatchNorm's
stored statistics. The payoff is eqs. 76-78: after folding BN into effective
parameters (eq. 1),

    w_in^eff = gamma/sqrt(sigma^2+eps) * w_raw,   b = beta - gamma*mu/sqrt(sigma^2+eps),

the marginal pre-activation is EXACTLY

    y_i ~ N(beta_i, gamma_i^2)                                        (eq. 78)

i.e. determined by the BN affine parameters alone -- no data, no forward pass.
For unnormalized networks the paper's App. E.1 box prescribes one calibration
pass to measure the marginal pre-activation statistics, setting gamma_i =
sigma_i and beta_i = mu_i + b_raw,i; we implement that fallback, so the method
runs on plain MLPs too (at the cost of HOPE's data-free claim, as the paper
concedes in its footnote 1).

Kernels (App. E.2, E.3), with c_i = beta_i/|gamma_i|:

    K(i,i) = (gamma_i^2+beta_i^2) Phi(c_i) + beta_i |gamma_i| phi(c_i)  (eq. 79)

    rho_eff = cos(w_in,i^eff, w_in,j^eff)
    kappa   = rho_eff/(1-rho_eff^2) * (|gamma_i|/||w_in,i^eff||) * (|gamma_j|/||w_in,j^eff||)
    rho_hat = 2 kappa / (1 + sqrt(1+4 kappa^2))                        (eqs. 80-81)

rho_hat is the "warped" correlation of a LOCAL pairwise max-entropy surrogate:
HOPE never forms or inverts the ambient covariance, it re-derives a 2x2 joint
per pair. The cross-kernel is then either the exact truncated-bivariate form
(eq. 83) or the zero-bias arc-cosine approximation (eq. 85), which is what the
paper actually uses at scale ("assuming bias shifts are negligible").

COSTS (Sec. 6.3, eq. 6) with N active units and layer capacity E_a = sum_k ||f_k||:

    J_prune(i)   = N ||f_i|| / (E_a - ||f_i||)
    J_merge(i,j) = N sqrt(||f_i-f_p||^2 + ||f_j-f_p||^2)
                   / (E_a - ||f_i|| - ||f_j|| + ||f_p||)

PARENT NEURON (Sec. 7, eqs. 12-15). With augmented weights wt_in = [w_in^eff; b]
and A = w_out^i (wt_in^i)^T + w_out^j (wt_in^j)^T (rank 2), the optimal parent
direction is the principal right-singular vector of A, its sign fixed by the
exact objective, its output direction v* the kernel-weighted combination of the
children's columns, and its scale s* the closed-form 1-D minimizer.

SCOPE OF THIS PORT
------------------
The repo's PruningMethod interface selects units within ONE prunable layer, so
we implement HOPE's granular per-layer operations: prune and merge, ranked by
the distortion rate J/dP. Within a single layer every granular operation
releases the same parameter footprint dP (one unit's incoming row plus outgoing
column), so the distortion-rate ranking of eq. 23 reduces exactly to ranking J
-- dP only matters when operations compete ACROSS layers.

Deliberately NOT ported: block eviction (Sec. 8), which needs residual-block
scope rather than a layer index; the cross-layer knapsack (Sec. 9), which needs
a global action set; and DEFT (Sec. 11.2), a transfer-learning method built on
top. BN write-back for the synthesized parent (Sec. 7.2.2, App. D) is not
implemented either -- the PrunableModel protocol cannot address BN parameters
-- so merging is supported on Linear layers whose units carry no BN. Pruning
works in both cases.

Run `python -c "from src.pruning.methods.hope import _selftest; _selftest()"`
for the numerical self-tests, which
check eq. 79 and eq. 83 against Monte Carlo and against an independent
derivation, plus gauge invariance, the paper's three cross-kernel axioms, and the parent
construction's behavior on duplicated neurons (which pins down HOPE's
clustering semantics -- see the note at self-test 6).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.special import ndtr, owens_t

from src.models.registry import PrunableModel
from src.pruning.registry import (
    PruneContext, PruneDecision, PruningMethod, register_pruning_method)

_SQRT2PI = math.sqrt(2.0 * math.pi)
_EPS_RHO = 1e-9      # |rho| beyond 1-_EPS_RHO uses the degenerate branch
_TINY = 1e-30


# ── standard normal helpers ──────────────────────────────────────────────────

def _phi(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(x)) / _SQRT2PI


def _Phi(x: np.ndarray) -> np.ndarray:
    return ndtr(x)


def _phi2(a: np.ndarray, b: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Standard bivariate normal PDF at (a, b) with correlation rho."""
    s = np.clip(1.0 - rho * rho, _TINY, None)
    q = (a * a - 2.0 * rho * a * b + b * b) / s
    return np.exp(-0.5 * q) / (2.0 * np.pi * np.sqrt(s))


def _Phi2(a: np.ndarray, b: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """P(X <= a, Y <= b) for a standard bivariate normal, via Owen's T.

    Owen (1956): P(X>h, Y>k) = (Phi(-h)+Phi(-k))/2 - T(h, r1) - T(k, r2) - beta,
    and Phi2(a,b;rho) = P(X > -a, Y > -b) by symmetry."""
    a, b, rho = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (a, b, rho)))
    h, k = -a, -b
    rc = np.clip(rho, -1.0 + _EPS_RHO, 1.0 - _EPS_RHO)
    s = np.sqrt(1.0 - rc * rc)
    hh = np.where(np.abs(h) < 1e-12, 1e-12, h)
    kk = np.where(np.abs(k) < 1e-12, 1e-12, k)
    t1 = owens_t(hh, (kk - rc * hh) / (hh * s))
    t2 = owens_t(kk, (hh - rc * kk) / (kk * s))
    beta = np.where(hh * kk < 0.0, 0.5, 0.0)
    gen = 0.5 * (ndtr(-hh) + ndtr(-kk)) - t1 - t2 - beta
    pos = ndtr(np.minimum(a, b))                              # rho -> +1
    neg = np.clip(ndtr(a) + ndtr(b) - 1.0, 0.0, None)         # rho -> -1
    out = np.where(rho >= 1.0 - _EPS_RHO, pos,
                   np.where(rho <= -1.0 + _EPS_RHO, neg, gen))
    return np.clip(out, 0.0, 1.0)


# ── HOPE kernels ─────────────────────────────────────────────────────────────

def self_kernel(beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """K(i,i) = E[ReLU(y)^2] for y ~ N(beta, gamma^2).   Paper eq. 79."""
    beta = np.asarray(beta, dtype=np.float64)
    g = np.abs(np.asarray(gamma, dtype=np.float64))
    c = np.where(g > _TINY, beta / np.maximum(g, _TINY), 0.0)
    out = (g * g + beta * beta) * _Phi(c) + beta * g * _phi(c)
    # gamma == 0: y is the constant beta, so E[ReLU(y)^2] = max(beta,0)^2
    return np.where(g > _TINY, np.clip(out, 0.0, None),
                    np.clip(beta, 0.0, None) ** 2)


def warped_correlation(rho_eff: np.ndarray, gamma_i: np.ndarray,
                       norm_i: np.ndarray, gamma_j: np.ndarray,
                       norm_j: np.ndarray) -> np.ndarray:
    """rho_hat of the local pairwise max-entropy surrogate. Paper eqs. 80-81.

    rho_eff is the cosine between EFFECTIVE input weights; norm_* are their
    2-norms. Note |gamma|/||w_in^eff|| = sqrt(sigma^2+eps)/||w_raw||, the ratio
    the paper's derivation uses."""
    rho_eff = np.clip(np.asarray(rho_eff, dtype=np.float64),
                      -1.0 + _EPS_RHO, 1.0 - _EPS_RHO)
    ri = np.abs(gamma_i) / np.maximum(norm_i, _TINY)
    rj = np.abs(gamma_j) / np.maximum(norm_j, _TINY)
    kappa = (rho_eff / (1.0 - rho_eff * rho_eff)) * ri * rj
    return 2.0 * kappa / (1.0 + np.sqrt(1.0 + 4.0 * kappa * kappa))


def cross_kernel_exact(beta_i, gamma_i, beta_j, gamma_j, rho_hat) -> np.ndarray:
    """K(i,j) = E[ReLU(y_i) ReLU(y_j)] under eq. 82's joint.  Paper eq. 83.

    The paper's expression is the general |rho_hat| < 1 branch. It converges to
    the perfectly-(anti)correlated limits only as O(sqrt(1-rho^2)) -- the
    bivariate PDF carries a 1/sqrt(1-rho^2) -- so clipping rho near +-1 leaves
    ~1e-5 error and breaks the diagonal-consistency axiom numerically. We add
    the two exact degenerate branches: at rho=+1 the pair is y_j = a linear
    increasing map of y_i, at rho=-1 a decreasing one, and both integrals are
    elementary."""
    bi, gi, bj, gj, r = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64)
          for v in (beta_i, gamma_i, beta_j, gamma_j, rho_hat)))
    gi, gj = np.abs(gi), np.abs(gj)
    ci = np.where(gi > _TINY, bi / np.maximum(gi, _TINY), 0.0)
    cj = np.where(gj > _TINY, bj / np.maximum(gj, _TINY), 0.0)

    rc = np.clip(r, -1.0 + _EPS_RHO, 1.0 - _EPS_RHO)
    s = np.sqrt(np.clip(1.0 - rc * rc, _TINY, None))
    general = ((ci * cj + rc) * _Phi2(ci, cj, rc)
               + ci * _phi(cj) * _Phi((ci - rc * cj) / s)
               + cj * _phi(ci) * _Phi((cj - rc * ci) / s)
               + (1.0 - rc * rc) * _phi2(ci, cj, rc))

    # rho -> +1: E[(z+ci)_+ (z+cj)_+], one shared standard normal z
    lo = np.minimum(ci, cj)
    pos = _Phi(lo) * (1.0 + ci * cj) + _phi(lo) * np.maximum(ci, cj)

    # rho -> -1: z_j = -z_i, so integrate over the band -ci < z < cj
    empty = (ci + cj) <= 0.0
    a_, b_ = np.where(empty, 0.0, -ci), np.where(empty, 0.0, cj)
    I0 = _Phi(b_) - _Phi(a_)
    I1 = _phi(a_) - _phi(b_)
    I2 = I0 - (b_ * _phi(b_) - a_ * _phi(a_))
    neg = np.where(empty, 0.0, -I2 + (cj - ci) * I1 + ci * cj * I0)

    val = np.where(r >= 1.0 - _EPS_RHO, pos,
                   np.where(r <= -1.0 + _EPS_RHO, neg, general))
    out = gi * gj * val
    dead = (gi <= _TINY) | (gj <= _TINY)
    return np.where(dead, 0.0, np.clip(out, 0.0, None))


def cross_kernel_zerobias(k_ii: np.ndarray, k_jj: np.ndarray,
                          rho_hat: np.ndarray) -> np.ndarray:
    """Zero-bias arc-cosine approximation.  Paper eqs. 84-85. This is the form
    the paper uses at scale, to avoid a bivariate CDF per neuron pair."""
    r = np.clip(np.asarray(rho_hat, dtype=np.float64), -1.0, 1.0)
    interaction = (np.sqrt(np.clip(1.0 - r * r, 0.0, None))
                   + (np.pi - np.arccos(r)) * r) / np.pi
    return interaction * np.sqrt(np.clip(k_ii * k_jj, 0.0, None))


# ── layer state ──────────────────────────────────────────────────────────────

class HopeLayer:
    """The surrogate statistics and Hilbert-space quantities of one prunable
    layer: effective input weights, (beta, gamma) per unit, outgoing columns."""

    def __init__(self, W_eff: np.ndarray, b_eff: np.ndarray, beta: np.ndarray,
                 gamma: np.ndarray, W_out: np.ndarray, exact_kernel: bool):
        self.W_eff = W_eff                      # [H, n]  effective input rows
        self.b_eff = b_eff                      # [H]
        self.beta = beta                         # [H] pre-activation mean
        self.gamma = gamma                       # [H] pre-activation std
        self.W_out = W_out                       # [H, c] outgoing columns
        self.exact = exact_kernel
        self.n_in = W_eff.shape[1]
        self.k_self = self_kernel(beta, gamma)                     # [H]
        self.cap = np.linalg.norm(W_out, axis=1) * np.sqrt(self.k_self)  # eq 75
        self.w_norm = np.linalg.norm(W_eff, axis=1)
        self.aug = np.concatenate([W_eff, b_eff[:, None]], axis=1)  # [H, n+1]

    # -- kernels between existing units -----------------------------------

    def rho_hat_row(self, i: int, js: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.w_norm[i] * self.w_norm[js], _TINY)
        rho_eff = (self.W_eff[js] @ self.W_eff[i]) / denom
        return warped_correlation(rho_eff, self.gamma[i], self.w_norm[i],
                                  self.gamma[js], self.w_norm[js])

    def cross_row(self, i: int, js: np.ndarray):
        """(K(i,j), rho_hat_ij) for j in js, under the configured kernel."""
        r = self.rho_hat_row(i, js)
        if self.exact:
            k = cross_kernel_exact(self.beta[i], self.gamma[i],
                                   self.beta[js], self.gamma[js], r)
        else:
            k = cross_kernel_zerobias(self.k_self[i], self.k_self[js], r)
        return k, r

    # -- the parent neuron (Sec. 7) ---------------------------------------

    def parent_geometry(self, i: int, j: int) -> dict:
        """E_rem-INDEPENDENT part of the optimal rank-1 parent (eqs. 13-14).

        Everything happens inside the 2-D span of the children's augmented
        weights, so the ambient dimension never enters (paper Sec. 7.1.1).
        Within that span a candidate u = c1*aug_i + c2*aug_j has pre-activation
        y_u = c1*y_i + c2*y_j, hence
            beta_u  = c1 beta_i + c2 beta_j
            gamma_u = sqrt(c1^2 g_i^2 + c2^2 g_j^2 + 2 c1 c2 |g_i||g_j| rho_hat)
        which is exactly the paper's eq. 18 for the parent's recovered BN scale.
        """
        B = self.aug[[i, j]].T                        # [n+1, 2]
        Wo = self.W_out[[i, j]].T                     # [c, 2]
        G_B, G_W = B.T @ B, Wo.T @ Wo
        # maximize ||A u||^2 s.t. ||u||=1 with u = B c, A = Wo B^T
        ev, Q = np.linalg.eigh(G_B)
        ev = np.clip(ev, 0.0, None)
        half = (Q * np.sqrt(ev)) @ Q.T                              # G_B^{1/2}
        inv_half = (Q * (1.0 / np.sqrt(np.maximum(ev, _TINY)))) @ Q.T
        S = half @ G_W @ half
        _, V = np.linalg.eigh(0.5 * (S + S.T))
        c = inv_half @ V[:, -1]                       # top eigenvector
        rho_ij = float(self.rho_hat_row(i, np.array([j]))[0])

        best = None
        for sign in (1.0, -1.0):
            cc = sign * c
            u = B @ cc
            nu = float(np.linalg.norm(u))
            if nu <= _TINY:
                continue
            cc = cc / nu                              # so that ||B cc|| == 1
            u = B @ cc
            beta_u = float(cc[0] * self.beta[i] + cc[1] * self.beta[j])
            var_u = (cc[0] ** 2 * self.gamma[i] ** 2
                     + cc[1] ** 2 * self.gamma[j] ** 2
                     + 2.0 * cc[0] * cc[1] * abs(self.gamma[i])
                     * abs(self.gamma[j]) * rho_ij)
            gamma_u = math.sqrt(max(var_u, 0.0))
            if gamma_u <= _TINY:
                continue
            k_uu = float(self_kernel(beta_u, gamma_u))
            # K(u, child_k): covariance of y_u with y_k inside the 2x2 block
            cov = np.array([
                cc[0] * self.gamma[i] ** 2
                + cc[1] * abs(self.gamma[i]) * abs(self.gamma[j]) * rho_ij,
                cc[1] * self.gamma[j] ** 2
                + cc[0] * abs(self.gamma[i]) * abs(self.gamma[j]) * rho_ij])
            gk = np.abs(self.gamma[[i, j]])
            rho_uk = np.clip(cov / np.maximum(gamma_u * gk, _TINY), -1.0, 1.0)
            if self.exact:
                k_uk = cross_kernel_exact(beta_u, gamma_u,
                                          self.beta[[i, j]], self.gamma[[i, j]],
                                          rho_uk)
            else:
                k_uk = cross_kernel_zerobias(k_uu, self.k_self[[i, j]], rho_uk)
            num = k_uk @ self.W_out[[i, j]]           # sum_k K(u, wt_k) w_out_k
            score = float(np.linalg.norm(num)) / math.sqrt(max(k_uu, _TINY))
            if best is None or score > best["score"]:
                best = {"score": score, "u": u, "c": cc, "k_uu": k_uu,
                        "beta": beta_u, "gamma": gamma_u, "num": num,
                        "k_uk": k_uk}
        if best is None:
            return {}

        nv = float(np.linalg.norm(best["num"]))
        if nv <= _TINY:
            return {}
        v_star = best["num"] / nv                                    # eq 13
        # <psi*, f_i + f_j> with psi* = g_u/sqrt(K(u,u)) (x) v*
        b_ip = float((best["k_uk"] / math.sqrt(max(best["k_uu"], _TINY)))
                     @ (self.W_out[[i, j]] @ v_star))
        return {"u": best["u"], "beta": best["beta"], "gamma": best["gamma"],
                "k_uu": best["k_uu"], "v": v_star,
                "a": float(self.cap[i]) ** 2 + float(self.cap[j]) ** 2,
                "b": b_ip}

    @staticmethod
    def close_parent(geo: dict, E_rem: float) -> dict:
        """Finish a cached geometry at the current layer state.  Paper eq. 12.

        The direction u*, output v* and the constants a = ||f_i||^2+||f_j||^2,
        b = <psi*, f_i+f_j> are properties of the PAIR alone. Only the scale s*
        and hence the distortion depend on the layer's residual capacity E_rem,
        which shrinks every step. Splitting them is what makes the paper's O(1)
        cost evaluation (Sec. 10, footnote 8) possible: the expensive rank-2
        eigenproblem and kernel evaluations are cached, and each greedy scan
        only re-closes them against the current E_rem."""
        a, b = geo["a"], geo["b"]
        denom = 2.0 * E_rem + b
        if denom <= _TINY:
            return {}
        s = (a + b * E_rem) / denom                                   # eq 12
        if not np.isfinite(s) or s <= 0.0:
            return {}
        dist2 = a - 2.0 * s * b + 2.0 * s * s
        out = dict(geo)
        out.update(s=s, dist2=max(dist2, 0.0), E_rem=E_rem)
        return out

    def parent(self, i: int, j: int) -> dict:
        """Cached geometry closed at the current layer state (test/API path)."""
        geo = self.parent_geometry(i, j)
        if not geo:
            return {}
        E_rem = float(self.cap.sum() - self.cap[i] - self.cap[j])
        return self.close_parent(geo, E_rem)

    def realize_parent(self, i: int, j: int, p: dict):
        """Map f_p back to physical parameters.  Paper eq. 15.

        wt_in* = sqrt(s* R_F) K_self^{-1/4} u*,  w_out* = sqrt(s*/R_F) K_self^{-1/4} v*
        with R_F the subspace Frobenius ratio ||W_in||_F/||W_out||_F, which
        splits the PH-1 gauge freedom so the parent inherits the pair's
        input/output balance."""
        W_in = self.aug[[i, j]]
        rf = float(np.linalg.norm(W_in) / max(np.linalg.norm(self.W_out[[i, j]]), _TINY))
        k4 = max(p["k_uu"], _TINY) ** 0.25
        aug_new = math.sqrt(p["s"] * rf) / k4 * p["u"]
        out_new = math.sqrt(p["s"] / max(rf, _TINY)) / k4 * p["v"]
        return aug_new[:self.n_in], float(aug_new[self.n_in]), out_new

    def install_parent(self, i: int, j: int, p: dict) -> None:
        """Write the realized parent into slot i (j is then removed).

        realize_parent applies the PH-1 gauge split, so the physical parent is
        aug_new = lam * u with u unit-norm; its pre-activation is lam*(u . xt),
        so (beta, gamma) scale by lam as well. Storing the unit-direction stats
        instead would make every later step score this unit with the wrong
        kernel -- keep this bookkeeping beside the math it must agree with."""
        w_row, b_row, w_out = self.realize_parent(i, j, p)
        aug_new = np.append(w_row, b_row)
        lam = float(np.linalg.norm(aug_new))
        self.W_eff[i] = w_row
        self.b_eff[i] = b_row
        self.aug[i] = aug_new
        self.W_out[i] = w_out
        self.beta[i] = lam * p["beta"]
        self.gamma[i] = lam * p["gamma"]
        self.k_self[i] = float(self_kernel(self.beta[i], self.gamma[i]))
        self.w_norm[i] = float(np.linalg.norm(w_row))
        self.cap[i] = float(np.linalg.norm(w_out)) * math.sqrt(max(self.k_self[i], 0.0))

    # -- costs (eq. 6) ------------------------------------------------------

    def prune_costs(self, active: np.ndarray) -> np.ndarray:
        E_a = float(self.cap[active].sum())
        n = len(active)
        rem = np.maximum(E_a - self.cap[active], _TINY)
        return n * self.cap[active] / rem

    def merge_cost_cached(self, geo: dict, E_rem: float,
                          n_active: int) -> tuple[float, dict]:
        """J_merge (eq. 6) from a cached geometry -- O(1)."""
        p = self.close_parent(geo, E_rem)
        if not p:
            return math.inf, {}
        denom = p["E_rem"] + p["s"]
        if denom <= _TINY:
            return math.inf, {}
        return n_active * math.sqrt(p["dist2"]) / denom, p

    def merge_cost(self, i: int, j: int, n_active: int) -> tuple[float, dict]:
        geo = self.parent_geometry(i, j)
        if not geo:
            return math.inf, {}
        E_rem = float(self.cap.sum() - self.cap[i] - self.cap[j])
        return self.merge_cost_cached(geo, E_rem, n_active)


# ── extraction ───────────────────────────────────────────────────────────────

def _following_bn(model: PrunableModel, layer_idx: int) -> nn.Module | None:
    """The BatchNorm applied to this layer's outputs, if the model exposes a
    module list in which it directly follows the prunable layer."""
    target = model.prunable_layer(layer_idx)
    mods = list(model.modules())
    for a, b in zip(mods, mods[1:]):
        if a is target and isinstance(b, (nn.BatchNorm1d, nn.BatchNorm2d)):
            return b
    return None


def build_layer(model: PrunableModel, layer_idx: int, ctx: PruneContext,
                exact_kernel: bool) -> HopeLayer:
    """Effective parameters and the surrogate's (beta, gamma) per unit.

    BN present -> eqs. 1 and 78: gamma = bn.weight, beta = bn.bias, entirely
    data-free. No BN -> the App. E.1 fallback: one calibration pass gives the
    marginal pre-activation mean/std, gamma_i = sigma_i, beta_i = mu_i + b_i."""
    lin = model.prunable_layer(layer_idx)
    W = lin.weight.data.double().reshape(lin.weight.shape[0], -1).cpu().numpy()
    b = (lin.bias.data.double().cpu().numpy() if lin.bias is not None
         else np.zeros(W.shape[0]))
    W_out = model.outgoing_weights(layer_idx).double().cpu().numpy()
    if W_out.shape[0] != W.shape[0]:                # patch-major conv consumer
        W_out = W_out.reshape(W.shape[0], -1)

    bn = _following_bn(model, layer_idx)
    if bn is not None:
        std = np.sqrt(bn.running_var.detach().double().cpu().numpy() + bn.eps)
        g = bn.weight.detach().double().cpu().numpy()
        bta = bn.bias.detach().double().cpu().numpy()
        scale = g / std
        W_eff = scale[:, None] * W
        b_eff = bta - scale * bn.running_mean.detach().double().cpu().numpy()
        return HopeLayer(W_eff, b_eff, bta, g, W_out, exact_kernel)

    # unnormalized: measure the marginal pre-activation statistics once
    with torch.no_grad():
        x = ctx.train_inputs.to(next(model.parameters()).device)
        pre = _pre_activation(model, layer_idx, x).double().cpu().numpy()
    mu, sd = pre.mean(axis=0), pre.std(axis=0)
    return HopeLayer(W, b, mu, np.maximum(sd, 0.0), W_out, exact_kernel)


def _pre_activation(model: PrunableModel, layer_idx: int,
                    x: torch.Tensor) -> torch.Tensor:
    """Pre-activation of the prunable layer, flattened over any spatial dims."""
    target = model.prunable_layer(layer_idx)
    grabbed: list[torch.Tensor] = []
    h = target.register_forward_hook(lambda m, inp, out: grabbed.append(out.detach()))
    try:
        model(x)
    finally:
        h.remove()
    out = grabbed[0]
    if out.dim() == 4:                              # [N, C, H, W] -> [N*H*W, C]
        out = out.permute(0, 2, 3, 1).reshape(-1, out.shape[1])
    return out.reshape(-1, out.shape[-1])


# ── the pruning method ───────────────────────────────────────────────────────

@register_pruning_method("hope")
class HOPE(PruningMethod):
    """Greedy progressive encoding within one prunable layer.

    Each step takes the admissible operation of lowest distortion cost J
    (eq. 6) -- prune a unit, or merge a pair into its optimal rank-1 parent --
    until `n_remove` units are gone. Within a layer all granular operations
    release the same parameter footprint, so ranking by J equals ranking by the
    distortion rate J/dP of eq. 23.

    params:
      n_remove      units to remove from the layer
      allow_merge   enable merge operations (default True; requires a Linear
                    prunable layer with no BN, since the synthesized parent's
                    BN write-back of App. D is not implemented)
      exact_kernel  eq. 83 truncated-bivariate cross-kernel instead of the
                    paper's zero-bias arc-cosine approximation of eq. 85
                    (default False = what the paper uses at scale)
    """

    def __init__(self, n_remove: int = 1, allow_merge: bool = True,
                 exact_kernel: bool = False):
        self.n_remove = int(n_remove)
        self.allow_merge = bool(allow_merge)
        self.exact_kernel = bool(exact_kernel)

    def select(self, model: PrunableModel, layer_idx: int,
               ctx: PruneContext) -> PruneDecision:
        lay = build_layer(model, layer_idx, ctx, self.exact_kernel)
        H = lay.W_eff.shape[0]
        n_remove = min(self.n_remove, H - 1)
        if n_remove <= 0:
            return PruneDecision(remove=[])

        can_merge = (self.allow_merge
                     and isinstance(model.prunable_layer(layer_idx), nn.Linear)
                     and _following_bn(model, layer_idx) is None)
        active = np.ones(H, dtype=bool)
        active[list(ctx.already_selected)] = False
        removed: list[int] = []
        touched = False

        # Paper Sec. 10 step 1: precompute and cache every valid pair's parent
        # geometry once, then only re-derive the rows a merge actually touched
        # (step 3's "localized update"). O(N^2) init + O(N) per merge, versus
        # O(N^2) parent solves per step for a naive rescan.
        geo_cache: dict[tuple[int, int], dict] = {}
        if can_merge:
            act0 = np.flatnonzero(active)
            for ai in range(len(act0)):
                for aj in range(ai + 1, len(act0)):
                    i0, j0 = int(act0[ai]), int(act0[aj])
                    g = lay.parent_geometry(i0, j0)
                    if g:
                        geo_cache[(i0, j0)] = g

        while len(removed) < n_remove and active.sum() > 1:
            idx = np.flatnonzero(active)
            n = len(idx)
            j_prune = lay.prune_costs(idx)
            best_p = int(np.argmin(j_prune))
            best = ("prune", float(j_prune[best_p]), int(idx[best_p]), -1, {})

            if can_merge:
                E_tot = float(lay.cap[idx].sum())
                for (ii, jj), geo in geo_cache.items():
                    if not (active[ii] and active[jj]):
                        continue
                    E_rem = E_tot - float(lay.cap[ii]) - float(lay.cap[jj])
                    c, p = lay.merge_cost_cached(geo, E_rem, n)
                    if c < best[1]:
                        best = ("merge", c, ii, jj, p)

            if best[0] == "prune":
                i = best[2]
                active[i] = False
                removed.append(i)
                lay.cap[i] = 0.0
            else:
                i, j, p = best[2], best[3], best[4]
                lay.install_parent(i, j, p)
                # localized update: the parent replaced unit i, so every cached
                # pair touching i is stale; pairs touching j die with it.
                for key in [k for k in geo_cache if i in k or j in k]:
                    del geo_cache[key]
                for other in np.flatnonzero(active):
                    o = int(other)
                    if o in (i, j):
                        continue
                    g = lay.parent_geometry(min(i, o), max(i, o))
                    if g:
                        geo_cache[(min(i, o), max(i, o))] = g
                active[j] = False
                removed.append(j)
                lay.cap[j] = 0.0
                touched = True

        dec = PruneDecision(remove=sorted(removed))
        if touched:
            lin = model.prunable_layer(layer_idx)
            dec.new_incoming = (
                torch.from_numpy(lay.W_eff).to(lin.weight.dtype),
                torch.from_numpy(lay.b_eff).to(lin.weight.dtype))
            W_out = lay.W_out.copy()
            W_out[np.array(sorted(removed), dtype=int)] = 0.0
            dec.new_outgoing = torch.from_numpy(W_out).to(lin.weight.dtype)
        return dec


# ── self-tests ───────────────────────────────────────────────────────────────

def _selftest() -> None:  # pragma: no cover
    rng = np.random.default_rng(0)

    # 1. eq. 79 self-kernel vs Monte Carlo
    for beta, gamma in [(0.0, 1.0), (0.7, 0.4), (-1.3, 2.0), (2.5, 0.3)]:
        y = beta + gamma * rng.standard_normal(4_000_000)
        mc = float(np.mean(np.clip(y, 0, None) ** 2))
        cf = float(self_kernel(beta, gamma))
        assert abs(cf - mc) < 5e-3 * max(mc, 1e-3), (beta, gamma, cf, mc)
    # closed-form corner: beta=0 -> gamma^2/2
    assert abs(float(self_kernel(0.0, 2.0)) - 2.0) < 1e-12

    # 2. eq. 83 cross-kernel vs Monte Carlo, and vs an independent derivation
    #    K(i,j) = |g_i g_j| E[(z_i+c_i)_+ (z_j+c_j)_+], corr(z_i,z_j) = rho
    for bi, gi, bj, gj, r in [(0.0, 1.0, 0.0, 1.0, 0.5),
                              (0.6, 0.8, -0.4, 1.2, -0.3),
                              (1.5, 0.5, 0.9, 0.7, 0.85),
                              (-0.8, 1.1, 0.2, 0.6, 0.1)]:
        n = 4_000_000
        z1 = rng.standard_normal(n)
        z2 = r * z1 + math.sqrt(1 - r * r) * rng.standard_normal(n)
        mc = float(np.mean(np.clip(bi + gi * z1, 0, None)
                           * np.clip(bj + gj * z2, 0, None)))
        cf = float(cross_kernel_exact(bi, gi, bj, gj, r))
        tol = 6e-3 * max(mc, 1e-3) + 4.0 / math.sqrt(n)
        assert abs(cf - mc) < tol, ("eq83 vs MC", bi, gi, bj, gj, r, cf, mc)

    # 3. the paper's three cross-kernel axioms (App. E.3.1)
    b_, g_ = 0.7, 1.3
    kii = float(self_kernel(b_, g_))
    assert abs(float(cross_kernel_exact(b_, g_, b_, g_, 1.0)) - kii) < 1e-8, \
        "diagonal consistency"
    for r in [-0.9, -0.2, 0.3, 0.95]:
        kij = float(cross_kernel_exact(0.4, 0.9, -0.2, 1.4, r))
        bound = math.sqrt(float(self_kernel(0.4, 0.9)) * float(self_kernel(-0.2, 1.4)))
        assert kij <= bound + 1e-9, ("Cauchy-Schwarz", r, kij, bound)
    mono = [float(cross_kernel_exact(0.4, 0.9, -0.2, 1.4, r))
            for r in np.linspace(-0.95, 0.95, 25)]
    assert all(x <= y + 1e-12 for x, y in zip(mono, mono[1:])), "monotone in rho"
    # zero-bias approximation agrees with the exact form at zero bias
    for r in [-0.7, 0.0, 0.45, 0.9]:
        ex = float(cross_kernel_exact(0.0, 1.0, 0.0, 1.0, r))
        ap = float(cross_kernel_zerobias(self_kernel(0.0, 1.0),
                                         self_kernel(0.0, 1.0), r))
        assert abs(ex - ap) < 1e-8, ("zero-bias branch", r, ex, ap)

    # 4. warped correlation: bounded, odd, and rho_hat -> 0 with rho_eff
    assert abs(float(warped_correlation(0.0, 1.0, 1.0, 1.0, 1.0))) < 1e-12
    for re in [0.2, 0.6, 0.9]:
        a = float(warped_correlation(re, 1.0, 1.0, 1.0, 1.0))
        b2 = float(warped_correlation(-re, 1.0, 1.0, 1.0, 1.0))
        assert 0.0 < a < 1.0 and abs(a + b2) < 1e-12, (re, a, b2)

    # 5. capacity is gauge invariant: (w_in, b, gamma) -> t*, w_out -> /t
    d, H, c = 12, 6, 5
    W = rng.normal(size=(H, d)); bb = rng.normal(size=H)
    beta = rng.normal(size=H); gam = np.abs(rng.normal(size=H)) + 0.3
    Wo = rng.normal(size=(H, c))
    L1 = HopeLayer(W, bb, beta, gam, Wo, True)
    t = 3.7
    L2 = HopeLayer(t * W, t * bb, t * beta, t * gam, Wo / t, True)
    assert np.allclose(L1.cap, L2.cap, rtol=1e-10), "capacity must be gauge invariant"

    # 6. parent construction on duplicated neurons. NOTE the semantics: HOPE's
    #    merge objective (paper p13) is the CLUSTERING distortion
    #        D^2 = ||f_i - f_p||^2 + ||f_j - f_p||^2 ,
    #    i.e. the pair is replaced by the duplicated state [f_p, f_p] and one
    #    copy is then dropped. For f_i = f_j = f this is minimized at f_p = f
    #    (analytically s* = ||f|| since a = 2||f||^2, b = 2||f||), so the parent
    #    reproduces ONE child while the layer's sum f_i + f_j loses the other --
    #    and D^2 = 0, so the operation is scored as FREE. HOPE's cost is a
    #    dispersion of the children about the parent, not the layer's function
    #    error; contrast a sum-rule merge, which emits f_i + f_j and is exact
    #    here. We assert HOPE's actual behavior so the port stays faithful.
    W3 = rng.normal(size=(3, d)); b3 = rng.normal(size=3)
    W3[1] = W3[0]; b3[1] = b3[0]
    mu = rng.normal(size=d) * 0.4
    A_ = rng.normal(size=(d, d)) / math.sqrt(d)
    Cov = A_ @ A_.T + 0.05 * np.eye(d)
    pre_mu = W3 @ mu + b3
    pre_sd = np.sqrt(np.einsum("ij,jk,ik->i", W3, Cov, W3))
    Wo3 = rng.normal(size=(3, c)); Wo3[1] = Wo3[0]
    lay = HopeLayer(W3, b3, pre_mu, pre_sd, Wo3, True)
    p = lay.parent(0, 1)
    assert p, "parent must exist for duplicates"
    assert p["dist2"] < 1e-10 * max(lay.cap[0] ** 2, 1.0), \
        f"duplicates must be scored free, got D^2={p['dist2']:.3e}"
    assert abs(p["s"] - lay.cap[0]) < 1e-8 * lay.cap[0], \
        f"s* must equal one child's capacity, {p['s']} vs {lay.cap[0]}"
    w_row, b_row, w_out = lay.realize_parent(0, 1, p)
    Lc = np.linalg.cholesky(Cov)
    Z = mu + rng.standard_normal((200_000, d)) @ Lc.T
    one = np.clip(Z @ W3[0] + b3[0], 0, None)[:, None] * Wo3[0]
    got = np.clip(Z @ w_row + b_row, 0, None)[:, None] * w_out
    rel_one = np.abs(got - one).max() / max(np.abs(one).max(), 1e-12)
    assert rel_one < 1e-6, f"parent must reproduce ONE child, rel {rel_one:.2e}"
    rel_sum = np.abs(got - 2.0 * one).max() / max(2.0 * np.abs(one).max(), 1e-12)
    assert abs(rel_sum - 0.5) < 1e-6, (
        "HOPE's merge is not function-preserving on duplicates: the pair's "
        f"summed contribution is halved (rel err {rel_sum:.3f}, expected 0.5)")

    # 7. ||f_p||_H == s* : the realization preserves the parent's capacity
    lay2 = HopeLayer(rng.normal(size=(4, d)), rng.normal(size=4),
                     rng.normal(size=4), np.abs(rng.normal(size=4)) + 0.4,
                     rng.normal(size=(4, c)), True)
    q = lay2.parent(0, 1)
    wr, br, wo = lay2.realize_parent(0, 1, q)
    aug = np.append(wr, br)
    scale = np.linalg.norm(aug) / max(np.linalg.norm(q["u"]), _TINY)
    cap_p = np.linalg.norm(wo) * math.sqrt(
        float(self_kernel(q["beta"] * scale, q["gamma"] * scale)))
    assert abs(cap_p - q["s"]) < 1e-8 * max(q["s"], 1.0), (cap_p, q["s"])

    # 8. REGRESSION: install_parent must store stats consistent with the
    #    realized physical parameters (the gauge factor lam is easy to drop,
    #    which silently corrupts every step after the first merge).
    lay3 = HopeLayer(rng.normal(size=(5, d)), rng.normal(size=5),
                     rng.normal(size=5), np.abs(rng.normal(size=5)) + 0.4,
                     rng.normal(size=(5, c)), True)
    q3 = lay3.parent(1, 3)
    assert q3
    lay3.install_parent(1, 3, q3)
    # capacity must equal s*, and (beta, gamma) must match the stored row
    assert abs(lay3.cap[1] - q3["s"]) < 1e-8 * max(q3["s"], 1.0), \
        f"install_parent capacity {lay3.cap[1]} != s* {q3['s']}"
    lam3 = float(np.linalg.norm(lay3.aug[1]))
    assert abs(lay3.beta[1] - lam3 * q3["beta"]) < 1e-10 * max(abs(lay3.beta[1]), 1.0)
    assert abs(lay3.gamma[1] - lam3 * q3["gamma"]) < 1e-10 * max(lay3.gamma[1], 1.0)
    assert abs(lay3.w_norm[1] - np.linalg.norm(lay3.W_eff[1])) < 1e-12

    # 9. the Sec.10 cache must be EXACTLY equivalent to a naive rescan: a
    #    pair's geometry depends only on its own two members, so nothing but
    #    the rows touched by a merge can go stale. Verify by replaying the
    #    greedy with uncached parent solves and comparing the op sequence.
    torch_mod = __import__("torch")
    from src.models.mlp import MLP as _MLP
    torch_mod.manual_seed(3)
    mdl = _MLP(input_dim=16, hidden_sizes=[18, 8], output_dim=4).eval()
    with torch_mod.no_grad():
        li = mdl.prunable_layer(0)
        li.weight[4] = li.weight[1]
        li.bias[4] = li.bias[1]
        wo = mdl.outgoing_weights(0).clone()
        wo[4] = wo[1]
        mdl = mdl.set_outgoing_weights(0, wo)
    cx = PruneContext(train_inputs=torch_mod.randn(96, 16), bundle=None,
                      device=torch_mod.device("cpu"))
    cached = HOPE(n_remove=6, allow_merge=True).select(mdl, 0, cx)

    lay_n = build_layer(mdl, 0, cx, False)
    act = np.ones(lay_n.W_eff.shape[0], dtype=bool)
    naive_removed = []
    while len(naive_removed) < 6 and act.sum() > 1:
        ids = np.flatnonzero(act)
        nn_ = len(ids)
        jp = lay_n.prune_costs(ids)
        bp = int(np.argmin(jp))
        pick = ("prune", float(jp[bp]), int(ids[bp]), -1, {})
        for u_ in range(nn_):
            for v_ in range(u_ + 1, nn_):
                cst, pr = lay_n.merge_cost(int(ids[u_]), int(ids[v_]), nn_)
                if cst < pick[1]:
                    pick = ("merge", cst, int(ids[u_]), int(ids[v_]), pr)
        if pick[0] == "prune":
            act[pick[2]] = False
            naive_removed.append(pick[2])
            lay_n.cap[pick[2]] = 0.0
        else:
            lay_n.install_parent(pick[2], pick[3], pick[4])
            act[pick[3]] = False
            naive_removed.append(pick[3])
            lay_n.cap[pick[3]] = 0.0
    assert cached.remove == sorted(naive_removed), (
        f"cached greedy diverged from naive rescan: {cached.remove} vs "
        f"{sorted(naive_removed)}")

    print("hope.py self-tests passed:")
    print("  eq.79 self-kernel and eq.83 cross-kernel vs Monte Carlo")
    print("  diagonal consistency, Cauchy-Schwarz, monotonicity in rho_hat")
    print("  zero-bias arc-cosine branch matches exact at beta=0")
    print("  capacity gauge invariance; ||f_p|| == s*")
    print("  duplicates: D^2 == 0 and s* == one child (merge NOT")
    print("    function-preserving -- the pair's sum is halved)")
    print("  install_parent bookkeeping consistent with the realized gauge")
    print("  Sec.10 pair cache exactly reproduces a naive greedy rescan")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
