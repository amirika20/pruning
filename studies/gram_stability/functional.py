"""Exact expected-damage pair selection via the local functional (L2) metric.

Input model: z ~ N(x0, sigma^2 I) with sigma^2 = R0^2/(d+2) -- the Gaussian
whose projections match the uniform-ball moments exactly (and its d->inf
limit; for uniform z in B_R, u^T z has variance R^2/(d+2)). Under it, every
inner product between unit-slope ReLU neuron functions

    h_i(x) = ( u_i^T (x - x0) - r_i )_+ ,    r_i = rho_i - u_i^T x0 = -gamma_i

has a closed form: E[h_i h_j] = sigma^2 * G(r_i/sigma, r_j/sigma, u_i^T u_j),

    G(a,b,c) = (c + ab) L(a,b;c) - b phi(a) PhiBar(beta_a)
               - a phi(b) PhiBar(alpha_b) + s phi(b) phi(alpha_b),

with s = sqrt(1-c^2), beta_a = (b-ca)/s, alpha_b = (a-cb)/s, and L(a,b;c) the
standard bivariate normal survival probability (computed via Owen's T --
vectorized, no quadrature). Special cases:
    G(0,0,c)   = (s + (pi - theta) c) / (2 pi)      [Cho-Saul arc-cosine kernel]
    G(a,a,1)   = (1+a^2) PhiBar(a) - a phi(a)       [= A(a), the self term]

Expected squared layer-output error of merging clusters k,l with the sum-rule
surgery (outgoing columns w = S_c, merged unit hbar with w_bar = w_k + w_l):

    E|| w_k h_k + w_l h_l - w_bar hbar ||^2
      = sigma^2 [ |w_k|^2 (K_kk + K_cc - 2 K_kc)
                + |w_l|^2 (K_ll + K_cc - 2 K_lc)
                + 2 w_k.w_l (K_kl + K_cc - K_kc - K_lc) ]

-- EXACT under the Gaussian model, fan-out included via the outgoing Gram
w_k.w_l. This is the pair-selection cost. The merge RULE (covector addition,
sum-rule surgery) is identical to the Ward engine, so a comparison isolates
the selection metric. Numerical floor: near-duplicate costs cancel to
~1e-16 * scale; costs are clamped at 0 (ordering among free merges is then
arbitrary, which is harmless).

Run `python studies/gram_stability/functional.py` for self-tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr, owens_t

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from studies.gram_stability.merge import TINY, IterativeMerge, LayerUnits

_SQRT2PI = np.sqrt(2.0 * np.pi)
_EPS_C = 1e-9      # |correlation| beyond 1-_EPS_C uses the degenerate branches
_NUDGE = 1e-12     # avoids h=0 in Owen's formula


def _phi(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(x)) / _SQRT2PI


def _phibar(x: np.ndarray) -> np.ndarray:
    return ndtr(-x)


def bvn_sf(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """P(X > a, Y > b) for standard bivariate normal with correlation c.
    Vectorized; Owen's-T based (Owen 1956). Equals Phi2(-a,-b;c) by symmetry."""
    a, b, c = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (a, b, c)))
    h, k = -a, -b

    # general branch on clipped correlation (degenerate cases patched below)
    cc = np.clip(c, -1.0 + _EPS_C, 1.0 - _EPS_C)
    s = np.sqrt(1.0 - cc * cc)
    hh = np.where(np.abs(h) < _NUDGE, _NUDGE, h)
    kk = np.where(np.abs(k) < _NUDGE, _NUDGE, k)
    r1 = (kk - cc * hh) / (hh * s)
    r2 = (hh - cc * kk) / (kk * s)
    beta = np.where(hh * kk < 0.0, 0.5, 0.0)
    general = 0.5 * (ndtr(hh) + ndtr(kk)) - owens_t(hh, r1) - owens_t(kk, r2) - beta

    pos = ndtr(np.minimum(h, k))                                   # c -> +1
    neg = np.clip(ndtr(h) + ndtr(k) - 1.0, 0.0, None)              # c -> -1
    out = np.where(c >= 1.0 - _EPS_C, pos,
                   np.where(c <= -1.0 + _EPS_C, neg, general))
    return np.clip(out, 0.0, 1.0)


def relu_self(a: np.ndarray) -> np.ndarray:
    """A(a) = E[(t - a)_+^2], t ~ N(0,1)."""
    a = np.asarray(a, dtype=np.float64)
    return (1.0 + a * a) * _phibar(a) - a * _phi(a)


def relu_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """G(a,b,c) = E[(x - a)_+ (y - b)_+], (x,y) standard bivariate, corr c."""
    a, b, c = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (a, b, c)))

    # general branch
    cc = np.clip(c, -1.0 + _EPS_C, 1.0 - _EPS_C)
    s = np.sqrt(1.0 - cc * cc)
    L = bvn_sf(a, b, cc)
    beta_a = (b - cc * a) / s
    alpha_b = (a - cc * b) / s
    general = ((cc + a * b) * L
               - b * _phi(a) * _phibar(beta_a)
               - a * _phi(b) * _phibar(alpha_b)
               + s * _phi(b) * _phi(alpha_b))

    # c -> +1: y = x, integrate (x-a)(x-b) over x > max(a,b)
    m = np.maximum(a, b)
    pos = _phibar(m) * (1.0 + a * b) + _phi(m) * (m - a - b)

    # c -> -1: y = -x, integrate (x-a)(-x-b) over a < x < -b
    lo, hi = a, -b
    empty = lo >= hi
    lo_s, hi_s = np.where(empty, 0.0, lo), np.where(empty, 0.0, hi)
    I0 = ndtr(hi_s) - ndtr(lo_s)
    I1 = _phi(lo_s) - _phi(hi_s)
    I2 = I0 - (hi_s * _phi(hi_s) - lo_s * _phi(lo_s))
    neg = np.where(empty, 0.0, -I2 + (a - b) * I1 + a * b * I0)

    out = np.where(c >= 1.0 - _EPS_C, pos,
                   np.where(c <= -1.0 + _EPS_C, neg, general))
    return np.clip(out, 0.0, None)


# ── the functional-metric merge engine ───────────────────────────────────────

class FunctionalMerge(IterativeMerge):
    """Same state and merge rule as IterativeMerge (covector addition,
    sum-rule surgery, certified bound); only pair selection differs: the pair
    minimizing the EXACT expected squared layer-output error under the
    Gaussian input model. All per-step updates are O(K) via Gram recursions:
    realized-direction cosines and outgoing inner products both update from
    their own rows, with no d- or m-dimensional work after initialization."""

    def _metric_init(self) -> None:
        d = self.orig.u.shape[1]
        self.sigma = self.R0 / np.sqrt(d + 2.0)
        H = self.n_orig

        self._n = np.linalg.norm(self.S_u, axis=1)                 # realized-gain norms
        self._dot = self.S_u @ self.x0                             # additive under merges
        safe = np.where(self._n > TINY, self._n, 1.0)
        self._r = np.where(self._n > TINY, (self.S_rho - self._dot) / safe, 0.0)
        U = self.S_u / safe[:, None]
        U[self._n <= TINY] = self.orig.u[self._n <= TINY]
        self._Cg = np.clip(U @ U.T, -1.0, 1.0)                     # realized-dir cosines
        self._Wg = self.S_c @ self.S_c.T                           # outgoing Gram

    def _metric_update(self, k: int, l: int) -> None:
        # called with sums already merged into k; metric state still old
        nk, nl, ckl = self._n[k], self._n[l], self._Cg[k, l]
        n_new = float(np.sqrt(max(nk * nk + nl * nl + 2.0 * nk * nl * ckl, 0.0)))
        if n_new > TINY:
            self._Cg[k, :] = np.clip((nk * self._Cg[k, :] + nl * self._Cg[l, :]) / n_new, -1.0, 1.0)
            self._Cg[:, k] = self._Cg[k, :]
            self._Cg[k, k] = 1.0
        row = self._Wg[k, :] + self._Wg[l, :]
        row[k] = self._Wg[k, k] + 2.0 * self._Wg[k, l] + self._Wg[l, l]
        self._Wg[k, :] = row
        self._Wg[:, k] = row
        self._dot[k] += self._dot[l]
        self._n[k] = n_new
        self._r[k] = (self.S_rho[k] - self._dot[k]) / n_new if n_new > TINY else 0.0

    def _pair_costs(self, k: int, idx: np.ndarray) -> np.ndarray:
        nk, nl = self._n[k], self._n[idx]
        ckl = self._Cg[k, idx]
        n_cand = np.sqrt(np.clip(nk * nk + nl * nl + 2.0 * nk * nl * ckl, 0.0, None))
        safe = np.where(n_cand > TINY, n_cand, 1.0)
        ck_c = np.clip((nk + nl * ckl) / safe, -1.0, 1.0)
        cl_c = np.clip((nl + nk * ckl) / safe, -1.0, 1.0)
        r_cand = (self.S_rho[k] + self.S_rho[idx] - self._dot[k] - self._dot[idx]) / safe

        zk = np.full_like(ckl, self._r[k] / self.sigma)
        zl = self._r[idx] / self.sigma
        zc = r_cand / self.sigma

        Kkk, Kll, Kcc = relu_self(zk), relu_self(zl), relu_self(zc)
        Kkl = relu_cross(zk, zl, ckl)
        Kkc = relu_cross(zk, zc, ck_c)
        Klc = relu_cross(zl, zc, cl_c)

        wkk = self._Wg[k, k]
        wll = self._Wg[idx, idx]
        wkl = self._Wg[k, idx]
        cost = (wkk * np.clip(Kkk + Kcc - 2.0 * Kkc, 0.0, None)
                + wll * np.clip(Kll + Kcc - 2.0 * Klc, 0.0, None)
                + 2.0 * wkl * (Kkl + Kcc - Kkc - Klc))
        cost = self.sigma ** 2 * np.clip(cost, 0.0, None)
        return np.where(n_cand > TINY, cost, 0.0)


class GaussianMeasureMerge(FunctionalMerge):
    """Functional selection under z ~ N(mu, C) with arbitrary center and
    covariance (per-feature variance vector, or full [d, d] matrix).

    1B3 (data-free box-CLT):   mu = box center, C = diag(((hi-lo)/2)^2 / 3)
    1B4 (matched, data-light): mu, C = moments of ~100 calibration inputs

    A realized unit's pre-activation t_k = (S_u_k^T z - S_rho_k)/n_k is
    Gaussian with (unnormalized) mean dotmu_k - S_rho_k and variance
    D_kk = S_u_k^T C S_u_k, and pairs are jointly Gaussian with covariance
    D_kl -- so every kernel entry is s_k s_l G(z_k, z_l, corr_kl) with
    z_k = (S_rho_k - dotmu_k)/sqrt(D_kk), s_k = sqrt(D_kk)/n_k. Both D and
    dotmu are additive under covector merging, so per-step updates stay O(K)
    with no d-dimensional work after initialization."""

    def __init__(self, units: LayerUnits, lo: np.ndarray, hi: np.ndarray,
                 mu: np.ndarray, cov: np.ndarray):
        self._mu = np.asarray(mu, dtype=np.float64)
        self._covm = np.asarray(cov, dtype=np.float64)
        super().__init__(units, lo, hi)

    def _apply_cov(self, M: np.ndarray) -> np.ndarray:
        return M * self._covm[None, :] if self._covm.ndim == 1 else M @ self._covm

    def _metric_init(self) -> None:
        super()._metric_init()                      # _n, _Cg, _Wg (reused)
        self._dotmu = self.S_u @ self._mu           # additive under merges
        Dg = self.S_u @ self._apply_cov(self.S_u).T
        self._Dg = 0.5 * (Dg + Dg.T)                # symmetrize roundoff

    def _metric_update(self, k: int, l: int) -> None:
        row = self._Dg[k, :] + self._Dg[l, :]
        row[k] = self._Dg[k, k] + 2.0 * self._Dg[k, l] + self._Dg[l, l]
        super()._metric_update(k, l)
        self._Dg[k, :] = row
        self._Dg[:, k] = row
        self._dotmu[k] += self._dotmu[l]

    def _pair_costs(self, k: int, idx: np.ndarray) -> np.ndarray:
        nk, nl = self._n[k], self._n[idx]
        ckl = self._Cg[k, idx]
        n_cand = np.sqrt(np.clip(nk * nk + nl * nl + 2.0 * nk * nl * ckl, 0.0, None))

        Dkk = max(float(self._Dg[k, k]), 0.0)
        Dll = np.clip(self._Dg[idx, idx], 0.0, None)
        Dkl = self._Dg[k, idx]
        Dcc = np.clip(Dkk + 2.0 * Dkl + Dll, 0.0, None)
        SDk, SDl, SDc = np.sqrt(Dkk), np.sqrt(Dll), np.sqrt(Dcc)

        # standardized offsets (n cancels) and per-unit stds of t
        rmk = self.S_rho[k] - self._dotmu[k]
        rml = self.S_rho[idx] - self._dotmu[idx]
        zk = rmk / max(SDk, 1e-300)
        zl = rml / np.maximum(SDl, 1e-300)
        zc = (rmk + rml) / np.maximum(SDc, 1e-300)
        sk = SDk / max(self._n[k], TINY)
        sl = SDl / np.where(nl > TINY, nl, TINY)
        sc = SDc / np.where(n_cand > TINY, n_cand, TINY)

        corr_kl = np.clip(Dkl / np.maximum(SDk * SDl, 1e-300), -1.0, 1.0)
        corr_kc = np.clip((Dkk + Dkl) / np.maximum(SDk * SDc, 1e-300), -1.0, 1.0)
        corr_lc = np.clip((Dkl + Dll) / np.maximum(SDl * SDc, 1e-300), -1.0, 1.0)

        Kkk, Kll, Kcc = sk * sk * relu_self(zk), sl * sl * relu_self(zl), sc * sc * relu_self(zc)
        Kkl = sk * sl * relu_cross(zk, zl, corr_kl)
        Kkc = sk * sc * relu_cross(zk, zc, corr_kc)
        Klc = sl * sc * relu_cross(zl, zc, corr_lc)

        wkk = self._Wg[k, k]
        wll = self._Wg[idx, idx]
        wkl = self._Wg[k, idx]
        cost = (wkk * np.clip(Kkk + Kcc - 2.0 * Kkc, 0.0, None)
                + wll * np.clip(Kll + Kcc - 2.0 * Klc, 0.0, None)
                + 2.0 * wkl * (Kkl + Kcc - Kkc - Klc))
        return np.where(n_cand > TINY, np.clip(cost, 0.0, None), 0.0)


# ── self-tests ────────────────────────────────────────────────────────────────

def _selftest() -> None:  # pragma: no cover
    rng = np.random.default_rng(0)

    # 1. arc-cosine special case
    for theta in [0.0, 0.3, 1.0, np.pi / 2, 2.5, np.pi]:
        want = (np.sin(theta) + (np.pi - theta) * np.cos(theta)) / (2 * np.pi)
        got = float(relu_cross(0.0, 0.0, np.cos(theta)))
        assert abs(got - want) < 1e-12, (theta, got, want)

    # 2. c=1 self-consistency
    for a in [-2.0, -0.3, 0.0, 0.7, 3.0]:
        assert abs(float(relu_cross(a, a, 1.0)) - float(relu_self(a))) < 1e-12

    # 3. Monte Carlo, general (a, b, c)
    n = 4_000_000
    x = rng.standard_normal(n)
    e = rng.standard_normal(n)
    for a, b, c in [(-0.5, 0.3, 0.6), (1.2, -0.8, -0.4), (0.0, 0.9, 0.95),
                    (2.5, 2.0, 0.3), (-1.0, -1.0, -0.9)]:
        y = c * x + np.sqrt(1 - c * c) * e
        mc = float(np.mean(np.clip(x - a, 0, None) * np.clip(y - b, 0, None)))
        cf = float(relu_cross(a, b, c))
        tol = 5e-3 * max(mc, 1e-3) + 3.0 * np.sqrt(np.var(
            np.clip(x - a, 0, None) * np.clip(y - b, 0, None)) / n)
        assert abs(cf - mc) < tol, (a, b, c, cf, mc)

    # 4. Gaussian model vs the true uniform ball (moderate + high d)
    for d in [50, 784]:
        R = 1.0
        sigma = R / np.sqrt(d + 2.0)
        m = 2_000_000
        g = rng.standard_normal((m, 2))  # 2-d marginal of ball is enough:
        # (z1, z2) of a uniform ball point: radius^2 ~ Beta-like; sample full:
        zfull = rng.standard_normal((m, d))
        zfull /= np.linalg.norm(zfull, axis=1, keepdims=True)
        zfull *= R * rng.random(m)[:, None] ** (1.0 / d)
        c = 0.7
        u1 = np.zeros(d); u1[0] = 1.0
        u2 = np.zeros(d); u2[0] = c; u2[1] = np.sqrt(1 - c * c)
        r1, r2 = 0.5 * sigma, -0.8 * sigma
        mc = float(np.mean(np.clip(zfull @ u1 - r1, 0, None)
                           * np.clip(zfull @ u2 - r2, 0, None)))
        cf = sigma ** 2 * float(relu_cross(r1 / sigma, r2 / sigma, c))
        assert abs(cf - mc) / mc < 0.05, (d, cf, mc)

    # 5. engine: exact duplicates (arbitrary outgoing signs) cost ~0 and merge first
    d, H, m2 = 10, 12, 4
    W = rng.normal(size=(H, d)); b = rng.normal(size=H); C = rng.normal(size=(H, m2))
    for j, s in [(4, 2.5), (5, 0.7)]:
        W[j] = s * W[3]; b[j] = s * b[3]
    alpha = np.linalg.norm(W, axis=1)
    units = LayerUnits(W / alpha[:, None], -b / alpha, alpha, C)
    lo, hi = -np.ones(d), np.ones(d)
    eng = FunctionalMerge(units, lo, hi)
    r1_, r2_ = eng.step(), eng.step()
    assert r1_["ward_cost"] < 1e-12 and r2_["ward_cost"] < 1e-12
    Wr, br, cols, _ = eng.realize()
    X = rng.normal(size=(256, d))
    ref = np.maximum(X @ W.T + b, 0) @ C
    err = np.abs(np.maximum(X @ Wr.T + br, 0) @ cols - ref).max()
    assert err < 1e-9, err

    # 6. dead neurons with different directions are ~free to merge
    #    (the cylinder metric puts them 2*R0 apart; the functional metric ~0)
    W2 = rng.normal(size=(4, d)); b2 = np.array([-50.0, -60.0, 1.0, -1.5])
    a2 = np.linalg.norm(W2, axis=1)
    units2 = LayerUnits(W2 / a2[:, None], -b2 / a2, a2, rng.normal(size=(4, m2)))
    eng2 = FunctionalMerge(units2, lo, hi)
    rec = eng2.step()
    merged = set(eng2.members[rec["survivor"]])
    assert merged == {0, 1}, f"dead pair should merge first, got {merged}"
    assert rec["ward_cost"] < 1e-12

    # 7. full sweep runs to one cluster, m1_raw conserved
    eng3 = FunctionalMerge(units, lo, hi)
    m1_0 = eng3.m1_raw()
    while eng3.n_active > 1:
        eng3.step()
    assert np.abs(eng3.m1_raw() - m1_0).max() / np.abs(m1_0).max() < 1e-12

    # 8. GaussianMeasureMerge: pair cost == MC expected merge damage, N(mu, C)
    d3, m3 = 8, 5
    W3 = rng.normal(size=(3, d3)); b3 = rng.normal(size=3); C3 = rng.normal(size=(3, m3))
    al = np.linalg.norm(W3, axis=1)
    u3 = LayerUnits(W3 / al[:, None], -b3 / al, al, C3)
    mu = rng.normal(size=d3) * 0.5
    A_ = rng.normal(size=(d3, d3)) / np.sqrt(d3)
    Cov = A_ @ A_.T + 0.1 * np.eye(d3)
    eng4 = GaussianMeasureMerge(u3, -np.ones(d3), np.ones(d3), mu, Cov)
    i, j = 0, 1
    a_w = u3.a
    S_u = a_w[i] * u3.u[i] + a_w[j] * u3.u[j]
    ub, rb = S_u / np.linalg.norm(S_u), (a_w[i] * u3.rho[i] + a_w[j] * u3.rho[j]) / np.linalg.norm(S_u)
    S_c = al[i] * C3[i] + al[j] * C3[j]
    Lch = np.linalg.cholesky(Cov)
    Z = mu + rng.standard_normal((2_000_000, d3)) @ Lch.T
    err = (np.clip(Z @ u3.u[i] - u3.rho[i], 0, None)[:, None] * (al[i] * C3[i])
           + np.clip(Z @ u3.u[j] - u3.rho[j], 0, None)[:, None] * (al[j] * C3[j])
           - np.clip(Z @ ub - rb, 0, None)[:, None] * S_c)
    mc = float(np.mean((err ** 2).sum(axis=1)))
    cf = float(eng4._cost[i, j])
    assert abs(cf - mc) / max(mc, 1e-12) < 0.05, (cf, mc)

    # 9. box-CLT measure: duplicates free, sweep completes, m1 conserved
    r_box = (hi - lo) / 2.0
    eng5 = GaussianMeasureMerge(units, lo, hi, (lo + hi) / 2.0, r_box * r_box / 3.0)
    s1, s2 = eng5.step(), eng5.step()
    assert s1["ward_cost"] < 1e-10 and s2["ward_cost"] < 1e-10
    m1_0 = eng5.m1_raw()
    while eng5.n_active > 1:
        eng5.step()
    assert np.abs(eng5.m1_raw() - m1_0).max() / np.abs(m1_0).max() < 1e-12

    print("functional.py self-tests passed: arc-cosine identity, c=+/-1 branches,")
    print("MC agreement (Gaussian + uniform-ball d=50/784), duplicate exactness,")
    print("dead-neuron absorption, m1 conservation")
    print("GaussianMeasureMerge: pair cost == MC damage under N(mu, C); box-CLT ok")


if __name__ == "__main__":
    _selftest()
