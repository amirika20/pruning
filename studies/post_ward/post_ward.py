"""Study-only engine: exact Ward in POST-activation space.

Hypothesis (2026-09-17): the sampled delta_f score forms its cluster centroid
in PRE-activation space -- after a merge, MashEngine.step rebuilds the cluster
response as relu((Z g_C - r_C) / ||g_C||) from the summed covector g_C = sum
a_i u_i, and rescores every remaining pair against that synthesized ReLU
neuron. That is Ward on the pre-activation centroid, measured post-ReLU. It is
NOT Ward on the response rows themselves, because relu(mean) != mean(relu).

Exact Ward in response space uses the mass-weighted mean of the members' true
post-activation rows, phibar_C = sum_i a_i phi_i / A_C, as the representative.
Its increments satisfy the Lance--Williams recursion

    D(C u D, E) = [(a_C+a_E) D(C,E) + (a_D+a_E) D(D,E) - a_E D(C,D)]
                  / (a_C + a_D + a_E)

so after the initial [H, H] Gram no response is ever recomputed and the whole
greedy loop is O(H^2). The realizable centroid is gone from SELECTION only:
realization at the cut is unchanged (medoid + repair, from the stock code).

PROMOTED (2026-09-17): this is now `centroid='post'` on the stock MashEngine and
MASH (src/pruning/methods/mash.py, module docstring CENTROID). This file stays
as the study's record and as the reference the promoted knob is checked
against (`run.py --check`, bit for bit on the ResNet-20 curves).

`PostWardEngine` is a subclass of the stock MashEngine. The stock constructor
already computes the singleton matrix a_i a_j / (a_i + a_j) ||phi_i - phi_j||^2
/ N from one Gram (_all_costs); this class only flips the engine into its
Lance--Williams branch, which the stock step() already implements for the
cylinder score, and drops the response recompute. Nothing in src/ changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pruning.methods import mash  # noqa: E402
from src.pruning.methods.mash import TINY, MashEngine


class PostWardEngine(MashEngine):
    """delta_f / empirical, with the cluster centroid taken in response space
    and the costs propagated by Lance--Williams. Anything else passes through
    to the stock engine unchanged (cylinder, gaussian, exact_damage)."""

    def __init__(self, units, score="delta_f", **kw):
        super().__init__(units, score=score, **kw)
        self.post_ward = (score == "delta_f" and self.measure == "empirical")
        if self.post_ward:
            # Same flag the cylinder score uses; step() then applies the
            # mass-weighted Lance--Williams update instead of _pair_costs.
            self._lw = True
            # Keep the singleton response rows for the exactness check; the
            # loop itself no longer reads them.
            self._Phi0 = self._Phi.clone()

    def step(self) -> dict:
        if not self.post_ward:
            return super().step()
        # The stock step, in the non-functional empirical branch, rebuilds
        # _Phi[k] from the summed covector after the merge (the pre-activation
        # centroid pushed through the ReLU). Under Lance--Williams that row is
        # never read, so skip the recompute by presenting the engine as
        # 'functional' for the duration of the call: that branch at most copies
        # one row (when the medoid changes), which nothing reads either. The
        # certificate, if tracked, still goes through the unpatched _medoid.
        self._fn = True
        try:
            return super().step()
        finally:
            self._fn = False

    # -- exactness check ----------------------------------------------------

    def direct_ward_costs(self) -> np.ndarray:
        """Recompute every active pair's Ward increment from scratch, using the
        response-space centroids phibar_C = sum a_i phi_i / A_C of the CURRENT
        clusters. Used to certify the Lance--Williams matrix."""
        act = np.flatnonzero(self.active)
        dev = self._Phi0.device
        N = self._Phi0.shape[1]
        bars = []
        for k in act:
            mem = torch.as_tensor(np.asarray(self.members[k]), device=dev)
            a = torch.as_tensor(self._mass0[np.asarray(self.members[k])], device=dev)
            bars.append((a[:, None] * self._Phi0[mem]).sum(0) / a.sum())
        B = torch.stack(bars)                                    # [K, N]
        sq = (B * B).sum(1)
        d2 = (sq[:, None] + sq[None, :] - 2.0 * (B @ B.T)).clamp_min_(0.0).cpu().numpy() / N
        A = self.A[act]
        den = A[:, None] + A[None, :]
        w = np.where(den > TINY, np.outer(A, A) / np.maximum(den, TINY), 0.0)
        cost = w * d2
        np.fill_diagonal(cost, np.inf)
        return cost


class _Patched:
    """Context manager: make MASH.plan build a PostWardEngine."""

    def __enter__(self):
        self._orig = mash.MashEngine
        mash.MashEngine = PostWardEngine
        return self

    def __exit__(self, *exc):
        mash.MashEngine = self._orig
        return False


def post_ward_selection():
    return _Patched()


# -- self-test -----------------------------------------------------------------

def _selftest(H=40, d=12, N=600, seed=0) -> None:
    """(1) Lance--Williams == direct response-space Ward at every step.
    (2) On singletons the two engines agree exactly (same initial matrix).
    (3) The two engines DO diverge after merges (otherwise the study is moot)."""
    from src.pruning.methods.mash import Units

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((H, d))
    b = rng.standard_normal(H) * 0.3
    C = rng.standard_normal((H, 7))
    # a few near-duplicate units so merges are meaningful
    W[1] = W[0] * 1.3 + 0.01 * rng.standard_normal(d)
    W[3] = W[2] * 0.8 + 0.01 * rng.standard_normal(d)
    Z = rng.standard_normal((N, d))
    nrm = np.linalg.norm(W, axis=1)
    units = Units(u=W / nrm[:, None], rho=-b / nrm, alpha=nrm, C=C)

    pw = PostWardEngine(units, score="delta_f", measure="empirical", Z=Z,
                        dictionary="medoid", track_certificate=False)
    st = MashEngine(units, score="delta_f", measure="empirical", Z=Z,
                    dictionary="medoid", track_certificate=False)
    assert np.allclose(pw._cost, st._cost, equal_nan=True), "singleton matrices differ"

    worst = 0.0
    diverged = False
    for t in range(H - 2):
        pw.step(); st.step()
        act = np.flatnonzero(pw.active)
        lw = pw._cost[np.ix_(act, act)]
        direct = pw.direct_ward_costs()
        fin = np.isfinite(lw)
        worst = max(worst, float(np.max(np.abs(lw[fin] - direct[fin])
                                         / np.maximum(np.abs(direct[fin]), 1e-12))))
        if pw.active.tolist() != st.active.tolist() or \
                not np.allclose(pw._cost, st._cost, equal_nan=True):
            diverged = True
    assert worst < 1e-8, f"Lance-Williams disagrees with direct post-activation Ward: {worst:.2e}"
    assert diverged, "post-activation and pre-activation centroids never diverged"
    print(f"selftest ok: LW == direct response-space Ward (rel err {worst:.1e}); "
          f"diverges from stock delta_f after merges as expected")


if __name__ == "__main__":
    _selftest()
