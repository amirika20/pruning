"""Candidate B matrices and Gram (B^T B) properties tracked during merging.

Five definitions of B, rows = current units (merged clusters + frozen
originals). All are computed from the REALIZED units, i.e. what the pruned
layer actually contains at that step:

    anchor  rows = rhobar_k * ubar_k            [K, d]   the user's B: anchors
    cov     rows = [ubar_k ; rhobar_k]          [K, d+1] unweighted covectors
    wcov    rows = sqrt(A_k) [ubar_k; rhobar_k] [K, d+1] Gram = weighted 2nd
                                                         moment M = sum A p p^T
    xi      rows = A_k [ubar_k; rhobar_k]       [K, d+1] covector-sum rows
    wq      rows = sqrt(A_k) qt_k               [K, d+1] box-centered weighted
                                                         (theory-native: Ward
                                                         cost = its trace loss)

Predictions from the theory (the study tests these):
    * column sum of xi (the first moment) is conserved exactly -- tracked
      separately as m1 in run_study, not here.
    * trace of wq's Gram decays by ~ the accumulated Ward cost.
    * anchor's Gram is dominated by saturated (large |rho|) units, the least
      functionally relevant -- expected stable-but-unpredictive or noisy.

Properties per Gram, via SVD of B (cheap: K <= width):
    trace, opnorm (lambda_1), fro (||G||_F), stable_rank = trace/opnorm,
    eff_rank = exp(entropy of eigenvalue distribution), eig1..eig5,
    aff3 = top-3 right-singular-subspace affinity vs the unmerged layer
           (||V_ref^T V_cur||_F^2 / 3, 1 = same subspace -- "general
           positioning in space").
"""

from __future__ import annotations

import numpy as np

TINY = 1e-30

GRAM_DEFS = ["anchor", "cov", "wcov", "xi", "wq"]


def build_B(name: str, ubar: np.ndarray, rhobar: np.ndarray, A: np.ndarray,
            qt: np.ndarray) -> np.ndarray:
    p = np.concatenate([ubar, rhobar[:, None]], axis=1)
    if name == "anchor":
        return rhobar[:, None] * ubar
    if name == "cov":
        return p
    if name == "wcov":
        return np.sqrt(np.maximum(A, 0.0))[:, None] * p
    if name == "xi":
        return A[:, None] * p
    if name == "wq":
        return np.sqrt(np.maximum(A, 0.0))[:, None] * qt
    raise KeyError(name)


def gram_props(B: np.ndarray, ref_V: np.ndarray | None = None,
               top_k: int = 5) -> tuple[dict, np.ndarray]:
    """(properties dict, top-3 right singular vectors V [D, 3])."""
    # economy SVD of B [K, D]: eigenvalues of B^T B are s^2
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    lam = s ** 2
    trace = float(lam.sum())
    opnorm = float(lam[0]) if len(lam) else 0.0
    fro = float(np.sqrt((lam ** 2).sum()))
    q = lam / max(trace, TINY)
    q = q[q > 1e-15]
    eff_rank = float(np.exp(-(q * np.log(q)).sum())) if len(q) else 0.0

    props = {
        "trace": trace,
        "opnorm": opnorm,
        "fro": fro,
        "stable_rank": trace / max(opnorm, TINY),
        "eff_rank": eff_rank,
    }
    for i in range(top_k):
        props[f"eig{i + 1}"] = float(lam[i]) if i < len(lam) else 0.0

    V3 = Vt[: min(3, Vt.shape[0])].T  # [D, <=3]
    if ref_V is not None:
        r = min(ref_V.shape[1], V3.shape[1])
        if r:
            M = ref_V[:, :r].T @ V3[:, :r]
            props["aff3"] = float((M ** 2).sum() / r)
        else:
            props["aff3"] = np.nan
    return props, V3
