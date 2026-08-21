"""The neuron-similarity matrix K of a layer, and what it says about redundancy.

K_ij is how alike neurons i and j are. Everything the merge methods do is a
function of it, so it is worth looking at directly rather than only through the
pruning that consumes it.

WHICH K. Four, because they disagree and the disagreement is the interesting
part:

    response  K_ij = E[phi_i phi_j] over calibration inputs, phi = post-ReLU
              unit-gain response. The primitive; everything else derives from it.
    cosine    K_ij / sqrt(K_ii K_jj) in [-1, 1]. Scale-free, so the off-diagonal
              distribution is directly readable as "how duplicated is this layer".
    delta_f   the POST-ReLU merge score, a_i a_j/(a_i+a_j) * ||phi_i - phi_j||^2.
              What MASH's delta_f criterion actually minimizes.
    cylinder  the PRE-ReLU merge score, a_i a_j/(a_i+a_j) * d_rho^2, from the
              geometric codes. What the domain-only criterion minimizes.

The last two are the same Ward form on the same masses, differing only in
pre- vs post-ReLU distance, so `criterion_agreement` between them isolates
exactly what the ReLU clipping does to the ranking -- as one number per layer.

WHAT TO READ OFF IT.

  Spectrum. The participation ratio of K's eigenvalues is the effective number
  of distinct functions the layer computes. `pr_fraction` = PR/H is then a
  redundancy index: near 1 means every neuron is doing something of its own and
  there is nothing to merge; well below 1 means the layer is duplicating itself.
  This predicts merge capacity before any pruning is attempted.

  CENTER IT. Post-ReLU responses are non-negative, so the uncentered K is
  dominated by its rank-one mean component -- measured at 55-71% of the trace on
  a trained MLP, which pins PR(K) near 2 whatever the layer's real structure is.
  The criterion legitimately uses the uncentered K (it is the L2 inner product),
  so both are reported: `K_*` is what the criterion sees, `Kc_*` is the centered
  covariance and the only one to read as redundancy. On the same MLP they differ
  by an order of magnitude -- 2.9/1.8/1.9 uncentered against 19.2/8.0/4.5
  centered.

  Off-diagonal cosine. The blunt version of the same question. A layer with real
  clump structure has a heavy right tail; one without has everything near zero
  however wide it is -- which is what we see on BatchNorm-trained conv filters.

  Nearest-neighbour cost. The sorted per-unit nearest-neighbour merge score is
  what a greedy pass consumes, in order. Its low quantiles say how cheap the
  first merges are and how fast the cost rises, which is the shape of the
  accuracy-versus-width curve before you measure it. `clump_ratio` = median
  nearest-neighbour distance over median pairwise distance compresses it: small
  means tight clumps, near 1 means a diffuse cloud with no natural groups.

  Removed-pair similarity. Given a pruning report, the similarity between each
  removed unit and its survivor, against the null of all pairs. A method that
  is genuinely exploiting redundancy removes pairs from the right tail; one that
  is not shows a removed-pair distribution indistinguishable from the null.

  Cosine is blind to MAGNITUDE, and that matters here. Two barely-active units
  are interchangeable -- their post-ReLU distance is tiny -- while their cosine
  is near zero by degeneracy, both having almost no response to correlate.
  A method that harvests those shows a removed-pair lift BELOW 1, which looks
  like failure and is not. `removed_norm_ratio` disambiguates: well below 1
  means the method is removing low-energy units, which is the cheap and correct
  thing to do, not a mistake in unit choice.

All four work on Conv2d: responses come straight off the hook, and `cylinder`
uses the BN-folded filter codes with the box measured over im2col patches.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from src.analysis.geometry_shift import responses, spectrum
from src.models.registry import PrunableModel

TINY = 1e-30
KINDS = ("response", "cosine", "delta_f", "cylinder")


# ── building the matrices ────────────────────────────────────────────────────

def ward_weight(mass: np.ndarray) -> np.ndarray:
    """a_i a_j / (a_i + a_j), the mass-weighted Ward factor."""
    s = mass[:, None] + mass[None, :]
    return np.where(s > TINY, np.outer(mass, mass) / np.maximum(s, TINY), 0.0)


def similarity_matrices(model: PrunableModel, layer_idx: int, x: torch.Tensor,
                        kinds: Sequence[str] = KINDS) -> dict[str, np.ndarray]:
    """{kind: [H, H] matrix}. Similarities are LARGE for alike neurons;
    `delta_f` and `cylinder` are costs, so they are SMALL for alike neurons --
    the helpers below keep that straight per kind."""
    out: dict[str, np.ndarray] = {}
    Phi = responses(model, layer_idx, x)                 # [N, H], post-ReLU
    N, H = Phi.shape
    if "response" in kinds or "cosine" in kinds or "delta_f" in kinds:
        K = Phi.T @ Phi / max(N, 1)
        if "response" in kinds:
            out["response"] = K
        d = np.sqrt(np.maximum(np.diag(K), 0.0))
        if "cosine" in kinds:
            denom = np.maximum(np.outer(d, d), TINY)
            C = np.clip(K / denom, -1.0, 1.0)
            np.fill_diagonal(C, 1.0)
            out["cosine"] = C

    mass = None
    try:
        from src.pruning.methods.mash import extract_units
        units, _ = extract_units(model, layer_idx)
        mass = units.mass
    except Exception:
        pass

    if "delta_f" in kinds and mass is not None:
        # phi here is the RAW response (it carries alpha); the unit-gain
        # distance is what the score uses, so divide it out.
        alpha = np.maximum(units.alpha, TINY)
        Ku = (Phi / alpha[None, :]).T @ (Phi / alpha[None, :]) / max(N, 1)
        dg = np.diag(Ku)
        d2 = np.maximum(dg[:, None] + dg[None, :] - 2.0 * Ku, 0.0)
        out["delta_f"] = ward_weight(mass) * d2

    if "cylinder" in kinds and mass is not None:
        from src.pruning.methods.mash import _layer_inputs
        Z = _layer_inputs(model, layer_idx, x)
        lo, hi = Z.min(axis=0), Z.max(axis=0)
        x0 = (lo + hi) / 2.0
        R = float(np.linalg.norm((hi - lo) / 2.0))
        gamma = units.u @ x0 - units.rho
        cos = np.clip(units.u @ units.u.T, -1.0, 1.0)
        d2 = (2.0 * R ** 2 * np.maximum(1.0 - cos, 0.0)
              + (gamma[:, None] - gamma[None, :]) ** 2)
        out["cylinder"] = ward_weight(mass) * d2
    return out


def _centered_gram(model: PrunableModel, layer_idx: int,
                   x: torch.Tensor) -> np.ndarray:
    """Covariance of the post-ReLU responses -- K with its rank-one mean
    component removed. This is the spectrum to read as redundancy."""
    Phi = responses(model, layer_idx, x)
    P = Phi - Phi.mean(axis=0, keepdims=True)
    return P.T @ P / max(len(Phi), 1)


def _offdiag(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], 1)
    return M[iu]


# ── reading them ─────────────────────────────────────────────────────────────

def spectrum_stats(K: np.ndarray) -> dict[str, float]:
    """Effective dimension of K, plus PR as a fraction of the width.

    K is a Gram, so its eigenvalues are the squared singular values of any
    square root; we feed a symmetric square root to `spectrum` so the same
    definitions apply.
    """
    w, V = np.linalg.eigh(0.5 * (K + K.T))
    w = np.clip(w, 0.0, None)
    root = V * np.sqrt(w)
    st = spectrum(root.T)
    H = K.shape[0]
    if "participation_ratio" in st and H:
        st["pr_fraction"] = st["participation_ratio"] / H
    if H:
        tot = max(float(w.sum()), TINY)
        order = np.sort(w)[::-1]
        cum = np.cumsum(order) / tot
        st["n90"] = int(np.searchsorted(cum, 0.90) + 1)
        st["n90_fraction"] = st["n90"] / H
    return st


def cosine_stats(C: np.ndarray, thresholds: Sequence[float] = (0.5, 0.8, 0.95)
                 ) -> dict[str, float]:
    """Off-diagonal cosine distribution -- the blunt redundancy read."""
    v = _offdiag(C)
    if v.size == 0:
        return {}
    a = np.abs(v)
    out = {"cos_mean": float(v.mean()), "cos_absmean": float(a.mean()),
           "cos_p50": float(np.quantile(a, 0.5)),
           "cos_p99": float(np.quantile(a, 0.99)),
           "cos_max": float(a.max())}
    for t in thresholds:
        out[f"frac_above_{t}"] = float((a >= t).mean())
    # per-unit nearest neighbour in cosine: how duplicated is the MOST
    # duplicated partner of a typical neuron
    Cn = np.abs(C).copy()
    np.fill_diagonal(Cn, -np.inf)
    nn = Cn.max(axis=1)
    out["nn_cos_median"] = float(np.median(nn))
    return out


def cost_stats(D: np.ndarray) -> dict[str, float]:
    """Nearest-neighbour merge-cost profile: what a greedy pass consumes.

    `D` is a COST matrix (small = alike). Reported relative to the median
    pairwise cost, so the numbers are comparable across layers and criteria.
    """
    H = D.shape[0]
    if H < 2:
        return {}
    Dn = D.astype(float).copy()
    np.fill_diagonal(Dn, np.inf)
    nn = Dn.min(axis=1)
    pair = _offdiag(D)
    med = max(float(np.median(pair)), TINY)
    out = {"nn_cost_min": float(nn.min()),
           "nn_cost_p50": float(np.median(nn)),
           "nn_cost_p90": float(np.quantile(nn, 0.9)),
           "pair_cost_p50": med,
           "clump_ratio": float(np.median(nn) / med)}
    # how many units have a partner far cheaper than a typical pair
    for f in (0.01, 0.1):
        out[f"frac_nn_below_{f}x"] = float((nn < f * med).mean())
    return out


def criterion_agreement(D_a: np.ndarray, D_b: np.ndarray,
                        top_frac: float = 0.05) -> dict[str, float]:
    """Do two cost matrices rank pairs the same way?

    Spearman over all pairs, plus the overlap of the cheapest `top_frac` of
    pairs -- the ones a greedy pass would actually take. The overlap is the
    decision-relevant number: two criteria can correlate well overall and still
    disagree completely about which merges to make first.
    """
    a, b = _offdiag(D_a), _offdiag(D_b)
    if a.size < 3:
        return {}
    k = max(int(top_frac * a.size), 1)
    ia, ib = set(np.argsort(a)[:k].tolist()), set(np.argsort(b)[:k].tolist())
    return {"spearman": float(spearmanr(a, b).correlation),
            "cheap_overlap": len(ia & ib) / k,
            "cheap_chance": k / a.size}


def removed_pair_similarity(C: np.ndarray, removed: Sequence[int],
                            survivors: dict[int, int] | None = None,
                            norms: np.ndarray | None = None
                            ) -> dict[str, float]:
    """Similarity of removed units to their survivor, against the all-pairs null.

    Without an explicit removed->survivor map, each removed unit is scored
    against its most similar SURVIVING unit, which is the right null-free
    reading for methods that delete rather than merge.
    """
    H = C.shape[0]
    removed = [int(i) for i in removed]
    if not removed or len(removed) >= H:
        return {}
    kept = np.array(sorted(set(range(H)) - set(removed)), dtype=int)
    vals: list[float] = []
    for i in removed:
        if survivors and i in survivors:
            vals.append(abs(float(C[i, survivors[i]])))
        else:
            vals.append(float(np.abs(C[i, kept]).max()))
    null = np.abs(_offdiag(C))
    v = np.array(vals)
    out = {"removed_sim_median": float(np.median(v)),
           "removed_sim_p10": float(np.quantile(v, 0.1)),
           "null_sim_median": float(np.median(null)),
           "lift": float(np.median(v) / max(np.median(null), TINY))}
    if norms is not None and len(norms) == H:
        allm = max(float(np.median(norms)), TINY)
        out["removed_norm_ratio"] = float(np.median(norms[removed]) / allm)
        out["kept_norm_ratio"] = float(np.median(norms[kept]) / allm)
    return out


# ── per-layer table ──────────────────────────────────────────────────────────

def similarity_table(model: PrunableModel, x: torch.Tensor,
                     report: Sequence[dict] | None = None,
                     kinds: Sequence[str] = KINDS, **meta: Any) -> pd.DataFrame:
    """One row per layer: K's spectrum, cosine and cost profiles, the
    pre/post-ReLU criterion agreement, and (with a report) removed-pair lift."""
    removed_by = {}
    survivors_by: dict[int, dict[int, int]] = {}
    for e in (report or []):
        if "layer" not in e:
            continue
        removed_by[e["layer"]] = e.get("removed_indices", [])
        surv: dict[int, int] = {}
        for ops in (e.get("merge_ops", {}) or {}).values():
            for op in ops:
                surv[int(op["removed"])] = int(op["survivor"])
        for d in (e.get("diagnostics", {}) or {}).values():
            cl, role = d.get("cluster"), d.get("role")
            if cl is None or role is None:
                continue
            reps = {c: i for i, (c, r) in enumerate(zip(cl, role))
                    if r == "survivor"}
            for i, (c, r) in enumerate(zip(cl, role)):
                if r == "absorbed" and c in reps:
                    surv[i] = reps[c]
        survivors_by[e["layer"]] = surv

    rows: list[dict] = []
    for li in range(model.n_prunable_layers()):
        mats = similarity_matrices(model, li, x, kinds)
        row: dict[str, Any] = dict(meta)
        row.update(layer=li, width=model.prunable_layer(li).weight.shape[0])
        norms = None
        if "response" in mats:
            K = mats["response"]
            row.update({f"K_{k}": v for k, v in spectrum_stats(K).items()})
            norms = np.sqrt(np.maximum(np.diag(K), 0.0))
        if "response" in mats and "cosine" in mats:
            # centered = the covariance; see the note on mean domination
            d = np.sqrt(np.maximum(np.diag(mats["response"]), 0.0))
            Phi_mean_outer = None
            Kc = _centered_gram(model, li, x)
            row.update({f"Kc_{k}": v for k, v in spectrum_stats(Kc).items()})
            del d, Phi_mean_outer
        if "cosine" in mats:
            row.update(cosine_stats(mats["cosine"]))
        for kind in ("delta_f", "cylinder"):
            if kind in mats:
                row.update({f"{kind}_{k}": v
                            for k, v in cost_stats(mats[kind]).items()})
        if "delta_f" in mats and "cylinder" in mats:
            row.update({f"agree_{k}": v for k, v in
                        criterion_agreement(mats["delta_f"],
                                            mats["cylinder"]).items()})
        if "cosine" in mats and li in removed_by and removed_by[li]:
            row.update(removed_pair_similarity(
                mats["cosine"], removed_by[li], survivors_by.get(li), norms))
        rows.append(row)
    return pd.DataFrame(rows)


def format_similarity(df: pd.DataFrame) -> str:
    """Compact text view of the similarity structure."""
    if df.empty:
        return "no layers analysed\n"
    out = ["Neuron-similarity structure of K", "=" * 96]
    out.append(f"{'layer':>5}{'H':>6}{'PRc(K)':>9}{'PRc/H':>7}{'n90/H':>7}"
               f"{'|cos| p99':>10}{'>0.8':>7}{'nnCos':>7}"
               f"{'clump dF':>10}{'clump cyl':>11}{'agree rho':>10}{'cheap ov':>9}")
    for _, r in df.iterrows():
        def g(k, fmt="{:.3f}"):
            v = r.get(k, np.nan)
            return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)
        out.append(f"{int(r.layer):>5}{int(r.width):>6}"
                   f"{g('Kc_participation_ratio','{:.1f}'):>9}"
                   f"{g('Kc_pr_fraction'):>7}{g('Kc_n90_fraction'):>7}"
                   f"{g('cos_p99'):>10}{g('frac_above_0.8'):>7}"
                   f"{g('nn_cos_median'):>7}"
                   f"{g('delta_f_clump_ratio','{:.4f}'):>10}"
                   f"{g('cylinder_clump_ratio','{:.4f}'):>11}"
                   f"{g('agree_spearman'):>10}{g('agree_cheap_overlap'):>9}")
    if "lift" in df.columns and df.lift.notna().any():
        out.append("")
        out.append("removed-pair cosine vs all-pairs null (lift > 1 = the method "
                   "removed genuinely similar units):")
        for _, r in df.iterrows():
            if not pd.isna(r.get("lift", np.nan)):
                nr = r.get("removed_norm_ratio", float("nan"))
                extra = "" if pd.isna(nr) else f"   removed ||phi|| ratio {nr:.3f}"
                out.append(f"  layer {int(r.layer)}: removed median "
                           f"{r.removed_sim_median:.3f} vs null "
                           f"{r.null_sim_median:.3f}  lift {r.lift:.2f}{extra}")
    return "\n".join(out) + "\n"
