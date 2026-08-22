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
  removed unit and its survivor, against a PERMUTATION null -- random removal
  sets of the same size scored by the same statistic. A method genuinely
  exploiting redundancy lifts above 1; one that is not sits at 1. The null has to
  be drawn the same way as the observation, which was got wrong here at first:
  comparing a maximum-over-survivors against a median-over-all-pairs made a
  random selection score ~2.1, so every number below that was worse than chance
  while looking like a lift.

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

import logging
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
    except Exception as exc:                            # noqa: BLE001
        # Without masses the delta_f and cylinder matrices are simply absent, so
        # their columns vanish from the table and a broken layer reads as "not
        # applicable". Warn, so the difference is visible.
        logging.warning(
            f"similarity layer {layer_idx}: no unit masses "
            f"({type(exc).__name__}: {exc}); delta_f/cylinder columns omitted")

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


def _removed_stat(C: np.ndarray, removed: np.ndarray, kept: np.ndarray,
                  paired: dict[int, int] | None,
                  rng: np.random.Generator | None = None) -> float:
    """Median similarity of each removed unit to the survivor it is scored against.

    `paired` given  -> the named survivor (a merge), or a RANDOM survivor when
                       `rng` is set, which is the matched null for that case.
    `paired` None    -> the most similar surviving unit, for both the observation
                       and the null.

    The MODE MUST BE THE SAME on both sides. Deciding it per unit was the second
    incarnation of this bug: a deletion arm has an empty survivor map, so the
    observation fell through to max-over-survivors while the null still drew a
    single random survivor, and a random removal set scored 1.9-2.4 instead of 1.
    """
    vals = []
    for i in removed:
        i = int(i)
        if paired is not None:
            j = int(rng.choice(kept)) if rng is not None else paired[i]
            vals.append(abs(float(C[i, j])))
        else:
            vals.append(float(np.abs(C[i, kept]).max()))
    return float(np.median(vals)) if vals else float("nan")


def removed_pair_similarity(C: np.ndarray, removed: Sequence[int],
                            survivors: dict[int, int] | None = None,
                            norms: np.ndarray | None = None,
                            n_null: int = 64, seed: int = 0
                            ) -> dict[str, float]:
    """Similarity of removed units to their survivor, against a PERMUTATION null.

    The null is random removal sets of the same size, scored by the SAME
    statistic -- which is the whole point of a null and was got wrong at first
    here. The observed statistic is a maximum over survivors (or a specific
    named pair); the original null was the median over all pairs. Those are
    different quantities, so a purely random removal set already scored ~2.1 and
    every real number had to be read against an invisible, much-larger-than-one
    baseline. Anything below ~2 looked like a lift and was in fact worse than
    chance.

    With the null drawn the same way, `lift` is ~1.0 for a random selection by
    construction, and above 1 genuinely means "removed units more alike to their
    survivors than chance".

    IT IS A ONE-SIDED DETECTOR, and worth knowing before reading it as a score.
    The statistic is a maximum over survivors, so it SATURATES: on a layer whose
    clumps are larger than the removal count, both the observation and the null
    find a near-identical survivor and both sit at 1. Measured on a synthetic
    four-clump layer removing 12 of 48 --

        random                    lift 1.000
        half of every clump       lift 1.000   <- the ideal selection, and
                                                  indistinguishable from random
        one entire clump removed  lift 0.079
        two entire clumps         lift 0.075

    -- so lift well below 1 means the method stranded units with no similar
    survivor, while lift ~1 means only "not worse than chance". Do not read ~1
    as evidence of exploiting redundancy.

    Cosine is also blind to MAGNITUDE, so read `removed_norm_ratio` alongside: a
    method harvesting barely-active units shows a lift below 1 because two
    near-zero responses cannot correlate, and that is correct behaviour rather
    than a failure -- the low norm ratio is what distinguishes the two cases.
    """
    H = C.shape[0]
    removed = np.array(sorted({int(i) for i in removed}), dtype=int)
    if not len(removed) or len(removed) >= H:
        return {}
    kept = np.array(sorted(set(range(H)) - set(removed.tolist())), dtype=int)

    # One decision, used on both sides: pair only when every removed unit has a
    # named survivor. A partial map would mix the two statistics again.
    paired = (survivors if survivors and all(int(i) in survivors for i in removed)
              else None)
    observed = _removed_stat(C, removed, kept, paired)
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_null):
        r = rng.choice(H, size=len(removed), replace=False)
        k = np.array(sorted(set(range(H)) - set(r.tolist())), dtype=int)
        # For the paired case the null keeps the pairing but randomizes the
        # partner; for the unpaired case it is the same max-over-survivors on a
        # random removal set.
        nulls.append(_removed_stat(C, r, k,
                                   {int(i): 0 for i in r} if paired else None,
                                   rng=rng if paired else None))
    null = float(np.median(nulls))

    out = {"paired": bool(paired),
           "removed_sim_median": observed,
           "null_sim_median": null,
           "null_sim_p90": float(np.quantile(nulls, 0.9)),
           "lift": observed / max(null, TINY)}
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


# ── self-tests ───────────────────────────────────────────────────────────────

def _selftest() -> None:  # pragma: no cover
    rng = np.random.default_rng(0)

    # A synthetic layer with four tight clumps: within-clump cosine ~1,
    # across-clump ~0, so the expected answers are known.
    H = 48
    base = rng.normal(size=(H, 6)) * 0.15
    for k in range(4):
        base[k * 12:(k + 1) * 12] += 6 * np.eye(6)[k]
    u = base / np.linalg.norm(base, axis=1, keepdims=True)
    C = np.clip(u @ u.T, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)

    # 1. structural invariants of a cosine matrix
    assert np.allclose(C, C.T) and np.allclose(np.diag(C), 1.0)

    # 2. THE NULL MUST BE DRAWN THE SAME WAY AS THE OBSERVATION. Getting this
    # wrong twice is why it is pinned here: comparing a maximum-over-survivors
    # against a median-over-all-pairs put a random selection at ~2.1, and
    # deciding the mode per unit left deletion arms mismatched at ~1.9-2.4.
    lifts = [removed_pair_similarity(
        C, rng.choice(H, 12, replace=False).tolist())["lift"] for _ in range(40)]
    assert 0.9 < float(np.median(lifts)) < 1.1, \
        f"unpaired null mis-specified: random removal lifts to {np.median(lifts):.2f}"

    paired = {i: i - 6 for i in range(6, 12)}          # same-clump partners
    r = removed_pair_similarity(C, list(paired), paired)
    assert r["paired"] and r["lift"] > 5, f"paired lift should be large, got {r}"
    cross = {i: i + 20 for i in range(6)}              # cross-clump partners
    assert removed_pair_similarity(C, list(cross), cross)["lift"] < 0.6

    # 3. it is a ONE-SIDED detector: stranding units is caught, an ideal
    # selection is NOT distinguishable from chance (the statistic saturates).
    whole = removed_pair_similarity(C, list(range(12)))["lift"]
    ideal = removed_pair_similarity(
        C, [i for k in range(4) for i in range(k * 12 + 6, k * 12 + 12)])["lift"]
    assert whole < 0.2, f"a whole clump removed should read far below 1, got {whole}"
    assert 0.9 < ideal < 1.1, f"saturation expected at ~1, got {ideal}"

    # 4. spectra: known answers, and scale invariance
    st = spectrum_stats(np.eye(8))
    assert abs(st["participation_ratio"] - 8) < 1e-9
    assert abs(st["pr_fraction"] - 1.0) < 1e-9
    K = u @ u.T
    assert abs(spectrum_stats(K)["participation_ratio"]
               - spectrum_stats(9.7 * K)["participation_ratio"]) < 1e-6

    # 5. a cost matrix agrees perfectly with itself
    D = ward_weight(np.abs(rng.normal(size=H)) + 0.1) * (1.0 - C)
    ag = criterion_agreement(D, D)
    assert abs(ag["spearman"] - 1) < 1e-9 and abs(ag["cheap_overlap"] - 1) < 1e-9

    print("similarity.py self-tests passed:")
    print("  cosine matrix symmetric with unit diagonal")
    print(f"  permutation null matched to the observation "
          f"(random lift {np.median(lifts):.3f}, paired {r['lift']:.1f}, "
          f"cross-clump {removed_pair_similarity(C, list(cross), cross)['lift']:.2f})")
    print(f"  one-sided: whole clump {whole:.3f}, ideal selection {ideal:.3f} "
          f"(saturates at 1)")
    print("  PR(identity)=n, PR scale-invariant, self-agreement exact")


if __name__ == "__main__":
    _selftest()
