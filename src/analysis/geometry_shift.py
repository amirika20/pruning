"""How pruning changes a layer's geometry -- before/after, parameter and function.

Exploratory battery. The per-unit tables in `pruning_detail` say WHICH units
went; this says what happened to the layer's structure as a whole. Two spaces,
because they answer different questions:

PARAMETER SPACE. Rows of B are the units' geometric codes, so B^T B is a
(d+1)x(d+1) second-moment matrix of the arrangement of activation boundaries --
same shape no matter how many units survive, which is exactly what makes
before/after comparable. Definitions follow the earlier Gram study:

    wq      sqrt(a_i) * [radius * u_i ; u_i.x0 - rho_i]   mass-weighted, box-
                                                          centered. The default:
                                                          the Ward objective is
                                                          its trace loss.
    wcov    sqrt(a_i) * [u_i ; rho_i]                      origin-centered
    cov     [u_i ; rho_i]                                  unweighted
    xi      a_i * [u_i ; rho_i]                            covector-sum rows;
                                                           its column sum is the
                                                           conserved first moment
    anchor  rho_i * u_i                                    kept only as a known
                                                           BAD arm -- dominated
                                                           by saturated units, it
                                                           false-alarms early

FUNCTION SPACE. The post-ReLU response covariance on shared inputs. Its
effective dimension is what the layer actually computes, and unlike the
parameter side it is insensitive to how the merge happened to place
hyperplanes. `captured` is the honest bottom line: the fraction of the original
responses' energy that the survivors can still linearly represent, i.e. the
empirical ||Pi_S F||^2 / ||F||^2.

EFFECTIVE DIMENSION, three ways, because they disagree usefully:

    participation_ratio  (sum L)^2 / sum L^2   quadratic; dominated by the top
                                               of the spectrum
    effective_rank       exp(entropy of L/sum L)  entropic; sensitive to the tail
    stable_rank          sum L / L_max          the cheapest, and the one the
                                                earlier study found tracked
                                                error onset

A merge that removes redundant directions should barely move the participation
ratio; one that removes genuine capacity should drop it. Divergence between PR
and effective rank means the tail changed while the top did not.

ALIGNMENT. Principal angles between the top-k right-singular subspaces before
and after: `affinity` = ||V_a^T V_b||_F^2 / k in [0, 1] (1 = same subspace), and
per-component |cos| so a rotation of one principal direction is visible rather
than averaged away.

A NOTE ON AMBIENT DIMENSION. After a joint prune, layer L's INPUT shrank because
layer L-1 lost units, so its codes live in a smaller space and are not directly
comparable. We lift the pruned weights back by scattering columns into the
surviving input positions and zero-filling the rest -- which is exact, since a
removed input contributes nothing. That needs the previous layer's removal set,
which `prune_model`'s report carries.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.registry import PrunableModel

TINY = 1e-30
B_DEFS = ("wq", "wcov", "cov", "xi", "anchor")


# ── spectra ──────────────────────────────────────────────────────────────────

def spectrum(B: np.ndarray, top_k: int = 5) -> dict[str, float]:
    """Spectral summary of B^T B, from the singular values of B."""
    if B.size == 0 or B.shape[0] == 0:
        return {}
    s = np.linalg.svd(B, compute_uv=False)
    lam = np.maximum(s ** 2, 0.0)
    tot = float(lam.sum())
    out: dict[str, float] = {
        "trace": tot,
        "fro": float(np.sqrt((lam ** 2).sum())),
        "opnorm": float(lam[0]) if lam.size else 0.0,
    }
    if tot <= TINY:
        out.update(stable_rank=0.0, participation_ratio=0.0, effective_rank=0.0)
        return out
    out["stable_rank"] = tot / max(float(lam[0]), TINY)
    out["participation_ratio"] = tot ** 2 / max(float((lam ** 2).sum()), TINY)
    p = lam / tot
    nz = p[p > TINY]
    out["effective_rank"] = float(np.exp(-(nz * np.log(nz)).sum()))
    for i in range(min(top_k, lam.size)):
        out[f"eig{i + 1}_frac"] = float(lam[i] / tot)
    return out


def right_subspace(B: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors of B, as columns [D, k]."""
    if B.shape[0] == 0:
        return np.zeros((B.shape[1], 0))
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    return Vt[: min(k, Vt.shape[0])].T


def alignment(B_before: np.ndarray, B_after: np.ndarray, k: int = 3) -> dict[str, Any]:
    """Principal-angle agreement between the two top-k right subspaces.

    `affinity` in [0, 1] is the averaged squared overlap (1 = identical
    subspace); `cos_i` are the principal cosines, largest first, so a single
    rotated direction shows up instead of being averaged away.
    """
    Va, Vb = right_subspace(B_before, k), right_subspace(B_after, k)
    kk = min(Va.shape[1], Vb.shape[1])
    if kk == 0:
        return {"affinity": float("nan")}
    M = Va[:, :kk].T @ Vb[:, :kk]
    cos = np.clip(np.linalg.svd(M, compute_uv=False), 0.0, 1.0)
    out: dict[str, Any] = {
        "affinity": float((cos ** 2).sum() / kk),
        "cos_min": float(cos.min()),
        "grassmann": float(np.linalg.norm(np.arccos(cos))),
    }
    for i, c in enumerate(cos):
        out[f"cos{i + 1}"] = float(c)
    return out


# ── code matrices ────────────────────────────────────────────────────────────

def build_B(kind: str, u: np.ndarray, rho: np.ndarray, mass: np.ndarray,
            x0: np.ndarray | None = None, radius: float = 1.0) -> np.ndarray:
    """Rows of B for one of the definitions in B_DEFS."""
    p = np.concatenate([u, rho[:, None]], axis=1)
    if kind == "anchor":
        return rho[:, None] * u
    if kind == "cov":
        return p
    if kind == "wcov":
        return np.sqrt(np.maximum(mass, 0.0))[:, None] * p
    if kind == "xi":
        return mass[:, None] * p
    if kind == "wq":
        if x0 is None:
            raise ValueError("B kind 'wq' needs the box center x0")
        qt = np.concatenate([radius * u, (u @ x0 - rho)[:, None]], axis=1)
        return np.sqrt(np.maximum(mass, 0.0))[:, None] * qt
    raise KeyError(kind)


def layer_codes(model: PrunableModel, layer_idx: int,
                kept_inputs: np.ndarray | None = None,
                d_full: int | None = None):
    """(u, rho, mass) of a layer's units, optionally lifted into the original
    input space.

    `kept_inputs` are the surviving input coordinates (i.e. the previous
    layer's survivors) and `d_full` the pre-prune input width; with both given,
    orientations are scattered back and zero-filled, which leaves ||u|| and
    every inner product with a surviving coordinate unchanged.
    """
    from src.pruning.methods.mash import extract_units

    units, ok = extract_units(model, layer_idx)
    u, rho, mass = units.u, units.rho, units.mass
    if kept_inputs is not None and d_full is not None and u.shape[1] != d_full:
        lifted = np.zeros((u.shape[0], d_full))
        lifted[:, np.asarray(kept_inputs, dtype=int)] = u
        u = lifted
    return u, rho, mass, ok


# ── function space ───────────────────────────────────────────────────────────

def responses(model: PrunableModel, layer_idx: int, x: torch.Tensor) -> np.ndarray:
    """Post-ReLU responses of a layer, [N', H] (spatial/token axes flattened)."""
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
    if z.dim() == 4:
        z = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    z = z.reshape(-1, z.shape[-1]).double().cpu().numpy()
    return np.maximum(z, 0.0)


def captured_fraction(Phi_before: np.ndarray, Phi_after: np.ndarray,
                      ridge: float = 1e-10) -> float:
    """Fraction of the ORIGINAL responses' energy the survivors can linearly
    represent: 1 - ||Phi_b - Pi_S Phi_b||^2 / ||Phi_b||^2, with S the span of
    the surviving responses plus a constant.

    This is the empirical ||Pi_S F||^2/||F||^2 -- the quantity a global repair
    optimizes. 1.0 means the surviving units can reproduce the old layer
    exactly (given a re-solve); it is the ceiling on any repair.
    """
    n = len(Phi_before)
    if n == 0 or Phi_before.size == 0 or Phi_after.size == 0:
        return float("nan")
    A = np.concatenate([Phi_after, np.ones((n, 1))], axis=1)
    G = A.T @ A / n
    lam = ridge * max(np.trace(G) / max(len(G), 1), TINY)
    coef = np.linalg.solve(G + lam * np.eye(len(G)), A.T @ Phi_before / n)
    resid = Phi_before - A @ coef
    tot = float((Phi_before ** 2).sum())
    return float(1.0 - (resid ** 2).sum() / max(tot, TINY))


def response_stats(Phi: np.ndarray, top_k: int = 5) -> dict[str, float]:
    """Effective dimension of what the layer computes. Uses the CENTERED
    response covariance, so a constant offset is not counted as a direction."""
    if Phi.size == 0:
        return {}
    C = Phi - Phi.mean(axis=0, keepdims=True)
    return spectrum(C / np.sqrt(max(len(Phi), 1)), top_k=top_k)


# ── the before/after table ───────────────────────────────────────────────────

def geometry_shift(model_before: PrunableModel, model_after: PrunableModel,
                   report: Sequence[dict], x: torch.Tensor,
                   kinds: Sequence[str] = ("wq",), top_k: int = 3,
                   include_function: bool = True, **meta: Any) -> pd.DataFrame:
    """One row per (layer, B-definition): before/after spectra, drift, alignment.

    `report` is `prune_model`'s per-layer output -- used for the removal sets
    that make the ambient-dimension lift possible. `x` are shared model inputs
    for the function-space half.
    """
    removed_by_layer = {e["layer"]: set(e.get("removed_indices", []))
                        for e in report if "layer" in e}
    widths_before = {e["layer"]: e["neurons_before"] for e in report if "layer" in e}

    rows: list[dict] = []
    for li in range(model_before.n_prunable_layers()):
        linear = isinstance(model_before.prunable_layer(li), nn.Linear)
        # inputs of this layer are the previous layer's units
        kept_inputs = d_full = None
        if li > 0 and (li - 1) in removed_by_layer:
            d_full = widths_before[li - 1]
            kept_inputs = np.array(sorted(set(range(d_full))
                                          - removed_by_layer[li - 1]), dtype=int)

        base: dict[str, Any] = dict(meta)
        base.update(layer=li,
                    width_before=widths_before.get(li, np.nan),
                    width_after=widths_before.get(li, 0)
                    - len(removed_by_layer.get(li, ())))

        fn: dict[str, Any] = {}
        if include_function:
            Pb = responses(model_before, li, x)
            Pa = responses(model_after, li, x)
            sb, sa = response_stats(Pb), response_stats(Pa)
            fn = {f"resp_{k}_before": v for k, v in sb.items()}
            fn.update({f"resp_{k}_after": v for k, v in sa.items()})
            for k in ("participation_ratio", "effective_rank", "stable_rank"):
                if k in sb and k in sa:
                    fn[f"resp_{k}_ratio"] = sa[k] / max(sb[k], TINY)
            fn["captured"] = captured_fraction(Pb, Pa)

        if not linear:
            rows.append({**base, "B": "-", **fn,
                         "note": "conv: parameter-space codes live on im2col "
                                 "patches, not lifted"})
            continue

        ub, rb, mb, _ = layer_codes(model_before, li)
        ua, ra, ma, _ = layer_codes(model_after, li, kept_inputs, d_full)
        if ua.shape[1] != ub.shape[1]:
            rows.append({**base, "B": "-", **fn,
                         "note": f"input dim {ub.shape[1]}->{ua.shape[1]} "
                                 "without a removal set to lift with"})
            continue

        # a single, comparable box: the BEFORE layer's own input box, so the
        # embedding radius is identical on both sides
        x0 = np.zeros(ub.shape[1])
        radius = 1.0
        try:
            from src.pruning.methods.mash import _layer_inputs
            Z = _layer_inputs(model_before, li, x)
            lo, hi = Z.min(axis=0), Z.max(axis=0)
            x0 = (lo + hi) / 2.0
            radius = float(np.linalg.norm((hi - lo) / 2.0))
        except Exception:
            pass

        for kind in kinds:
            Bb = build_B(kind, ub, rb, mb, x0, radius)
            Ba = build_B(kind, ua, ra, ma, x0, radius)
            sb, sa = spectrum(Bb), spectrum(Ba)
            row = {**base, "B": kind, **fn}
            row.update({f"{k}_before": v for k, v in sb.items()})
            row.update({f"{k}_after": v for k, v in sa.items()})
            for k in ("trace", "fro", "stable_rank", "participation_ratio",
                      "effective_rank"):
                if k in sb and k in sa:
                    row[f"{k}_ratio"] = sa[k] / max(sb[k], TINY)
            # Effective dimension PER SURVIVING UNIT separates the two reasons
            # it can fall: fewer units, or a genuinely flatter arrangement. A
            # per-unit ratio near 1 says the merge removed only directions the
            # layer was duplicating; below 1 says it collapsed real spread.
            nb, na = len(ub), len(ua)
            if nb and na:
                pb = sb.get("participation_ratio", np.nan) / nb
                pa = sa.get("participation_ratio", np.nan) / na
                row["pr_per_unit_before"] = pb
                row["pr_per_unit_after"] = pa
                row["pr_per_unit_ratio"] = pa / max(pb, TINY)
            row.update({f"align_{k}": v
                        for k, v in alignment(Bb, Ba, top_k).items()})
            # REALIZED first-moment drift. Not an invariant: the exactly
            # conserved quantity is the ADDITIVE covector sum with the original
            # masses, which cannot be recovered from the pruned weights -- the
            # emitted column is eta_C * sum v_i (and a global repair re-solves
            # it outright), so the realized mass is not the additive one. What
            # this measures is the drift of the realized moment, which the
            # earlier study found to be a usable stopping signal in its own
            # right. It moves even for an exactly-free removal, because a dead
            # unit still carries a covector.
            m_b = (mb[:, None] * np.concatenate([ub, rb[:, None]], 1)).sum(0)
            m_a = (ma[:, None] * np.concatenate([ua, ra[:, None]], 1)).sum(0)
            denom = max(float(np.abs(m_b).max()), TINY)
            row["m1_realized_drift"] = float(np.abs(m_a - m_b).max() / denom)
            rows.append(row)
    return pd.DataFrame(rows)


def format_geometry_shift(df: pd.DataFrame, kind: str = "wq") -> str:
    """Compact text view of the default B and the function-space columns."""
    if df.empty:
        return "no layers analysed\n"
    sub = df[df.B.isin([kind, "-"])] if "B" in df.columns else df
    out = [f"Geometry shift (B = {kind})", "=" * 92]
    out.append(f"{'layer':>5}{'H->K':>10}{'PR before':>11}{'PR after':>10}"
               f"{'PR ratio':>10}{'PR/unit r.':>11}{'align':>8}{'cos_min':>9}"
               f"{'respPR r.':>10}{'captured':>10}")
    for _, r in sub.iterrows():
        def g(k, fmt="{:.3f}"):
            v = r.get(k, np.nan)
            return "-".rjust(0) if v is None or (isinstance(v, float) and np.isnan(v)) \
                else fmt.format(v)
        out.append(
            f"{int(r.layer):>5}"
            f"{f'{int(r.width_before)}->{int(r.width_after)}':>10}"
            f"{g('participation_ratio_before'):>11}"
            f"{g('participation_ratio_after'):>10}"
            f"{g('participation_ratio_ratio'):>10}"
            f"{g('pr_per_unit_ratio'):>11}"
            f"{g('align_affinity'):>8}{g('align_cos_min'):>9}"
            f"{g('resp_participation_ratio_ratio'):>10}"
            f"{g('captured'):>10}")
    if "m1_realized_drift" in sub.columns and sub.m1_realized_drift.notna().any():
        out.append("")
        out.append(f"realized first-moment drift (a signal, NOT an invariant "
                   f"-- see the note in geometry_shift): "
                   f"max {sub.m1_realized_drift.max():.3f}")
    return "\n".join(out) + "\n"
