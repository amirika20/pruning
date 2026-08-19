#!/usr/bin/env python
"""Phase B: merge-rule x fan-out-surgery grid (design-space slots 3 and 4).

Partitions are FIXED across variants (func_matched dendrograms on the intact
model, per E7), so differences are purely the surgery. Variants:

    mean+sum          3M1 + 4F1  (the baseline used in Phases A/C)
    survivor+sum      3M2 + 4F1  (keep the loudest member's hyperplane)
    mean+proj         3M1 + 4F2  kernel least-squares column per cluster:
                      c_bar = sum_i alpha_i c_i E[h_i hbar] / E[hbar^2]
    mean+global       3M1 + 4F4  re-solve ALL surviving columns from the
                      kernel normal equations G_kk C_new = G_ko C  (closed
                      form under the matched Gaussian: same 128-sample
                      moments as selection, no extra data)
    mean+sum+bias     3M1 + 4F1 + 4F5 (expected residual folded into the
    mean+global+bias  3M1 + 4F4 + 4F5  next layer's bias, closed form)

All expectations use the layer-input moments (mu, Sigma) of the CURRENT
(partially pruned) model. Evaluation: joint equal-fraction pruning of all
layers, val-accuracy curve -> capacity at -1pt / -0.5pt.

Usage:
    python studies/gram_stability/phase_b.py \
        --run-dir studies/gram_stability/outputs/gram_mnist_mlp_... [more]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.special import ndtr

from src.config import ExperimentConfig
from src.data import build_dataset
from src.models import build_model
from src.models.mlp import MLP

from studies.gram_stability.merge import extract_units
from studies.gram_stability.functional import relu_cross, _phi
from studies.gram_stability.phase_c2 import (
    N_CALIB, apply_layer, dendrogram, evaluate, make_engine, partition_at)

STUDY_ROOT = Path(__file__).resolve().parent
VARIANTS = ["mean+sum", "survivor+sum", "mean+proj", "mean+global",
            "mean+sum+bias", "mean+global+bias"]


# ── Gaussian expectations for affine units t ~ N(m, s^2) ─────────────────────

def relu_mean(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """E[sigma(t)] = s*(phi(a) - a*PhiBar(a)), a = -m/s."""
    s = np.maximum(s, 1e-300)
    a = -m / s
    return s * (_phi(a) - a * ndtr(-a))


def cross_gram(mA, sA, mB, sB, corr) -> np.ndarray:
    """E[sigma(tA_i) sigma(tB_j)] elementwise on broadcast grids."""
    sA = np.maximum(sA, 1e-300)
    sB = np.maximum(sB, 1e-300)
    return sA * sB * relu_cross(-mA / sA, -mB / sB, np.clip(corr, -1, 1))


class LayerMoments:
    """t-statistics of a set of affine units under z ~ N(mu, Sigma)."""

    def __init__(self, U: np.ndarray, rho: np.ndarray, mu: np.ndarray, Sigma: np.ndarray):
        self.m = U @ mu - rho                    # [n]
        SU = U @ Sigma                           # [n, d]
        self.cov_half = SU                       # for cross terms
        var = np.einsum("ij,ij->i", SU, U)
        self.s = np.sqrt(np.clip(var, 0.0, None))
        self.U = U

    def corr_with(self, other: "LayerMoments") -> np.ndarray:
        cov = self.cov_half @ other.U.T          # [n, n2]
        denom = np.maximum(np.outer(self.s, other.s), 1e-300)
        return cov / denom


# ── surgery variants ─────────────────────────────────────────────────────────

def realize_variant(model: MLP, layer_idx: int, clusters: list[list[int]],
                    variant: str, mu: np.ndarray, Sigma: np.ndarray):
    """(W_rows, biases, cols, delta_next_bias) for one layer under `variant`."""
    units, ok = extract_units(model, layer_idx)
    a = units.a
    merge_rule = variant.split("+")[0]
    fanout = "global" if "global" in variant else ("proj" if "proj" in variant else "sum")
    bias_fix = variant.endswith("+bias")

    W_rows, biases, cols, members = [], [], [], []
    for mem in clusters:
        if len(mem) == 1 and not ok[mem[0]]:
            i = mem[0]
            W_rows.append(units.u[i]); biases.append(-units.rho[i])
            cols.append(units.C[i]); members.append([i])
            continue
        mem_a = np.array(mem)
        S_c = (units.alpha[mem_a, None] * units.C[mem_a]).sum(axis=0)
        if merge_rule == "survivor":
            rep = mem_a[np.argmax(a[mem_a])]
            ubar, rhobar = units.u[rep], units.rho[rep]
        else:
            S_u = (a[mem_a, None] * units.u[mem_a]).sum(axis=0)
            n = np.linalg.norm(S_u)
            if n < 1e-12:
                W_rows.append(np.zeros(units.u.shape[1])); biases.append(0.0)
                cols.append(S_c); members.append(list(mem_a)); continue
            ubar, rhobar = S_u / n, float((a[mem_a] * units.rho[mem_a]).sum()) / n
        W_rows.append(ubar); biases.append(-rhobar)
        cols.append(S_c); members.append(list(mem_a))
    W_rows, biases, cols = np.array(W_rows), np.array(biases), np.array(cols)

    need_kernel = fanout in ("proj", "global") or bias_fix
    if need_kernel:
        orig = LayerMoments(units.u, units.rho, mu, Sigma)      # unit-slope
        kept = LayerMoments(W_rows, -biases, mu, Sigma)
        # original activations h_i = alpha_i * sigma(t_i)
        alpha = units.alpha.copy()

    if fanout == "proj":
        corr = kept.corr_with(orig)
        for k, mem in enumerate(members):
            if len(mem) == 1 and not ok[mem[0]]:
                continue
            mem_a = np.array(mem)
            Kih = cross_gram(orig.m[mem_a], orig.s[mem_a],
                             kept.m[k], kept.s[k], corr[k, mem_a].T)
            Khh = float(cross_gram(kept.m[k], kept.s[k], kept.m[k], kept.s[k], 1.0))
            if Khh > 1e-30:
                w = alpha[mem_a] * Kih / Khh
                cols[k] = (w[:, None] * units.C[mem_a]).sum(axis=0)
    elif fanout == "global":
        Gkk = cross_gram(kept.m[:, None], kept.s[:, None],
                         kept.m[None, :], kept.s[None, :],
                         kept.corr_with(kept))
        Gko = cross_gram(kept.m[:, None], kept.s[:, None],
                         orig.m[None, :], orig.s[None, :],
                         kept.corr_with(orig)) * alpha[None, :]
        lam = 1e-8 * max(np.trace(Gkk) / max(len(Gkk), 1), 1e-30)
        cols = np.linalg.solve(Gkk + lam * np.eye(len(Gkk)), Gko @ units.C)

    delta_bias = None
    if bias_fix:
        Eh = alpha * relu_mean(orig.m, orig.s)
        Ehat = relu_mean(kept.m, kept.s)
        delta_bias = Eh @ units.C - Ehat @ cols
    return W_rows, biases, cols, delta_bias


def apply_cuts_variant(model: MLP, bundle, cuts: list[int], dendros, variant: str) -> MLP:
    cur = model
    for li, k in enumerate(cuts):
        pairs, idx_map, frozen = dendros[li]
        k = min(k, len(pairs))
        sub = partition_at(len(idx_map), pairs, k)
        clusters = [[int(idx_map[i]) for i in cl] for cl in sub] \
            + [[int(f)] for f in frozen]
        X = bundle.train_ds.tensors[0][:N_CALIB]
        with torch.no_grad():
            Xl = cur.net[: 2 * li](X).double().numpy()
        mu, Sigma = Xl.mean(axis=0), np.atleast_2d(np.cov(Xl.T))
        W_rows, biases, cols, dbias = realize_variant(cur, li, clusters, variant, mu, Sigma)
        cur = apply_layer(cur, li, W_rows, biases, cols)
        if dbias is not None:
            nxt = cur.net[2 * (li + 1)]
            nxt.bias.data += torch.from_numpy(dbias).float()
    return cur


# ── experiment ────────────────────────────────────────────────────────────────

def run_seed(config, run_dir: Path, seed: int, n_ckpt: int = 25) -> list[dict]:
    torch.manual_seed(seed)
    bundle = build_dataset(config.data.kind, data_seed=seed, **config.data.params)
    model = build_model(config.model.kind, bundle, **config.model.params)
    model.load_state_dict(torch.load(run_dir / "models" / f"seed_{seed}.pt",
                                     weights_only=True))
    model = model.cpu().eval()
    L = model.n_prunable_layers()
    H = [model.prunable_layer(li).out_features for li in range(L)]
    acc0 = evaluate(model, bundle)

    dendros = []
    for li in range(L):
        eng, idx_map, frozen = make_engine("func_matched", model, bundle, li)
        pairs, _ = dendrogram(eng)
        dendros.append((pairs, idx_map, frozen))

    records = []
    for variant in VARIANTS:
        # sanity: zero cuts must reproduce the model exactly
        base = apply_cuts_variant(model, bundle, [0] * L, dendros, variant)
        assert abs(evaluate(base, bundle) - acc0) < 1e-6, variant
        for f in np.linspace(0.05, 0.97, n_ckpt):
            cuts = [int(round(f * (h - 1))) for h in H]
            acc = evaluate(apply_cuts_variant(model, bundle, cuts, dendros, variant), bundle)
            records.append({"seed": seed, "variant": variant,
                            "frac_removed": sum(cuts) / sum(H),
                            "val_acc": acc, "acc0": acc0})
    return records


def capacity(g: pd.DataFrame, drop: float) -> float:
    ok = g[g.val_acc >= g.acc0.iloc[0] - drop]
    return float(ok.frac_removed.max()) if len(ok) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", nargs="+", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    for rd in args.run_dir:
        rd = Path(rd)
        config = ExperimentConfig.from_yaml(rd / "config.yaml")
        out_dir = STUDY_ROOT / "outputs" / \
            f"b_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True)
        records = []
        for seed in config.seeds:
            logging.info(f"[{config.name}] seed {seed}")
            records.extend(run_seed(config, rd, seed))
        df = pd.DataFrame(records)
        df.to_csv(out_dir / "curves.csv", index=False)

        lines = [f"Phase B surgery grid: {config.name}", "=" * 60, "",
                 "joint equal-fraction pruning, func_matched partitions fixed",
                 "capacity = max fraction of all hidden units removed (median over seeds)",
                 "", f"  {'variant':<20}{'cap@-0.5pt':>12}{'cap@-1pt':>12}{'cap@-2pt':>12}"]
        for variant in VARIANTS:
            sub = df[df.variant == variant]
            caps = {d: np.median([capacity(g, d) for _, g in sub.groupby("seed")])
                    for d in (0.005, 0.01, 0.02)}
            lines.append(f"  {variant:<20}{caps[0.005]:>12.3f}{caps[0.01]:>12.3f}"
                         f"{caps[0.02]:>12.3f}")
        report = "\n".join(lines)
        (out_dir / "report.txt").write_text(report)
        logging.info("\n" + report)


if __name__ == "__main__":
    main()
