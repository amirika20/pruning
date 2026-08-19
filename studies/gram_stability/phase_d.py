#!/usr/bin/env python
"""Phase D: our pipeline vs OSSCAR on the MLP configs, matched widths AND
matched data budgets. All arms prune every layer jointly (equal fractions,
sequential layer order) and are scored on the same val split.

    ours          func_matched selection + mean merge + KERNEL global repair
                  (4F4); data budget = 128 unlabeled inputs (moments only)
    hybrid128     our partitions/merged units + EMPIRICAL ridge LS repair on
                  the same 128 inputs (parametric vs empirical, structure fixed)
    osscar128     OSSCAR (fastprune + local search, official port) with the
                  same 128 calibration inputs
    osscar_full   OSSCAR with the full training set (its ceiling)

Usage:
    python studies/gram_stability/phase_d.py \
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

from src.config import ExperimentConfig
from src.data import build_dataset
from src.models import build_model
from src.models.mlp import MLP
from src.pruning.methods.osscar import OSSCAR
from src.pruning.registry import PruneContext

from studies.gram_stability.merge import extract_units
from studies.gram_stability.phase_c2 import (
    N_CALIB, apply_layer, dendrogram, evaluate, make_engine, partition_at)
from studies.gram_stability.phase_b import apply_cuts_variant, realize_variant

STUDY_ROOT = Path(__file__).resolve().parent


# ── hybrid: our structure + empirical ridge LS repair ────────────────────────

def apply_cuts_hybrid(model: MLP, bundle, cuts, dendros, n_calib: int) -> MLP:
    Xc = bundle.train_ds.tensors[0][:n_calib]
    cur = model
    for li, k in enumerate(cuts):
        pairs, idx_map, frozen = dendros[li]
        k = min(k, len(pairs))
        sub = partition_at(len(idx_map), pairs, k)
        clusters = [[int(idx_map[i]) for i in cl] for cl in sub] \
            + [[int(f)] for f in frozen]
        with torch.no_grad():
            Xl = cur.net[: 2 * li](Xc).double().numpy()
        mu, Sigma = Xl.mean(axis=0), np.atleast_2d(np.cov(Xl.T))
        # mean-merge structure (sum columns as init; repair below replaces them)
        W_rows, biases, cols, _ = realize_variant(cur, li, clusters, "mean+sum", mu, Sigma)
        units, _ = extract_units(cur, li)
        lin = cur.net[2 * li]
        W = lin.weight.data.double().numpy()
        b = lin.bias.data.double().numpy()
        H_orig = np.maximum(Xl @ W.T + b, 0.0)                 # [N, H]
        H_kept = np.maximum(Xl @ W_rows.T + biases, 0.0)       # [N, K]
        target = H_orig @ units.C                              # [N, m] (C raw cols)
        Gkk = H_kept.T @ H_kept
        lam = 1e-6 * max(np.trace(Gkk) / max(len(Gkk), 1), 1e-30)
        cols = np.linalg.solve(Gkk + lam * np.eye(len(Gkk)), H_kept.T @ target)
        cur = apply_layer(cur, li, W_rows, biases, cols)
    return cur


# ── OSSCAR arm: sequential per layer at the same cut vector ──────────────────

def apply_cuts_osscar(model: MLP, bundle, cuts, calib_x: torch.Tensor) -> MLP:
    cur = model
    for li, k in enumerate(cuts):
        H = cur.prunable_layer(li).out_features
        k = min(k, H - 1)
        if k <= 0:
            continue
        ctx = PruneContext(train_inputs=calib_x, bundle=bundle,
                           device=torch.device("cpu"))
        dec = OSSCAR(n_remove=int(k)).select(cur, li, ctx)
        cur = cur.set_outgoing_weights(li, dec.new_outgoing)
        cur = cur.prune_layer(li, dec.remove)
    return cur


# ── experiment ────────────────────────────────────────────────────────────────

def run_seed(config, run_dir: Path, seed: int, n_ckpt: int = 12) -> list[dict]:
    torch.manual_seed(seed)
    bundle = build_dataset(config.data.kind, data_seed=seed, **config.data.params)
    model = build_model(config.model.kind, bundle, **config.model.params)
    model.load_state_dict(torch.load(run_dir / "models" / f"seed_{seed}.pt",
                                     weights_only=True))
    model = model.cpu().eval()
    L = model.n_prunable_layers()
    Hs = [model.prunable_layer(li).out_features for li in range(L)]
    acc0 = evaluate(model, bundle)
    calib128 = bundle.train_ds.tensors[0][:N_CALIB]
    calib_full = bundle.train_ds.tensors[0]

    dendros = []
    for li in range(L):
        eng, idx_map, frozen = make_engine("func_matched", model, bundle, li)
        pairs, _ = dendrogram(eng)
        dendros.append((pairs, idx_map, frozen))

    records = []
    for f in np.linspace(0.1, 0.95, n_ckpt):
        cuts = [int(round(f * (h - 1))) for h in Hs]
        frac = sum(cuts) / sum(Hs)
        arms = {
            "ours": lambda: apply_cuts_variant(model, bundle, cuts, dendros, "mean+global"),
            "hybrid128": lambda: apply_cuts_hybrid(model, bundle, cuts, dendros, N_CALIB),
            "osscar128": lambda: apply_cuts_osscar(model, bundle, cuts, calib128),
            "osscar_full": lambda: apply_cuts_osscar(model, bundle, cuts, calib_full),
        }
        for arm, fn in arms.items():
            acc = evaluate(fn(), bundle)
            records.append({"seed": seed, "arm": arm, "frac_removed": frac,
                            "val_acc": acc, "acc0": acc0})
        logging.info(f"  seed {seed} f={f:.2f}: " + "  ".join(
            f"{r['arm']}={r['val_acc']:.3f}" for r in records[-4:]))
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
            f"d_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True)
        records = []
        for seed in config.seeds:
            logging.info(f"[{config.name}] seed {seed}")
            records.extend(run_seed(config, rd, seed))
        df = pd.DataFrame(records)
        df.to_csv(out_dir / "curves.csv", index=False)

        lines = [f"Phase D: ours vs OSSCAR: {config.name}", "=" * 60, "",
                 "joint equal-fraction pruning of all layers, sequential order",
                 "capacity = max fraction of all hidden units removed (median over seeds)",
                 "data budgets: ours/hybrid/osscar128 = 128 unlabeled inputs;",
                 f"osscar_full = full train set ({config.data.params.get('n_samples', '?')})",
                 "", f"  {'arm':<14}{'cap@-0.5pt':>12}{'cap@-1pt':>12}{'cap@-2pt':>12}"]
        for arm in ["ours", "hybrid128", "osscar128", "osscar_full"]:
            sub = df[df.arm == arm]
            caps = {d: np.median([capacity(g, d) for _, g in sub.groupby("seed")])
                    for d in (0.005, 0.01, 0.02)}
            lines.append(f"  {arm:<14}{caps[0.005]:>12.3f}{caps[0.01]:>12.3f}"
                         f"{caps[0.02]:>12.3f}")
        report = "\n".join(lines)
        (out_dir / "report.txt").write_text(report)
        logging.info("\n" + report)


if __name__ == "__main__":
    main()
