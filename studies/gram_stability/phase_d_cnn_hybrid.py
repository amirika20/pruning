#!/usr/bin/env python
"""The E11-motivated hybrid on CNNs: OUR selection + OSSCAR's weight update.

Per layer (sequential): the func_matched dendrogram at cut k defines the
removal set (each cluster keeps its loudest member -- medoid-style; survivors
keep their ORIGINAL filters and BN), then the consumer weights are re-solved
exactly as OSSCAR does it (damped XtX from the same 128 calibration images,
exact least squares on the support), and the removed channels are pruned.

Compares against the stored E10 arms at the same fractions/seeds.

Usage:
    python studies/gram_stability/phase_d_cnn_hybrid.py
"""

from __future__ import annotations

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
from src.pruning.methods.osscar import OSSCAR, _expand, _solve_support
from src.pruning.registry import PruneContext

from studies.gram_stability.functional import GaussianMeasureMerge
from studies.gram_stability.phase_c2 import dendrogram, partition_at
from studies.gram_stability.phase_d_cnn import (
    N_CALIB_IMGS, SEEDS, capacity, conv_units, get_model, sample_patches, val_acc)
from studies.gram_stability.overlap_analysis import ours_removed

STUDY_ROOT = Path(__file__).resolve().parent
CONFIGS = [
    "configs/experiments/mnist/cnn/mnist_pixel_cnn.yaml",
    "configs/experiments/fashion_mnist/cnn/fashion_mnist_pixel_cnn.yaml",
]


def osscar_update(model, bundle, li: int, removed: list[int], calib: torch.Tensor,
                  swap: bool = False):
    """OSSCAR's exact repair for a GIVEN removal set (damped XtX over the
    consumer's calibration inputs, least squares on the support); with
    swap=True, the removal set is first refined by OSSCAR's local search
    (equal-count swaps, accept on objective improvement)."""
    helper = OSSCAR(n_remove=1)  # only used for XtX collector + conventions
    ctx = PruneContext(train_inputs=calib, bundle=bundle, device=torch.device("cpu"))
    XtX = helper._collect_XtX(model, li, ctx)
    R = XtX.shape[0]
    H = model.prunable_layer(li).out_channels
    kk = R // H
    damp = helper.lambda2 * float(torch.diag(XtX).mean())
    XtX = XtX + damp * torch.eye(R, dtype=XtX.dtype)
    B = model.outgoing_weights(li).to(XtX.dtype)
    XtY = XtX @ B
    mask = torch.zeros(H, dtype=torch.bool)
    mask[list(removed)] = True
    if swap:
        mask = helper._local_search_stage(XtX, XtY, mask, kk)
    W_new = _solve_support(XtX, XtY, _expand(mask, kk))
    model = model.set_outgoing_weights(li, W_new)
    return model.prune_layer(li, sorted(mask.nonzero(as_tuple=True)[0].tolist()))


def apply_cuts_hybrid_v2(model, bundle, cuts, dendros, calib: torch.Tensor,
                         swap: bool = False):
    cur = model
    for li, k in enumerate(cuts):
        pairs, idx_map, frozen, a_full = dendros[li]
        k = min(k, len(pairs))
        removed = ours_removed(a_full, idx_map, frozen, pairs, k)
        if not removed:
            continue
        cur = osscar_update(cur, bundle, li, sorted(removed), calib, swap=swap)
    return cur


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    ckpt_dir = STUDY_ROOT / "outputs" / "d_cnn_ckpts"
    for cpath in CONFIGS:
        config = ExperimentConfig.from_yaml(Path(cpath))
        records = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            bundle = build_dataset(config.data.kind, data_seed=seed, **config.data.params)
            model = get_model(config, bundle, seed, ckpt_dir)
            L = model.n_prunable_layers()
            Hs = [model.prunable_layer(li).out_channels for li in range(L)]
            acc0 = val_acc(model, bundle)
            calib = bundle.train_ds.tensors[0][:N_CALIB_IMGS]

            dendros = []
            for li in range(L):
                units, ok, _, _ = conv_units(model, li)
                idx_map, frozen = np.flatnonzero(ok), np.flatnonzero(~ok)
                P = sample_patches(model, li, calib, 4096, seed=li)
                eng = GaussianMeasureMerge(units.subset(idx_map), P.min(0), P.max(0),
                                           P.mean(0), np.atleast_2d(np.cov(P.T)))
                pairs, _ = dendrogram(eng)
                dendros.append((pairs, idx_map, frozen, units.a))

            from studies.gram_stability.phase_d_cnn import apply_cuts_osscar
            for f in np.linspace(0.35, 0.80, 14):
                cuts = [int(round(f * (h - 1))) for h in Hs]
                frac = sum(cuts) / sum(Hs)
                arms = {
                    "hybrid": lambda: apply_cuts_hybrid_v2(model, bundle, cuts, dendros, calib),
                    "hybrid_swap": lambda: apply_cuts_hybrid_v2(model, bundle, cuts, dendros, calib, swap=True),
                    "osscar128": lambda: apply_cuts_osscar(model, bundle, cuts, calib),
                }
                for arm, fn in arms.items():
                    records.append({"seed": seed, "arm": arm, "frac_removed": frac,
                                    "val_acc": val_acc(fn(), bundle), "acc0": acc0})
                logging.info(f"[{config.name}] seed {seed} f={f:.2f}: " + "  ".join(
                    f"{r['arm']}={r['val_acc']:.3f}" for r in records[-3:]))
        df = pd.DataFrame(records)
        lines = []
        for arm in ["hybrid", "hybrid_swap", "osscar128"]:
            sub = df[df.arm == arm]
            caps = {d: np.median([capacity(g, d) for _, g in sub.groupby("seed")])
                    for d in (0.005, 0.01, 0.02)}
            lines.append(f"  {arm:<12} cap@-0.5pt={caps[0.005]:.3f} "
                         f"cap@-1pt={caps[0.01]:.3f} cap@-2pt={caps[0.02]:.3f}")
        out = f"fine-grid gap test on {config.name} (NOTE: caps floor at 0.35 = grid start):\n" + "\n".join(lines)
        logging.info("\n" + out)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        (STUDY_ROOT / "outputs" / f"dcnn_hybrid_{config.name}_{stamp}.txt").write_text(
            out + "\n")
        df.to_csv(STUDY_ROOT / "outputs" / f"dcnn_hybrid_{config.name}_{stamp}.csv",
                  index=False)


if __name__ == "__main__":
    main()
