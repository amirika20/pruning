#!/usr/bin/env python
"""Do our method and OSSCAR agree on WHICH units are expendable?

Per layer and removal fraction, compare:
  ours    = union over multi-member clusters of (members minus the loudest
            one) -- the units our dendrogram absorbs at cut k
  osscar  = OSSCAR's removal set at n_remove = k (same layer, intact model,
            same 128-image calibration)
Overlap = |ours & osscar| / k, against the chance baseline k/H.

High overlap => both identify the same expendable units and the capacity gap
is surgery. Low overlap => the methods disagree about expendability itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.data import build_dataset
from src.models import build_model
from src.pruning.methods.osscar import OSSCAR
from src.pruning.registry import PruneContext

from studies.gram_stability.functional import GaussianMeasureMerge
from studies.gram_stability.phase_c2 import dendrogram, make_engine, partition_at
from studies.gram_stability.phase_d_cnn import conv_units, sample_patches

STUDY_ROOT = Path(__file__).resolve().parent
FRACS = [0.25, 0.5, 0.75]
N_CALIB = 128


def ours_removed(units_a: np.ndarray, idx_map, frozen, pairs, k: int) -> set[int]:
    sub = partition_at(len(idx_map), pairs, k)
    removed = set()
    for cl in sub:
        if len(cl) > 1:
            full = [int(idx_map[i]) for i in cl]
            keep = full[int(np.argmax(units_a[full]))]
            removed.update(i for i in full if i != keep)
    return removed


def analyze(kind: str, config_path: str, ckpt: Path, seed: int) -> list[str]:
    config = ExperimentConfig.from_yaml(Path(config_path))
    torch.manual_seed(seed)
    bundle = build_dataset(config.data.kind, data_seed=seed, **config.data.params)
    model = build_model(config.model.kind, bundle, **config.model.params)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model = model.cpu().eval()
    calib = bundle.train_ds.tensors[0][:N_CALIB]

    lines = []
    for li in range(model.n_prunable_layers()):
        if kind == "cnn":
            units, ok, _, _ = conv_units(model, li)
            idx_map, frozen = np.flatnonzero(ok), np.flatnonzero(~ok)
            P = sample_patches(model, li, calib, 4096, seed=li)
            eng = GaussianMeasureMerge(units.subset(idx_map), P.min(0), P.max(0),
                                       P.mean(0), np.atleast_2d(np.cov(P.T)))
            H = model.prunable_layer(li).out_channels
        else:
            eng, idx_map, frozen = make_engine("func_matched", model, bundle, li)
            units = eng.orig
            H = model.prunable_layer(li).out_features
        pairs, _ = dendrogram(eng)
        a_full = np.zeros(H)
        a_full[idx_map] = units.a if kind == "mlp" else units.a[idx_map] \
            if len(units.a) == H else units.a
        # units for cnn path is already full-size; for mlp eng.orig is subset
        if kind == "cnn":
            a_full = units.a
        for f in FRACS:
            k = min(int(round(f * (H - 1))), len(pairs))
            mine = ours_removed(a_full, idx_map, frozen, pairs, k)
            ctx = PruneContext(train_inputs=calib, bundle=bundle,
                               device=torch.device("cpu"))
            dec = OSSCAR(n_remove=int(len(mine)) or 1).select(model, li, ctx)
            theirs = set(dec.remove)
            n = max(len(mine), 1)
            inter = len(mine & theirs)
            lines.append(f"  layer {li} (H={H:3d}) f={f:.2f}: removed {len(mine):3d}, "
                         f"overlap {inter / n:5.2f}  (chance {len(theirs) / H:.2f})")
    return lines


def main() -> None:
    ckpts = STUDY_ROOT / "outputs" / "d_cnn_ckpts"
    jobs = [
        ("cnn", "configs/experiments/mnist/cnn/mnist_pixel_cnn.yaml",
         ckpts / "mnist_pixel_cnn_seed_0.pt"),
        ("cnn", "configs/experiments/fashion_mnist/cnn/fashion_mnist_pixel_cnn.yaml",
         ckpts / "fashion_mnist_pixel_cnn_seed_0.pt"),
        ("mlp", "studies/gram_stability/outputs/gram_mnist_mlp_20260817_150158/config.yaml",
         Path("studies/gram_stability/outputs/gram_mnist_mlp_20260817_150158/models/seed_0.pt")),
    ]
    for kind, cfg, ck in jobs:
        print(f"\n=== {Path(cfg).stem} ({kind}) seed 0")
        for line in analyze(kind, cfg, ck, seed=0):
            print(line)


if __name__ == "__main__":
    main()
