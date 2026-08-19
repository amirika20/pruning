#!/usr/bin/env python
"""Phase C2: cross-layer budget allocation (design-space 2S9) on MLPs.

All layers are pruned JOINTLY; strategies decide how a global budget spreads
across layers, traced as accuracy-vs-total-remaining-width curves:

    equal_frac    same fraction merged in every layer (baseline)
    greedy_norm   globally cheapest next merge, costs normalized by the
                  layer's output scale squared (the 2S7 currency)
    greedy_raw    same without normalization (tests the scaling decision)
    grid_oracle   envelope of a full per-layer-fraction grid (reference)
    sequential    greedy_norm's cut vectors, but each layer's dendrogram is
                  RECOMPUTED on the already-pruned model (composition test)

Joint application semantics (both modes): each layer's merge PARTITION comes
from its dendrogram; the realization (covector sums -> merged rows/columns)
is always computed in the CURRENT model's coordinates, layer by layer in
order -- exact regardless of how earlier layers changed the geometry.

Usage:
    python studies/gram_stability/phase_c2.py \
        --run-dir studies/gram_stability/outputs/gram_mnist_mlp_... [more]
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.data import build_dataset
from src.models import build_model
from src.models.mlp import MLP

from studies.gram_stability.merge import IterativeMerge, extract_units, input_boxes
from studies.gram_stability.functional import GaussianMeasureMerge

STUDY_ROOT = Path(__file__).resolve().parent
N_CALIB = 128
ACC_DROP = 0.01


# ── engines / dendrograms ────────────────────────────────────────────────────

def make_engine(arm: str, model: MLP, bundle, layer_idx: int):
    """(engine over the mergeable units, their full-layer indices, frozen
    indices). Zero-norm units (possible in already-pruned models) are frozen
    out of merging and carried as exact singletons."""
    units, ok = extract_units(model, layer_idx)
    idx_map = np.flatnonzero(ok)
    frozen = np.flatnonzero(~ok)
    sub = units.subset(idx_map)
    x_train = bundle.train_ds.tensors[0].double().numpy()
    lo, hi = input_boxes(model, x_train)[layer_idx]
    if arm == "ward":
        return IterativeMerge(sub, lo, hi), idx_map, frozen
    X = bundle.train_ds.tensors[0][:N_CALIB]
    with torch.no_grad():
        Xl = model.net[: 2 * layer_idx](X).double().numpy()
    eng = GaussianMeasureMerge(sub, lo, hi, Xl.mean(axis=0),
                               np.atleast_2d(np.cov(Xl.T)))
    return eng, idx_map, frozen


def dendrogram(engine) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Full merge sequence: [(survivor, removed)...], per-step costs."""
    pairs, costs = [], []
    while engine.n_active > 1:
        rec = engine.step()
        pairs.append((rec["survivor"], rec["removed"]))
        costs.append(rec["ward_cost"])
    return pairs, np.array(costs)


def partition_at(H: int, pairs: list[tuple[int, int]], k: int) -> list[list[int]]:
    parent = list(range(H))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs[:k]:
        parent[find(j)] = find(i)
    clusters: dict[int, list[int]] = {}
    for x in range(H):
        clusters.setdefault(find(x), []).append(x)
    return list(clusters.values())


def realize_partition(model: MLP, layer_idx: int, clusters: list[list[int]]):
    """Merged (rows, biases, cols) computed in the CURRENT model's coords.
    Zero-norm singletons are realized exactly (raw row/bias/column, so a
    constant sigma(b) unit keeps its contribution)."""
    units, ok = extract_units(model, layer_idx)
    a = units.a
    W_rows, biases, cols = [], [], []
    for mem in clusters:
        if len(mem) == 1 and not ok[mem[0]]:
            i = mem[0]
            W_rows.append(units.u[i])          # raw row (safe-norm 1 fold)
            biases.append(-units.rho[i])       # raw bias
            cols.append(units.C[i])            # raw column, unscaled
            continue
        mem = np.array(mem)
        S_u = (a[mem, None] * units.u[mem]).sum(axis=0)
        S_rho = float((a[mem] * units.rho[mem]).sum())
        S_c = (units.alpha[mem, None] * units.C[mem]).sum(axis=0)
        n = np.linalg.norm(S_u)
        if n < 1e-12:
            W_rows.append(np.zeros_like(S_u)); biases.append(0.0); cols.append(S_c)
        else:
            W_rows.append(S_u / n); biases.append(-S_rho / n); cols.append(S_c)
    return np.array(W_rows), np.array(biases), np.array(cols)


def apply_layer(model: MLP, layer_idx: int, W_rows, biases, cols) -> MLP:
    pruned = copy.deepcopy(model)
    old = pruned.net[2 * layer_idx]
    nxt = pruned.net[2 * (layer_idx + 1)]
    K = W_rows.shape[0]
    new = nn.Linear(old.in_features, K)
    new.weight.data = torch.from_numpy(W_rows).float()
    new.bias.data = torch.from_numpy(biases).float()
    new_nxt = nn.Linear(K, nxt.out_features)
    new_nxt.weight.data = torch.from_numpy(cols).float().t().contiguous()
    new_nxt.bias.data = nxt.bias.data.clone()
    pruned.net[2 * layer_idx] = new
    pruned.net[2 * (layer_idx + 1)] = new_nxt
    return pruned


def apply_cuts(model: MLP, bundle, cuts: list[int], arm: str,
               dendros: list[tuple] | None) -> MLP:
    """dendros = [(pairs, idx_map, frozen), ...] -> independent-joint
    (intact-model partitions); None -> sequential (recompute each layer's
    dendrogram on the current, already-pruned model)."""
    cur = model
    for li, k in enumerate(cuts):
        if dendros is not None:
            pairs, idx_map, frozen = dendros[li]
        else:
            eng, idx_map, frozen = make_engine(arm, cur, bundle, li)
            pairs, _ = dendrogram(eng)
        k = min(k, len(pairs))
        sub_clusters = partition_at(len(idx_map), pairs, k)
        clusters = [[int(idx_map[i]) for i in cl] for cl in sub_clusters] \
            + [[int(f)] for f in frozen]
        cur = apply_layer(cur, li, *realize_partition(cur, li, clusters))
    return cur


@torch.no_grad()
def evaluate(model: MLP, bundle) -> float:
    X, y = bundle.val_ds.tensors
    out = model(X)
    return float((out.argmax(dim=1) == y).float().mean()) if bundle.task == "multiclass" \
        else -float(F.mse_loss(out, y))


# ── strategies ────────────────────────────────────────────────────────────────

def greedy_order(costs: list[np.ndarray], scales: list[float] | None) -> list[int]:
    """Layer index of each global merge event (k-way merge on next costs)."""
    ptr = [0] * len(costs)
    order = []
    norm = [(s ** 2 if scales else 1.0) for s in (scales or [1.0] * len(costs))]
    while True:
        cands = [(costs[l][ptr[l]] / norm[l], l) for l in range(len(costs))
                 if ptr[l] < len(costs[l])]
        if not cands:
            return order
        _, l = min(cands)
        order.append(l)
        ptr[l] += 1


def cuts_from_order(order: list[int], n_layers: int, upto: int) -> list[int]:
    cuts = [0] * n_layers
    for l in order[:upto]:
        cuts[l] += 1
    return cuts


def run_seed(config, run_dir: Path, seed: int, arm: str, n_ckpt: int = 30) -> list[dict]:
    torch.manual_seed(seed)
    bundle = build_dataset(config.data.kind, data_seed=seed, **config.data.params)
    model = build_model(config.model.kind, bundle, **config.model.params)
    model.load_state_dict(torch.load(run_dir / "models" / f"seed_{seed}.pt",
                                     weights_only=True))
    model = model.cpu().eval()
    L = model.n_prunable_layers()
    H = [model.prunable_layer(li).out_features for li in range(L)]
    total = sum(H)
    acc0 = evaluate(model, bundle)

    # intact-model dendrograms + output scales
    dendros, cost_lists, scales = [], [], []
    for li in range(L):
        eng, idx_map, frozen = make_engine(arm, model, bundle, li)
        pairs, costs = dendrogram(eng)
        dendros.append((pairs, idx_map, frozen))
        cost_lists.append(costs)
        with torch.no_grad():
            Xl = model.net[: 2 * li](bundle.val_ds.tensors[0])
            h = torch.relu(model.prunable_layer(li)(Xl))
            Z0 = model.net[2 * (li + 1)](h)
        scales.append(float(Z0.norm(dim=1).mean()))

    records = []

    def emit(strategy: str, cuts: list[int], acc: float):
        records.append({"seed": seed, "arm": arm, "strategy": strategy,
                        "cuts": str(cuts), "removed": sum(cuts),
                        "frac_removed": sum(cuts) / total, "val_acc": acc,
                        "acc0": acc0})

    orders = {"greedy_norm": greedy_order(cost_lists, scales),
              "greedy_raw": greedy_order(cost_lists, None)}
    total_events = sum(len(d[0]) for d in dendros)
    ckpts = np.unique(np.linspace(1, total_events, n_ckpt, dtype=int))

    for name, order in orders.items():
        for upto in ckpts:
            cuts = cuts_from_order(order, L, int(upto))
            emit(name, cuts, evaluate(apply_cuts(model, bundle, cuts, arm, dendros), bundle))

    for f in np.linspace(0.05, 0.97, n_ckpt):
        cuts = [int(round(f * (h - 1))) for h in H]
        emit("equal_frac", cuts, evaluate(apply_cuts(model, bundle, cuts, arm, dendros), bundle))

    # sequential composition test at a subset of greedy_norm checkpoints
    for upto in ckpts[::3]:
        cuts = cuts_from_order(orders["greedy_norm"], L, int(upto))
        emit("sequential", cuts, evaluate(apply_cuts(model, bundle, cuts, arm, None), bundle))

    # grid oracle (func_matched only -- it's the operating arm)
    if arm == "func_matched":
        fr = [0.0, 0.25, 0.5, 0.625, 0.75, 0.8125, 0.875, 0.9375]
        for f0 in fr:
            for f1 in fr:
                for f2 in (fr if L > 2 else [0.0]):
                    cuts = [int(round(f * (h - 1))) for f, h in zip([f0, f1, f2][:L], H)]
                    emit("grid", cuts, evaluate(apply_cuts(model, bundle, cuts, arm, dendros), bundle))
    return records


# ── reporting ────────────────────────────────────────────────────────────────

def capacity_at_drop(g: pd.DataFrame, drop: float) -> float:
    g = g.sort_values("frac_removed")
    ok = g[g.val_acc >= g.acc0.iloc[0] - drop]
    return float(ok.frac_removed.max()) if len(ok) else 0.0


def report(df: pd.DataFrame, name: str) -> str:
    lines = [f"Phase C2 cross-layer allocation: {name}", "=" * 64, "",
             "capacity = max fraction of ALL hidden units removed (jointly)",
             f"with val acc within {ACC_DROP*100:.0f} pt of baseline (median over seeds)", ""]
    for arm in sorted(df.arm.unique()):
        lines.append(f"arm = {arm}")
        sub = df[df.arm == arm]
        strategies = [s for s in ["equal_frac", "greedy_raw", "greedy_norm",
                                  "sequential", "grid"] if s in set(sub.strategy)]
        for strat in strategies:
            caps = [capacity_at_drop(h, ACC_DROP)
                    for _, h in sub[sub.strategy == strat].groupby("seed")]
            if strat == "grid":  # envelope: best cuts per removal bucket
                caps = []
                for _, h in sub[sub.strategy == strat].groupby("seed"):
                    h = h.copy()
                    h["bucket"] = (h.frac_removed * 50).round()
                    env = h.groupby("bucket").val_acc.max().reset_index()
                    env["frac_removed"] = env.bucket / 50
                    env["acc0"] = h.acc0.iloc[0]
                    caps.append(capacity_at_drop(env, ACC_DROP))
            lines.append(f"  {strat:<12} capacity@-1pt = {np.median(caps):.3f}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", default=["func_matched", "ward"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    for rd in args.run_dir:
        rd = Path(rd)
        config = ExperimentConfig.from_yaml(rd / "config.yaml")
        out_dir = STUDY_ROOT / "outputs" / \
            f"c2_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True)
        records = []
        for seed in config.seeds:
            for arm in args.arms:
                logging.info(f"[{config.name}] seed {seed} arm {arm}")
                records.extend(run_seed(config, rd, seed, arm))
        df = pd.DataFrame(records)
        df.to_csv(out_dir / "allocations.csv", index=False)
        rep = report(df, config.name)
        (out_dir / "report.txt").write_text(rep)
        logging.info("\n" + rep)


if __name__ == "__main__":
    main()
