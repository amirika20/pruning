#!/usr/bin/env python
"""Phase D on CNNs: ours vs OSSCAR on Conv+BN models trained on MNIST /
Fashion-MNIST (the `cnn` model: Conv->BN->ReLU blocks -> GAP -> Linear).

Conv adaptation (as in resnet_phase_a): BN folded into effective patch-space
weights; one covector per filter over its conv's im2col patches; outgoing
c_i = the consumer's kernel slice for channel i (or the head column after
GAP). Surgery = mean merge + KERNEL global repair in patch space (E8 winner),
realized exactly via the identity-BN trick.

Arms (equal-fraction joint pruning of all blocks, sequential order):
    ours          func_matched selection + mean+global; 128 images (moments)
    hybrid128     same structure, EMPIRICAL ridge LS repair on patches from
                  the same 128 images
    osscar128     OSSCAR with the same 128 calibration images
    osscar_full   OSSCAR with the full train set

Trains its own checkpoints (cached under outputs/d_cnn_ckpts/).

Usage:
    python studies/gram_stability/phase_d_cnn.py \
        --config configs/experiments/mnist/cnn/mnist_pixel_cnn.yaml \
                 configs/experiments/fashion_mnist/cnn/fashion_mnist_pixel_cnn.yaml
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
from src.models.cnn import CNN
from src.pruning.methods.osscar import OSSCAR
from src.pruning.registry import PruneContext
from src.training.trainer import train

from studies.gram_stability.merge import ZERO_NORM, LayerUnits
from studies.gram_stability.functional import GaussianMeasureMerge
from studies.gram_stability.resnet_phase_a import fold_bn
from studies.gram_stability.phase_b import LayerMoments, cross_gram
from studies.gram_stability.phase_c2 import dendrogram, partition_at

STUDY_ROOT = Path(__file__).resolve().parent
N_CALIB_IMGS = 128
N_PATCHES = 4096
SEEDS = [0, 1, 2]


# ── conv-layer extraction ─────────────────────────────────────────────────────

def conv_units(model: CNN, li: int) -> tuple[LayerUnits, np.ndarray, np.ndarray, np.ndarray]:
    """(units, ok, w_eff, b_eff): folded patch-space parameterization +
    outgoing rows (consumer kernel slices / head columns)."""
    conv, bn = model.prunable_layer(li), model.prunable_bn(li)
    w_eff, b_eff = fold_bn(conv, bn)
    alpha = np.linalg.norm(w_eff, axis=1)
    ok = alpha > ZERO_NORM
    safe = np.where(ok, alpha, 1.0)
    consumer = model.outgoing_module(li)
    if isinstance(consumer, nn.Conv2d):
        C = (consumer.weight.detach().double().permute(1, 0, 2, 3)
             .reshape(consumer.in_channels, -1).cpu().numpy())
    else:
        C = consumer.weight.detach().double().t().cpu().numpy()   # [K, out]
    return LayerUnits(w_eff / safe[:, None], -b_eff / safe, alpha, C), ok, w_eff, b_eff


def sample_patches(model: CNN, li: int, images: torch.Tensor, n: int, seed: int) -> np.ndarray:
    conv = model.prunable_layer(li)
    grabbed: list[torch.Tensor] = []
    h = conv.register_forward_pre_hook(lambda m, a: grabbed.append(a[0].detach()))
    with torch.no_grad():
        model(images)
    h.remove()
    P = F.unfold(grabbed[0], kernel_size=conv.kernel_size, stride=conv.stride,
                 padding=conv.padding)                       # [B, d, L]
    P = P.permute(0, 2, 1).reshape(-1, P.shape[1])           # [B*L, d]
    rng = np.random.default_rng(seed)
    take = rng.permutation(P.shape[0])[:n]
    return P[take].double().cpu().numpy()


# ── surgery: mean merge + kernel/empirical global repair, exact realization ──

def realize_conv(model: CNN, li: int, clusters, mu, Sigma, repair: str,
                 P_emp: np.ndarray | None = None):
    units, ok, w_eff, b_eff = conv_units(model, li)
    a = units.a
    W_rows, biases, cols = [], [], []
    for mem in clusters:
        if len(mem) == 1 and not ok[mem[0]]:
            i = mem[0]
            W_rows.append(w_eff[i]); biases.append(b_eff[i]); cols.append(units.C[i])
            continue
        mem_a = np.array(mem)
        S_u = (a[mem_a, None] * units.u[mem_a]).sum(axis=0)
        S_c = (units.alpha[mem_a, None] * units.C[mem_a]).sum(axis=0)
        n = np.linalg.norm(S_u)
        if n < 1e-12:
            W_rows.append(np.zeros(units.u.shape[1])); biases.append(0.0); cols.append(S_c)
            continue
        W_rows.append(S_u / n)
        biases.append(-float((a[mem_a] * units.rho[mem_a]).sum()) / n)
        cols.append(S_c)
    W_rows, biases, cols = np.array(W_rows), np.array(biases), np.array(cols)

    if repair == "kernel":
        orig = LayerMoments(units.u, units.rho, mu, Sigma)
        kept = LayerMoments(W_rows, -biases, mu, Sigma)
        Gkk = cross_gram(kept.m[:, None], kept.s[:, None],
                         kept.m[None, :], kept.s[None, :], kept.corr_with(kept))
        Gko = cross_gram(kept.m[:, None], kept.s[:, None],
                         orig.m[None, :], orig.s[None, :],
                         kept.corr_with(orig)) * units.alpha[None, :]
        lam = 1e-8 * max(np.trace(Gkk) / max(len(Gkk), 1), 1e-30)
        cols = np.linalg.solve(Gkk + lam * np.eye(len(Gkk)), Gko @ units.C)
    elif repair == "empirical":
        H_orig = np.maximum(P_emp @ w_eff.T + b_eff, 0.0)
        H_kept = np.maximum(P_emp @ W_rows.T + biases, 0.0)
        Gkk = H_kept.T @ H_kept
        lam = 1e-6 * max(np.trace(Gkk) / max(len(Gkk), 1), 1e-30)
        cols = np.linalg.solve(Gkk + lam * np.eye(len(Gkk)),
                               H_kept.T @ (H_orig @ units.C))
    return W_rows, biases, cols


def apply_conv_layer(model: CNN, li: int, W_rows, biases, cols) -> CNN:
    pruned = copy.deepcopy(model)
    block = pruned.blocks[li]
    conv = block.conv
    K = W_rows.shape[0]
    new_conv = nn.Conv2d(conv.in_channels, K, conv.kernel_size,
                         padding=conv.padding, bias=False)
    new_conv.weight.data = torch.from_numpy(W_rows).float().reshape(
        K, conv.in_channels, *conv.kernel_size)
    new_bn = nn.BatchNorm2d(K)
    new_bn.running_mean.zero_(); new_bn.running_var.fill_(1.0)
    new_bn.weight.data.fill_(float(np.sqrt(1.0 + new_bn.eps)))
    new_bn.bias.data = torch.from_numpy(biases).float()
    block.conv, block.bn = new_conv, new_bn

    consumer = pruned.outgoing_module(li)
    if isinstance(consumer, nn.Conv2d):
        new_c = nn.Conv2d(K, consumer.out_channels, consumer.kernel_size,
                          padding=consumer.padding, bias=False)
        new_c.weight.data = torch.from_numpy(cols).float().reshape(
            K, consumer.out_channels, *consumer.kernel_size).permute(1, 0, 2, 3).contiguous()
        pruned.blocks[li + 1].conv = new_c
    else:
        new_h = nn.Linear(K, consumer.out_features)
        new_h.weight.data = torch.from_numpy(cols).float().t().contiguous()
        new_h.bias.data = consumer.bias.data.clone()
        pruned.head = new_h
    return pruned.eval()


def apply_cuts_ours(model: CNN, calib: torch.Tensor, cuts, dendros, repair: str) -> CNN:
    cur = model
    for li, k in enumerate(cuts):
        pairs, idx_map, frozen = dendros[li]
        k = min(k, len(pairs))
        sub = partition_at(len(idx_map), pairs, k)
        clusters = [[int(idx_map[i]) for i in cl] for cl in sub] \
            + [[int(f)] for f in frozen]
        P = sample_patches(cur, li, calib, N_PATCHES, seed=li)
        mu, Sigma = P.mean(axis=0), np.cov(P.T)
        cur = apply_conv_layer(cur, li, *realize_conv(cur, li, clusters, mu,
                                                      np.atleast_2d(Sigma), repair, P))
    return cur


def apply_cuts_osscar(model: CNN, bundle, cuts, calib_x: torch.Tensor) -> CNN:
    cur = model
    for li, k in enumerate(cuts):
        H = cur.prunable_layer(li).out_channels
        k = min(k, H - 1)
        if k <= 0:
            continue
        ctx = PruneContext(train_inputs=calib_x, bundle=bundle,
                           device=torch.device("cpu"))
        dec = OSSCAR(n_remove=int(k)).select(cur, li, ctx)
        cur = cur.set_outgoing_weights(li, dec.new_outgoing)
        cur = cur.prune_layer(li, dec.remove)
    return cur


@torch.no_grad()
def val_acc(model: CNN, bundle) -> float:
    X, y = bundle.val_ds.tensors
    return float((model(X).argmax(dim=1) == y).float().mean())


# ── experiment ────────────────────────────────────────────────────────────────

def get_model(config, bundle, seed: int, ckpt_dir: Path) -> CNN:
    ckpt = ckpt_dir / f"{config.name}_seed_{seed}.pt"
    model = build_model(config.model.kind, bundle, **config.model.params)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        return model.cpu().eval()
    device = torch.device(config.training.device)
    model = model.to(device)
    tl, vl = bundle.loaders(config.training.batch_size)
    train(model, tl, vl, config.training, task=bundle.task,
          desc=f"[{config.name}] seed {seed}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    torch.save(model.state_dict(), ckpt)
    return model


def run_seed(config, seed: int, ckpt_dir: Path, n_ckpt: int = 12) -> list[dict]:
    torch.manual_seed(seed)
    bundle = build_dataset(config.data.kind, data_seed=seed, **config.data.params)
    model = get_model(config, bundle, seed, ckpt_dir)
    L = model.n_prunable_layers()
    Hs = [model.prunable_layer(li).out_channels for li in range(L)]
    acc0 = val_acc(model, bundle)
    calib128 = bundle.train_ds.tensors[0][:N_CALIB_IMGS]
    calib_full = bundle.train_ds.tensors[0]

    dendros = []
    for li in range(L):
        units, ok, _, _ = conv_units(model, li)
        idx_map = np.flatnonzero(ok)
        frozen = np.flatnonzero(~ok)
        P = sample_patches(model, li, calib128, N_PATCHES, seed=li)
        lo, hi = P.min(axis=0), P.max(axis=0)
        eng = GaussianMeasureMerge(units.subset(idx_map), lo, hi,
                                   P.mean(axis=0), np.atleast_2d(np.cov(P.T)))
        pairs, _ = dendrogram(eng)
        dendros.append((pairs, idx_map, frozen))

    # sanity: zero cuts reproduce the model through the fold/identity-BN trick
    base = apply_cuts_ours(model, calib128, [0] * L, dendros, "kernel")
    assert abs(val_acc(base, bundle) - acc0) < 5e-3, "zero-cut realization drifted"

    records = []
    for f in np.linspace(0.1, 0.95, n_ckpt):
        cuts = [int(round(f * (h - 1))) for h in Hs]
        frac = sum(cuts) / sum(Hs)
        arms = {
            "ours": lambda: apply_cuts_ours(model, calib128, cuts, dendros, "kernel"),
            "hybrid128": lambda: apply_cuts_ours(model, calib128, cuts, dendros, "empirical"),
            "osscar128": lambda: apply_cuts_osscar(model, bundle, cuts, calib128),
            "osscar_full": lambda: apply_cuts_osscar(model, bundle, cuts, calib_full),
        }
        for arm, fn in arms.items():
            records.append({"seed": seed, "arm": arm, "frac_removed": frac,
                            "val_acc": val_acc(fn(), bundle), "acc0": acc0})
        logging.info(f"  seed {seed} f={f:.2f}: " + "  ".join(
            f"{r['arm']}={r['val_acc']:.3f}" for r in records[-4:]))
    return records


def capacity(g: pd.DataFrame, drop: float) -> float:
    ok = g[g.val_acc >= g.acc0.iloc[0] - drop]
    return float(ok.frac_removed.max()) if len(ok) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    ckpt_dir = STUDY_ROOT / "outputs" / "d_cnn_ckpts"

    for cpath in args.config:
        config = ExperimentConfig.from_yaml(Path(cpath))
        out_dir = STUDY_ROOT / "outputs" / \
            f"dcnn_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True)
        records = []
        for seed in SEEDS:
            logging.info(f"[{config.name}] seed {seed}")
            records.extend(run_seed(config, seed, ckpt_dir))
        df = pd.DataFrame(records)
        df.to_csv(out_dir / "curves.csv", index=False)

        lines = [f"Phase D (CNN): ours vs OSSCAR: {config.name}", "=" * 60, "",
                 "Conv->BN->ReLU blocks, joint equal-fraction filter pruning",
                 "capacity = max fraction of all filters removed (median over seeds)",
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
