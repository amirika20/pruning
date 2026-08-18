#!/usr/bin/env python
"""Phase A scoring-rule comparison on ResNet (pretrained, e.g. ResNet-18 /
Imagenette). Same engines and merge surgery as the MLP harness; this file
supplies the Conv+BN adaptation:

* A prunable slot is a block-internal channel set (conv1 of a BasicBlock),
  whose sole consumer is conv2 -- the scoping src/models/resnet.py exposes.
* "Input space" is PATCH space: the im2col vectors conv1 sees. Each channel
  is one hyperplane on patches; weight sharing means one covector per filter.
* BN is folded into effective patch weights (exact in eval mode):
      w_eff = (gamma/std) * w,   b_eff = beta - gamma*mu_run/std.
* Outgoing c_i = conv2's kernel slice for channel i (flattened), the exact
  vector through which channel i's activation map feeds every consumer unit.
* Boxes/moments for the metrics come from calibration patches (train images
  only): per-dim ranges -> (x0, R0) / ellipsoid radii; mean+covariance -> the
  matched-Gaussian arm. NOTE: on ResNet the "box" is empirical (IBP through a
  deep BN/skip stack is uselessly loose), so the DOM arms here are data-light
  in the box, data-free in everything else.
* Evaluation: layer-local relative error of the consumer-weighted patch
  response V(p) = sum_i c_i h_i(p) on held-out val patches (every snapshot),
  plus full-model val accuracy of the REALIZED pruned network (BN replaced by
  an exact identity-with-offset, merged columns written into conv2) at a
  coarser stride.

Usage (repo root, .ml venv python; needs the pretrained checkpoint + data):
    python studies/gram_stability/resnet_phase_a.py \
        --config configs/experiments/imagenet/resnet/imagenette_resnet18_pretrained.yaml
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from dataclasses import dataclass
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

from studies.gram_stability.merge import (
    ZERO_NORM, EllipsoidMerge, IterativeMerge, LayerUnits)
from studies.gram_stability.functional import FunctionalMerge, GaussianMeasureMerge
from studies.gram_stability.compare_metrics import capacities, format_report, make_plots

STUDY_ROOT = Path(__file__).resolve().parent

ARMS = {
    "ward": lambda u, lo, hi, ex: IterativeMerge(u, lo, hi),
    "ellipsoid": lambda u, lo, hi, ex: EllipsoidMerge(u, lo, hi),
    "func_iso": lambda u, lo, hi, ex: FunctionalMerge(u, lo, hi),
    "func_box": lambda u, lo, hi, ex: GaussianMeasureMerge(
        u, lo, hi, (lo + hi) / 2.0, ((hi - lo) / 2.0) ** 2 / 3.0),
    "func_matched": lambda u, lo, hi, ex: GaussianMeasureMerge(
        u, lo, hi, ex["mu"], ex["cov"]),
}
DEFAULT_ARMS = ["ward", "ellipsoid", "func_box", "func_matched"]


# ── slot extraction ───────────────────────────────────────────────────────────

def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    """Effective patch-space affine of Conv->BN in eval mode: [K, d], [K]."""
    std = torch.sqrt(bn.running_var.detach().double() + bn.eps)
    scale = bn.weight.detach().double() / std
    W = conv.weight.detach().double().reshape(conv.out_channels, -1)
    w_eff = (scale[:, None] * W).cpu().numpy()
    b_eff = (bn.bias.detach().double() - scale * bn.running_mean.detach().double()).cpu().numpy()
    return w_eff, b_eff


def _sample_patches(model, layer_idx: int, images: torch.Tensor, n_patches: int,
                    device, batch: int = 32, seed: int = 0) -> torch.Tensor:
    """[n_patches, d] float64 im2col patches at slot `layer_idx`'s conv input,
    sampled uniformly over (image, spatial position)."""
    conv = model.prunable_layer(layer_idx)
    grabbed: list[torch.Tensor] = []

    def hook(module, args):
        grabbed.append(args[0].detach())

    h = conv.register_forward_pre_hook(hook)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    out: list[torch.Tensor] = []
    n_batches = (len(images) + batch - 1) // batch
    per_batch = n_patches // n_batches + 1
    with torch.no_grad():
        for i in range(0, len(images), batch):
            grabbed.clear()
            model(images[i:i + batch].to(device))
            x = grabbed[0]
            P = F.unfold(x, kernel_size=conv.kernel_size, stride=conv.stride,
                         padding=conv.padding)              # [B, d, L]
            P = P.permute(0, 2, 1).reshape(-1, P.shape[1])  # [B*L, d]
            take = torch.randperm(P.shape[0], generator=rng)[:per_batch]
            out.append(P[take].double().cpu())
    h.remove()
    P = torch.cat(out)[:n_patches]
    return P


@dataclass
class SlotData:
    units: LayerUnits           # folded (u, rho, alpha) + outgoing kernel slices
    ok: np.ndarray
    w_eff: np.ndarray           # [K, d] folded rows (for frozen units / V0)
    b_eff: np.ndarray
    C: np.ndarray               # [K, m] outgoing slices
    lo: np.ndarray
    hi: np.ndarray
    extras: dict                # mu, cov for func_matched
    P_eval: torch.Tensor        # [N, d] float64 on device
    V0: torch.Tensor            # [N, m] float64 on device
    V0_norm: float


def collect_slot(model, layer_idx: int, train_x: torch.Tensor, val_x: torch.Tensor,
                 device, n_calib: int, n_eval: int) -> SlotData:
    conv = model.prunable_layer(layer_idx)
    bn = model.prunable_bn(layer_idx)
    nxt = model.outgoing_module(layer_idx)

    w_eff, b_eff = fold_bn(conv, bn)
    alpha = np.linalg.norm(w_eff, axis=1)
    ok = alpha > ZERO_NORM
    safe = np.where(ok, alpha, 1.0)
    C = (nxt.weight.detach().double().permute(1, 0, 2, 3)
         .reshape(nxt.in_channels, -1).cpu().numpy())       # [K, out2*k2*k2]
    units = LayerUnits(w_eff / safe[:, None], -b_eff / safe, alpha, C)

    P_cal = _sample_patches(model, layer_idx, train_x, n_calib, device, seed=layer_idx)
    lo = P_cal.min(dim=0).values.numpy()
    hi = P_cal.max(dim=0).values.numpy()
    Pg = P_cal.to(device)
    mu = Pg.mean(dim=0)
    Xc = Pg - mu
    cov = (Xc.T @ Xc) / (Pg.shape[0] - 1)
    extras = {"mu": mu.cpu().numpy(), "cov": cov.cpu().numpy()}
    del Pg, Xc, cov

    P_eval = _sample_patches(model, layer_idx, val_x, n_eval, device,
                             seed=1000 + layer_idx).to(device)
    Wg = torch.from_numpy(w_eff).to(device)
    Cg = torch.from_numpy(C).to(device)
    V0 = torch.relu(P_eval @ Wg.T + torch.from_numpy(b_eff).to(device)) @ Cg
    return SlotData(units, ok, w_eff, b_eff, C, lo, hi, extras,
                    P_eval, V0, float(torch.linalg.norm(V0)))


# ── evaluation ────────────────────────────────────────────────────────────────

def v_rel_err(sd: SlotData, W_rows, biases, cols, device) -> float:
    """Relative L2 error of the consumer-weighted patch response."""
    if (~sd.ok).any():
        W_rows = np.concatenate([W_rows, sd.w_eff[~sd.ok]])
        biases = np.concatenate([biases, sd.b_eff[~sd.ok]])
        cols = np.concatenate([cols, sd.C[~sd.ok]])
    Wt = torch.from_numpy(W_rows).to(device)
    bt = torch.from_numpy(biases).to(device)
    ct = torch.from_numpy(cols).to(device)
    V = torch.relu(sd.P_eval @ Wt.T + bt) @ ct
    return float(torch.linalg.norm(V - sd.V0) / max(sd.V0_norm, 1e-30))


def realize_resnet(model, layer_idx: int, sd: SlotData,
                   W_rows, biases, cols) -> nn.Module:
    """Pruned copy: conv1 rows = merged folded weights, BN = exact identity
    plus offset, conv2 input slices = merged outgoing columns."""
    if (~sd.ok).any():
        W_rows = np.concatenate([W_rows, sd.w_eff[~sd.ok]])
        biases = np.concatenate([biases, sd.b_eff[~sd.ok]])
        cols = np.concatenate([cols, sd.C[~sd.ok]])
    pruned = copy.deepcopy(model)
    block, slot = pruned._slot(layer_idx)
    cn, bnn, nxtn = block.slot_modules(slot)
    conv, nxt = getattr(block, cn), getattr(block, nxtn)
    K = W_rows.shape[0]
    dev = conv.weight.device

    new_conv = nn.Conv2d(conv.in_channels, K, conv.kernel_size, stride=conv.stride,
                         padding=conv.padding, bias=False).to(dev)
    new_conv.weight.data = torch.from_numpy(W_rows).float().reshape(
        K, conv.in_channels, *conv.kernel_size).to(dev)

    new_bn = nn.BatchNorm2d(K).to(dev)
    new_bn.running_mean.zero_()
    new_bn.running_var.fill_(1.0)
    new_bn.weight.data.fill_(float(np.sqrt(1.0 + new_bn.eps)))
    new_bn.bias.data = torch.from_numpy(biases).float().to(dev)

    new_next = nn.Conv2d(K, nxt.out_channels, nxt.kernel_size, stride=nxt.stride,
                         padding=nxt.padding, bias=False).to(dev)
    new_next.weight.data = torch.from_numpy(cols).float().reshape(
        K, nxt.out_channels, *nxt.kernel_size).permute(1, 0, 2, 3).contiguous().to(dev)

    setattr(block, cn, new_conv)
    setattr(block, bnn, new_bn)
    setattr(block, nxtn, new_next)
    return pruned.eval()


@torch.no_grad()
def model_val_acc(model, val_x, val_y, device, batch: int = 64) -> float:
    correct = 0
    for i in range(0, len(val_x), batch):
        out = model(val_x[i:i + batch].to(device))
        correct += int((out.argmax(dim=1).cpu() == val_y[i:i + batch]).sum())
    return correct / len(val_x)


# ── sweep ─────────────────────────────────────────────────────────────────────

def sweep_slot(arm: str, model, sd: SlotData, layer_idx: int, device,
               val_x, val_y, acc_every: int) -> list[dict]:
    engine = ARMS[arm](sd.units.subset(np.flatnonzero(sd.ok)), sd.lo, sd.hi, sd.extras)
    H = engine.n_orig
    stride = max(1, H // 64)

    r_orig = engine.orig.rho - engine.orig.u @ engine.x0
    sat = np.abs(r_orig) > engine.R0

    def n_sat_absorbed() -> int:
        total = 0
        for k in np.flatnonzero(engine.active):
            mem = engine.members[k]
            if len(mem) > 1 and sat[np.array(mem)].all():
                total += len(mem) - 1
        return total

    n_snap = 0

    def snapshot(step: int, rec: dict) -> dict:
        nonlocal n_snap
        W_rows, biases, cols, _ = engine.realize()
        row = {
            "metric": arm, "seed": 0, "layer": layer_idx, "step": step,
            "frac_merged": step / max(H - 1, 1),
            "cost": rec.get("ward_cost", 0.0),
            "bound_total": rec.get("bound_total", 0.0),
            "n_sat_absorbed": n_sat_absorbed(),
            "layer_rel_err": v_rel_err(sd, W_rows, biases, cols, device),
            "val_loss": np.nan,
            "val_acc": np.nan,
        }
        if n_snap % acc_every == 0 or engine.n_active == 1:
            pruned = realize_resnet(model, layer_idx, sd, W_rows, biases, cols)
            row["val_acc"] = model_val_acc(pruned, val_x, val_y, device)
            del pruned
        n_snap += 1
        return row

    records = [snapshot(0, {})]
    assert records[0]["layer_rel_err"] < 1e-6, \
        f"step-0 folded realization must be exact, got {records[0]['layer_rel_err']:.2e}"
    step = 0
    while engine.n_active > 1:
        rec = engine.step()
        step += 1
        if step % stride == 0 or engine.n_active == 1:
            records.append(snapshot(step, rec))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/experiments/imagenet/resnet/"
                                            "imagenette_resnet18_pretrained.yaml")
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=list(ARMS))
    parser.add_argument("--slots", nargs="+", type=int, default=None,
                        help="prunable slot indices (default: all)")
    parser.add_argument("--n-calib-patches", type=int, default=8192)
    parser.add_argument("--n-eval-patches", type=int, default=2048)
    parser.add_argument("--n-calib-imgs", type=int, default=128)
    parser.add_argument("--n-eval-imgs", type=int, default=64)
    parser.add_argument("--acc-every", type=int, default=4,
                        help="full-model val accuracy every Nth snapshot")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ExperimentConfig.from_yaml(Path(args.config))
    out_dir = STUDY_ROOT / "outputs" / \
        f"resnetA_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True)

    torch.manual_seed(config.seeds[0])
    bundle = build_dataset(config.data.kind, data_seed=config.seeds[0], **config.data.params)
    model = build_model(config.model.kind, bundle, **config.model.params)
    model = model.to(device).eval()

    train_x = bundle.train_ds.tensors[0][:args.n_calib_imgs]
    val_x, val_y = bundle.val_ds.tensors
    eval_x = val_x[:args.n_eval_imgs]
    acc0 = model_val_acc(model, val_x, val_y, device)
    logging.info(f"=== {config.name}: {model.n_prunable_layers()} slots, "
                 f"baseline val acc {acc0:.4f}, device={device}")

    slots = args.slots or list(range(model.n_prunable_layers()))
    all_records = []
    for layer_idx in slots:
        sd = collect_slot(model, layer_idx, train_x, eval_x, device,
                          args.n_calib_patches, args.n_eval_patches)
        K, d = sd.units.u.shape
        logging.info(f"slot {layer_idx}: K={K} d={d} m={sd.C.shape[1]} "
                     f"R0={np.linalg.norm((sd.hi - sd.lo) / 2):.2f}")
        for arm in args.arms:
            recs = sweep_slot(arm, model, sd, layer_idx, device, val_x, val_y,
                              args.acc_every)
            logging.info(f"  slot {layer_idx} {arm}: final rel err "
                         f"{recs[-1]['layer_rel_err']:.3f}, "
                         f"final acc {recs[-1]['val_acc']:.3f}")
            all_records.extend(recs)
        del sd
        torch.cuda.empty_cache() if device.type == "cuda" else None

    df = pd.DataFrame(all_records)
    df.to_csv(out_dir / "steps.csv", index=False)
    caps = capacities(df)
    caps.to_json(out_dir / "capacities.json", orient="records", indent=2)
    make_plots(df, out_dir / "plots", args.arms)
    report = format_report(config.name, caps, args.arms)
    (out_dir / "report.txt").write_text(report)
    logging.info("\n" + report)


if __name__ == "__main__":
    main()
