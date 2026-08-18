#!/usr/bin/env python
"""Train models, iteratively merge each hidden layer to a single unit, and
track Gram-matrix properties + actual error at every step.

Usage (from the repo root, with the .ml venv python):
    python studies/gram_stability/run_study.py --config studies/gram_stability/configs/sine_mlp.yaml
    python studies/gram_stability/run_study.py --config studies/gram_stability/configs   # every yaml
    python studies/gram_stability/run_study.py --config a.yaml --epochs 20 --stride 4    # smoke run

Per config the output directory contains:
    config.yaml
    steps.csv      one row per (seed, layer, merge step): merge diagnostics,
                   certified bound, actual errors, and per-B-definition Gram
                   properties (columns <def>_<prop>, defs in gram.GRAM_DEFS)
    summary.json   per-(seed, layer) stability + predictiveness of each property
    report.txt     readable ranking: which Gram properties are stable, and
                   which predict the actual error
    plots/         trajectories, stability-vs-error, elbow curves
    models/        trained checkpoints
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.data import build_dataset
from src.data.registry import DatasetBundle
from src.models import build_model
from src.models.mlp import MLP
from src.training.trainer import train

from studies.gram_stability.merge import IterativeMerge, extract_units, input_boxes
from studies.gram_stability.gram import GRAM_DEFS, build_B, gram_props
from studies.gram_stability.analyze import analyze_steps, format_report, make_plots

STUDY_ROOT = Path(__file__).resolve().parent


# ── evaluation context: everything needed to score a merged layer ────────────

@dataclass
class EvalCtx:
    Xl: np.ndarray        # [N, d] float64 layer input on the val set
    y: torch.Tensor       # val labels
    task: str
    next_bias: np.ndarray  # [m]
    suffix: torch.nn.Module  # rest of the net after the consumer Linear
    Z0: np.ndarray        # [N, m] original next-layer pre-activations
    frozen_W: np.ndarray  # [F, d] rows excluded from merging (zero-norm)
    frozen_b: np.ndarray  # [F]
    frozen_C: np.ndarray  # [F, m]


def make_eval_ctx(model: MLP, bundle: DatasetBundle, layer_idx: int,
                  frozen: tuple[np.ndarray, np.ndarray, np.ndarray]) -> EvalCtx:
    X_val, y_val = bundle.val_ds.tensors
    with torch.no_grad():
        Xl = model.net[: 2 * layer_idx](X_val).double().numpy()

    layer = model.prunable_layer(layer_idx)
    W = layer.weight.data.double().numpy()
    b = layer.bias.data.double().numpy()
    consumer = model.outgoing_module(layer_idx)
    C = model.outgoing_weights(layer_idx).double().numpy()          # [H, m]
    next_bias = consumer.bias.data.double().numpy()
    suffix = model.net[2 * (layer_idx + 1) + 1:]

    Z0 = np.maximum(Xl @ W.T + b, 0.0) @ C + next_bias
    return EvalCtx(Xl, y_val, bundle.task, next_bias, suffix, Z0, *frozen)


def eval_merged(ctx: EvalCtx, W_rows: np.ndarray, biases: np.ndarray,
                cols: np.ndarray) -> dict:
    """Layer-local relative error + full-model val loss/acc for the merged
    layer given by (rows, biases, outgoing cols) + the frozen units."""
    if len(ctx.frozen_W):
        W_rows = np.concatenate([W_rows, ctx.frozen_W])
        biases = np.concatenate([biases, ctx.frozen_b])
        cols = np.concatenate([cols, ctx.frozen_C])
    Z = np.maximum(ctx.Xl @ W_rows.T + biases, 0.0) @ cols + ctx.next_bias
    rel = float(np.linalg.norm(Z - ctx.Z0) / max(np.linalg.norm(ctx.Z0), 1e-30))

    with torch.no_grad():
        out = ctx.suffix(torch.from_numpy(Z).float())
    if ctx.task == "multiclass":
        loss = float(F.cross_entropy(out, ctx.y))
        acc = float((out.argmax(dim=1) == ctx.y).float().mean())
    else:
        loss = float(F.mse_loss(out, ctx.y))
        acc = np.nan
    return {"layer_rel_err": rel, "val_loss": loss, "val_acc": acc}


# ── per-layer merge sweep ─────────────────────────────────────────────────────

def _m1_stats(v: np.ndarray, v0: np.ndarray, prefix: str) -> dict:
    n0 = np.linalg.norm(v0)
    return {
        f"{prefix}_ratio": float(np.linalg.norm(v) / max(n0, 1e-30)),
        f"{prefix}_cos": float(v @ v0 / max(np.linalg.norm(v) * n0, 1e-30)),
    }


def sweep_layer(model: MLP, bundle: DatasetBundle, layer_idx: int,
                lo: np.ndarray, hi: np.ndarray, seed: int, stride: int) -> list[dict]:
    units, ok = extract_units(model, layer_idx)
    layer = model.prunable_layer(layer_idx)
    W = layer.weight.data.double().numpy()
    b = layer.bias.data.double().numpy()
    C = model.outgoing_weights(layer_idx).double().numpy()
    frozen = (W[~ok], b[~ok], C[~ok])

    engine = IterativeMerge(units.subset(np.flatnonzero(ok)), lo, hi)
    ctx = make_eval_ctx(model, bundle, layer_idx, frozen)
    H = engine.n_orig

    m1r0, m1z0 = engine.m1_raw(), engine.m1_realized()
    ref_V: dict[str, np.ndarray] = {}

    def snapshot(step: int, merge_rec: dict) -> dict:
        W_rows, biases, cols, A = engine.realize()
        rec = {
            "seed": seed, "layer": layer_idx, "step": step,
            "n_clusters": engine.n_active + len(frozen[0]),
            "frac_merged": step / max(H - 1, 1),
            **merge_rec,
            **eval_merged(ctx, W_rows, biases, cols),
            **_m1_stats(engine.m1_raw(), m1r0, "m1_raw"),
            **_m1_stats(engine.m1_realized(), m1z0, "m1_real"),
        }
        ubar, rhobar = W_rows, -biases
        qt = np.concatenate(
            [engine.R0 * ubar, (ubar @ engine.x0 - rhobar)[:, None]], axis=1)
        for name in GRAM_DEFS:
            B = build_B(name, ubar, rhobar, A, qt)
            props, V3 = gram_props(B, ref_V.get(name))
            if name not in ref_V:
                ref_V[name] = V3
                props["aff3"] = 1.0
            rec.update({f"{name}_{k}": v for k, v in props.items()})
        return rec

    empty = {"survivor": -1, "removed": -1, "cluster_size": 0, "ward_cost": 0.0,
             "cum_ward": 0.0, "bound_cluster": 0.0, "bound_total": 0.0,
             "overshoot": 1.0}
    records = [snapshot(0, empty)]
    if records[0]["layer_rel_err"] > 1e-6:
        raise AssertionError(
            f"step-0 realization must be exact, got rel err {records[0]['layer_rel_err']:.2e}")

    step = 0
    while engine.n_active > 1:
        merge_rec = engine.step()
        step += 1
        if step % stride == 0 or engine.n_active == 1:
            records.append(snapshot(step, merge_rec))

    if abs(records[-1]["m1_raw_ratio"] - 1) > 1e-9 or records[-1]["m1_raw_cos"] < 1 - 1e-12:
        raise AssertionError("m1_raw must be conserved exactly")
    return records


# ── orchestration ─────────────────────────────────────────────────────────────

def run_config(config_path: Path, epochs_override: int | None, stride: int) -> Path:
    config = ExperimentConfig.from_yaml(config_path)
    if epochs_override is not None:
        config.training.epochs = epochs_override

    out_dir = STUDY_ROOT / "outputs" / f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True)
    config.save(out_dir / "config.yaml")
    device = torch.device(config.training.device)

    logging.info(f"=== {config.name}: {len(config.seeds)} seed(s), device={device} ===")
    logging.info(f"Output: {out_dir}")

    all_records = []
    for seed in config.seeds:
        torch.manual_seed(seed)
        bundle = build_dataset(config.data.kind, data_seed=seed, **config.data.params)
        train_loader, val_loader = bundle.loaders(config.training.batch_size)
        model = build_model(config.model.kind, bundle, **config.model.params).to(device)
        if not isinstance(model, MLP):
            raise TypeError("gram_stability targets plain MLPs (IBP boxes + Linear surgery)")

        train(model, train_loader, val_loader, config.training, task=bundle.task,
              desc=f"[{config.name}] seed {seed}")
        model = model.cpu().eval()

        ckpt_dir = out_dir / "models"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"seed_{seed}.pt")

        x_train = bundle.train_ds.tensors[0].double().numpy()
        boxes = input_boxes(model, x_train)
        for layer_idx in range(model.n_prunable_layers()):
            lo, hi = boxes[layer_idx]
            recs = sweep_layer(model, bundle, layer_idx, lo, hi, seed, stride)
            logging.info(
                f"[{config.name}] seed {seed} layer {layer_idx}: "
                f"{len(recs)} snapshots, final rel err {recs[-1]['layer_rel_err']:.3f}")
            all_records.extend(recs)

    df = pd.DataFrame(all_records)
    df.to_csv(out_dir / "steps.csv", index=False)

    summary = analyze_steps(df)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"name": config.name, **summary}, f, indent=2)

    make_plots(df, out_dir / "plots")
    report = format_report(config.name, summary)
    (out_dir / "report.txt").write_text(report)
    logging.info("\n" + report)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", required=True,
                        help="yaml file(s) or a directory of yamls")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override training epochs (for quick smoke runs)")
    parser.add_argument("--stride", type=int, default=1,
                        help="snapshot every N merge steps (default 1 = every step)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    config_paths: list[Path] = []
    for c in args.config:
        p = Path(c)
        config_paths.extend(sorted(p.rglob("*.yaml")) if p.is_dir() else [p])

    for path in config_paths:
        run_config(path, args.epochs, args.stride)


if __name__ == "__main__":
    main()
