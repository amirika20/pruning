#!/usr/bin/env python
"""Head-to-head: Ward/Euclidean (cylinder) pair selection vs exact
expected-damage (Gaussian functional / arc-cosine kernel) selection.

Both arms use the IDENTICAL merge rule (covector addition + sum-rule fan-out
surgery) -- only the pair-selection metric differs, so any gap is purely the
metric's doing.

Usage (from the repo root, with the .ml venv python):
    python studies/gram_stability/compare_metrics.py \
        --run-dir studies/gram_stability/outputs/gram_mnist_mlp_YYYYMMDD_HHMMSS ...

Each --run-dir must contain config.yaml and models/seed_<s>.pt (as produced by
run_study.py); models are reloaded, never retrained.

Output per run-dir: outputs/compare_<name>_<timestamp>/
    steps.csv    (metric, seed, layer, step) rows: selection cost, certified
                 bound, actual layer error, val loss/acc, saturated-absorption
    report.txt   capacity table: how much of each layer each metric can merge
                 at matched error/accuracy tolerances
    plots/       error-vs-fraction curves, both metrics overlaid
"""

from __future__ import annotations

import argparse
import json
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

from studies.gram_stability.merge import IterativeMerge, extract_units, input_boxes
from studies.gram_stability.functional import FunctionalMerge
from studies.gram_stability.run_study import make_eval_ctx, eval_merged

STUDY_ROOT = Path(__file__).resolve().parent
ARMS = {"ward": IterativeMerge, "functional": FunctionalMerge}
REL_ERR_TOLS = [0.01, 0.05, 0.10]
ACC_DROP = 0.01  # 1 accuracy point


def sweep_arm(arm: str, model, bundle, layer_idx: int, lo, hi, seed: int,
              stride: int, ctx) -> list[dict]:
    units, ok = extract_units(model, layer_idx)
    engine = ARMS[arm](units.subset(np.flatnonzero(ok)), lo, hi)
    H = engine.n_orig

    # ball-saturation flags of the ORIGINAL units (|local offset| > R0)
    r_orig = engine.orig.rho - engine.orig.u @ engine.x0
    sat = np.abs(r_orig) > engine.R0

    def n_sat_absorbed() -> int:
        """Saturated neurons removed by all-saturated merges so far."""
        total = 0
        for k in np.flatnonzero(engine.active):
            mem = engine.members[k]
            if len(mem) > 1 and sat[np.array(mem)].all():
                total += len(mem) - 1
        return total

    def snapshot(step: int, rec: dict) -> dict:
        W_rows, biases, cols, _ = engine.realize()
        return {
            "metric": arm, "seed": seed, "layer": layer_idx, "step": step,
            "frac_merged": step / max(H - 1, 1),
            "cost": rec.get("ward_cost", 0.0),
            "bound_total": rec.get("bound_total", 0.0),
            "n_sat_absorbed": n_sat_absorbed(),
            **eval_merged(ctx, W_rows, biases, cols),
        }

    records = [snapshot(0, {})]
    assert records[0]["layer_rel_err"] < 1e-6, "step-0 realization must be exact"
    step = 0
    while engine.n_active > 1:
        rec = engine.step()
        step += 1
        if step % stride == 0 or engine.n_active == 1:
            records.append(snapshot(step, rec))
    return records


def capacities(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (metric, seed, layer), g in df.groupby(["metric", "seed", "layer"]):
        g = g.sort_values("step")
        row = {"metric": metric, "seed": seed, "layer": layer}
        for tol in REL_ERR_TOLS:
            bad = g.loc[g.layer_rel_err > tol, "frac_merged"]
            row[f"cap_err{int(tol * 100)}"] = float(bad.iloc[0]) if len(bad) else 1.0
        if g["val_acc"].notna().any():
            acc0 = float(g["val_acc"].iloc[0])
            bad = g.loc[g.val_acc < acc0 - ACC_DROP, "frac_merged"]
            row["cap_acc1pt"] = float(bad.iloc[0]) if len(bad) else 1.0
        row["sat_absorbed_at_25pct"] = int(
            g.loc[g.frac_merged <= 0.25, "n_sat_absorbed"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def format_report(name: str, caps: pd.DataFrame) -> str:
    med = caps.groupby(["metric", "layer"]).median(numeric_only=True).reset_index()
    cols = [c for c in med.columns if c.startswith("cap_")] + ["sat_absorbed_at_25pct"]
    lines = [
        f"Metric comparison: {name}",
        "=" * 72,
        "",
        "Capacity = fraction of the layer merged before crossing the tolerance",
        "(medians over seeds; higher = the metric merges more before damage).",
        "cap_errX: layer output rel err > X%.  cap_acc1pt: val acc drops 1 pt.",
        "sat_absorbed_at_25pct: saturated neurons removed by all-saturated",
        "merges within the first 25% of the merge sequence.",
        "",
        f"  {'layer':>5} {'metric':<12}" + "".join(f"{c:>16}" for c in cols),
    ]
    for layer in sorted(med["layer"].unique()):
        for metric in ["ward", "functional"]:
            r = med[(med.layer == layer) & (med.metric == metric)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"  {layer:>5} {metric:<12}"
                         + "".join(f"{r[c]:>16.3f}" for c in cols))
    # overall verdict
    piv = caps.pivot_table(index=["seed", "layer"], columns="metric",
                           values="cap_err5", aggfunc="first")
    if {"ward", "functional"} <= set(piv.columns):
        wins = int((piv["functional"] > piv["ward"]).sum())
        ties = int((piv["functional"] == piv["ward"]).sum())
        n = len(piv)
        lines += ["", f"cap_err5 head-to-head over {n} (seed, layer) runs: "
                      f"functional wins {wins}, ties {ties}, ward wins {n - wins - ties}"]
    return "\n".join(lines)


def make_plots(df: pd.DataFrame, plot_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"ward": "tab:blue", "functional": "tab:red"}
    layers = sorted(df["layer"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(4.6 * len(layers), 3.6),
                             squeeze=False)
    for ax, layer in zip(axes[0], layers):
        for (metric, seed), g in df[df.layer == layer].groupby(["metric", "seed"]):
            g = g.sort_values("step")
            ax.plot(g["frac_merged"], g["layer_rel_err"].clip(lower=1e-9),
                    color=colors[metric], alpha=0.55, lw=1.2,
                    label=metric if seed == df["seed"].min() else None)
        ax.set_yscale("log")
        ax.set_xlabel("fraction merged")
        ax.set_ylabel("layer rel err")
        ax.set_title(f"layer {layer}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("pair-selection metric: Ward/cylinder (blue) vs functional (red)")
    fig.tight_layout()
    fig.savefig(plot_dir / "rel_err_curves.png", dpi=130)
    plt.close(fig)


def run_dir_compare(run_dir: Path, stride: int) -> Path:
    config = ExperimentConfig.from_yaml(run_dir / "config.yaml")
    out_dir = STUDY_ROOT / "outputs" / \
        f"compare_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True)
    logging.info(f"=== {config.name}: comparing metrics, checkpoints from {run_dir}")

    all_records = []
    for seed in config.seeds:
        torch.manual_seed(seed)
        bundle = build_dataset(config.data.kind, data_seed=seed, **config.data.params)
        model = build_model(config.model.kind, bundle, **config.model.params)
        state = torch.load(run_dir / "models" / f"seed_{seed}.pt", weights_only=True)
        model.load_state_dict(state)
        model = model.cpu().eval()

        x_train = bundle.train_ds.tensors[0].double().numpy()
        boxes = input_boxes(model, x_train)
        for layer_idx in range(model.n_prunable_layers()):
            lo, hi = boxes[layer_idx]
            units, ok = extract_units(model, layer_idx)
            layer = model.prunable_layer(layer_idx)
            W = layer.weight.data.double().numpy()
            b = layer.bias.data.double().numpy()
            C = model.outgoing_weights(layer_idx).double().numpy()
            ctx = make_eval_ctx(model, bundle, layer_idx, (W[~ok], b[~ok], C[~ok]))
            for arm in ARMS:
                recs = sweep_arm(arm, model, bundle, layer_idx, lo, hi, seed, stride, ctx)
                logging.info(f"[{config.name}] seed {seed} layer {layer_idx} {arm}: "
                             f"final rel err {recs[-1]['layer_rel_err']:.3f}")
                all_records.extend(recs)

    df = pd.DataFrame(all_records)
    df.to_csv(out_dir / "steps.csv", index=False)
    caps = capacities(df)
    caps.to_json(out_dir / "capacities.json", orient="records", indent=2)
    make_plots(df, out_dir / "plots")
    report = format_report(config.name, caps)
    (out_dir / "report.txt").write_text(report)
    logging.info("\n" + report)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", nargs="+", required=True,
                        help="run_study.py output dir(s) with config.yaml + models/")
    parser.add_argument("--stride", type=int, default=1,
                        help="snapshot every N merge steps")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    for rd in args.run_dir:
        run_dir_compare(Path(rd), args.stride)


if __name__ == "__main__":
    main()
