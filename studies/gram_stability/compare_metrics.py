#!/usr/bin/env python
"""Phase A of the design-space search (studies/DESIGN_SPACE.md): rank the
scoring rules. Every arm uses the IDENTICAL merge rule (covector addition +
sum-rule fan-out surgery) -- only pair selection differs.

Arms (design-space IDs):
    ward          1A1  Ward on covectors, sphere R0            [DOM]
    ellipsoid     1A2  Ward, per-feature ellipsoid metric      [DOM]
    func_iso      1B1  expected damage, isotropic Gaussian     [DOM]
    func_box      1B3  expected damage, box-CLT measure        [DOM]
    func_matched  1B4  expected damage, matched Gaussian       [DL: 128 calib samples]

Usage (from the repo root, with the .ml venv python):
    python studies/gram_stability/compare_metrics.py \
        --run-dir studies/gram_stability/outputs/gram_mnist_mlp_... [more dirs] \
        [--arms ward ellipsoid func_box func_matched] [--stride N]

Each --run-dir must contain config.yaml and models/seed_<s>.pt (as produced by
run_study.py); models are reloaded, never retrained.

Output per run-dir: outputs/compare_<name>_<timestamp>/
    steps.csv, capacities.json, report.txt, plots/
Stopping rules are NOT compared here -- they are post-hoc cuts of these
recorded trajectories (evaluate offline from steps.csv).
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

from studies.gram_stability.merge import EllipsoidMerge, IterativeMerge, extract_units, input_boxes
from studies.gram_stability.functional import FunctionalMerge, GaussianMeasureMerge
from studies.gram_stability.run_study import make_eval_ctx, eval_merged

STUDY_ROOT = Path(__file__).resolve().parent
N_CALIB = 128
REL_ERR_TOLS = [0.01, 0.05, 0.10]
ACC_DROP = 0.01  # 1 accuracy point
BASELINE = "ward"

ARMS = {
    "ward": lambda u, lo, hi, ex: IterativeMerge(u, lo, hi),
    "ellipsoid": lambda u, lo, hi, ex: EllipsoidMerge(u, lo, hi),
    "func_iso": lambda u, lo, hi, ex: FunctionalMerge(u, lo, hi),
    "func_box": lambda u, lo, hi, ex: GaussianMeasureMerge(
        u, lo, hi, (lo + hi) / 2.0, ((hi - lo) / 2.0) ** 2 / 3.0),
    "func_matched": lambda u, lo, hi, ex: GaussianMeasureMerge(
        u, lo, hi, ex["mu"], ex["cov"]),
}


def calib_moments(model, bundle, layer_idx: int) -> dict:
    """Layer-input mean/covariance from N_CALIB training samples (data-light
    tier; val/test never touched)."""
    X = bundle.train_ds.tensors[0][:N_CALIB]
    with torch.no_grad():
        Xl = model.net[: 2 * layer_idx](X).double().numpy()
    mu = Xl.mean(axis=0)
    cov = np.atleast_2d(np.cov(Xl.T))
    return {"mu": mu, "cov": cov}


def sweep_arm(arm: str, model, bundle, layer_idx: int, lo, hi, seed: int,
              stride: int, ctx, extras: dict) -> list[dict]:
    units, ok = extract_units(model, layer_idx)
    engine = ARMS[arm](units.subset(np.flatnonzero(ok)), lo, hi, extras)
    H = engine.n_orig

    r_orig = engine.orig.rho - engine.orig.u @ engine.x0
    sat = np.abs(r_orig) > engine.R0

    def n_sat_absorbed() -> int:
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


def format_report(name: str, caps: pd.DataFrame, arms: list[str]) -> str:
    med = caps.groupby(["metric", "layer"]).median(numeric_only=True).reset_index()
    cols = [c for c in med.columns if c.startswith("cap_")] + ["sat_absorbed_at_25pct"]
    lines = [
        f"Phase A scoring-rule comparison: {name}",
        "=" * 76,
        "",
        "Arms: ward=1A1  ellipsoid=1A2  func_iso=1B1  func_box=1B3  func_matched=1B4",
        "Capacity = fraction of the layer merged before crossing the tolerance",
        "(medians over seeds; higher = better). Same merge surgery in all arms.",
        "",
        f"  {'layer':>5} {'metric':<14}" + "".join(f"{c:>15}" for c in cols),
    ]
    for layer in sorted(med["layer"].unique()):
        for metric in arms:
            r = med[(med.layer == layer) & (med.metric == metric)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"  {layer:>5} {metric:<14}"
                         + "".join(f"{r[c]:>15.3f}" for c in cols))
        lines.append("")

    piv = caps.pivot_table(index=["seed", "layer"], columns="metric",
                           values="cap_err5", aggfunc="first")
    lines.append(f"cap_err5 head-to-head vs {BASELINE} per (seed, layer):")
    for metric in arms:
        if metric == BASELINE or metric not in piv.columns:
            continue
        w = int((piv[metric] > piv[BASELINE]).sum())
        t = int((piv[metric] == piv[BASELINE]).sum())
        n = len(piv)
        lines.append(f"  {metric:<14} wins {w:>3}  ties {t:>3}  losses {n - w - t:>3}")
    return "\n".join(lines)


def make_plots(df: pd.DataFrame, plot_dir: Path, arms: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    palette = plt.get_cmap("tab10")
    colors = {a: palette(i) for i, a in enumerate(arms)}
    layers = sorted(df["layer"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(4.8 * len(layers), 3.8),
                             squeeze=False)
    for ax, layer in zip(axes[0], layers):
        for (metric, seed), g in df[df.layer == layer].groupby(["metric", "seed"]):
            g = g.sort_values("step")
            ax.plot(g["frac_merged"], g["layer_rel_err"].clip(lower=1e-9),
                    color=colors[metric], alpha=0.5, lw=1.1,
                    label=metric if seed == df["seed"].min() else None)
        ax.set_yscale("log")
        ax.set_xlabel("fraction merged")
        ax.set_ylabel("layer rel err")
        ax.set_title(f"layer {layer}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Phase A: scoring rules (same surgery, different pair selection)")
    fig.tight_layout()
    fig.savefig(plot_dir / "rel_err_curves.png", dpi=130)
    plt.close(fig)


def run_dir_compare(run_dir: Path, stride: int, arms: list[str]) -> Path:
    config = ExperimentConfig.from_yaml(run_dir / "config.yaml")
    out_dir = STUDY_ROOT / "outputs" / \
        f"compare_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True)
    logging.info(f"=== {config.name}: Phase A arms={arms}, checkpoints from {run_dir}")

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
            extras = calib_moments(model, bundle, layer_idx) \
                if "func_matched" in arms else {}
            for arm in arms:
                recs = sweep_arm(arm, model, bundle, layer_idx, lo, hi, seed,
                                 stride, ctx, extras)
                logging.info(f"[{config.name}] seed {seed} layer {layer_idx} "
                             f"{arm}: final rel err {recs[-1]['layer_rel_err']:.3f}")
                all_records.extend(recs)

    df = pd.DataFrame(all_records)
    df.to_csv(out_dir / "steps.csv", index=False)
    caps = capacities(df)
    caps.to_json(out_dir / "capacities.json", orient="records", indent=2)
    make_plots(df, out_dir / "plots", arms)
    report = format_report(config.name, caps, arms)
    (out_dir / "report.txt").write_text(report)
    logging.info("\n" + report)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", nargs="+", required=True,
                        help="run_study.py output dir(s) with config.yaml + models/")
    parser.add_argument("--arms", nargs="+", default=list(ARMS),
                        choices=list(ARMS), help="scoring arms to run")
    parser.add_argument("--stride", type=int, default=1,
                        help="snapshot every N merge steps")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    for rd in args.run_dir:
        run_dir_compare(Path(rd), args.stride, args.arms)


if __name__ == "__main__":
    main()
