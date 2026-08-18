"""Which Gram properties are stable under iterative merging -- and which
predict actual damage?

For every tracked property we compute, per (seed, layer):

    drift(t)      |prop(t)/prop(0) - 1|   (aff3: 1 - aff3;  m1: 1 - cos)
    frac@5%       fraction of the layer merged when drift first exceeds 5%
                  (1.0 = never drifted -- maximally stable)
    spearman_err  Spearman correlation of drift with the actual layer output
                  error across the merge trajectory -- a property can only
                  serve as a data-free stopping signal if this is high

plus three theory checks:
    m1_raw        conserved exactly (assert-level)
    ward=trace    predicted: trace loss of the wq Gram ~ accumulated Ward cost
    bound vs err  Spearman of the certified bound with the actual error
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from studies.gram_stability.gram import GRAM_DEFS

RATIO_PROPS = ["trace", "opnorm", "fro", "stable_rank", "eff_rank", "eig1"]
TRACKED = [f"{d}_{p}" for d in GRAM_DEFS for p in RATIO_PROPS] + \
          [f"{d}_aff3" for d in GRAM_DEFS] + ["m1_real_cos", "m1_real_ratio"]


def _drift(g: pd.DataFrame, col: str) -> pd.Series:
    x = g[col].astype(float)
    if col.endswith("aff3") or col.endswith("_cos"):
        return (1.0 - x).clip(lower=0)
    x0 = x.iloc[0]
    if not np.isfinite(x0) or abs(x0) < 1e-30:
        return pd.Series(np.nan, index=x.index)
    return (x / x0 - 1.0).abs()


def _frac_at(g: pd.DataFrame, drift: pd.Series, thresh: float) -> float:
    hit = g.loc[drift > thresh, "frac_merged"]
    return float(hit.iloc[0]) if len(hit) else 1.0


def analyze_steps(df: pd.DataFrame) -> dict:
    per_run: dict[str, dict] = {}
    rows = []
    checks_rows = []

    for (seed, layer), g in df.groupby(["seed", "layer"]):
        g = g.sort_values("step").reset_index(drop=True)
        err = g["layer_rel_err"].astype(float)
        run_key = f"seed{seed}_layer{layer}"
        run: dict[str, dict] = {}
        for col in TRACKED:
            if col not in g:
                continue
            d = _drift(g, col)
            entry = {
                "frac_at_5pct": _frac_at(g, d, 0.05),
                "frac_at_1pct": _frac_at(g, d, 0.01),
                "spearman_err": float(d.corr(err, method="spearman")),
            }
            run[col] = entry
            rows.append({"prop": col, **entry})
        per_run[run_key] = run

        # theory checks on this trajectory
        wq0 = float(g["wq_trace"].iloc[0])
        trace_loss = wq0 - g["wq_trace"].astype(float)
        cum = g["cum_ward"].astype(float)
        mask = cum > 1e-12 * max(wq0, 1e-30)
        ratio = (trace_loss[mask] / cum[mask]).median() if mask.any() else np.nan
        checks_rows.append({
            "run": run_key,
            "m1_raw_max_drift": float((g["m1_raw_ratio"] - 1).abs().max()),
            "ward_trace_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
            "bound_vs_err_spearman": float(
                g["bound_total"].astype(float).corr(err, method="spearman")),
            "max_overshoot": float(g["overshoot"].max()),
        })

    agg = (pd.DataFrame(rows).groupby("prop")
           .median(numeric_only=True).reset_index()
           .sort_values("frac_at_5pct", ascending=False))
    checks = pd.DataFrame(checks_rows)

    return {
        "ranking": agg.to_dict(orient="records"),
        "checks": {
            "m1_raw_max_drift": float(checks["m1_raw_max_drift"].max()),
            "ward_trace_ratio_median": float(checks["ward_trace_ratio"].median()),
            "bound_vs_err_spearman_median": float(checks["bound_vs_err_spearman"].median()),
            "max_overshoot": float(checks["max_overshoot"].max()),
        },
        "per_run": per_run,
    }


# ── report ────────────────────────────────────────────────────────────────────

def format_report(name: str, summary: dict) -> str:
    lines = [
        f"Gram-stability study: {name}",
        "=" * 64,
        "",
        "Ranking (medians over seeds x layers).",
        "  frac@5%: fraction of the layer merged before the property drifts",
        "           by 5% -- higher = more stable under merging.",
        "  rho_err: Spearman(drift, actual layer error) -- higher = better",
        "           data-free stopping signal.",
        "",
        f"  {'property':<24}{'frac@5%':>9}{'frac@1%':>9}{'rho_err':>9}",
        f"  {'-' * 51}",
    ]
    for r in summary["ranking"]:
        lines.append(
            f"  {r['prop']:<24}{r['frac_at_5pct']:>9.3f}{r['frac_at_1pct']:>9.3f}"
            f"{r['spearman_err']:>9.3f}")

    c = summary["checks"]
    lines += [
        "",
        "Theory checks",
        f"  m1_raw conservation: max drift {c['m1_raw_max_drift']:.2e}  (predicted: exact)",
        f"  wq trace loss / cumulative Ward cost: median {c['ward_trace_ratio_median']:.3f}"
        "  (predicted ~1)",
        f"  certified bound vs actual error: median Spearman "
        f"{c['bound_vs_err_spearman_median']:.3f}",
        f"  max overshoot A_C/||g_C||: {c['max_overshoot']:.3f}"
        "  (1 = slope-matched; grows with cluster angular spread)",
        "",
        "Reading: a property is a useful invariant if it is BOTH stable early",
        "(large frac@5%) AND predictive when it finally moves (high rho_err).",
        "Stable-but-blind properties (frac@5% ~ 1, rho_err ~ 0) certify nothing.",
    ]
    return "\n".join(lines)


# ── plots ────────────────────────────────────────────────────────────────────

def make_plots(df: pd.DataFrame, plot_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted(df["layer"].unique())
    cmap = plt.get_cmap("viridis", max(len(layers), 2))

    def by_run(ax, col, transform=lambda s: s, logy=False):
        for (seed, layer), g in df.groupby(["seed", "layer"]):
            g = g.sort_values("step")
            y = transform(g[col].astype(float))
            ax.plot(g["frac_merged"], y, color=cmap(layers.index(layer)),
                    alpha=0.6, lw=1.2)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("fraction of layer merged")
        ax.set_title(col, fontsize=9)
        ax.grid(alpha=0.25)

    # per-definition property trajectories (normalized to step 0)
    for name in GRAM_DEFS:
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
        for ax, prop in zip(axes, ["trace", "opnorm", "eff_rank", "aff3"]):
            col = f"{name}_{prop}"
            if prop == "aff3":
                by_run(ax, col)
                ax.set_ylim(-0.02, 1.02)
            else:
                for (seed, layer), g in df.groupby(["seed", "layer"]):
                    g = g.sort_values("step")
                    x0 = float(g[col].iloc[0])
                    if abs(x0) < 1e-30:
                        continue
                    ax.plot(g["frac_merged"], g[col].astype(float) / x0,
                            color=cmap(layers.index(layer)), alpha=0.6, lw=1.2)
                ax.set_title(f"{col} / step0", fontsize=9)
                ax.set_xlabel("fraction of layer merged")
                ax.grid(alpha=0.25)
        fig.suptitle(f"B = {name}: Gram properties under iterative merging (color = layer)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"trajectories_{name}.png", dpi=130)
        plt.close(fig)

    # error / certificate curves
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    by_run(axes[0], "layer_rel_err", transform=lambda s: s.clip(lower=1e-12), logy=True)
    by_run(axes[1], "bound_total", transform=lambda s: s.clip(lower=1e-12), logy=True)
    by_run(axes[2], "ward_cost", transform=lambda s: s.clip(lower=1e-18), logy=True)
    if df["val_acc"].notna().any():
        by_run(axes[3], "val_acc")
    else:
        by_run(axes[3], "val_loss", logy=True)
    fig.suptitle("actual error, certified bound, per-step Ward cost (elbow), model quality")
    fig.tight_layout()
    fig.savefig(plot_dir / "error_and_certificates.png", dpi=130)
    plt.close(fig)

    # drift-vs-error scatter for a few key properties
    keys = ["wq_trace", "wcov_trace", "anchor_trace", "cov_eff_rank", "anchor_aff3", "wq_aff3"]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2))
    for ax, col in zip(np.atleast_1d(axes), keys):
        for (seed, layer), g in df.groupby(["seed", "layer"]):
            g = g.sort_values("step")
            d = _drift(g, col)
            ax.scatter(d, g["layer_rel_err"].astype(float).clip(lower=1e-12),
                       s=6, alpha=0.4, color=cmap(layers.index(layer)))
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_yscale("log")
        ax.set_xlabel(f"{col} drift")
        ax.set_ylabel("layer rel err")
        ax.grid(alpha=0.25)
    fig.suptitle("does Gram drift track actual error?")
    fig.tight_layout()
    fig.savefig(plot_dir / "drift_vs_error.png", dpi=130)
    plt.close(fig)
