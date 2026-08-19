#!/usr/bin/env python
"""Figures for method.tex, generated from the experiment outputs
(studies/gram_stability/outputs/). Colors: validated categorical palette,
fixed slot order; recessive chrome."""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "gram_stability" / "outputs"
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed order
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False,
})


def style(ax, xlabel, ylabel, title):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, loc="left")


def med(df, keys, x, y):
    return df.groupby(keys, as_index=False)[[x, y]].median()


# ── A: scoring metrics, MNIST-MLP layer 0 (E2/E4) ────────────────────────────
df = pd.read_csv(sorted(glob.glob(str(ROOT / "compare_gram_mnist_mlp_20260818_17*/steps.csv")))[-1])
df = df[df.layer == 0]
fig, ax = plt.subplots(figsize=(4.4, 3.0))
for arm, label, c in [("ward", "Ward (domain only)", SLOT[0]),
                      ("func_iso", "expected damage, isotropic $\\mu$", SLOT[1]),
                      ("func_matched", "expected damage, matched $\\mu$", SLOT[2])]:
    g = med(df[df.metric == arm], ["step"], "frac_merged", "layer_rel_err")
    ax.plot(g.frac_merged, g.layer_rel_err.clip(lower=1e-6), color=c, lw=1.8, label=label)
ax.set_yscale("log")
ax.set_ylim(1e-6, 2)
style(ax, "fraction of layer merged", "layer output rel. error",
      "Scoring measures, MNIST layer 0 (3-seed median)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIGS / "fig_metric.pdf")
plt.close(fig)

# ── B: surgery grid, MNIST-MLP joint pruning (E8) ────────────────────────────
df = pd.read_csv(sorted(glob.glob(str(ROOT / "b_gram_mnist_mlp_*/curves.csv")))[-1])
fig, ax = plt.subplots(figsize=(4.4, 3.0))
for variant, label, c in [("survivor+sum", "survivor + sum (3M2+4F1)", SLOT[1]),
                          ("mean+sum", "mean + sum (3M1+4F1)", SLOT[0]),
                          ("mean+proj", "mean + projection (4F2)", SLOT[2]),
                          ("mean+global", "mean + global repair (4F4)", SLOT[3])]:
    g = med(df[df.variant == variant], ["frac_removed"], "frac_removed", "val_acc")
    ax.plot(g.frac_removed, g.val_acc, color=c, lw=1.8, label=label)
acc0 = float(df.acc0.iloc[0])
ax.axhline(acc0 - 0.01, color=AXIS, lw=0.8, ls=(0, (4, 3)))
ax.text(0.02, acc0 - 0.008, "$-1$pt budget", color=MUTED, fontsize=7.5)
ax.set_ylim(0.55, 1.0)
style(ax, "fraction of ALL hidden units removed (jointly)", "val. accuracy",
      "Fan-out surgery, MNIST-MLP (3-seed median)")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(FIGS / "fig_surgery.pdf")
plt.close(fig)

# ── C: CNN gap decomposition (E10/E12/E13) ───────────────────────────────────
base = pd.read_csv(sorted(glob.glob(str(ROOT / "dcnn_mnist_pixel_cnn_*/curves.csv")))[-1])
fine = pd.read_csv(sorted(glob.glob(str(ROOT / "dcnn_hybrid_mnist_pixel_cnn_*.csv")))[-1])
fig, ax = plt.subplots(figsize=(4.4, 3.0))
series = [
    (base[base.arm == "ours"], "merged units + kernel repair", SLOT[0]),
    (fine[fine.arm == "hybrid"], "our support + exact LS", SLOT[2]),
    (fine[fine.arm == "hybrid_swap"], "$+$ swap refinement", SLOT[3]),
    (base[base.arm == "osscar128"], "OSSCAR (128 images)", SLOT[1]),
]
for sub, label, c in series:
    g = med(sub, ["frac_removed"], "frac_removed", "val_acc")
    ax.plot(g.frac_removed, g.val_acc, color=c, lw=1.8, label=label)
acc0 = float(base.acc0.iloc[0])
ax.axhline(acc0 - 0.01, color=AXIS, lw=0.8, ls=(0, (4, 3)))
ax.text(0.11, acc0 - 0.008, "$-1$pt budget", color=MUTED, fontsize=7.5)
ax.set_ylim(0.4, 1.0)
style(ax, "fraction of all filters removed (jointly)", "val. accuracy",
      "CNN gap decomposition, MNIST (3-seed median)")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(FIGS / "fig_cnn.pdf")
plt.close(fig)

# ── D: certificate vs actual error (E1) ──────────────────────────────────────
df = pd.read_csv(ROOT / "gram_mnist_mlp_20260817_150158" / "steps.csv")
df = df[(df.bound_total > 0) & (df.layer_rel_err > 0)]
fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.scatter(df.bound_total, df.layer_rel_err, s=6, alpha=0.35, color=SLOT[0],
           edgecolors="none")
ax.set_xscale("log")
ax.set_yscale("log")
style(ax, "accumulated certified bound (Thm. 4.1)", "measured layer rel. error",
      "Certificate vs. truth (MNIST-MLP, all runs)")
fig.tight_layout()
fig.savefig(FIGS / "fig_bound.pdf")
plt.close(fig)

print("wrote", sorted(p.name for p in FIGS.glob("*.pdf")))

# ── E: first-moment conservation (Prop 3.3) vs realized drift ────────────────
df = pd.read_csv(ROOT / "gram_mnist_mlp_20260817_150158" / "steps.csv")
fig, ax = plt.subplots(figsize=(4.4, 3.0))
for (s, l), g in df.groupby(["seed", "layer"]):
    g = g.sort_values("step")
    raw = (g.m1_raw_ratio - 1).abs().clip(lower=1e-17)
    real = (g.m1_real_ratio - 1).abs().clip(lower=1e-17)
    ax.plot(g.frac_merged, raw, color=SLOT[0], lw=1.1, alpha=0.55,
            label="raw covector sum (exact invariant)" if (s, l) == (0, 0) else None)
    ax.plot(g.frac_merged, real, color=SLOT[1], lw=1.1, alpha=0.55,
            label="realized first moment" if (s, l) == (0, 0) else None)
ax.set_yscale("log")
ax.set_ylim(1e-17, 3)
style(ax, "fraction of layer merged", "relative drift of first moment",
      "First-moment conservation (all layers, seeds)")
ax.legend(loc="center left")
fig.tight_layout()
fig.savefig(FIGS / "fig_moment.pdf")
plt.close(fig)
