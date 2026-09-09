"""Accuracy / perplexity vs fraction of units removed, four arms, three models.

    python studies/paper/figs/make_curves_figure.py

Reads outputs/benchmark/<cell>/seed_*/curve.csv. Mean over seeds, min-max band.
MASH is the scale-tier recipe for each family: merge dictionary on the OPT FFN,
medoid on the BatchNorm ResNets (merge is refused under BN).
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "benchmark"
HERE = Path(__file__).resolve().parent

PANELS = [  # (cell prefix, title, mash arm, metric, ylabel)
    ("wikitext_opt350m", "OPT-350m / WikiText-2", "mash_merge_empirical_delta_f", "ppl", "perplexity"),
    ("imagenet_resnet18", "ResNet-18 / ImageNet-1k", "mash_medoid_empirical_delta_f", "acc", "top-1 accuracy"),
    ("cifar10_resnet56", "ResNet-56 / CIFAR-10", "mash_medoid_empirical_delta_f", "acc", "top-1 accuracy"),
]
# fixed categorical order: slot 1 blue, 2 orange, 3 aqua, 4 yellow
ARMS = [("MASH", None, "#2a78d6", "-"), ("OSSCAR", "osscar", "#eb6834", "-"),
        ("magnitude", "magnitude_mass", "#1baf7a", "-"), ("random", "random", "#eda100", "--")]
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"


def load(cell):
    frames = [pd.read_csv(p) for p in sorted((OUT / cell).glob("seed_*/curve.csv"))]
    if not frames:
        return None
    df = pd.concat(frames)
    df["ppl"] = np.exp(df.val_loss.clip(upper=60))
    g = df.groupby("fraction")
    return g.agg(removed=("removed", "mean"), acc=("val_acc", "mean"), acc_lo=("val_acc", "min"),
                 acc_hi=("val_acc", "max"), ppl=("ppl", "mean"), ppl_lo=("ppl", "min"),
                 ppl_hi=("ppl", "max"), n=("seed", "nunique")).reset_index()


plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
for ax, (cell, title, mash_arm, metric, ylabel) in zip(axes, PANELS):
    for label, arm, color, ls in ARMS:
        arm = arm or mash_arm
        d = load(f"{cell}__{arm}")
        if d is None:
            print(f"missing {cell}__{arm}"); continue
        x = d.removed * 100
        ax.fill_between(x, d[f"{metric}_lo"], d[f"{metric}_hi"], color=color, alpha=0.15, lw=0)
        ax.plot(x, d[metric], color=color, ls=ls, lw=2, marker="o", ms=3.5, mec="white", mew=0.8,
                label=label)
        n_seeds = int(d.n.max())
    if metric == "ppl":
        ax.set_yscale("log")
        ax.set_ylim(15, 2000)
        ax.set_yticks([20, 50, 100, 300, 1000]); ax.set_yticklabels(["20", "50", "100", "300", "1000"])
        ax.minorticks_off()
    else:
        ax.set_ylim(0, max(0.96, ax.get_ylim()[1]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(f"{title}   ({n_seeds} seed{'s' if n_seeds > 1 else ''})", loc="left", color=INK)
    ax.text(0.99, 0.02, "MASH: " + mash_arm.replace("mash_", "").replace("_", " "),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK2)
    ax.set_xlabel("units removed (%)")
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, handlelength=1.8,
              loc="upper left" if metric == "ppl" else "lower left")
    ax.set_xlim(left=0)
fig.tight_layout(w_pad=1.5)
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig_curves_four_arms.{ext}", dpi=200, bbox_inches="tight")
print("wrote", HERE / "fig_curves_four_arms.{pdf,png}")
