"""Paper figures for the two selection-vs-repair experiments (repair tier).

    python studies/paper/figs/make_repair_figures.py [--models a b ...]

Reads outputs/benchmark/<model>__<arm>/seed_*/curve.csv, mean over seeds with
a min-max band, and writes three figures beside this script:

  fig_repair_effect      experiment 1 -- each method with (solid) and without
                         (dashed) repair. Hue = method, line style = repair.
  fig_repair_ridge       experiment 2 -- the empirical repair at ridge 1e-8
                         (solid) vs 1e-2 (dotted) for random, magnitude, MASH.
  fig_repair_measure     MASH's delta_f score from Gaussian moments vs sample
                         Grams, under the empirical repair, medoid and merge.

Panels are models; a panel is skipped (with a note) when a model has none of
the arms, and an arm is skipped when its cell has not run yet, so the script
can be run against a partial sweep. The MASH arm in experiments 1 and 2 is the
MEDOID dictionary, because "without repair" is only defined for survivors that
are original units; merge appears in the measure figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "benchmark"
HERE = Path(__file__).resolve().parent

MODELS = [  # (cell prefix, title, metric)
    ("cifar10_resnet20", "ResNet-20 / CIFAR-10", "acc"),
    ("cifar10_resnet56", "ResNet-56 / CIFAR-10", "acc"),
    ("imagenet_resnet18", "ResNet-18 / ImageNet-1k", "acc"),
    ("imagenet_resnet50", "ResNet-50 / ImageNet-1k", "acc"),
    ("imagenet_mobilenetv2", "MobileNetV2 / ImageNet-1k", "acc"),
    ("imagenet_vit_b16", "ViT-B/16 / ImageNet-1k", "acc"),
    ("wikitext_opt125m", "OPT-125m / WikiText-2", "ppl"),
    ("wikitext_opt350m", "OPT-350m / WikiText-2", "ppl"),
    ("wikitext_opt1.3b", "OPT-1.3b / WikiText-2", "ppl"),
]
# fixed categorical order (validated default palette): blue, orange, aqua, yellow, magenta
HUE = {"MASH": "#2a78d6", "OSSCAR": "#eb6834", "magnitude": "#1baf7a",
       "random": "#eda100", "MASH (merge)": "#e87ba4"}
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"

# experiment 1: (label, arm without repair, arm with repair)
REPAIR = [("random", "random", "random_empirical"),
          ("magnitude", "magnitude_mass", "magnitude_mass_empirical"),
          ("OSSCAR", "osscar_norepair", "osscar"),
          ("MASH", "mash_medoid_none_delta_f", "mash_medoid_empirical_delta_f")]
# experiment 2: (label, ridge 1e-8 arm, ridge 1e-2 arm)
RIDGE = [("random", "random_empirical", "random_ridge"),
         ("magnitude", "magnitude_mass_empirical", "magnitude_mass_ridge"),
         ("MASH", "mash_medoid_empirical_delta_f", "mash_ridge_medoid_empirical_delta_f")]
# measure: (label, gaussian arm, sampled arm)
MEASURE = [("MASH", "mash_medoid_empirical_delta_f", "mash_sampled_medoid_empirical_delta_f"),
           ("MASH (merge)", "mash_merge_empirical_delta_f", "mash_sampled_merge_empirical_delta_f")]


def load(cell: str) -> pd.DataFrame | None:
    frames = [pd.read_csv(p) for p in sorted((OUT / cell).glob("seed_*/curve.csv"))]
    if not frames:
        return None
    df = pd.concat(frames)
    df["ppl"] = np.exp(df.val_loss.clip(upper=60))
    g = df.groupby("fraction")
    return g.agg(removed=("removed", "mean"), acc=("val_acc", "mean"),
                 acc_lo=("val_acc", "min"), acc_hi=("val_acc", "max"),
                 ppl=("ppl", "mean"), ppl_lo=("ppl", "min"), ppl_hi=("ppl", "max"),
                 n=("seed", "nunique")).reset_index()


def draw(ax, d, metric, color, ls, label):
    x = d.removed * 100
    ax.fill_between(x, d[f"{metric}_lo"], d[f"{metric}_hi"], color=color, alpha=0.12, lw=0)
    ax.plot(x, d[metric], color=color, ls=ls, lw=2, marker="o", ms=3, mec="white",
            mew=0.7, label=label)


def style(ax, title, metric, n_seeds):
    if metric == "ppl":
        ax.set_yscale("log"); ax.set_ylim(15, 2000); ax.minorticks_off()
        ax.set_yticks([20, 50, 100, 300, 1000])
        ax.set_yticklabels(["20", "50", "100", "300", "1000"])
        ax.set_ylabel("perplexity")
    else:
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("top-1 accuracy")
    seeds = f"{n_seeds} seed{'s' if n_seeds != 1 else ''}" if n_seeds else "no cells yet"
    ax.set_title(f"{title}   ({seeds})", loc="left", color=INK)
    ax.set_xlabel("units removed (%)")
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def figure(models, pairs, styles, name, note):
    """One panel per model; each pair draws two arms in one hue with two styles."""
    n = len(models)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.1 * nrow), squeeze=False)
    drawn_any = False
    for ax, (cell, title, metric) in zip(axes.flat, models):
        n_seeds = 0
        for label, arm_a, arm_b in pairs:
            for arm, (ls, suffix) in zip((arm_a, arm_b), styles):
                d = load(f"{cell}__{arm}")
                if d is None:
                    continue
                draw(ax, d, metric, HUE[label], ls, f"{label} {suffix}")
                n_seeds = max(n_seeds, int(d.n.max()))
                drawn_any = True
        style(ax, title, metric, n_seeds)
        if n_seeds:
            # perplexity curves rise from the lower left, so the empty corner
            # is the lower right; accuracy curves fall, so it is the lower left
            ax.legend(frameon=False, fontsize=6.5, ncol=2, handlelength=2.2,
                      loc="lower right" if metric == "ppl" else "lower left")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.text(0.01, -0.01, note, fontsize=7, color=INK2, ha="left", va="top")
    fig.tight_layout(w_pad=1.2, h_pad=1.5)
    if drawn_any:
        for ext in ("pdf", "png"):
            fig.savefig(HERE / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        print("wrote", HERE / f"{name}.{{pdf,png}}")
    else:
        print(f"{name}: no cells found, nothing written")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="cell prefixes to include (default: all nine)")
    args = ap.parse_args()
    models = [m for m in MODELS if not args.models or m[0] in set(args.models)]

    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.labelcolor": INK, "text.color": INK, "font.family": "DejaVu Sans"})
    figure(models, REPAIR, [("--", "(no repair)"), ("-", "(repaired)")], "fig_repair_effect",
           "Dashed: the arm's removed set deleted outright. Solid: the consumer re-solved -- "
           "OSSCAR by its own damped least squares, the others by the empirical repair. "
           "MASH = medoid dictionary, delta_f score (Gaussian moments).")
    figure(models, RIDGE, [("-", "ridge 1e-8"), (":", "ridge 1e-2")], "fig_repair_ridge",
           "Empirical repair on each arm's removed set; ridge relative to the mean diagonal "
           "of the Gram. 1e-2 is OSSCAR's damping strength. MASH = medoid, delta_f.")
    figure(models, MEASURE, [("-", "Gaussian"), ("-.", "sampled")], "fig_repair_measure",
           "MASH delta_f score with the Grams from rectified-Gaussian moments (solid) or "
           "sample averages (dash-dot); empirical repair. Merge appears where BatchNorm allows it.")


if __name__ == "__main__":
    main()
