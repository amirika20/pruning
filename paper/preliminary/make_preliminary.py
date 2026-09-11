"""Preliminary figures for reading the paper-tier results. NOT paper figures.

    python paper/preliminary/make_preliminary.py [--out outputs/benchmark]

Writes, beside this script:

  prelim_repaired.{pdf,png}    every method under its repair, per model
  prelim_norepair.{pdf,png}    every method with no data-fitted repair, per model
  prelim_measure.{pdf,png}     MASH: sampled (solid) vs Gaussian (dashed) score,
                               same emission, FC models only
  prelim_dictionary.{pdf,png}  MASH, no repair: merge vs medoid emission
  prelim_timing.{pdf,png}      plan / solve / overhead seconds per arm on the
                               big models, plus prelim_timing.csv
  prelim_overlap.{pdf,png}     removed-unit overlap of the best MASH arm with
                               OSSCAR, magnitude and MASH-cylinder, vs chance

Mean over seeds with a min-max band; an arm whose cell has not run is skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

MODELS = [  # (cell prefix, title, metric)
    ("mnist_lenet", "LeNet-5 / MNIST", "acc"),
    ("cifar10_resnet20", "ResNet-20 / CIFAR-10", "acc"),
    ("cifar10_resnet56", "ResNet-56 / CIFAR-10", "acc"),
    ("imagenet_resnet18", "ResNet-18 / ImageNet", "acc"),
    ("imagenet_resnet50", "ResNet-50 / ImageNet", "acc"),
    ("imagenet_mobilenetv2", "MobileNetV2 / ImageNet", "acc"),
    ("imagenet_vit_b16", "ViT-B/16 / ImageNet", "acc"),
    ("wikitext_opt125m", "OPT-125m / WikiText-2", "ppl"),
    ("wikitext_opt350m", "OPT-350m / WikiText-2", "ppl"),
    ("wikitext_opt1.3b", "OPT-1.3b / WikiText-2", "ppl"),
    ("wikitext_opt2.7b", "OPT-2.7b / WikiText-2", "ppl"),
    ("wikitext_opt6.7b", "OPT-6.7b / WikiText-2", "ppl"),
]
FC = {"imagenet_vit_b16", "wikitext_opt125m", "wikitext_opt350m", "wikitext_opt1.3b",
      "wikitext_opt2.7b", "wikitext_opt6.7b"}
# fixed categorical order (validated default palette)
HUE = {"random": "#eda100", "magnitude": "#1baf7a", "OSSCAR": "#eb6834",
       "MASH delta_f": "#2a78d6", "MASH cylinder": "#e87ba4", "MASH delete": "#008300",
       "merge": "#2a78d6", "medoid": "#eb6834"}
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"
OUT = ROOT / "outputs" / "benchmark"


def dict_for(model: str) -> str:
    return "merge" if model in FC else "medoid"


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
    ax.plot(x, d[metric], color=color, ls=ls, lw=1.8, marker="o", ms=2.6, mec="white",
            mew=0.6, label=label)


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


def curves_figure(name, note, arms_for, models=MODELS, ncol=4):
    """arms_for(model) -> list of (label, arm, linestyle, hue key)."""
    models = [m for m in models if any(load(f"{m[0]}__{a}") is not None
                                       for _, a, _, _ in arms_for(m[0]))]
    if not models:
        print(f"{name}: nothing to draw"); return
    ncol = min(ncol, len(models)); nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.1 * nrow), squeeze=False)
    for ax, (cell, title, metric) in zip(axes.flat, models):
        n_seeds = 0
        for label, arm, ls, hue in arms_for(cell):
            d = load(f"{cell}__{arm}")
            if d is None:
                continue
            draw(ax, d, metric, HUE[hue], ls, label)
            n_seeds = max(n_seeds, int(d.n.max()))
        style(ax, title, metric, n_seeds)
        if n_seeds:
            ax.legend(frameon=False, fontsize=6.3, ncol=1 if metric == "ppl" else 2,
                      handlelength=2.2, loc="lower right" if metric == "ppl" else "lower left")
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    fig.text(0.01, -0.01, note, fontsize=7, color=INK2, ha="left", va="top")
    fig.tight_layout(w_pad=1.2, h_pad=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig); print("wrote", HERE / f"{name}.{{pdf,png}}")


def repaired(m):
    dc = dict_for(m)
    return [("random + ridge repair", "random_ridge", "-", "random"),
            ("magnitude + ridge repair", "magnitude_mass_ridge", "-", "magnitude"),
            ("OSSCAR (own repair)", "osscar", "-", "OSSCAR"),
            (f"MASH delta_f ({dc}) + ridge", f"mash_ridge_{dc}_empirical_delta_f", "-", "MASH delta_f"),
            (f"MASH cylinder ({dc}) + ridge", f"mash_ridge_{dc}_empirical_cylinder", "-", "MASH cylinder")]


def norepair(m):
    dc = dict_for(m)
    return [("random, delete", "random", "-", "random"),
            ("magnitude, delete", "magnitude_mass", "-", "magnitude"),
            ("OSSCAR, delete", "osscar_norepair", "-", "OSSCAR"),
            ("MASH, delete (medoid, none)", "mash_medoid_none_delta_f", "--", "MASH delete"),
            ("MASH, drop both of each pair", "mash_drop_none_delta_f", ":", "MASH delete"),
            (f"MASH delta_f, {dc} + sum", f"mash_{dc}_sum_delta_f", "-", "MASH delta_f"),
            (f"MASH cylinder, {dc} + sum", f"mash_{dc}_sum_cylinder", "-", "MASH cylinder")]


def measure(m):
    return [("delete: sampled", "mash_medoid_none_delta_f", "-", "MASH delete"),
            ("delete: Gaussian", "mash_gaussian_medoid_none_delta_f", "--", "MASH delete"),
            ("merge+sum: sampled", "mash_merge_sum_delta_f", "-", "MASH delta_f"),
            ("merge+sum: Gaussian", "mash_gaussian_merge_sum_delta_f", "--", "MASH delta_f"),
            ("merge+ridge: sampled", "mash_ridge_merge_empirical_delta_f", "-", "MASH cylinder"),
            ("merge+ridge: Gaussian", "mash_ridge_gaussian_merge_empirical_delta_f", "--", "MASH cylinder")]


def dictionary(m):
    return [("merge + sum, delta_f", "mash_merge_sum_delta_f", "-", "merge"),
            ("medoid + sum, delta_f", "mash_medoid_sum_delta_f", "-", "medoid"),
            ("merge + sum, cylinder", "mash_merge_sum_cylinder", "--", "merge"),
            ("medoid + sum, cylinder", "mash_medoid_sum_cylinder", "--", "medoid"),
            ("medoid, delete", "mash_medoid_none_delta_f", ":", "medoid")]


TIMING_MODELS = ["wikitext_opt350m", "imagenet_vit_b16", "imagenet_resnet50", "wikitext_opt1.3b",
                 "wikitext_opt2.7b", "wikitext_opt6.7b"]
TIMING_ARMS = [("random", "random_ridge"), ("magnitude", "magnitude_mass_ridge"),
               ("OSSCAR", "osscar"), ("MASH delete", "mash_medoid_none_delta_f"),
               ("MASH merge+sum", "mash_merge_sum_delta_f"),
               ("MASH medoid+sum", "mash_medoid_sum_delta_f"),
               ("MASH merge+ridge", "mash_ridge_merge_empirical_delta_f"),
               ("MASH medoid+ridge", "mash_ridge_medoid_empirical_delta_f"),
               ("MASH Gaussian merge+ridge", "mash_ridge_gaussian_merge_empirical_delta_f"),
               ("MASH cylinder merge+sum", "mash_merge_sum_cylinder")]


def timing():
    rows = []
    for m in TIMING_MODELS:
        for label, arm in TIMING_ARMS:
            reps = [json.loads(p.read_text()) for p in sorted((OUT / f"{m}__{arm}").glob("seed_*/report.json"))]
            if not reps:
                continue
            plan = np.mean([r.get("plan_seconds", 0) for r in reps])
            total = np.mean([r.get("total_seconds", 0) for r in reps])
            wall = np.mean([r.get("wall_seconds", 0) for r in reps])
            widths = max(1, len(reps[0].get("grid", [])) or 1)
            rows.append(dict(model=m, arm=label, cell=arm, seeds=len(reps), plan_s=plan,
                             solve_s=total - plan, overhead_s=max(wall - total, 0), wall_s=wall,
                             reused=any(r.get("plan_reused_from") for r in reps)))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "prelim_timing.csv", index=False)
    models = [m for m in TIMING_MODELS if (df.model == m).any()]
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.6), squeeze=False)
    for ax, m in zip(axes.flat, models):
        d = df[df.model == m].set_index("arm").reindex([a for a, _ in TIMING_ARMS]).dropna(subset=["wall_s"])
        y = np.arange(len(d))
        ax.barh(y, d.plan_s / 3600, color="#2a78d6", label="plan (one pass, all widths)")
        ax.barh(y, d.solve_s / 3600, left=d.plan_s / 3600, color="#eb6834", label="solve (per width: select/repair/eval)")
        ax.barh(y, d.overhead_s / 3600, left=(d.plan_s + d.solve_s) / 3600, color="#c3c2b7", label="load + dense eval")
        for yi, (arm, r) in zip(y, d.iterrows()):
            if r.reused:
                ax.text(r.wall_s / 3600, yi, " plan reused", va="center", fontsize=6, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels(d.index, fontsize=7); ax.invert_yaxis()
        ax.set_xlabel("hours per seed"); ax.set_title(m.replace("wikitext_", "").replace("imagenet_", ""), loc="left")
        ax.grid(True, axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.text(0.01, -0.02, "Mean over seeds from report.json. MASH cells whose dendrogram came from a sibling "
             "(plan reuse) show no plan bar. OSSCAR has no plan: its Hessian is rebuilt at every width.",
             fontsize=7, color=INK2, ha="left", va="top")
    fig.tight_layout(w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"prelim_timing.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig); print("wrote", HERE / "prelim_timing.{pdf,png,csv}")


def overlap():
    """|A n B| / min(|A|,|B|) per layer, averaged over layers, one curve per pair,
    against the chance level (the removed fraction itself). Seed 0 of each cell;
    removal sets do not depend on the repair, so the sum-rule / no-repair cells
    are the canonical source. MASH's removed set is the same for merge and medoid
    (both keep each cluster's heaviest member), so the FC and BN arms compare alike."""
    import sys
    sys.path.insert(0, str(ROOT))
    from src.analysis.pruning_detail import overlap_from_removals
    models = [m for m in MODELS if (OUT / f"{m[0]}__osscar_norepair").exists()]
    ncol = 4; nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
    pairs = [("MASH vs OSSCAR", "osscar_norepair", "#eb6834"),
             ("MASH vs magnitude", "magnitude_mass", "#1baf7a"),
             ("MASH vs MASH-cylinder", None, "#e87ba4")]
    for ax, (cell, title, _) in zip(axes.flat, models):
        dc = dict_for(cell)
        runs = {"mash": f"{cell}__mash_{dc}_sum_delta_f", "osscar_norepair": f"{cell}__osscar_norepair",
                "magnitude_mass": f"{cell}__magnitude_mass", "cyl": f"{cell}__mash_{dc}_sum_cylinder"}
        runs = {k: OUT / v / "seed_0" for k, v in runs.items() if (OUT / v / "seed_0" / "removals.json").exists()}
        if "mash" not in runs:
            ax.axis("off"); continue
        df = overlap_from_removals(runs)
        drawn = False
        for label, other, color in pairs:
            key = other or "cyl"
            if key not in runs:
                continue
            sub = df[((df.a == "mash") & (df.b == key)) | ((df.a == key) & (df.b == "mash"))]
            g = sub.groupby("fraction").agg(ov=("overlap", "mean"), lo=("overlap", "min"), hi=("overlap", "max"),
                                            chance=("chance", "mean")).reset_index()
            ax.fill_between(g.fraction * 100, g.lo, g.hi, color=color, alpha=0.12, lw=0)
            ax.plot(g.fraction * 100, g.ov, color=color, lw=1.8, marker="o", ms=2.6, mec="white", mew=0.6, label=label)
            drawn = True
        if drawn:
            ax.plot(g.fraction * 100, g.chance, color=INK2, ls="--", lw=1, label="chance (= fraction)")
        ax.set_ylim(0, 1); ax.set_xlim(left=0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(f"{title}   (seed 0)", loc="left", color=INK)
        ax.set_xlabel("units removed (%)"); ax.set_ylabel("overlap  |A∩B| / min(|A|,|B|)")
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if drawn:
            ax.legend(frameon=False, fontsize=6.3, loc="upper left")
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    fig.text(0.01, -0.01, "Removed-unit overlap per layer (mean line, min-max band over layers) between MASH "
             "delta_f (sampled; merge on FC, medoid under BN) and each other selection. The dashed line is "
             "the overlap two independent random sets of that size would have.", fontsize=7, color=INK2, ha="left", va="top")
    fig.tight_layout(w_pad=1.2, h_pad=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"prelim_overlap.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig); print("wrote", HERE / "prelim_overlap.{pdf,png}")


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(); OUT = Path(args.out)
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.labelcolor": INK, "text.color": INK, "font.family": "DejaVu Sans"})
    curves_figure("prelim_repaired",
                  "Every method under its repair: OSSCAR its own damped least squares, the rest the "
                  "empirical repair at ridge 1e-2. MASH dictionary: merge on FC models, medoid under BatchNorm.",
                  repaired)
    curves_figure("prelim_norepair",
                  "No data-fitted repair anywhere. Deletion arms remove their set outright; MASH's sum-rule "
                  "arms move the absorbed units' columns onto the survivor (merge on FC, medoid under BN).",
                  norepair)
    curves_figure("prelim_measure",
                  "MASH delta_f score from sample Grams (solid, the default) vs rectified-Gaussian moments "
                  "(dashed), same emission. FC models only: the Gaussian score has no conv form.",
                  measure, models=[m for m in MODELS if m[0] in FC], ncol=3)
    curves_figure("prelim_dictionary",
                  "MASH with no data-fitted repair: merged hyperplane (blue) vs kept medoid (orange), sum "
                  "rule; dotted = medoid deleted outright. Merge exists only where no BatchNorm.",
                  dictionary, ncol=4)
    timing()
    overlap()


if __name__ == "__main__":
    main()
