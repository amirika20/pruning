"""Main-text figures for the paper.

    python paper/main_text_figures/make_main_text.py [--out outputs/benchmark]

Writes, beside this script:

  fig_norepair.{pdf,png}   no data-fitted repair: random, magnitude, OSSCAR (delete)
                           and MASH delta_f + sum rule (merge on the ReLU FC
                           models, medoid elsewhere; tagged per panel)
  fig_repair.{pdf,png}     every method under its repair: random + ridge, magnitude
                           + ridge, OSSCAR (own damped LS), MASH delta_f + ridge
                           (merge, 20k rows on the ReLU FC models; medoid, all rows
                           elsewhere; tagged per panel). The baselines use the
                           all-rows ridge cells where they exist (Pythia, Qwen).
  fig_overlap.{pdf,png}    removed-unit overlap of MASH with OSSCAR and with
                           magnitude, per layer, against chance

Six models, one panel each: ResNet-50, ViT-B/16 (ImageNet); OPT-2.7b, OPT-6.7b,
Pythia-2.8b, Qwen2.5-7B (WikiText-2). Mean over seeds with a min-max band.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "benchmark"

MODELS = [  # (cell prefix, title, metric)
    ("imagenet_resnet50", "ResNet-50", "acc"),
    ("imagenet_vit_b16", "ViT-B/16", "acc"),
    ("wikitext_opt2.7b", "OPT-2.7b", "ppl"),
    ("wikitext_opt6.7b", "OPT-6.7b", "ppl"),
    ("wikitext_pythia2.8b", "Pythia-2.8b", "ppl"),
    ("wikitext_qwen2.5_7b", "Qwen2.5-7B", "ppl"),
]

# validated default categorical palette, fixed order: MASH, OSSCAR, magnitude, random
HUE = {"MASH": "#2a78d6", "OSSCAR": "#eb6834", "magnitude": "#1baf7a", "random": "#eda100"}
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"

NOREPAIR = [  # (legend label, arm, hue)
    ("Random", "random", "random"),
    ("Magnitude", "magnitude_mass", "magnitude"),
    ("OSSCAR", "osscar_norepair", "OSSCAR"),
    ("MASH (ours) + sum rule (variant tagged per panel)", "mash_medoid_sum_delta_f", "MASH"),
]
REPAIR = [
    ("Random + ridge", "random_ridge", "random"),
    ("Magnitude + ridge", "magnitude_mass_ridge", "magnitude"),
    ("OSSCAR", "osscar", "OSSCAR"),
    ("MASH (ours) + ridge (variant tagged per panel)", "mash_full_medoid_empirical_delta_f", "MASH"),
]
OVERLAP = [  # (legend label, arm whose removals to compare against, hue)
    ("MASH vs OSSCAR", "osscar_norepair", "OSSCAR"),
    ("MASH vs magnitude", "magnitude_mass", "magnitude"),
]
MASH_REMOVALS = "mash_medoid_sum_delta_f"


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


def draw(ax, d, metric, color, emphasis=False):
    x = d.removed * 100
    ax.fill_between(x, d[f"{metric}_lo"], d[f"{metric}_hi"], color=color, alpha=0.15, lw=0)
    ax.plot(x, d[metric], color=color, lw=2.2 if emphasis else 1.6, marker="o",
            ms=3.2 if emphasis else 2.6, mec="white", mew=0.6, zorder=3 if emphasis else 2)


def style(ax, title, metric, n_seeds, lo=None, ymax_ppl=2000):
    if metric == "ppl":
        floor = 15.0 if lo is None else min(15.0, 0.85 * lo)
        ax.set_yscale("log"); ax.set_ylim(floor, ymax_ppl); ax.minorticks_off()
        ticks = [t for t in (5, 10, 20, 50, 100, 300, 1000) if floor <= t <= ymax_ppl]
        ax.set_yticks(ticks); ax.set_yticklabels([str(t) for t in ticks])
        ax.set_ylabel("WikiText-2 perplexity")
    else:
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("ImageNet top-1")
    seeds = f"{n_seeds} seed{'s' if n_seeds != 1 else ''}"
    ax.set_title(f"{title}  ({seeds})", loc="left", color=INK, fontweight="bold", pad=4)
    ax.set_xlabel("units removed (%)")
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def legend_below(fig, handles, labels, ncol):
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               fontsize=8, handlelength=2.4, columnspacing=1.8,
               bbox_to_anchor=(0.5, -0.005))


def save(fig, name):
    name = name + SUFFIX
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote", HERE / f"{name}.{{pdf,png}}")


# ReLU fully-connected models: MASH's best arm uses the MERGE dictionary, both
# with no repair (merge + sum; OPT-6.7b ppl 37 vs 58 at 50%, ViT 43% vs 0.8%) and
# under the 20k-row ridge. The medoid + all-rows sum-centred ridge collapses
# there (OPT-6.7b: ppl 63 vs 27 at 50%; ViT: 0.7% vs 61% top-1 at 50%). Merge is
# undefined under BatchNorm (ResNet-50) and on the GELU / gated LMs (Pythia,
# Qwen), where the medoid all-rows arm is the one that exists and works.
MERGE_MODELS = {"imagenet_vit_b16", "wikitext_opt2.7b", "wikitext_opt6.7b"}
TAG = {"mash_ridge_merge_empirical_delta_f": "merge + ridge",
       "mash_full_medoid_empirical_delta_f": "medoid + ridge",
       "mash_merge_sum_delta_f": "merge + sum",
       "mash_medoid_sum_delta_f": "medoid + sum"}
# --centroid post: the MASH arms' `centroid: post` twins (exact Ward in
# response space, arms.yaml `post` tier, SUBMIT_NEXT.md §18). Baselines are
# unchanged; the figures get a `_post` suffix so both centroids stay on disk.
POST = {a: a.replace("mash_", "mash_post_", 1) for a in TAG}
TAG.update({POST[a]: t + " (post)" for a, t in TAG.items()})
CENTROID = "pre"
SUFFIX = ""


def tag(ax, arm, metric):
    """Name the MASH variant drawn in this panel, in the empty corner: top-right
    for accuracy (curves fall to the right), top-left for perplexity (curves sit
    at the floor on the left and rise). The legend says the variant is tagged."""
    if arm not in TAG:
        return
    x, ha = (0.97, "right") if metric == "acc" else (0.03, "left")
    ax.text(x, 0.96, TAG[arm], transform=ax.transAxes, ha=ha, va="top",
            fontsize=7, color=HUE["MASH"], fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))


def resolve(cell: str, arm: str) -> str:
    """Per-model arm substitutions (see MERGE_MODELS and the all-rows note)."""
    if arm == "mash_full_medoid_empirical_delta_f" and cell in MERGE_MODELS:
        return "mash_ridge_merge_empirical_delta_f"
    if arm == "mash_medoid_sum_delta_f" and cell in MERGE_MODELS:
        return "mash_merge_sum_delta_f"
    # Prefer the all-rows ridge baseline where it was run (the functional LMs), so
    # the baselines get the same repair budget as MASH's all-rows ridge.
    if arm in ("random_ridge", "magnitude_mass_ridge") and (OUT / f"{cell}__{arm}_full").exists():
        return f"{arm}_full"
    if CENTROID == "post":
        return POST.get(arm, arm)
    return arm


def curves_figure(name, arms, tagged=False):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), squeeze=False)
    for ax, (cell, title, metric) in zip(axes.flat, MODELS):
        n_seeds, lo = 0, None
        for label, arm, hue in arms:
            arm = resolve(cell, arm)
            d = load(f"{cell}__{arm}")
            if d is None:
                print(f"  missing {cell}__{arm}"); continue
            draw(ax, d, metric, HUE[hue], emphasis=(hue == "MASH"))
            if hue == "MASH" and tagged:
                tag(ax, arm, metric)
            n_seeds = max(n_seeds, int(d.n.max()))
            lo = float(d[metric].min()) if lo is None else min(lo, float(d[metric].min()))
        style(ax, title, metric, n_seeds, lo)
    # y-label only on the left column
    for ax in axes[:, 1:].flat:
        ax.set_ylabel("")
    handles = [Line2D([], [], color=HUE[h], lw=2.2 if h == "MASH" else 1.6, marker="o",
                      ms=3, mec="white", mew=0.6) for _, _, h in arms]
    legend_below(fig, handles, [l for l, _, _ in arms], ncol=len(arms))
    fig.tight_layout(w_pad=1.0, h_pad=1.2, rect=(0, 0.06, 1, 1))
    save(fig, name)


def overlap_figure(name):
    """|A n B| / min(|A|,|B|) per layer: mean line, min-max band over layers,
    against the chance level (the removed fraction). Seed 0; removal sets do not
    depend on the repair, so the no-repair cells are the source."""
    sys.path.insert(0, str(ROOT))
    from src.analysis.pruning_detail import overlap_from_removals
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), squeeze=False)
    for ax, (cell, title, _) in zip(axes.flat, MODELS):
        runs = {"mash": f"{cell}__{POST.get(MASH_REMOVALS, MASH_REMOVALS) if CENTROID == 'post' else MASH_REMOVALS}"}
        runs.update({arm: f"{cell}__{arm}" for _, arm, _ in OVERLAP})
        runs = {k: OUT / v / "seed_0" for k, v in runs.items()
                if (OUT / v / "seed_0" / "removals.json").exists()}
        if "mash" not in runs:
            print(f"  missing removals for {cell}"); ax.axis("off"); continue
        df = overlap_from_removals(runs)
        g = None
        for label, other, hue in OVERLAP:
            if other not in runs:
                print(f"  missing removals {cell}__{other}"); continue
            sub = df[((df.a == "mash") & (df.b == other)) | ((df.a == other) & (df.b == "mash"))]
            g = sub.groupby("fraction").agg(ov=("overlap", "mean"), lo=("overlap", "min"),
                                            hi=("overlap", "max"), chance=("chance", "mean")).reset_index()
            x = g.fraction * 100
            ax.fill_between(x, g.lo, g.hi, color=HUE[hue], alpha=0.15, lw=0)
            ax.plot(x, g.ov, color=HUE[hue], lw=1.8, marker="o", ms=2.8, mec="white", mew=0.6)
        if g is not None:
            ax.plot(g.fraction * 100, g.chance, color=INK2, ls="--", lw=1.0)
        ax.set_ylim(0, 1); ax.set_xlim(left=0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(f"{title}  (seed 0)", loc="left", color=INK, fontweight="bold")
        ax.set_xlabel("units removed (%)"); ax.set_ylabel("removed-set overlap")
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 1:].flat:
        ax.set_ylabel("")
    handles = [Line2D([], [], color=HUE[h], lw=1.8, marker="o", ms=3, mec="white", mew=0.6)
               for _, _, h in OVERLAP]
    handles.append(Line2D([], [], color=INK2, ls="--", lw=1.0))
    legend_below(fig, handles, [l for l, _, _ in OVERLAP] + ["chance (two random sets)"], ncol=3)
    fig.tight_layout(w_pad=1.0, h_pad=1.2, rect=(0, 0.06, 1, 1))
    save(fig, name)


def main() -> None:
    global OUT, CENTROID, SUFFIX
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--centroid", choices=("pre", "post"), default="pre",
                    help="post: draw MASH from the mash_post_* cells, write fig_*_post")
    args = ap.parse_args(); OUT = Path(args.out)
    CENTROID = args.centroid
    SUFFIX = "" if CENTROID == "pre" else f"_{CENTROID}"
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.labelcolor": INK, "text.color": INK, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42})
    curves_figure("fig_norepair", NOREPAIR, tagged=True)
    curves_figure("fig_repair", REPAIR, tagged=True)
    overlap_figure("fig_overlap")


if __name__ == "__main__":
    main()
