#!/usr/bin/env python
"""Re-run the analysis of existing study runs at a different eps -- no
retraining, everything is recomputed from each run's neurons.csv.

Usage:
    python studies/beta_saturation/reanalyze.py --eps 0.01                  # every run in outputs/
    python studies/beta_saturation/reanalyze.py --eps 0.01 --runs outputs/beta_sat_mnist_mlp_*

Writes report_eps<eps>.txt, summary_eps<eps>.json and plots/eps<eps>_* next
to the originals (originals are left untouched).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from studies.beta_saturation.analyze import (
    analyze, categorize, format_report,
    plot_deep_profile, plot_group_box, plot_rank, plot_scatter, plot_score_auroc,
)

STUDY_ROOT = Path(__file__).resolve().parent


def reanalyze_run(run_dir: Path, eps: float) -> None:
    df = pd.read_csv(run_dir / "neurons.csv")
    df = df.drop(columns=["saturation", "category", "abs_b"], errors="ignore")
    df = categorize(df, eps)
    name = json.loads((run_dir / "summary.json").read_text())["name"]
    tag = f"eps{eps:g}"

    per_seed = {int(s): analyze(g, eps) for s, g in df.groupby("seed")}
    pooled = analyze(df, eps)
    with open(run_dir / f"summary_{tag}.json", "w") as f:
        json.dump({"name": name, "per_seed": per_seed, "pooled": pooled}, f, indent=2)

    plot_score_auroc(pooled, run_dir / "plots" / f"{tag}_score_auroc.png",
                     f"{name}: score comparison ({tag})")
    for seed, g in df.groupby("seed"):
        g = categorize(g, eps)
        plot_scatter(g, run_dir / "plots" / f"{tag}_seed_{seed}_scatter.png", f"{name} (seed {seed}, {tag})")
        plot_rank(g, run_dir / "plots" / f"{tag}_seed_{seed}_rank.png", f"{name} (seed {seed}, {tag})")
        plot_group_box(g, run_dir / "plots" / f"{tag}_seed_{seed}_box.png", f"{name} (seed {seed}, {tag})")
        plot_deep_profile(g, run_dir / "plots" / f"{tag}_seed_{seed}_deep_profile.png", f"{name} (seed {seed}, {tag})")

    report = format_report(name, per_seed, pooled)
    (run_dir / f"report_{tag}.txt").write_text(report)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--runs", nargs="*", default=None,
                        help="run directories (default: every dir in outputs/ with a neurons.csv)")
    args = parser.parse_args()

    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        run_dirs = sorted(d for d in (STUDY_ROOT / "outputs").iterdir()
                          if (d / "neurons.csv").exists())

    for run_dir in run_dirs:
        print(f"########## {run_dir.name}")
        reanalyze_run(run_dir, args.eps)


if __name__ == "__main__":
    main()
