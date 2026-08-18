#!/usr/bin/env python
"""Train models and test the beta-saturation hypothesis.

Usage (from the repo root, with the .ml venv python):
    python studies/beta_saturation/run_study.py --config studies/beta_saturation/configs/sine_mlp.yaml
    python studies/beta_saturation/run_study.py --config studies/beta_saturation/configs   # every yaml
    python studies/beta_saturation/run_study.py --config a.yaml --epochs 20               # quick smoke run

Configs are ordinary ExperimentConfig yamls; only data/model/training/seeds
are used (pruning and finetune sections are ignored -- nothing is pruned here,
we only measure trained networks).

Output per config:
    studies/beta_saturation/outputs/<name>_<timestamp>/
        config.yaml
        neurons.csv        one row per (seed, layer, neuron)
        summary.json       per-seed and pooled hypothesis metrics
        report.txt         readable verdict table
        plots/seed_<s>_{scatter,rank,box}.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.data import build_dataset
from src.models import build_model
from src.training.trainer import train

from studies.beta_saturation.collect import collect_neuron_stats
from studies.beta_saturation.analyze import (
    analyze, categorize, format_report,
    plot_deep_profile, plot_group_box, plot_rank, plot_scatter, plot_score_auroc,
)

STUDY_ROOT = Path(__file__).resolve().parent


def run_config(config_path: Path, eps: float, epochs_override: int | None) -> Path:
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

        train(model, train_loader, val_loader, config.training, task=bundle.task,
              desc=f"[{config.name}] seed {seed}")
        all_records.extend(collect_neuron_stats(model, bundle, device, seed))

        # Checkpoint so follow-up analyses can reload without retraining.
        ckpt_dir = out_dir / "models"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"seed_{seed}.pt")

    df = pd.DataFrame(all_records)
    df = categorize(df, eps)
    df.to_csv(out_dir / "neurons.csv", index=False)

    # Per-seed metrics (stability) + all seeds' neurons pooled (power).
    per_seed = {int(s): analyze(g, eps) for s, g in df.groupby("seed")}
    pooled = analyze(df, eps)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"name": config.name, "per_seed": per_seed, "pooled": pooled}, f, indent=2)

    plot_score_auroc(pooled, out_dir / "plots" / "score_auroc.png", f"{config.name}: score comparison")
    for seed, g in df.groupby("seed"):
        g = categorize(g, eps)
        plot_scatter(g, out_dir / "plots" / f"seed_{seed}_scatter.png", f"{config.name} (seed {seed})")
        plot_rank(g, out_dir / "plots" / f"seed_{seed}_rank.png", f"{config.name} (seed {seed})")
        plot_group_box(g, out_dir / "plots" / f"seed_{seed}_box.png", f"{config.name} (seed {seed})")
        plot_deep_profile(g, out_dir / "plots" / f"seed_{seed}_deep_profile.png", f"{config.name} (seed {seed})")

    report = format_report(config.name, per_seed, pooled)
    (out_dir / "report.txt").write_text(report)
    logging.info("\n" + report)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", required=True,
                        help="yaml file(s) or a directory of yamls")
    parser.add_argument("--eps", type=float, default=0.0,
                        help="saturation tolerance: act_freq <= eps is 'never', >= 1-eps is 'always' (default 0 = strict)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override training epochs (for quick smoke runs)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    config_paths: list[Path] = []
    for c in args.config:
        p = Path(c)
        config_paths.extend(sorted(p.rglob("*.yaml")) if p.is_dir() else [p])

    for path in config_paths:
        run_config(path, args.eps, args.epochs)


if __name__ == "__main__":
    main()
