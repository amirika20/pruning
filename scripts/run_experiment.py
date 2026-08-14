#!/usr/bin/env python
"""Run one or more experiments from YAML configs.

Usage:
    python scripts/run_experiment.py --config configs/experiments/mnist/mlp/mnist_flat_mlp.yaml
    python scripts/run_experiment.py --config configs/experiments/mnist       # every yaml under a directory
    python scripts/run_experiment.py --config a.yaml b.yaml                   # several in sequence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig
from src.experiments.runner import run_experiment


def collect_config_paths(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        p = Path(target)
        if p.is_dir():
            found = sorted(p.rglob("*.yaml"))
            if not found:
                raise FileNotFoundError(f"no .yaml configs found under {p}")
            paths.extend(found)
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f"config path does not exist: {p}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, nargs="+",
        help="experiment YAML config file(s), or directories to search for them",
    )
    args = parser.parse_args()

    config_paths = collect_config_paths(args.config)
    for path in config_paths:
        config = ExperimentConfig.from_yaml(path)
        exp_dir = run_experiment(config)
        print(f"experiment complete: {exp_dir}")


if __name__ == "__main__":
    main()
