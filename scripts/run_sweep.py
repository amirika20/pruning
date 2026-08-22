#!/usr/bin/env python
"""Run one benchmark cell: an accuracy-versus-width sweep for a single arm.

    python scripts/run_sweep.py --config configs/benchmark/generated/mnist_mlp/mash_merge_kernel_delta_f.yaml
    python scripts/run_sweep.py --config C --seed 1 --grid 16
    python scripts/run_sweep.py --config C --fractions 0.2 0.4 0.6

This is the benchmark's unit of work, and it is NOT the same as
scripts/run_experiment.py: that prunes once, at whatever width the config names,
which cannot produce a capacity number. Capacity is read off a curve, so the
width belongs to the sweep rather than the config -- which is why the generated
configs carry an arm with no width in it.

Per (config, seed) it writes, under
`outputs/benchmark/<config-name>/seed_<s>/`:

    curve.csv       one row per width: fraction, widths, accuracy, loss, timings
    report.json     first-crossing capacities at each tolerance, AUC, grid
                    spacing, plan/solve seconds, and the run fingerprints
    geometry_shift.csv / similarity.csv   at the reference width, when
                    config.analyze_geometry is set

A seed whose DENSE model is at chance accuracy is skipped, not swept: every unit
is removable from a network that does nothing, so the capacity would come out
near 1.0 and read as the best result in the table. Pass --allow-untrained to
override, which records dense_above_chance=False in the report.

Capacity is FIRST-CROSSING, not max-passing (see src/analysis/metrics). The
max-passing value is recorded alongside as a diagnostic: a large gap between
them means the curve oscillates through the tolerance, so the grid is too coarse
or the evaluation set too small to support a capacity claim at that tolerance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.geometry_shift import geometry_shift
from src.analysis.similarity import similarity_table
from src.config import ExperimentConfig
from src.data import build_dataset
from src.experiments.sweep import format_sweep, sweep_report, sweep_widths
from src.models import build_model
from src.reproducibility import run_fingerprint, seed_everything
from src.training.trainer import evaluate, train


def _accuracy_of(model, bundle, device):
    """(loss, accuracy) of a model on its validation split."""
    _, val_loader = bundle.loaders(batch_size=None)
    return evaluate(model, val_loader, bundle.task)


def load_model(config: ExperimentConfig, seed: int, device: torch.device):
    """(model, bundle, fingerprints). Trains only when the config asks for it --
    every pretrained entry has training.epochs == 0, so the weights come from
    the downloaded checkpoint and nothing is fitted here."""
    seed_everything(seed, deterministic=config.deterministic)
    bundle = build_dataset(config.data.kind, data_seed=seed, **config.data.params)
    seed_everything(seed, deterministic=config.deterministic)
    model = build_model(config.model.kind, bundle, **config.model.params).to(device)
    init = run_fingerprint(bundle, model, None, seed, config)

    if config.training.epochs > 0:
        logging.info(f"training {config.training.epochs} epochs "
                     f"({config.model.kind}, no published checkpoint)")
        tl, vl = bundle.loaders(config.training.batch_size, seed=seed)
        train(model, tl, vl, config.training, task=bundle.task,
              desc=f"{config.name} s{seed}")
    else:
        logging.info("no training (pretrained checkpoint or epochs=0)")

    model = model.eval()
    fp = run_fingerprint(bundle, None, model, seed, config)
    fp["init"] = init.get("init")
    return model, bundle, fp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="one seed (default: every seed in the config)")
    ap.add_argument("--grid", type=int, default=16,
                    help="number of widths, evenly spaced in (0, 0.95]")
    ap.add_argument("--fractions", type=float, nargs="*", default=None,
                    help="explicit grid, overrides --grid")
    ap.add_argument("--out", default=None, help="override output_root")
    ap.add_argument("--allow-untrained", action="store_true",
                    help="sweep even when the dense model is at chance accuracy "
                         "(the capacity it reports is meaningless -- see below)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    config = ExperimentConfig.from_yaml(args.config)
    device = torch.device(config.training.device)
    fractions = (list(args.fractions) if args.fractions
                 else list(np.linspace(0.95 / args.grid, 0.95, args.grid)))
    seeds = [args.seed] if args.seed is not None else list(config.seeds)
    root = Path(args.out or config.output_root) / config.name

    method = config.pruning.methods[0]
    logging.info(f"cell: {config.name}")
    logging.info(f"arm : {method.kind}  {method.params}")
    logging.info(f"grid: {len(fractions)} widths, "
                 f"{fractions[0]:.3f}..{fractions[-1]:.3f}")

    swept, skipped = [], []
    for seed in seeds:
        t0 = time.perf_counter()
        model, bundle, fp = load_model(config, seed, device)

        # A MODEL AT CHANCE MUST NOT PRODUCE A CAPACITY NUMBER. Everything is
        # removable from a network that does nothing, so a training failure
        # reads as the best result in the table rather than as a failure -- which
        # is exactly what happened on the modular-arithmetic entry, where two of
        # three seeds never groked and each recorded cap01 = 0.947 across 42
        # arms. Refuse loudly instead; --allow-untrained overrides.
        chance = (1.0 / bundle.output_dim if bundle.task == "multiclass"
                  and bundle.output_dim else None)
        if chance is not None:
            _, dense_acc = _accuracy_of(model, bundle, device)
            floor = max(2.0 * chance, chance + 0.02)
            if dense_acc is not None and dense_acc < floor:
                msg = (f"seed {seed}: dense accuracy {dense_acc:.4f} is at chance "
                       f"({chance:.4f}; floor {floor:.4f}) -- the model did not "
                       f"train, so any capacity from it is meaningless")
                if not args.allow_untrained:
                    logging.error(msg + "; skipping this seed "
                                  "(--allow-untrained to override)")
                    skipped.append(seed)
                    continue
                logging.warning(msg + "; sweeping anyway as requested")
                fp["dense_above_chance"] = False

        widths = [model.prunable_layer(i).weight.shape[0]
                  for i in range(model.n_prunable_layers())]
        logging.info(f"seed {seed}: {len(widths)} prunable layers, "
                     f"{sum(widths)} units, fingerprint trained={fp.get('trained')}")

        curve = sweep_widths(model, bundle, method.kind, method.params,
                             fractions=fractions, device=device,
                             seed=seed, name=config.name, arm=method.kind)
        rep = sweep_report(curve)
        rep.update(cell=config.name, arm=method.kind, arm_params=method.params,
                   seed=seed, units=int(sum(widths)), widths=widths,
                   fingerprints=fp, wall_seconds=time.perf_counter() - t0)

        out = root / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        curve.to_csv(out / "curve.csv", index=False)
        (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
        logging.info("\n" + format_sweep({f"{config.name} s{seed}": rep}))

        if config.analyze_geometry:
            # At the reference width only: these batteries are per-model-pair,
            # not per-width, and running them at every point would dominate.
            try:
                from src.pruning.surgery import prune_model
                pruned, prep = prune_model(model, config.pruning, bundle, device)
                gx = bundle.val_ds.tensors[0][:256].to(device)
                geometry_shift(model, pruned, prep, gx, seed=seed,
                               name=config.name).to_csv(
                    out / "geometry_shift.csv", index=False)
                similarity_table(model, gx, prep, seed=seed,
                                 name=config.name).to_csv(
                    out / "similarity.csv", index=False)
            except Exception as exc:                      # noqa: BLE001
                logging.warning(f"analysis skipped: {type(exc).__name__}: {exc}")

        swept.append(seed)
        logging.info(f"seed {seed} done in {time.perf_counter() - t0:.1f}s -> {out}")

    # EXIT NON-ZERO WHEN NOTHING WAS SWEPT. Returning normally here would let
    # run_manifest print "ok" for a cell that produced no curve at all, which is
    # the same silent-success bug the chance guard exists to prevent -- one level
    # up. A partial cell (some seeds groked, some did not) also exits non-zero so
    # the tier summary names it.
    if skipped:
        print(f"cell INCOMPLETE: {root} -- swept {swept}, "
              f"skipped {skipped} (at chance)", file=sys.stderr)
        sys.exit(1)
    print(f"cell complete: {root}")


if __name__ == "__main__":
    main()
