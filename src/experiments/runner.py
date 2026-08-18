"""Top-level orchestration: config -> multi-seed train/prune/finetune ->
aggregation -> plots.

This is the single entry point scripts should call (`run_experiment(config)`);
everything else in the package is a building block it composes.

Each run gets a fresh timestamped folder grouped by dataset:

    outputs/<data.kind>/<name>_<YYYYMMDD_HHMMSS>/
        config.yaml            exact config that produced the run
        metadata.json          git commit, timestamp, torch/device info
        run.log                full log of the run
        seeds/seed_<s>/        per-seed results.json + pruning_summary.png
        aggregated/results.json  mean +- std across seeds
        plots/curves.png       aggregate loss/accuracy curves
"""

from __future__ import annotations

import copy
import json
import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import torch

from src.analysis.metrics import print_efficiency_report
from src.analysis.plots import plot_aggregate_curves, plot_pruning_summary
from src.analysis.report import format_report
from src.config import ExperimentConfig
from src.data import build_dataset
from src.experiments.aggregate import aggregate_seed_results
from src.models import build_model
from src.pruning import prune_model
from src.training.trainer import evaluate, train

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── experiment folder / logging / metadata ───────────────────────────────────

def create_experiment_dir(config: ExperimentConfig) -> Path:
    """outputs/<data.kind>/<name>_<timestamp>/ -- always a fresh folder."""
    root = Path(config.output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = root / config.data.kind / f"{config.name}_{timestamp}"
    exp_dir.mkdir(parents=True)
    return exp_dir


def setup_logging(exp_dir: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    # Console handler: add once
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)

    # File handler: swap per experiment so each run gets its own log
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            root.removeHandler(h)
    fh = logging.FileHandler(exp_dir / "run.log")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def collect_run_metadata() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {
        "timestamp": datetime.now().isoformat(),
        "git_commit": commit,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
        "hostname": platform.node(),
    }


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ── single seed ──────────────────────────────────────────────────────────────

def run_single_seed(config: ExperimentConfig, seed: int, device: torch.device, desc: str):
    """One full train -> prune -> finetune pipeline. The same `seed` drives
    torch.manual_seed (model init + batching) and the dataset builder's
    data_seed. Returns (result_dict, models_dict, bundle)."""
    torch.manual_seed(seed)

    bundle = build_dataset(config.data.kind, data_seed=seed, **config.data.params)
    train_loader, val_loader = bundle.loaders(config.training.batch_size)
    test_loader = bundle.test_loader()
    model = build_model(config.model.kind, bundle, **config.model.params).to(device)

    def eval_stage(m) -> dict:
        """val (+ test when the dataset provides one) loss/accuracy of `m`."""
        val_loss, val_acc = evaluate(m, val_loader, bundle.task)
        stage = {"val_loss": val_loss, "val_acc": val_acc}
        if test_loader is not None:
            stage["test_loss"], stage["test_acc"] = evaluate(m, test_loader, bundle.task)
        return stage

    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model      : {config.model.kind}  ({n_params:,} params)")
    logging.info(f"Task       : {bundle.task}  ({bundle.output_dim} output dim)")
    logging.info(f"Data       : {config.data.kind}  "
                 f"(train={len(bundle.train_ds)}, val={len(bundle.val_ds)}, "
                 f"test={len(bundle.test_ds) if bundle.test_ds is not None else 0})")

    # --- Train ---
    logging.info("=== Training ===")
    train_losses, val_losses, train_accs, val_accs = train(
        model, train_loader, val_loader, config.training, task=bundle.task,
        desc=f"{desc} train",
    )
    if val_accs:
        logging.info(f"Final val accuracy: {val_accs[-1]:.4f}")
    model_before = copy.deepcopy(model)
    stage_trained = eval_stage(model_before)

    # --- Prune ---
    logging.info("=== Pruning ===")
    model_pruned, pruning_per_layer = prune_model(model, config.pruning, bundle, device)
    stage_pruned = eval_stage(model_pruned)
    pruned_val_loss, pruned_val_acc = stage_pruned["val_loss"], stage_pruned["val_acc"]
    msg = f"Val loss after pruning (no retraining): {pruned_val_loss:.4f}"
    if pruned_val_acc is not None:
        msg += f" | val accuracy: {pruned_val_acc:.4f}"
    logging.info(msg)
    metrics_pruned = print_efficiency_report(model_before, model_pruned, label="pruned")

    # --- Fine-tune (finetune.epochs == 0 skips the phase entirely, e.g. to
    # reproduce retraining-free protocols like Srinivas & Babu 2015) ---
    skip_finetune = config.finetune.epochs <= 0
    if skip_finetune:
        logging.info("=== Fine-tuning skipped (finetune.epochs == 0) ===")
        model_finetuned = model_pruned
        ft_train_losses, ft_val_losses, ft_train_accs, ft_val_accs = [], [], [], []
        metrics_finetuned = None
        stage_finetuned = None
    else:
        logging.info("=== Fine-tuning ===")
        model_finetuned = copy.deepcopy(model_pruned)
        ft_train_losses, ft_val_losses, ft_train_accs, ft_val_accs = train(
            model_finetuned, train_loader, val_loader, config.finetune, task=bundle.task,
            desc=f"{desc} finetune",
        )
        if ft_val_accs:
            logging.info(f"Final val accuracy (finetuned): {ft_val_accs[-1]:.4f}")
        metrics_finetuned = print_efficiency_report(model_before, model_finetuned, label="fine-tuned")
        stage_finetuned = eval_stage(model_finetuned)

    if "test_loss" in stage_trained:
        end = stage_finetuned or stage_pruned
        msg = f"Test loss: {end['test_loss']:.4f}"
        if end["test_acc"] is not None:
            msg += f" | test accuracy: {end['test_acc']:.4f}"
        logging.info(msg)

    is_multiclass = bundle.task == "multiclass"
    result = {
        "seed": seed,
        "task": bundle.task,
        "history": {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_acc": train_accs,
            "val_acc": val_accs,
            "ft_train_loss": ft_train_losses,
            "ft_val_loss": ft_val_losses,
            "ft_train_acc": ft_train_accs,
            "ft_val_acc": ft_val_accs,
        },
        "final": {
            # Empty when training.epochs == 0 (e.g. starting from a
            # pretrained model) -- then there is no training history.
            **({} if not train_losses else {
                "train_loss": train_losses[-1],
                "val_loss": val_losses[-1],
                **({"val_acc": val_accs[-1]} if is_multiclass else {}),
            }),
            "pruned_val_loss": pruned_val_loss,
            **({"pruned_val_acc": pruned_val_acc} if is_multiclass else {}),
            **({} if skip_finetune else {
                "ft_train_loss": ft_train_losses[-1],
                "ft_val_loss": ft_val_losses[-1],
                **({"ft_val_acc": ft_val_accs[-1]} if is_multiclass else {}),
            }),
        },
        # val (+ test) of each model stage -- the basis of report.txt.
        "stages": {
            "trained": stage_trained,
            "pruned": stage_pruned,
            "finetuned": stage_finetuned,
        },
        "pruning_per_layer": pruning_per_layer,
        "pruned": metrics_pruned,
        "finetuned": metrics_finetuned,
    }
    models = {"before": model_before, "pruned": model_pruned, "finetuned": model_finetuned}
    return result, models, bundle


# ── full experiment ──────────────────────────────────────────────────────────

def run_experiment(config: ExperimentConfig) -> Path:
    """Run every seed in `config.seeds`, save per-seed results and plots,
    then a cross-seed aggregate (mean +- std). Returns the experiment folder."""
    exp_dir = create_experiment_dir(config)
    setup_logging(exp_dir)
    config.save(exp_dir / "config.yaml")
    save_json(collect_run_metadata(), exp_dir / "metadata.json")

    device = torch.device(config.training.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    logging.info(f"{'=' * 60}")
    logging.info(f"Experiment: {config.name}  ({len(config.seeds)} seed(s), device={device})")
    logging.info(f"Output    : {exp_dir}")
    logging.info(f"{'=' * 60}")

    seed_results = []
    for i, seed in enumerate(config.seeds):
        desc = f"[{config.name}] seed {i + 1}/{len(config.seeds)} (seed={seed})"
        logging.info(f"--- {desc} ---")
        result, models, bundle = run_single_seed(config, seed, device, desc)

        # Persist before any plotting: training time must never hinge on
        # cosmetic code being exception-free.
        s_dir = exp_dir / "seeds" / f"seed_{seed}"
        save_json(result, s_dir / "results.json")
        (s_dir / "report.txt").write_text(format_report(config, [result], title_suffix=f" (seed {seed})"))

        plot_pruning_summary(
            models["before"], models["pruned"], models["finetuned"],
            result["history"], bundle,
            save_path=s_dir / "pruning_summary.png",
        )
        seed_results.append(result)

    aggregated = aggregate_seed_results(seed_results)
    aggregated["name"] = config.name
    save_json(aggregated, exp_dir / "aggregated" / "results.json")

    report = format_report(config, seed_results)
    (exp_dir / "report.txt").write_text(report)
    logging.info("\n" + report)

    plot_aggregate_curves(
        [r["history"] for r in seed_results],
        task=seed_results[0]["task"],
        save_path=exp_dir / "plots" / "curves.png",
        title=config.name,
    )

    # Report the run's endpoint: fine-tuned metrics when the phase ran,
    # otherwise the post-prune (no-retraining) metrics.
    loss_key = "ft_val_loss" if "ft_val_loss" in aggregated["final"] else "pruned_val_loss"
    acc_key = "ft_val_acc" if "ft_val_acc" in aggregated["final"] else "pruned_val_acc"
    msg = (f"[{config.name}] mean over {len(config.seeds)} seed(s): "
           f"{loss_key}={aggregated['final'][loss_key]['mean']:.4f}"
           f"+/-{aggregated['final'][loss_key]['std']:.4f}")
    if acc_key in aggregated["final"]:
        msg += (f" {acc_key}={aggregated['final'][acc_key]['mean']:.4f}"
                f"+/-{aggregated['final'][acc_key]['std']:.4f}")
    logging.info(msg)

    return exp_dir
