import copy
import json
import logging
import torch
from config import Config, DataConfig, ModelConfig, TrainConfig, PruneConfig
from data import make_loaders, get_input_dim, get_task, get_n_classes
from model import build_model
from train import train
from prune import find_silent_neurons, find_redundant_neurons, prune_layer, layer_width
from plot_pruning import plot_pruning_summary
from metrics import print_efficiency_report
from paths import build_experiment_dirs, setup_logging


def main():
    cfg = Config(
        data=DataConfig(
            function="mnist",
            n_samples=2000,   # 1600 train / 400 val
            train_ratio=0.8,
        ),
        model=ModelConfig(
            hidden_sizes=[128, 128],
            arch="rescnn",
            input_channels=1,
        ),
        train=TrainConfig(optimizer="adam", epochs=500, lr=1e-3, batch_size=64,
                          log_every=10, weight_decay=1e-4),
        prune=PruneConfig(threshold=0.1, prune_silent=True, prune_redundant=True),
        finetune=TrainConfig(epochs=10, lr=1e-4, batch_size=64,
                             log_every=5, weight_decay=1e-4),
        experiment_name="mnist_rescnn",
        seed=0,
    )

    output_dir, log_dir, already_exists = build_experiment_dirs(cfg)
    setup_logging(log_dir)

    if already_exists:
        logging.info(f"Experiment already exists at {output_dir} — skipping.")
        return

    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        logging.info(f"Device     : cuda ({torch.cuda.get_device_name(0)},"
                     f" {torch.cuda.device_count()} device(s))")
    else:
        device = torch.device("cpu")
        logging.warning(
            "CUDA not available — running on CPU. "
            "If you have a GPU, reinstall PyTorch with CUDA support."
        )
    logging.info(f"Output dir : {output_dir}")
    logging.info(f"Log dir    : {log_dir}")

    task      = get_task(cfg.data)
    n_classes = get_n_classes(cfg.data)

    train_loader, val_loader, train_ds, val_ds = make_loaders(cfg.data, cfg.train.batch_size)
    model = build_model(cfg.model, get_input_dim(cfg.data), n_classes).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model      : {cfg.model.arch.upper()}  ({n_params:,} params)")
    logging.info(f"Task       : {task}  ({n_classes} class{'es' if n_classes > 1 else ''})")
    logging.info(f"Data       : {cfg.data.function}  "
                 f"(train={len(train_ds)}, val={len(val_ds)})")

    # --- Train ---
    logging.info("=== Training ===")
    train_losses, val_losses = train(model, train_loader, val_loader, cfg.train, task=task)
    model_before = copy.deepcopy(model)

    # --- Prune all hidden layers sequentially ---
    logging.info("=== Pruning ===")
    train_inputs = train_ds.tensors[0].to(device)
    current = model
    total_removed = 0
    pruning_per_layer = []

    for layer_idx in range(len(cfg.model.hidden_sizes)):
        n_before = layer_width(current, layer_idx)
        to_remove: set = set()
        n_silent = n_redundant = 0

        if cfg.prune.prune_silent:
            silent = find_silent_neurons(current, layer_idx, train_inputs)
            n_silent = len(silent)
            to_remove.update(silent)

        if cfg.prune.prune_redundant:
            redundant = find_redundant_neurons(
                current, layer_idx, cfg.prune.threshold, cfg.prune.threshold
            )
            new_redundant = [i for i in redundant if i not in to_remove]
            n_redundant = len(new_redundant)
            to_remove.update(new_redundant)

        n_removed = len(to_remove)
        pruning_per_layer.append({
            "layer":             layer_idx,
            "neurons_before":    n_before,
            "silent_removed":    n_silent,
            "redundant_removed": n_redundant,
            "total_removed":     n_removed,
            "neurons_after":     n_before - n_removed,
        })
        logging.info(
            f"  Layer {layer_idx}: silent={n_silent}  redundant={n_redundant}"
            f"  | removed={n_removed}  remaining={n_before - n_removed}"
        )
        if to_remove:
            current = prune_layer(current, layer_idx, list(to_remove))
        total_removed += n_removed

    logging.info(f"  Total removed: {total_removed}")
    model_pruned = current
    metrics_pruned = print_efficiency_report(model_before, model_pruned, label="pruned")

    # --- Fine-tune ---
    logging.info("=== Fine-tuning ===")
    model_finetuned = copy.deepcopy(model_pruned)
    ft_train_losses, ft_val_losses = train(
        model_finetuned, train_loader, val_loader, cfg.finetune, task=task
    )
    metrics_finetuned = print_efficiency_report(model_before, model_finetuned, label="fine-tuned")

    # --- Save results ---
    results = {
        "training": {
            "final_train_loss": train_losses[-1],
            "final_val_loss":   val_losses[-1],
        },
        "finetuning": {
            "final_train_loss": ft_train_losses[-1],
            "final_val_loss":   ft_val_losses[-1],
        },
        "pruning_per_layer": pruning_per_layer,
        "pruned":            metrics_pruned,
        "finetuned":         metrics_finetuned,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved results to {output_dir / 'results.json'}")

    # --- Plot ---
    plot_pruning_summary(
        model_before, model_pruned, model_finetuned,
        train_losses, val_losses,
        ft_train_losses, ft_val_losses,
        train_ds, val_ds,
        cfg.data, str(output_dir),
    )


if __name__ == "__main__":
    main()
