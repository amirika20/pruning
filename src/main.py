import copy
import json
import logging
import torch
from config import Config, DataConfig, ModelConfig, TrainConfig, PruneConfig
from config import TransformerExperimentConfig, TransformerModelConfig, ModularDataConfig
from data import make_loaders, get_input_dim, get_task, get_n_classes, make_modular_loaders
from model import build_model, build_transformer
from train import train
from prune import find_silent_neurons, find_redundant_neurons, prune_layer, layer_width
from plot_pruning import plot_pruning_summary
from metrics import print_efficiency_report
from paths import build_experiment_dirs, setup_logging


# ── Experiment definitions ────────────────────────────────────────────────────

# Case 1a: vectorised MNIST (flat [N, 784]) → MLP
MNIST_FLAT_MLP = Config(
    data=DataConfig(
        function="mnist_flat",
        n_samples=2000,
        train_ratio=0.8,
    ),
    model=ModelConfig(
        hidden_sizes=[512, 256, 128],
        arch="mlp",
    ),
    train=TrainConfig(optimizer="adam", epochs=500, lr=1e-3, batch_size=64,
                      log_every=10, weight_decay=1e-4),
    prune=PruneConfig(threshold=0.1, prune_silent=True, prune_redundant=True),
    finetune=TrainConfig(epochs=500, lr=1e-4, batch_size=64, log_every=10, weight_decay=1e-4),
    experiment_name="mnist_flat_mlp",
    seed=0,
)

# Case 1b: vectorised MNIST (flat [N, 784]) → ResMLP (ResNet over linear layers)
MNIST_FLAT_RESMLP = Config(
    data=DataConfig(
        function="mnist_flat",
        n_samples=2000,
        train_ratio=0.8,
    ),
    model=ModelConfig(
        hidden_sizes=[512, 256, 128],
        arch="resnet",
    ),
    train=TrainConfig(optimizer="adam", epochs=500, lr=1e-3, batch_size=64,
                      log_every=10, weight_decay=1e-4),
    prune=PruneConfig(threshold=0.1, prune_silent=True, prune_redundant=True),
    finetune=TrainConfig(epochs=500, lr=1e-4, batch_size=64, log_every=10, weight_decay=1e-4),
    experiment_name="mnist_flat_resmlp",
    seed=0,
)

# Case 2: pixel-space MNIST ([N, 1, 28, 28]) → CNN
MNIST_PIXEL_CNN = Config(
    data=DataConfig(
        function="mnist",
        n_samples=2000,
        train_ratio=0.8,
    ),
    model=ModelConfig(
        hidden_sizes=[32, 64, 128],
        arch="cnn",
        input_channels=1,
    ),
    train=TrainConfig(optimizer="adam", epochs=500, lr=1e-3, batch_size=64,
                      log_every=10, weight_decay=1e-4),
    prune=PruneConfig(threshold=0.1, prune_silent=True, prune_redundant=True),
    finetune=TrainConfig(epochs=500, lr=1e-4, batch_size=64, log_every=10, weight_decay=1e-4),
    experiment_name="mnist_pixel_cnn",
    seed=0,
)

# Case 3: modular arithmetic (a+b) mod p  →  2-layer transformer
MODULAR_TRANSFORMER = TransformerExperimentConfig(
    data=ModularDataConfig(p=97, train_ratio=0.5, seed=42),
    model=TransformerModelConfig(d_model=128, n_heads=4, d_ff=256, n_layers=2),
    train=TrainConfig(optimizer="adamw", epochs=3000, lr=1e-3, batch_size=512,
                      log_every=500, weight_decay=1.0),
    prune=PruneConfig(threshold=0.1, prune_silent=False, prune_redundant=True),
    finetune=TrainConfig(optimizer="adamw", epochs=1000, lr=1e-4, batch_size=512,
                         log_every=200, weight_decay=1.0),
    experiment_name="modular_transformer",
    seed=0,
)

EXPERIMENTS = [MNIST_FLAT_MLP, MNIST_FLAT_RESMLP, MNIST_PIXEL_CNN]
TRANSFORMER_EXPERIMENTS = [MODULAR_TRANSFORMER]


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(cfg: Config, device: torch.device) -> None:
    output_dir, log_dir, already_exists = build_experiment_dirs(cfg)
    setup_logging(log_dir)

    logging.info(f"{'='*60}")
    logging.info(f"Experiment: {cfg.experiment_name}")
    logging.info(f"{'='*60}")

    if already_exists:
        logging.info(f"Already exists at {output_dir} — skipping.")
        return

    torch.manual_seed(cfg.seed)

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
    train_losses, val_losses, train_accs, val_accs = train(
        model, train_loader, val_loader, cfg.train, task=task,
        desc=f"[{cfg.experiment_name}] train",
    )
    if val_accs:
        logging.info(f"Final val accuracy: {val_accs[-1]:.4f}")
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
    ft_train_losses, ft_val_losses, ft_train_accs, ft_val_accs = train(
        model_finetuned, train_loader, val_loader, cfg.finetune, task=task,
        desc=f"[{cfg.experiment_name}] finetune",
    )
    if ft_val_accs:
        logging.info(f"Final val accuracy (finetuned): {ft_val_accs[-1]:.4f}")
    metrics_finetuned = print_efficiency_report(model_before, model_finetuned, label="fine-tuned")

    # --- Save results ---
    results = {
        "training": {
            "final_train_loss": train_losses[-1],
            "final_val_loss":   val_losses[-1],
            **({"final_val_acc": val_accs[-1]} if val_accs else {}),
        },
        "finetuning": {
            "final_train_loss": ft_train_losses[-1],
            "final_val_loss":   ft_val_losses[-1],
            **({"final_val_acc": ft_val_accs[-1]} if ft_val_accs else {}),
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
        train_accs, val_accs,
        ft_train_accs, ft_val_accs,
        train_ds, val_ds,
        cfg.data, str(output_dir),
    )


# ── Transformer experiment runner ─────────────────────────────────────────────

def run_transformer_experiment(cfg: TransformerExperimentConfig, device: torch.device) -> None:
    output_dir, log_dir, already_exists = build_experiment_dirs(cfg)
    setup_logging(log_dir)

    logging.info(f"{'='*60}")
    logging.info(f"Experiment: {cfg.experiment_name}")
    logging.info(f"{'='*60}")

    if already_exists:
        logging.info(f"Already exists at {output_dir} — skipping.")
        return

    torch.manual_seed(cfg.seed)

    p       = cfg.data.p
    seq_len = 4          # [a, op, b, =]
    vocab_size = p + 2   # digits 0..p-1, op token p, eq token p+1

    train_loader, val_loader, train_ds, val_ds = make_modular_loaders(cfg.data, cfg.train.batch_size)
    model = build_transformer(cfg.model, vocab_size=vocab_size,
                              n_classes=p, seq_len=seq_len).to(device)

    n_params = sum(p_ .numel() for p_ in model.parameters())
    logging.info(f"Model      : ModularTransformer  ({n_params:,} params)")
    logging.info(f"Task       : modular_add  ({p} classes, p={p})")
    logging.info(f"Data       : {len(train_ds)} train  /  {len(val_ds)} val  (all {p}² pairs)")

    # --- Train ---
    logging.info("=== Training ===")
    train_losses, val_losses, train_accs, val_accs = train(
        model, train_loader, val_loader, cfg.train, task="multiclass",
        desc=f"[{cfg.experiment_name}] train",
    )
    logging.info(f"Final val accuracy: {val_accs[-1]:.4f}")
    model_before = copy.deepcopy(model)

    # --- Prune FFN layers ---
    logging.info("=== Pruning ===")
    train_inputs = train_ds.tensors[0].to(device)
    current      = model
    total_removed = 0
    pruning_per_layer = []

    for layer_idx in range(cfg.model.n_layers):
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
            f"  FFN layer {layer_idx}: silent={n_silent}  redundant={n_redundant}"
            f"  | removed={n_removed}  remaining={n_before - n_removed}"
        )
        if to_remove:
            current = prune_layer(current, layer_idx, list(to_remove))
        total_removed += n_removed

    logging.info(f"  Total removed: {total_removed}")
    model_pruned  = current
    metrics_pruned = print_efficiency_report(model_before, model_pruned, label="pruned")

    # --- Fine-tune ---
    logging.info("=== Fine-tuning ===")
    model_finetuned = copy.deepcopy(model_pruned)
    ft_train_losses, ft_val_losses, ft_train_accs, ft_val_accs = train(
        model_finetuned, train_loader, val_loader, cfg.finetune, task="multiclass",
        desc=f"[{cfg.experiment_name}] finetune",
    )
    logging.info(f"Final val accuracy (finetuned): {ft_val_accs[-1]:.4f}")
    metrics_finetuned = print_efficiency_report(model_before, model_finetuned, label="fine-tuned")

    # --- Save results ---
    results = {
        "training": {
            "final_train_loss": train_losses[-1],
            "final_val_loss":   val_losses[-1],
            "final_val_acc":    val_accs[-1],
        },
        "finetuning": {
            "final_train_loss": ft_train_losses[-1],
            "final_val_loss":   ft_val_losses[-1],
            "final_val_acc":    ft_val_accs[-1],
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
        train_accs, val_accs,
        ft_train_accs, ft_val_accs,
        train_ds, val_ds,
        cfg.data, str(output_dir),
    )


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: cpu (CUDA not available)")

    # for cfg in EXPERIMENTS:
    #     run_experiment(cfg, device)

    for cfg in TRANSFORMER_EXPERIMENTS:
        run_transformer_experiment(cfg, device)


if __name__ == "__main__":
    main()
