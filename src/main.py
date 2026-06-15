import copy
from config import Config, DataConfig, ModelConfig, TrainConfig, PruneConfig
from data import make_loaders
from model import build_model
from train import train
from prune import find_silent_neurons, find_redundant_neurons, prune_layer, layer_width
from plot_pruning import plot_pruning_summary
from metrics import print_efficiency_report


def main():
    cfg = Config(
        data=DataConfig(function="complex", n_samples=100, noise_std=0.05),
        model=ModelConfig(hidden_sizes=[128,128], arch='resnet'),
        train=TrainConfig(optimizer='adam',epochs=5000, lr=1e-3, batch_size=16, log_every=50),
        prune=PruneConfig(threshold=0.01,prune_silent=True, prune_redundant=True),
        finetune=TrainConfig(epochs=10000, lr=1e-4, log_every=50),
        output_dir="outputs",
    )

    train_loader, val_loader, train_ds, val_ds = make_loaders(cfg.data)
    model = build_model(cfg.model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model}")
    print(f"Fitting {cfg.data.function}(x) with {n_params} parameters\n")

    # --- Train ---
    print("=== Training ===")
    train_losses, val_losses = train(model, train_loader, val_loader, cfg.train)
    model_before = copy.deepcopy(model)

    # --- Prune all hidden layers sequentially ---
    print("\n=== Pruning ===")
    train_inputs = train_ds.tensors[0]
    current = model
    total_removed = 0
    for layer_idx in range(len(cfg.model.hidden_sizes)):
        n_before = layer_width(current, layer_idx)
        to_remove: set = set()

        if cfg.prune.prune_silent:
            silent = find_silent_neurons(current, layer_idx, train_inputs)
            to_remove.update(silent)
            print(f"  Layer {layer_idx}: silent={len(silent)}", end="")

        if cfg.prune.prune_redundant:
            redundant = find_redundant_neurons(current, layer_idx, cfg.prune.threshold, cfg.prune.threshold)
            new_redundant = [i for i in redundant if i not in to_remove]
            to_remove.update(new_redundant)
            print(f"  redundant={len(new_redundant)}", end="")

        print(f"  | removed={len(to_remove)}  remaining={n_before - len(to_remove)}")
        if to_remove:
            current = prune_layer(current, layer_idx, list(to_remove))
        total_removed += len(to_remove)
    print(f"  Total removed: {total_removed}")
    model_pruned = current
    print_efficiency_report(model_before, model_pruned, label="pruned")

    # --- Fine-tune ---
    print("\n=== Fine-tuning ===")
    model_finetuned = copy.deepcopy(model_pruned)
    ft_train_losses, ft_val_losses = train(model_finetuned, train_loader, val_loader, cfg.finetune)
    print_efficiency_report(model_before, model_finetuned, label="fine-tuned")

    # --- Plot ---
    plot_pruning_summary(
        model_before, model_pruned, model_finetuned,
        train_losses, val_losses,
        ft_train_losses, ft_val_losses,
        train_ds, val_ds,
        cfg.data, cfg.output_dir,
    )


if __name__ == "__main__":
    main()
