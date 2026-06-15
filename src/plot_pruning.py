import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List
from torch.utils.data import TensorDataset
from config import DataConfig


def _true_fn(x: np.ndarray, cfg: DataConfig) -> np.ndarray:
    if cfg.function == "sin":
        return np.sin(x)
    elif cfg.function == "cos":
        return np.cos(x)
    elif cfg.function == "complex":
        return 2 * np.cos(0.5 * x) + np.cos(x)
    else:
        raise ValueError(f"Unknown function: {cfg.function}")


def _predict(model: nn.Module, x_line: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_line, dtype=torch.float32).unsqueeze(1)
        return model(x_t).numpy().squeeze()


def _draw_fit(ax, x_train, y_train, x_val, y_val, x_line, y_true, y_pred, title):
    ax.scatter(x_train, y_train, s=12, alpha=0.5, color="steelblue", label="train")
    ax.scatter(x_val, y_val, s=12, alpha=0.5, color="tomato", label="val")
    ax.plot(x_line, y_true, color="black", linewidth=1.5, label="true")
    ax.plot(x_line, y_pred, color="darkorange", linewidth=2, linestyle="--", label="model")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_pruning_summary(
    model_before: nn.Module,
    model_pruned: nn.Module,
    model_finetuned: nn.Module,
    train_losses: List[float],
    val_losses: List[float],
    ft_train_losses: List[float],
    ft_val_losses: List[float],
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    cfg: DataConfig,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    x_train = train_ds.tensors[0].numpy().squeeze()
    y_train = train_ds.tensors[1].numpy().squeeze()
    x_val = val_ds.tensors[0].numpy().squeeze()
    y_val = val_ds.tensors[1].numpy().squeeze()

    x_line = np.linspace(cfg.x_range[0], cfg.x_range[1], 500)
    y_true = _true_fn(x_line, cfg)
    y_before = _predict(model_before, x_line)
    y_pruned = _predict(model_pruned, x_line)
    y_ft = _predict(model_finetuned, x_line)

    e1 = len(train_losses)
    e2 = len(ft_train_losses)
    epochs = list(range(1, e1 + e2 + 1))
    all_train = train_losses + ft_train_losses
    all_val = val_losses + ft_val_losses

    # Layout: loss panel takes left half (full height); 3 fit panels on the right
    #   row 0 right: before pruning | after pruning
    #   row 1 right: after fine-tuning (spanning both columns)
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    ax_loss = fig.add_subplot(gs[:, :2])
    ax_before = fig.add_subplot(gs[0, 2])
    ax_pruned = fig.add_subplot(gs[0, 3])
    ax_ft = fig.add_subplot(gs[1, 2:])

    # Loss panel
    ax_loss.plot(epochs, all_train, label="train")
    ax_loss.plot(epochs, all_val, label="val")
    ax_loss.axvline(x=e1 + 0.5, color="red", linestyle="--", linewidth=1.5, label="pruning")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Function panels
    _draw_fit(ax_before, x_train, y_train, x_val, y_val, x_line, y_true, y_before, "Before pruning")
    _draw_fit(ax_pruned, x_train, y_train, x_val, y_val, x_line, y_true, y_pruned, "After pruning")
    _draw_fit(ax_ft, x_train, y_train, x_val, y_val, x_line, y_true, y_ft, "After fine-tuning")

    path = os.path.join(output_dir, "pruning_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pruning summary to {path}")
