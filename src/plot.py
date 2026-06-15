import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List
from torch.utils.data import TensorDataset
from config import DataConfig


def plot_losses(train_losses: List[float], val_losses: List[float], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "loss.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss plot to {path}")


def plot_fit(
    model: nn.Module,
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
    y_true = np.sin(x_line) if cfg.function == "sin" else np.cos(x_line)

    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_line, dtype=torch.float32).unsqueeze(1)
        y_pred = model(x_t).numpy().squeeze()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(x_train, y_train, s=8, alpha=0.4, color="steelblue", label="train data")
    ax.scatter(x_val, y_val, s=8, alpha=0.4, color="tomato", label="val data")
    ax.plot(x_line, y_true, color="black", linewidth=1.5, label=f"true {cfg.function}(x)")
    ax.plot(x_line, y_pred, color="darkorange", linewidth=2, linestyle="--", label="model")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Learned Function vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "fit.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fit plot to {path}")
