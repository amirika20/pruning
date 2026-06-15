import logging
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from typing import List
from torch.utils.data import TensorDataset
from config import DataConfig
from data import classify_points, get_task


# ── helpers ──────────────────────────────────────────────────────────────────

def _true_fn(x: np.ndarray, cfg: DataConfig) -> np.ndarray:
    if cfg.function == "sin":
        return np.sin(x)
    elif cfg.function == "cos":
        return np.cos(x)
    elif cfg.function == "complex":
        return 2 * np.cos(0.5 * x) + np.cos(x)
    elif cfg.function == "complex2":
        return 2 * x * np.cos(0.5 * x) + np.cos(x)
    else:
        raise ValueError(f"Unknown function: {cfg.function}")


def _predict_1d(model: nn.Module, x_line: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_line, dtype=torch.float32).unsqueeze(1).to(device)
        return model(x_t).cpu().numpy().squeeze()


def _predict_grid(model: nn.Module, pts: np.ndarray) -> np.ndarray:
    """pts: [N, 2] float32 → [N] model outputs."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(pts, dtype=torch.float32).to(device)).cpu().numpy().squeeze()


# ── 1-D regression panel ─────────────────────────────────────────────────────

def _draw_fit(ax, x_train, y_train, x_val, y_val, x_line, y_true, y_pred, title):
    ax.scatter(x_train, y_train, s=12, alpha=0.5, color="steelblue", label="train")
    ax.scatter(x_val,   y_val,   s=12, alpha=0.5, color="tomato",    label="val")
    ax.plot(x_line, y_true, color="black",      linewidth=1.5, label="true")
    ax.plot(x_line, y_pred, color="darkorange", linewidth=2, linestyle="--", label="model")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


# ── 2-D classification panel ─────────────────────────────────────────────────

def _draw_classification(ax, x_train, y_train, x_val, y_val, model, cfg, title):
    grid_res = 300
    xs = np.linspace(cfg.x_range[0], cfg.x_range[1], grid_res)
    xx, yy = np.meshgrid(xs, xs)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    z_model = _predict_grid(model, pts).reshape(grid_res, grid_res)
    z_true  = classify_points(pts, cfg).reshape(grid_res, grid_res)

    # Background heatmap of model output
    ax.contourf(xx, yy, z_model, levels=50, cmap="RdBu", alpha=0.75, vmin=-1.5, vmax=1.5)

    # Model decision boundary — solid black
    ax.contour(xx, yy, z_model, levels=[0], colors="black",  linewidths=2.0)
    # True decision boundary — dashed gray
    ax.contour(xx, yy, z_true,  levels=[0], colors="gray",   linewidths=1.5,
               linestyles="--")

    # Scatter data points, colored by class
    for x_pts, y_pts, marker, alpha in [
        (x_train, y_train, "o", 0.9),
        (x_val,   y_val,   "^", 0.6),
    ]:
        mask_pos = y_pts > 0
        ax.scatter(x_pts[mask_pos,  0], x_pts[mask_pos,  1],
                   c="red",  s=14, marker=marker, alpha=alpha, linewidths=0.3,
                   edgecolors="k", zorder=3)
        ax.scatter(x_pts[~mask_pos, 0], x_pts[~mask_pos, 1],
                   c="blue", s=14, marker=marker, alpha=alpha, linewidths=0.3,
                   edgecolors="k", zorder=3)

    ax.set_xlim(cfg.x_range)
    ax.set_ylim(cfg.x_range)
    ax.set_title(title)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    handles = [
        mlines.Line2D([0], [0], color="black", lw=2,            label="model boundary"),
        mlines.Line2D([0], [0], color="gray",  lw=1.5, ls="--", label="true boundary"),
    ]
    ax.legend(handles=handles, fontsize=7)


# ── main entry point ─────────────────────────────────────────────────────────

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

    x_train = train_ds.tensors[0].numpy()          # [N, 1] or [N, 2]
    y_train = train_ds.tensors[1].numpy().squeeze() # [N]
    x_val   = val_ds.tensors[0].numpy()
    y_val   = val_ds.tensors[1].numpy().squeeze()

    e1 = len(train_losses)
    e2 = len(ft_train_losses)
    epochs    = list(range(1, e1 + e2 + 1))
    all_train = train_losses + ft_train_losses
    all_val   = val_losses   + ft_val_losses

    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    ax_loss   = fig.add_subplot(gs[:, :2])
    ax_before = fig.add_subplot(gs[0,  2])
    ax_pruned = fig.add_subplot(gs[0,  3])
    ax_ft     = fig.add_subplot(gs[1, 2:])

    # Loss panel (same for both tasks)
    ax_loss.plot(epochs, all_train, label="train")
    ax_loss.plot(epochs, all_val,   label="val")
    ax_loss.axvline(x=e1 + 0.5, color="red", linestyle="--", linewidth=1.5, label="pruning")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    task = get_task(cfg)
    if task == "regression":
        x_train_1d = x_train.squeeze()
        x_val_1d   = x_val.squeeze()
        x_line  = np.linspace(cfg.x_range[0], cfg.x_range[1], 500)
        y_true  = _true_fn(x_line, cfg)
        _draw_fit(ax_before, x_train_1d, y_train, x_val_1d, y_val,
                  x_line, y_true, _predict_1d(model_before,   x_line), "Before pruning")
        _draw_fit(ax_pruned, x_train_1d, y_train, x_val_1d, y_val,
                  x_line, y_true, _predict_1d(model_pruned,   x_line), "After pruning")
        _draw_fit(ax_ft,     x_train_1d, y_train, x_val_1d, y_val,
                  x_line, y_true, _predict_1d(model_finetuned, x_line), "After fine-tuning")
    elif task == "classification":
        _draw_classification(ax_before, x_train, y_train, x_val, y_val,
                             model_before,   cfg, "Before pruning")
        _draw_classification(ax_pruned, x_train, y_train, x_val, y_val,
                             model_pruned,   cfg, "After pruning")
        _draw_classification(ax_ft,     x_train, y_train, x_val, y_val,
                             model_finetuned, cfg, "After fine-tuning")
    else:
        # multiclass (e.g. MNIST): no spatial visualisation — show accuracy summary
        for ax, label in [(ax_before, "Before pruning"),
                          (ax_pruned, "After pruning"),
                          (ax_ft,     "After fine-tuning")]:
            ax.axis("off")
            ax.text(0.5, 0.5, label, ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")

    path = os.path.join(output_dir, "pruning_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved pruning summary to {path}")
