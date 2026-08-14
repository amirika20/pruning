"""Per-seed pruning summary plot + cross-seed aggregate curves.

All dataset specifics (true function, decision-boundary classifier, x range)
come from `DatasetBundle.task`/`.extra`, so these functions never
special-case dataset kinds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.data.registry import DatasetBundle


# ── helpers ──────────────────────────────────────────────────────────────────

def _predict_1d(model: nn.Module, x_line: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_line, dtype=torch.float32).unsqueeze(1).to(device)
        return model(x_t).cpu().numpy().squeeze()


def _predict_grid(model: nn.Module, pts: np.ndarray) -> np.ndarray:
    """pts: [N, 2] float32 -> [N] model outputs."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(pts, dtype=torch.float32).to(device)).cpu().numpy().squeeze()


# ── 1-D regression panel ─────────────────────────────────────────────────────

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


# ── 2-D classification panel ─────────────────────────────────────────────────

def _draw_classification(ax, x_train, y_train, x_val, y_val, model, bundle: DatasetBundle, title):
    x_range = bundle.extra["x_range"]
    classify_fn = bundle.extra["classify_fn"]
    grid_res = 300
    xs = np.linspace(x_range[0], x_range[1], grid_res)
    xx, yy = np.meshgrid(xs, xs)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    z_model = _predict_grid(model, pts).reshape(grid_res, grid_res)
    z_true = classify_fn(pts).reshape(grid_res, grid_res)

    # Background heatmap of model output
    ax.contourf(xx, yy, z_model, levels=50, cmap="RdBu", alpha=0.75, vmin=-1.5, vmax=1.5)

    # Model decision boundary -- solid black; true boundary -- dashed gray
    ax.contour(xx, yy, z_model, levels=[0], colors="black", linewidths=2.0)
    ax.contour(xx, yy, z_true, levels=[0], colors="gray", linewidths=1.5, linestyles="--")

    for x_pts, y_pts, marker, alpha in [
        (x_train, y_train, "o", 0.9),
        (x_val, y_val, "^", 0.6),
    ]:
        mask_pos = y_pts > 0
        ax.scatter(x_pts[mask_pos, 0], x_pts[mask_pos, 1],
                   c="red", s=14, marker=marker, alpha=alpha, linewidths=0.3,
                   edgecolors="k", zorder=3)
        ax.scatter(x_pts[~mask_pos, 0], x_pts[~mask_pos, 1],
                   c="blue", s=14, marker=marker, alpha=alpha, linewidths=0.3,
                   edgecolors="k", zorder=3)

    ax.set_xlim(x_range)
    ax.set_ylim(x_range)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    handles = [
        mlines.Line2D([0], [0], color="black", lw=2, label="model boundary"),
        mlines.Line2D([0], [0], color="gray", lw=1.5, ls="--", label="true boundary"),
    ]
    ax.legend(handles=handles, fontsize=7)


# ── per-seed pruning summary ─────────────────────────────────────────────────

def plot_pruning_summary(
    model_before: nn.Module,
    model_pruned: nn.Module,
    model_finetuned: nn.Module,
    history: dict,
    bundle: DatasetBundle,
    save_path: str | Path,
):
    """One figure per seed: loss (and accuracy for multiclass) across the
    train -> prune -> finetune timeline, plus before/pruned/finetuned fit or
    decision-boundary panels for the low-dimensional synthetic tasks.

    `history` is the dict produced by the runner: keys train_loss/val_loss/
    ft_train_loss/ft_val_loss (+ *_acc variants for multiclass).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    e1 = len(history["train_loss"])
    e2 = len(history["ft_train_loss"])
    epochs = list(range(1, e1 + e2 + 1))
    all_train = history["train_loss"] + history["ft_train_loss"]
    all_val = history["val_loss"] + history["ft_val_loss"]

    if bundle.task == "multiclass":
        # Two-panel layout: loss (left) | accuracy (right)
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
        fig.subplots_adjust(wspace=0.3)

        ax_loss.plot(epochs, all_train, label="train loss")
        ax_loss.plot(epochs, all_val, label="val loss")
        ax_loss.axvline(x=e1 + 0.5, color="red", linestyle="--", linewidth=1.5, label="pruning")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        all_train_acc = history["train_acc"] + history["ft_train_acc"]
        all_val_acc = history["val_acc"] + history["ft_val_acc"]
        ax_acc.plot(epochs[:len(all_train_acc)], all_train_acc, label="train acc")
        ax_acc.plot(epochs[:len(all_val_acc)], all_val_acc, label="val acc")
        ax_acc.axvline(x=e1 + 0.5, color="red", linestyle="--", linewidth=1.5, label="pruning")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title("Accuracy")
        ax_acc.set_ylim(0, 1)
        ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

    else:
        x_train = bundle.train_ds.tensors[0].numpy()
        y_train = bundle.train_ds.tensors[1].numpy().squeeze()
        x_val = bundle.val_ds.tensors[0].numpy()
        y_val = bundle.val_ds.tensors[1].numpy().squeeze()

        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

        ax_loss = fig.add_subplot(gs[:, :2])
        ax_before = fig.add_subplot(gs[0, 2])
        ax_pruned = fig.add_subplot(gs[0, 3])
        ax_ft = fig.add_subplot(gs[1, 2:])

        ax_loss.plot(epochs, all_train, label="train")
        ax_loss.plot(epochs, all_val, label="val")
        ax_loss.axvline(x=e1 + 0.5, color="red", linestyle="--", linewidth=1.5, label="pruning")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        if bundle.task == "regression":
            x_range = bundle.extra["x_range"]
            true_fn = bundle.extra["true_fn"]
            x_train_1d = x_train.squeeze()
            x_val_1d = x_val.squeeze()
            x_line = np.linspace(x_range[0], x_range[1], 500)
            y_true = true_fn(x_line)
            _draw_fit(ax_before, x_train_1d, y_train, x_val_1d, y_val,
                      x_line, y_true, _predict_1d(model_before, x_line), "Before pruning")
            _draw_fit(ax_pruned, x_train_1d, y_train, x_val_1d, y_val,
                      x_line, y_true, _predict_1d(model_pruned, x_line), "After pruning")
            _draw_fit(ax_ft, x_train_1d, y_train, x_val_1d, y_val,
                      x_line, y_true, _predict_1d(model_finetuned, x_line), "After fine-tuning")
        else:  # classification
            _draw_classification(ax_before, x_train, y_train, x_val, y_val,
                                 model_before, bundle, "Before pruning")
            _draw_classification(ax_pruned, x_train, y_train, x_val, y_val,
                                 model_pruned, bundle, "After pruning")
            _draw_classification(ax_ft, x_train, y_train, x_val, y_val,
                                 model_finetuned, bundle, "After fine-tuning")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved pruning summary to {save_path}")


# ── cross-seed aggregate curves ──────────────────────────────────────────────

def _mean_std(histories: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array(histories)  # [n_seeds, n_epochs]
    return arr.mean(axis=0), arr.std(axis=0)


def plot_aggregate_curves(seed_histories: list[dict], task: str, save_path: str | Path, title: str):
    """Mean +- std loss (and accuracy for multiclass) across seeds, over the
    concatenated train -> prune -> finetune timeline."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    e1 = len(seed_histories[0]["train_loss"])
    curves = {
        "train": [h["train_loss"] + h["ft_train_loss"] for h in seed_histories],
        "val": [h["val_loss"] + h["ft_val_loss"] for h in seed_histories],
    }
    is_multiclass = task == "multiclass"
    n_panels = 2 if is_multiclass else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)
    axes = axes[0]

    ax = axes[0]
    for label, hists in curves.items():
        mean, std = _mean_std(hists)
        epochs = np.arange(1, len(mean) + 1)
        (line,) = ax.plot(epochs, mean, label=f"{label} (mean of {len(hists)} seeds)")
        ax.fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=0.2)
    ax.axvline(x=e1 + 0.5, color="red", linestyle="--", linewidth=1.5, label="pruning")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title} — loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if is_multiclass:
        ax = axes[1]
        acc_curves = {
            "train": [h["train_acc"] + h["ft_train_acc"] for h in seed_histories],
            "val": [h["val_acc"] + h["ft_val_acc"] for h in seed_histories],
        }
        for label, hists in acc_curves.items():
            mean, std = _mean_std(hists)
            epochs = np.arange(1, len(mean) + 1)
            (line,) = ax.plot(epochs, mean, label=f"{label} acc")
            ax.fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=0.2)
        ax.axvline(x=e1 + 0.5, color="red", linestyle="--", linewidth=1.5, label="pruning")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title} — accuracy")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved aggregate curves to {save_path}")
