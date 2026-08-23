import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys

from tqdm import tqdm

from src.config import TrainingConfig


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    if name == "gd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}. Choose from: adam, adamw, sgd, gd, rmsprop")


def evaluate(model: nn.Module, loader: DataLoader, task: str) -> tuple[float, float | None]:
    """(mean loss, accuracy) over one pass; accuracy is None unless multiclass.
    For task "causal_lm" the loss is next-token cross-entropy per token
    (labels are the inputs, shifted here) -- perplexity is exp(loss)."""
    device = next(model.parameters()).device
    is_multiclass = task == "multiclass"
    loss_fn = nn.CrossEntropyLoss() if is_multiclass else nn.MSELoss()
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            if task == "causal_lm":
                logits = pred[:, :-1].reshape(-1, pred.shape[-1])
                loss = nn.functional.cross_entropy(logits, y_batch[:, 1:].reshape(-1))
            else:
                loss = loss_fn(pred, y_batch)
            total_loss += loss.item() * len(x_batch)
            if is_multiclass:
                correct += (pred.argmax(1) == y_batch).sum().item()
    n = len(loader.dataset)
    return total_loss / n, (correct / n if is_multiclass else None)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainingConfig,
    task: str = "regression",
    desc: str = "Training",
):
    """Train the model.

    task: "regression"      -> MSELoss,          labels float [B, 1]
          "classification"  -> MSELoss,          labels float [B, 1] (+-1 targets)
          "multiclass"      -> CrossEntropyLoss, labels long  [B]

    Returns (train_losses, val_losses, train_accs, val_accs).
    The accuracy lists are per-epoch floats for multiclass tasks, else [].

    Task "causal_lm" (e.g. the `opt` model) is pretrained-only and cannot be
    trained here -- any epochs > 0 raise.
    """
    if task == "causal_lm" and cfg.epochs > 0:
        raise NotImplementedError(
            "causal-LM models are pretrained-only in this repo: there is no LM training "
            "loop. Set training: {epochs: 0} and finetune: {epochs: 0} (one-shot protocol)."
        )
    device = next(model.parameters()).device
    optimizer = build_optimizer(model, cfg)
    is_multiclass = task == "multiclass"
    loss_fn = nn.CrossEntropyLoss() if is_multiclass else nn.MSELoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # tqdm REDRAWS ON EVERY ITERATION, and a redraw is a full line in a file.
    # Unthrottled, one 3000-epoch cell wrote 54k progress lines and the modular
    # array produced 131 MB of logs, 92% of it bars -- which also buries the
    # report tables the logs exist for. On a tty, keep the live bar; when stdout
    # is redirected, throttle to one line every 30s, which still answers "how far
    # in are we" (~10 lines per cell) without the flood.
    interactive = sys.stderr.isatty()
    pbar = tqdm(range(1, cfg.epochs + 1), desc=desc, leave=True,
                mininterval=0.1 if interactive else 30.0,
                disable=False)
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        correct_train = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            if cfg.l1 > 0:
                # L1 on Linear weight matrices only (biases excluded), as in
                # Serra et al. 2021's stability-inducing training.
                loss = loss + cfg.l1 * sum(
                    m.weight.abs().sum() for m in model.modules() if isinstance(m, nn.Linear)
                )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x_batch)
            if is_multiclass:
                correct_train += (pred.detach().argmax(1) == y_batch).sum().item()
        train_losses.append(epoch_loss / len(train_loader.dataset))
        if is_multiclass:
            train_accs.append(correct_train / len(train_loader.dataset))

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct_val = 0
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                val_loss += loss_fn(pred, y_batch).item() * len(x_batch)
                if is_multiclass:
                    correct_val += (pred.argmax(1) == y_batch).sum().item()
            val_losses.append(val_loss / len(val_loader.dataset))
            if is_multiclass:
                val_accs.append(correct_val / len(val_loader.dataset))

        postfix = {"loss": f"{train_losses[-1]:.4f}", "val": f"{val_losses[-1]:.4f}"}
        if val_accs:
            postfix["acc"] = f"{val_accs[-1]:.3f}"
        # refresh=False when piped: set_postfix defaults to refresh=True, which
        # forces a redraw every epoch and BYPASSES mininterval -- that, not the
        # interval, is what wrote a line per epoch into the job logs. With it
        # off, mininterval governs and tqdm still emits a periodic line.
        pbar.set_postfix(postfix, refresh=interactive)

        if epoch % cfg.log_every == 0:
            msg = (f"Epoch {epoch:4d} | train {train_losses[-1]:.6f}"
                   f" | val {val_losses[-1]:.6f}")
            if val_accs:
                msg += f" | acc {val_accs[-1]:.4f}"
            logging.info(msg)

    return train_losses, val_losses, train_accs, val_accs
