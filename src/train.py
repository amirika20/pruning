import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import TrainConfig


def _build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
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


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    task: str = "regression",
    desc: str = "Training",
):
    """Train the model.

    task: "regression"      → MSELoss,          labels float [B, 1]
          "classification"  → MSELoss,          labels float [B, 1]  (±1 targets)
          "multiclass"      → CrossEntropyLoss, labels long  [B]

    Returns (train_losses, val_losses, val_accs).
    val_accs is a list of per-epoch accuracy floats for multiclass tasks, else [].
    """
    device = next(model.parameters()).device
    optimizer = _build_optimizer(model, cfg)
    is_multiclass = task == "multiclass"
    loss_fn = nn.CrossEntropyLoss() if is_multiclass else nn.MSELoss()

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    pbar = tqdm(range(1, cfg.epochs + 1), desc=desc, leave=True)
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
        pbar.set_postfix(postfix)

        if epoch % cfg.log_every == 0:
            msg = (f"Epoch {epoch:4d} | train {train_losses[-1]:.6f}"
                   f" | val {val_losses[-1]:.6f}")
            if val_accs:
                msg += f" | acc {val_accs[-1]:.4f}"
            logging.info(msg)

    return train_losses, val_losses, train_accs, val_accs
