import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import TrainConfig


def _build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    if name == "gd":
        # Full-batch gradient descent: SGD with no momentum and batch_size == dataset size
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}. Choose from: adam, sgd, gd, rmsprop")


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainConfig):
    optimizer = _build_optimizer(model, cfg)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x_batch)
        train_losses.append(epoch_loss / len(train_loader.dataset))

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for x_batch, y_batch in val_loader:
                pred = model(x_batch)
                val_loss += loss_fn(pred, y_batch).item() * len(x_batch)
            val_losses.append(val_loss / len(val_loader.dataset))

        if epoch % cfg.log_every == 0:
            print(f"Epoch {epoch:4d} | train {train_losses[-1]:.6f} | val {val_losses[-1]:.6f}")

    return train_losses, val_losses
