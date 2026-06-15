import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from config import DataConfig


def generate_data(cfg: DataConfig):
    rng = np.random.default_rng(cfg.seed)
    x = rng.uniform(cfg.x_range[0], cfg.x_range[1], cfg.n_samples)
    if cfg.function == "sin":
        y = np.sin(x)
    elif cfg.function == "cos":
        y = np.cos(x)
    elif cfg.function == "complex":
        y = 2*np.cos(0.5*x)+np.cos(x)
    else:
        raise ValueError(f"Unknown function: {cfg.function}")
    y += rng.normal(0, cfg.noise_std, size=y.shape)

    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    split = int(cfg.n_samples * cfg.train_ratio)
    x_train, x_val = x_t[:split], x_t[split:]
    y_train, y_val = y_t[:split], y_t[split:]

    return (
        TensorDataset(x_train, y_train),
        TensorDataset(x_val, y_val),
    )


def make_loaders(cfg: DataConfig):
    train_ds, val_ds = generate_data(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.n_samples, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    return train_loader, val_loader, train_ds, val_ds
