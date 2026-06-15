from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import DataConfig

_2D_FUNCTIONS    = {"square", "circle", "bullseye"}
_MNIST_FUNCTIONS = {"mnist"}


def get_input_dim(cfg: DataConfig) -> int:
    if cfg.function in _2D_FUNCTIONS: return 2
    return 1   # 1D functions and MNIST (channel dim handled by model)


def get_task(cfg: DataConfig) -> str:
    if cfg.function in _MNIST_FUNCTIONS: return "multiclass"
    if cfg.function in _2D_FUNCTIONS:    return "classification"
    return "regression"


def get_n_classes(cfg: DataConfig) -> int:
    if cfg.function == "mnist": return 10
    return 1


def is_image_data(cfg: DataConfig) -> bool:
    return cfg.function in _MNIST_FUNCTIONS


def classify_points(pts: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """pts: [N, 2] → [N] labels of +1 / -1 (no noise)."""
    half = (cfg.x_range[1] - cfg.x_range[0]) / 2
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)

    if cfg.function == "circle":
        return np.where(r < half * 0.5, 1.0, -1.0)

    elif cfg.function == "square":
        side = half * 0.5
        inside = (np.abs(pts[:, 0]) < side) & (np.abs(pts[:, 1]) < side)
        return np.where(inside, 1.0, -1.0)

    elif cfg.function == "bullseye":
        r1, r2 = half * 0.333, half * 0.666
        inside = (r < r1) | ((r >= r2))
        return np.where(inside, 1.0, -1.0)

    raise ValueError(f"Unknown 2D function: {cfg.function!r}")


def _load_mnist(cfg: DataConfig) -> tuple[TensorDataset, TensorDataset]:
    """Download (or load) MNIST and return a stratified subset as TensorDatasets.

    Labels are torch.long [N] (for CrossEntropyLoss).
    Images are float32 [N, 1, 28, 28], normalised to MNIST mean/std.
    """
    from torchvision import datasets, transforms

    root = Path(cfg.mnist_root).expanduser()
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train = datasets.MNIST(root, train=True,  download=True, transform=tfm)
    full_test  = datasets.MNIST(root, train=False, download=True, transform=tfm)

    rng = torch.Generator().manual_seed(cfg.seed)
    n_train = int(cfg.n_samples * cfg.train_ratio)
    n_val   = cfg.n_samples - n_train

    train_idx = torch.randperm(len(full_train), generator=rng)[:n_train]
    val_idx   = torch.randperm(len(full_test),  generator=rng)[:n_val]

    x_train = torch.stack([full_train[int(i)][0] for i in train_idx])
    y_train = torch.tensor([full_train[int(i)][1] for i in train_idx], dtype=torch.long)

    x_val = torch.stack([full_test[int(i)][0] for i in val_idx])
    y_val = torch.tensor([full_test[int(i)][1] for i in val_idx], dtype=torch.long)

    return TensorDataset(x_train, y_train), TensorDataset(x_val, y_val)


def generate_data(cfg: DataConfig):
    if cfg.function in _MNIST_FUNCTIONS:
        return _load_mnist(cfg)

    rng = np.random.default_rng(cfg.seed)

    if get_task(cfg) == "classification":
        x1 = rng.uniform(cfg.x_range[0], cfg.x_range[1], cfg.n_samples)
        x2 = rng.uniform(cfg.x_range[0], cfg.x_range[1], cfg.n_samples)
        x = np.stack([x1, x2], axis=1)         # [N, 2]
        y = classify_points(x, cfg)
        if cfg.noise_std > 0:                   # noise_std = label flip probability
            flip = rng.random(cfg.n_samples) < cfg.noise_std
            y = np.where(flip, -y, y)
    else:
        x = rng.uniform(cfg.x_range[0], cfg.x_range[1], cfg.n_samples)
        if cfg.function == "sin":
            y = np.sin(x)
        elif cfg.function == "cos":
            y = np.cos(x)
        elif cfg.function == "complex":
            y = 2 * np.cos(0.5 * x) + np.cos(x)
        elif cfg.function == "complex2":
            y = 2 * x * np.cos(0.5 * x) + np.cos(x)
        else:
            raise ValueError(f"Unknown function: {cfg.function!r}")
        y += rng.normal(0, cfg.noise_std, size=y.shape)

    x_t = torch.tensor(x, dtype=torch.float32)
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(1)                 # [N] → [N, 1]
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    split = int(cfg.n_samples * cfg.train_ratio)
    return (
        TensorDataset(x_t[:split], y_t[:split]),
        TensorDataset(x_t[split:], y_t[split:]),
    )


def make_loaders(cfg: DataConfig, batch_size: int = None):
    train_ds, val_ds = generate_data(cfg)
    bs = batch_size if batch_size is not None else len(train_ds)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=len(val_ds), shuffle=False)
    return train_loader, val_loader, train_ds, val_ds
