"""Synthetic 1-D regression and 2-D shape-classification datasets.

Ported from the original flat data.py; the function/shape choice is a builder
param instead of a dataset-name suffix, and each bundle carries its true
function (or classifier) in `extra` so the plots never re-derive it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.data.registry import DatasetBundle, register_dataset

_DEFAULT_RANGE = (4 * -3.14159, 4 * 3.14159)

_1D_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "complex": lambda x: 2 * np.cos(0.5 * x) + np.cos(x),
    "complex2": lambda x: 2 * x * np.cos(0.5 * x) + np.cos(x),
}


def classify_points(pts: np.ndarray, shape: str, x_range: tuple[float, float]) -> np.ndarray:
    """pts: [N, 2] -> [N] labels of +1 / -1 (no noise)."""
    half = (x_range[1] - x_range[0]) / 2
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)

    if shape == "circle":
        return np.where(r < half * 0.5, 1.0, -1.0)
    if shape == "square":
        side = half * 0.5
        inside = (np.abs(pts[:, 0]) < side) & (np.abs(pts[:, 1]) < side)
        return np.where(inside, 1.0, -1.0)
    if shape == "bullseye":
        r1, r2 = half * 0.333, half * 0.666
        inside = (r < r1) | (r >= r2)
        return np.where(inside, 1.0, -1.0)
    raise ValueError(f"Unknown 2D shape: {shape!r}. Choose from: circle, square, bullseye")


def _split(x_t: torch.Tensor, y_t: torch.Tensor, train_ratio: float) -> tuple[TensorDataset, TensorDataset]:
    split = int(len(x_t) * train_ratio)
    return TensorDataset(x_t[:split], y_t[:split]), TensorDataset(x_t[split:], y_t[split:])


@register_dataset("sine")
def sine(
    data_seed: int = 0,
    function: str = "sin",  # "sin", "cos", "complex", "complex2"
    n_samples: int = 50,
    x_range: tuple[float, float] = _DEFAULT_RANGE,
    noise_std: float = 0.05,
    train_ratio: float = 0.8,
) -> DatasetBundle:
    if function not in _1D_FUNCTIONS:
        raise ValueError(f"Unknown 1D function: {function!r}. Choose from: {list(_1D_FUNCTIONS)}")
    true_fn = _1D_FUNCTIONS[function]

    rng = np.random.default_rng(data_seed)
    x = rng.uniform(x_range[0], x_range[1], n_samples)
    y = true_fn(x) + rng.normal(0, noise_std, size=x.shape)

    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # [N, 1]
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    train_ds, val_ds = _split(x_t, y_t, train_ratio)

    # Held-out test set: an independent draw from the same distribution,
    # sized like the val split.
    n_test = max(1, n_samples - int(n_samples * train_ratio))
    x_test = rng.uniform(x_range[0], x_range[1], n_test)
    y_test = true_fn(x_test) + rng.normal(0, noise_std, size=x_test.shape)
    test_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
    )

    return DatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        input_shape=(1,),
        output_dim=1,
        task="regression",
        extra={"true_fn": true_fn, "x_range": tuple(x_range)},
    )


@register_dataset("shape2d")
def shape2d(
    data_seed: int = 0,
    shape: str = "circle",  # "circle", "square", "bullseye"
    n_samples: int = 50,
    x_range: tuple[float, float] = _DEFAULT_RANGE,
    noise_std: float = 0.05,  # label-flip probability
    train_ratio: float = 0.8,
) -> DatasetBundle:
    rng = np.random.default_rng(data_seed)
    x1 = rng.uniform(x_range[0], x_range[1], n_samples)
    x2 = rng.uniform(x_range[0], x_range[1], n_samples)
    x = np.stack([x1, x2], axis=1)  # [N, 2]
    y = classify_points(x, shape, x_range)
    if noise_std > 0:
        flip = rng.random(n_samples) < noise_std
        y = np.where(flip, -y, y)

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    train_ds, val_ds = _split(x_t, y_t, train_ratio)

    # Held-out test set: an independent draw (same distribution, same
    # label-flip noise), sized like the val split.
    n_test = max(1, n_samples - int(n_samples * train_ratio))
    xt1 = rng.uniform(x_range[0], x_range[1], n_test)
    xt2 = rng.uniform(x_range[0], x_range[1], n_test)
    x_test = np.stack([xt1, xt2], axis=1)
    y_test = classify_points(x_test, shape, x_range)
    if noise_std > 0:
        flip = rng.random(n_test) < noise_std
        y_test = np.where(flip, -y_test, y_test)
    test_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
    )

    return DatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        input_shape=(2,),
        output_dim=1,
        task="classification",
        extra={
            "classify_fn": lambda pts: classify_points(pts, shape, tuple(x_range)),
            "x_range": tuple(x_range),
        },
    )
