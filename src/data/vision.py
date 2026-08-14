"""MNIST / Fashion-MNIST builders: stratified random subsets as tensors.

`flatten=True` yields vectors [N, 784] (for MLP/ResMLP); `flatten=False`
keeps pixel space [N, 1, 28, 28] (for CNN/ResCNN).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from src.data.registry import DatasetBundle, register_dataset


def _subset_as_tensors(dataset, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([dataset[int(i)][0] for i in indices])
    y = torch.tensor([dataset[int(i)][1] for i in indices], dtype=torch.long)
    return x, y


def _build_vision_bundle(
    dataset_cls,
    normalize: tuple[tuple[float, ...], tuple[float, ...]],
    data_seed: int,
    flatten: bool,
    n_samples: int,
    train_ratio: float,
    root: str,
) -> DatasetBundle:
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalize),
    ])
    root_path = Path(root).expanduser()
    full_train = dataset_cls(root_path, train=True, download=True, transform=tfm)
    full_test = dataset_cls(root_path, train=False, download=True, transform=tfm)

    rng = torch.Generator().manual_seed(data_seed)
    n_train = int(n_samples * train_ratio)
    n_val = n_samples - n_train
    train_idx = torch.randperm(len(full_train), generator=rng)[:n_train]
    val_idx = torch.randperm(len(full_test), generator=rng)[:n_val]

    x_train, y_train = _subset_as_tensors(full_train, train_idx)
    x_val, y_val = _subset_as_tensors(full_test, val_idx)

    if flatten:
        x_train = x_train.view(len(x_train), -1)  # [N, 784]
        x_val = x_val.view(len(x_val), -1)

    return DatasetBundle(
        train_ds=TensorDataset(x_train, y_train),
        val_ds=TensorDataset(x_val, y_val),
        input_shape=tuple(x_train.shape[1:]),
        output_dim=10,
        task="multiclass",
    )


@register_dataset("mnist")
def mnist(
    data_seed: int = 0,
    flatten: bool = True,
    n_samples: int = 2000,
    train_ratio: float = 0.8,
    root: str = "~/.cache/mnist",
) -> DatasetBundle:
    from torchvision import datasets

    return _build_vision_bundle(
        datasets.MNIST, ((0.1307,), (0.3081,)),
        data_seed, flatten, n_samples, train_ratio, root,
    )


@register_dataset("fashion_mnist")
def fashion_mnist(
    data_seed: int = 0,
    flatten: bool = True,
    n_samples: int = 2000,
    train_ratio: float = 0.8,
    root: str = "~/.cache/mnist",
) -> DatasetBundle:
    from torchvision import datasets

    return _build_vision_bundle(
        datasets.FashionMNIST, ((0.2860,), (0.3530,)),
        data_seed, flatten, n_samples, train_ratio, root,
    )
