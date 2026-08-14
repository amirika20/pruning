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
    n_test: int | None,
    root: str,
) -> DatasetBundle:
    """`n_samples` examples are drawn from the official TRAIN split and cut
    into train/val by `train_ratio`; the test set is `n_test` examples (all
    10k by default) from the official TEST split -- so val is genuinely
    disjoint from test and never comes from the test data."""
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
    trainval_idx = torch.randperm(len(full_train), generator=rng)[:n_samples]
    test_idx = torch.randperm(len(full_test), generator=rng)[: (n_test if n_test is not None else len(full_test))]

    x_trainval, y_trainval = _subset_as_tensors(full_train, trainval_idx)
    x_test, y_test = _subset_as_tensors(full_test, test_idx)

    if flatten:
        x_trainval = x_trainval.view(len(x_trainval), -1)  # [N, 784]
        x_test = x_test.view(len(x_test), -1)

    return DatasetBundle(
        train_ds=TensorDataset(x_trainval[:n_train], y_trainval[:n_train]),
        val_ds=TensorDataset(x_trainval[n_train:], y_trainval[n_train:]),
        test_ds=TensorDataset(x_test, y_test),
        input_shape=tuple(x_trainval.shape[1:]),
        output_dim=10,
        task="multiclass",
    )


@register_dataset("mnist")
def mnist(
    data_seed: int = 0,
    flatten: bool = True,
    n_samples: int = 2000,
    train_ratio: float = 0.8,
    n_test: int | None = None,  # None = the full official test split
    root: str = "~/.cache/mnist",
) -> DatasetBundle:
    from torchvision import datasets

    return _build_vision_bundle(
        datasets.MNIST, ((0.1307,), (0.3081,)),
        data_seed, flatten, n_samples, train_ratio, n_test, root,
    )


@register_dataset("fashion_mnist")
def fashion_mnist(
    data_seed: int = 0,
    flatten: bool = True,
    n_samples: int = 2000,
    train_ratio: float = 0.8,
    n_test: int | None = None,  # None = the full official test split
    root: str = "~/.cache/mnist",
) -> DatasetBundle:
    from torchvision import datasets

    return _build_vision_bundle(
        datasets.FashionMNIST, ((0.2860,), (0.3530,)),
        data_seed, flatten, n_samples, train_ratio, n_test, root,
    )
