"""Vision dataset builders: random subsets materialized as tensors.

MNIST / Fashion-MNIST / CIFAR-10 / CIFAR-100 download via torchvision;
ImageNet reads a local copy (see `imagenet`). `flatten=True` yields flat
vectors (for MLP/ResMLP); `flatten=False` keeps pixel space (for CNN/ResNet).
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
    output_dim: int = 10,
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
        output_dim=output_dim,
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


@register_dataset("cifar10")
def cifar10(
    data_seed: int = 0,
    flatten: bool = False,
    n_samples: int = 2000,
    train_ratio: float = 0.8,
    n_test: int | None = None,  # None = the full official test split
    root: str = "~/.cache/cifar",
) -> DatasetBundle:
    from torchvision import datasets

    return _build_vision_bundle(
        datasets.CIFAR10, ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        data_seed, flatten, n_samples, train_ratio, n_test, root,
    )


@register_dataset("cifar100")
def cifar100(
    data_seed: int = 0,
    flatten: bool = False,
    n_samples: int = 2000,
    train_ratio: float = 0.8,
    n_test: int | None = None,  # None = the full official test split
    root: str = "~/.cache/cifar",
) -> DatasetBundle:
    from torchvision import datasets

    return _build_vision_bundle(
        datasets.CIFAR100, ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        data_seed, flatten, n_samples, train_ratio, n_test, root,
        output_dim=100,
    )


@register_dataset("imagenet")
def imagenet(
    data_seed: int = 0,
    n_samples: int = 2000,
    train_ratio: float = 0.8,
    n_test: int = 2000,
    image_size: int = 224,
    root: str = "~/datasets/imagenet",
) -> DatasetBundle:
    """ImageNet-1k from a LOCAL copy (no download): `root` must contain
    `train/` and `val/` in the usual ImageFolder layout (one subdirectory per
    class). Like the other vision builders, a random `n_samples`-example
    subset of train/ is materialized as tensors and split into train/val by
    `train_ratio`; `n_test` examples from val/ form the test split. Keep
    n_samples/n_test modest -- each 224x224 example is ~0.6 MB as a tensor."""
    from torchvision import datasets, transforms

    root_path = Path(root).expanduser()
    train_dir, val_dir = root_path / "train", root_path / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"imagenet root {root_path} must contain train/ and val/ directories "
            "in ImageFolder layout (one subdirectory per class); ImageNet cannot "
            "be downloaded automatically"
        )

    tfm = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    full_train = datasets.ImageFolder(train_dir, transform=tfm)
    full_test = datasets.ImageFolder(val_dir, transform=tfm)

    rng = torch.Generator().manual_seed(data_seed)
    n_train = int(n_samples * train_ratio)
    trainval_idx = torch.randperm(len(full_train), generator=rng)[:n_samples]
    test_idx = torch.randperm(len(full_test), generator=rng)[:n_test]

    x_trainval, y_trainval = _subset_as_tensors(full_train, trainval_idx)
    x_test, y_test = _subset_as_tensors(full_test, test_idx)

    return DatasetBundle(
        train_ds=TensorDataset(x_trainval[:n_train], y_trainval[:n_train]),
        val_ds=TensorDataset(x_trainval[n_train:], y_trainval[n_train:]),
        test_ds=TensorDataset(x_test, y_test),
        input_shape=tuple(x_trainval.shape[1:]),
        output_dim=len(full_train.classes),
        task="multiclass",
        # Folder names, i.e. WNIDs for ImageNet-style trees -- lets pretrained
        # builders map classes back to ImageNet-1k indices (see src/models/resnet.py).
        extra={"class_names": list(full_train.classes)},
    )
