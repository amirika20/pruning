"""Vision dataset builders: random subsets materialized as tensors.

MNIST / Fashion-MNIST / CIFAR-10 / CIFAR-100 download via torchvision;
ImageNet reads a local copy (see `imagenet`). `flatten=True` yields flat
vectors (for MLP/ResMLP); `flatten=False` keeps pixel space (for CNN/ResNet).
"""

from __future__ import annotations

from pathlib import Path

import hashlib
import logging
import os
import torch
from torch.utils.data import TensorDataset

from src.data.registry import DatasetBundle, register_dataset


def _dataset_cache_dir() -> Path:
    """Where materialized dataset tensors live. Shares the scratch tree the job
    scripts already export, so a cache written by one cell is visible to all."""
    base = os.environ.get("PRUNING_DATA_CACHE")
    if not base:
        scratch = os.environ.get("PRUNING_SCRATCH")
        base = (f"{scratch}/cache/datasets" if scratch
                else str(Path.home() / ".cache" / "pruning-datasets"))
    return Path(base).expanduser()


def _dataset_cache_key(kind: str, **params) -> str:
    blob = kind + "|" + "|".join(f"{k}={params[k]}" for k in sorted(params))
    return f"{kind}_{hashlib.sha256(blob.encode()).hexdigest()[:16]}.pt"


def _load_dataset_cache(key: str):
    p = _dataset_cache_dir() / key
    if not p.is_file():
        return None
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception as exc:                       # noqa: BLE001
        # A truncated cache (a job killed mid-write) must not poison every later
        # run -- fall through and rebuild.
        logging.warning(f"ignoring unreadable dataset cache {p}: {exc}")
        return None


def _save_dataset_cache(key: str, payload: dict) -> None:
    d = _dataset_cache_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / (key + f".tmp{os.getpid()}")
        torch.save(payload, tmp)
        tmp.replace(d / key)                        # atomic: concurrent tasks race safely
    except OSError as exc:
        logging.warning(f"could not write dataset cache to {d}: {exc}")


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

    # DECODE ONCE, NOT 108 TIMES. Materializing the subset means JPEG-decoding
    # every selected image, and the smoke run measured 1230s of a 1330s seed
    # doing exactly that -- 92% of the wall clock. Nothing about it depends on
    # the pruning arm, so all 36 ImageNet cells and all 3 seeds were repeating
    # the same work: ~37 GPU-hours of decoding across the tier.
    #
    # Cached as uint8 BEFORE normalization: 150 KB per 224x224 example against
    # 602 KB as float32, and x.float()/255 then Normalize is bit-identical to
    # ToTensor + Normalize, so the cache changes nothing but the clock.
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    raw_tfm = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.PILToTensor(),                       # uint8 [C, H, W]
    ])
    key = _dataset_cache_key("imagenet", root=str(root_path), n_samples=n_samples,
                             n_test=n_test, image_size=image_size, seed=data_seed)
    cached = _load_dataset_cache(key)
    if cached is None:
        full_train = datasets.ImageFolder(train_dir, transform=raw_tfm)
        full_test = datasets.ImageFolder(val_dir, transform=raw_tfm)
        rng = torch.Generator().manual_seed(data_seed)
        trainval_idx = torch.randperm(len(full_train), generator=rng)[:n_samples]
        test_idx = torch.randperm(len(full_test), generator=rng)[:n_test]
        xa, ya = _subset_as_tensors(full_train, trainval_idx)
        xb, yb = _subset_as_tensors(full_test, test_idx)
        cached = {"x_trainval": xa, "y_trainval": ya, "x_test": xb, "y_test": yb,
                  "classes": list(full_train.classes)}
        _save_dataset_cache(key, cached)
    classes = cached["classes"]

    def _norm(u8: torch.Tensor) -> torch.Tensor:
        x = u8.float().div_(255.0)
        m = torch.tensor(MEAN).view(1, -1, 1, 1)
        s = torch.tensor(STD).view(1, -1, 1, 1)
        return x.sub_(m).div_(s)

    n_train = int(n_samples * train_ratio)
    x_trainval, y_trainval = _norm(cached["x_trainval"]), cached["y_trainval"]
    x_test, y_test = _norm(cached["x_test"]), cached["y_test"]

    return DatasetBundle(
        train_ds=TensorDataset(x_trainval[:n_train], y_trainval[:n_train]),
        val_ds=TensorDataset(x_trainval[n_train:], y_trainval[n_train:]),
        test_ds=TensorDataset(x_test, y_test),
        input_shape=tuple(x_trainval.shape[1:]),
        output_dim=len(classes),
        task="multiclass",
        # Folder names, i.e. WNIDs for ImageNet-style trees -- lets pretrained
        # builders map classes back to ImageNet-1k indices (see src/models/resnet.py).
        extra={"class_names": list(classes)},
    )
