"""Dataset registry: maps a config's `data.kind` string to a builder function.

A dataset builder has signature (data_seed: int, **params) -> DatasetBundle
and declares its own input_dim/output_dim/task -- the trainer picks the right
loss and evaluation metric from `task`, and model builders read input/output
sizes from the bundle, so nothing downstream ever string-matches on dataset
names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from torch.utils.data import DataLoader, TensorDataset

DATASET_REGISTRY: dict[str, Callable[..., "DatasetBundle"]] = {}


@dataclass
class DatasetBundle:
    """Everything the trainer, model builders, and plots need from a dataset.

    task: "regression"     -> MSELoss, float labels [N, 1]
          "classification" -> MSELoss, float +-1 labels [N, 1]
          "multiclass"     -> CrossEntropyLoss, long labels [N]

    extra holds task-specific info so downstream code never special-cases
    dataset kinds: synthetic datasets stash {"true_fn": ..., "x_range": ...}
    (or {"classify_fn": ...}) for the fit plots; modular arithmetic stashes
    {"vocab_size": ..., "seq_len": ..., "p": ...} for the transformer builder.
    """

    train_ds: TensorDataset
    val_ds: TensorDataset
    input_shape: tuple[int, ...]  # per-sample shape, e.g. (784,), (1, 28, 28), (4,)
    output_dim: int
    task: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def input_dim(self) -> int:
        """Flat feature dimension -- meaningful for vector inputs (MLP-style)."""
        return self.input_shape[-1] if len(self.input_shape) == 1 else self.input_shape[0]

    def loaders(self, batch_size: int | None = None) -> tuple[DataLoader, DataLoader]:
        """Shuffled train loader + single-batch val loader (val order is irrelevant)."""
        bs = batch_size if batch_size is not None else len(self.train_ds)
        train_loader = DataLoader(self.train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=len(self.val_ds), shuffle=False)
        return train_loader, val_loader


def register_dataset(name: str) -> Callable[[Callable[..., DatasetBundle]], Callable[..., DatasetBundle]]:
    def decorator(builder: Callable[..., DatasetBundle]) -> Callable[..., DatasetBundle]:
        if name in DATASET_REGISTRY:
            raise ValueError(f"dataset kind {name!r} already registered")
        DATASET_REGISTRY[name] = builder
        return builder

    return decorator


def build_dataset(kind: str, data_seed: int = 0, **params) -> DatasetBundle:
    if kind not in DATASET_REGISTRY:
        raise KeyError(f"unknown dataset kind {kind!r}. available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[kind](data_seed=data_seed, **params)
