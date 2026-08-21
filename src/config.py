"""Experiment configuration schema.

An `ExperimentConfig` is the single source of truth for a run: what data to
build, what model to build, how to train it, which pruning methods to apply,
and how to fine-tune afterwards. Every experiment folder stores a copy of the
exact config that produced it (see `src.experiments.runner`), so results are
always reproducible from disk.

Add new dataset/model/pruning *kinds* in their respective registries
(`src.data.registry`, `src.models.registry`, `src.pruning.registry`) -- this
file only needs to change if a genuinely new top-level section is introduced.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DataConfig:
    # Name registered in src.data.registry, e.g. "mnist", "fashion_mnist",
    # "sine", "shape2d", "modular_add".
    kind: str
    # Keyword arguments forwarded to the registered dataset builder
    # (flatten, n_samples, train_ratio, p, ...).
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    # Name registered in src.models.registry: "mlp", "resmlp", "cnn",
    # "rescnn", "transformer".
    kind: str
    # Keyword arguments forwarded to the registered model builder
    # (hidden_sizes, d_model, n_layers, ...).
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    optimizer: str = "adam"  # "adam", "adamw", "sgd", "gd", "rmsprop"
    epochs: int = 500
    lr: float = 1e-3
    momentum: float = 0.9  # used by sgd with momentum
    batch_size: int = 64
    weight_decay: float = 0.0
    # L1 penalty on all Linear weight matrices (not biases), added to the
    # loss. Serra et al. (NeurIPS 2021) use it to induce the ReLU stability
    # that lossless compression (leo_pp) exploits.
    l1: float = 0.0
    log_every: int = 50
    device: str = field(default_factory=_default_device)


@dataclass
class PruneMethodConfig:
    # Name registered in src.pruning.registry, e.g. "saturated", "mash".
    kind: str
    # Keyword arguments forwarded to the registered method's constructor.
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PruningConfig:
    # Applied in order to every prunable layer; each method's selection
    # excludes indices already chosen by the methods before it.
    methods: list[PruneMethodConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PruningConfig":
        return cls(methods=[PruneMethodConfig(**m) for m in d.get("methods", [])])


@dataclass
class ExperimentConfig:
    name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    finetune: TrainingConfig = field(default_factory=lambda: TrainingConfig(epochs=200))
    # One full train->prune->finetune run per entry -- the same seed value
    # drives torch.manual_seed (model init) and the dataset builder's
    # data_seed. Results across all seeds are averaged (see
    # src.experiments.aggregate).
    seeds: list[int] = field(default_factory=lambda: [0])
    output_root: str = "outputs"
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            name=d["name"],
            data=DataConfig(**d["data"]),
            model=ModelConfig(**d["model"]),
            training=TrainingConfig(**d.get("training", {})),
            pruning=PruningConfig.from_dict(d.get("pruning", {})),
            finetune=TrainingConfig(**d.get("finetune", {})),
            seeds=d.get("seeds", [0]),
            output_root=d.get("output_root", "outputs"),
            notes=d.get("notes", ""),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
