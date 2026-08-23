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
    # Stop once validation accuracy reaches this, treating `epochs` as a BUDGET
    # rather than a duration. Grokking time is strongly seed-dependent -- at 3000
    # epochs two modular seeds groked and one reached ~0.26 -- so a fixed count
    # either starves the slow seed or wastes epochs on the fast ones. Stopping on
    # the outcome does both correctly and usually costs less.
    #
    # This reads validation accuracy to decide when training is DONE, which is
    # legitimate here (nothing is tuned on it, and capacity is already measured
    # against dense val accuracy) but would not be if it selected between models.
    stop_at_accuracy: float | None = None
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
    # Pin every RNG and disable cudnn autotuning, so the data split and the
    # trained weights are reproducible from the seed alone. Required for any
    # comparison of WHICH neurons different pruning methods remove -- the run's
    # recorded fingerprints are what prove two runs started from the same
    # place (see src.reproducibility). Turn off only to trade reproducibility
    # for throughput.
    deterministic: bool = True
    # Run the exploratory before/after geometry battery (participation ratio,
    # principal-subspace alignment, response-space effective dimension and the
    # captured fraction) and write geometry_shift.csv per seed. A couple of
    # forward passes plus SVDs -- turn off for very wide models.
    analyze_geometry: bool = True
    output_root: str = "outputs"
    # Minimum dense validation accuracy for this cell to mean anything. Capacity
    # is measured relative to the dense model's own accuracy, so a model that did
    # not train has a tolerance band just above chance, every width passes, and
    # the capacity comes out NEAR ONE -- a training failure enters the results as
    # the best row in the table. run_sweep refuses a seed below this floor.
    #
    # None falls back to a generic just-above-chance check, which catches a model
    # that learned nothing but NOT one that half-learned: the modular-arithmetic
    # entry needs grokking, and a seed sitting at 30% clears any chance-based
    # floor while being useless to prune. Set this for any cell where a training
    # threshold, not mere non-randomness, is what makes the measurement valid.
    require_accuracy: float | None = None
    # Width the geometry/similarity batteries compare dense against. They used
    # the cell's own pruning spec, which for a benchmark arm carries no budget --
    # so the method's default applied, n_remove=1: ONE unit per layer, 12 of
    # OPT-125m's 36864 (0.03%). Every participation-ratio, subspace-alignment and
    # K-matrix number was a comparison against a model barely distinguishable
    # from dense.
    analysis_fraction: float = 0.5
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
            # deterministic and analyze_geometry were absent here, so a config
            # setting either to false was silently overridden by the dataclass
            # default -- the generator has always written both.
            deterministic=bool(d.get("deterministic", True)),
            analyze_geometry=bool(d.get("analyze_geometry", True)),
            output_root=d.get("output_root", "outputs"),
            require_accuracy=d.get("require_accuracy"),
            analysis_fraction=float(d.get("analysis_fraction", 0.5)),
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
