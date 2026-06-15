from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    function: str = "sin"       # "sin" or "cos"
    n_samples: int = 50
    x_range: tuple = (4*-3.14159, 4*3.14159)
    noise_std: float = 0.05
    train_ratio: float = 0.8
    seed: int = 42


@dataclass
class ModelConfig:
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64, 64])
    arch: str = "mlp"   # "mlp" or "resnet"
    activation: str = "relu"


@dataclass
class TrainConfig:
    optimizer: str = "adam"     # "adam", "sgd", "gd", "rmsprop"
    epochs: int = 500
    lr: float = 1e-3
    momentum: float = 0.9       # used by sgd/gd with momentum
    batch_size: int = 64
    weight_decay: float = 0.0
    log_every: int = 50


@dataclass
class PruneConfig:
    threshold: float = 0.3
    prune_silent: bool = True    # remove neurons never activated on training data
    prune_redundant: bool = True # remove neurons with nearly identical hyperplanes


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)
    finetune: TrainConfig = field(default_factory=lambda: TrainConfig(epochs=200, log_every=50))
    output_dir: str = "outputs"
