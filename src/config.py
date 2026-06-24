from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    function: str = "sin"       # "sin","cos","complex","circle","square","bullseye"
                                #   "mnist"      → pixel space [N,1,28,28] → use CNN/ResCNN
                                #   "mnist_flat" → vectorised  [N,784]     → use MLP/ResNet
    n_samples: int = 50         # for mnist: total subset size (train+val split by train_ratio)
    x_range: tuple = (4*-3.14159, 4*3.14159)
    noise_std: float = 0.05
    train_ratio: float = 0.8
    seed: int = 42
    mnist_root: str = "~/.cache/mnist"   # where torchvision downloads MNIST


@dataclass
class ModelConfig:
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64, 64])
    arch: str = "mlp"          # "mlp", "resnet", or "cnn"
    activation: str = "relu"
    input_channels: int = 1    # CNN only: number of input image channels


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
    experiment_name: str = "default"
    seed: int = 0


# ── Transformer / modular-arithmetic configs ──────────────────────────────────

@dataclass
class TransformerModelConfig:
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    n_layers: int = 2


@dataclass
class ModularDataConfig:
    function: str = "modular_add"  # task identifier used by get_task / get_n_classes
    p: int = 97                    # prime modulus
    train_ratio: float = 0.5
    seed: int = 42


@dataclass
class TransformerExperimentConfig:
    data: ModularDataConfig = field(default_factory=ModularDataConfig)
    model: TransformerModelConfig = field(default_factory=TransformerModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)
    finetune: TrainConfig = field(default_factory=lambda: TrainConfig(epochs=200, log_every=50))
    experiment_name: str = "modular_transformer"
    seed: int = 0
