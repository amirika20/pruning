import torch
import torch.nn as nn
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from config import ModelConfig


class MLP(nn.Module):
    def __init__(self, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        in_size = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size),
        )
        self.shortcut = (
            nn.Linear(in_size, out_size, bias=False)
            if in_size != out_size
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.branch(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, hidden_sizes: List[int]):
        super().__init__()
        sizes = [1] + hidden_sizes
        self.blocks = nn.ModuleList([
            ResidualBlock(sizes[i], sizes[i + 1])
            for i in range(len(hidden_sizes))
        ])
        self.head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def build_model(cfg: "ModelConfig") -> nn.Module:
    if cfg.arch == "mlp":
        return MLP(cfg.hidden_sizes)
    elif cfg.arch == "resnet":
        return ResNet(cfg.hidden_sizes)
    else:
        raise ValueError(f"Unknown arch: {cfg.arch}")
