import torch
import torch.nn as nn
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from config import ModelConfig


class MLP(nn.Module):
    def __init__(self, hidden_sizes: List[int], input_dim: int = 1, output_dim: int = 1):
        super().__init__()
        layers = []
        in_size = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_dim))
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
    def __init__(self, hidden_sizes: List[int], input_dim: int = 1, output_dim: int = 1):
        super().__init__()
        sizes = [input_dim] + hidden_sizes
        self.blocks = nn.ModuleList([
            ResidualBlock(sizes[i], sizes[i + 1])
            for i in range(len(hidden_sizes))
        ])
        self.head = nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class CNNBlock(nn.Module):
    """Conv2d (no bias) → BatchNorm2d → ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class CNN(nn.Module):
    """Stack of CNNBlocks → global average pool → Linear head.
    Input : [B, input_channels, H, W]
    Output: [B, output_dim]
    hidden_sizes defines the output channels of each conv block."""
    def __init__(self, hidden_sizes: List[int], input_channels: int = 1, output_dim: int = 1):
        super().__init__()
        channels = [input_channels] + hidden_sizes
        self.blocks = nn.ModuleList([
            CNNBlock(channels[i], channels[i + 1])
            for i in range(len(hidden_sizes))
        ])
        self.head = nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=[2, 3])   # global average pooling → [B, C]
        return self.head(x)


class ResCNNBlock(nn.Module):
    """CNN block with a residual shortcut.

    Branch: Conv2d → BN → ReLU → Conv2d → BN   (intermediate dim is prunable)
    Shortcut: identity or 1×1 Conv2d+BN when channels change
    Output: ReLU(branch + shortcut)

    The prunable dimension is the intermediate channel count (output of branch[0],
    input of branch[3]).  The block's output channels and shortcut are untouched
    during pruning — exactly analogous to the MLP ResNet.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=False),  # [0]
            nn.BatchNorm2d(out_channels),                                                  # [1]
            nn.ReLU(),                                                                     # [2]
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad, bias=False),  # [3]
            nn.BatchNorm2d(out_channels),                                                  # [4]
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.branch(x) + self.shortcut(x))


class ResCNN(nn.Module):
    """Stack of ResCNNBlocks → global average pool → Linear head.
    Input : [B, input_channels, H, W]
    Output: [B, output_dim]
    hidden_sizes defines the output channels of each residual block."""
    def __init__(self, hidden_sizes: List[int], input_channels: int = 1, output_dim: int = 1):
        super().__init__()
        channels = [input_channels] + hidden_sizes
        self.blocks = nn.ModuleList([
            ResCNNBlock(channels[i], channels[i + 1])
            for i in range(len(hidden_sizes))
        ])
        self.head = nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=[2, 3])   # global average pooling → [B, C]
        return self.head(x)


def build_model(cfg: "ModelConfig", input_dim: int = 1, output_dim: int = 1) -> nn.Module:
    if cfg.arch == "mlp":
        return MLP(cfg.hidden_sizes, input_dim, output_dim)
    elif cfg.arch == "resnet":
        return ResNet(cfg.hidden_sizes, input_dim, output_dim)
    elif cfg.arch == "cnn":
        return CNN(cfg.hidden_sizes, cfg.input_channels, output_dim)
    elif cfg.arch == "rescnn":
        return ResCNN(cfg.hidden_sizes, cfg.input_channels, output_dim)
    else:
        raise ValueError(f"Unknown arch: {cfg.arch!r}. Choose from: mlp, resnet, cnn, rescnn")
