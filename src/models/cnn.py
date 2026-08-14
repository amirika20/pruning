from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.data.registry import DatasetBundle
from src.models.registry import PrunableModel, register_model


class CNNBlock(nn.Module):
    """Conv2d (no bias) -> BatchNorm2d -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class CNN(PrunableModel):
    """Stack of CNNBlocks -> global average pool -> Linear head.
    Input : [B, input_channels, H, W]
    Output: [B, output_dim]
    hidden_sizes defines the output channels of each conv block."""

    def __init__(self, hidden_sizes: list[int], input_channels: int = 1, output_dim: int = 1):
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
        x = x.mean(dim=[2, 3])  # global average pooling -> [B, C]
        return self.head(x)

    # ── prunable protocol ─────────────────────────────────────────────────

    def n_prunable_layers(self) -> int:
        return len(self.blocks)

    def prunable_layer(self, idx: int) -> nn.Module:
        return self.blocks[idx].conv

    def prunable_bn(self, idx: int) -> nn.Module:
        return self.blocks[idx].bn

    def prune_layer(self, idx: int, indices_to_remove: list[int]) -> "CNN":
        """Remove output filters from CNNBlock `idx`.

        Shrinks block.conv output channels + block.bn, then fixes the next
        consumer: the next block's conv in_channels, or the head Linear's
        in_features for the last block."""
        pruned = copy.deepcopy(self)
        block = pruned.blocks[idx]
        conv, bn = block.conv, block.bn
        device = conv.weight.device
        keep = sorted(set(range(conv.out_channels)) - set(indices_to_remove))
        n_keep = len(keep)

        new_conv = nn.Conv2d(conv.in_channels, n_keep, conv.kernel_size,
                             padding=conv.padding, bias=False).to(device)
        new_conv.weight.data = conv.weight.data[keep].clone()

        new_bn = nn.BatchNorm2d(n_keep).to(device)
        new_bn.weight.data = bn.weight.data[keep].clone()
        new_bn.bias.data = bn.bias.data[keep].clone()
        new_bn.running_mean.data = bn.running_mean.data[keep].clone()
        new_bn.running_var.data = bn.running_var.data[keep].clone()
        new_bn.num_batches_tracked = bn.num_batches_tracked.clone()

        block.conv = new_conv
        block.bn = new_bn

        if idx < len(pruned.blocks) - 1:
            nc = pruned.blocks[idx + 1].conv
            new_nc = nn.Conv2d(n_keep, nc.out_channels, nc.kernel_size,
                               padding=nc.padding, bias=False).to(device)
            new_nc.weight.data = nc.weight.data[:, keep].clone()
            pruned.blocks[idx + 1].conv = new_nc
        else:
            h = pruned.head
            new_h = nn.Linear(n_keep, h.out_features).to(device)
            new_h.weight.data = h.weight.data[:, keep].clone()
            new_h.bias.data = h.bias.data.clone()
            pruned.head = new_h

        return pruned


@register_model("cnn")
def build_cnn(
    bundle: DatasetBundle, hidden_sizes: list[int] = [32, 64, 128], input_channels: int | None = None
) -> CNN:
    if input_channels is None:
        input_channels = bundle.input_shape[0] if len(bundle.input_shape) == 3 else 1
    return CNN(hidden_sizes, input_channels=input_channels, output_dim=bundle.output_dim)
