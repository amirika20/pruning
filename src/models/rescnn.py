from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.data.registry import DatasetBundle
from src.models.registry import PrunableModel, register_model


class ResCNNBlock(nn.Module):
    """CNN block with a residual shortcut.

    Branch: Conv2d -> BN -> ReLU -> Conv2d -> BN   (intermediate dim is prunable)
    Shortcut: identity or 1x1 Conv2d+BN when channels change
    Output: ReLU(branch + shortcut)

    The prunable dimension is the intermediate channel count (output of
    branch[0], input of branch[3]). The block's output channels and shortcut
    are untouched during pruning -- exactly analogous to ResMLP.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=False),  # [0]
            nn.BatchNorm2d(out_channels),                                                 # [1]
            nn.ReLU(),                                                                    # [2]
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad, bias=False),  # [3]
            nn.BatchNorm2d(out_channels),                                                 # [4]
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


class ResCNN(PrunableModel):
    """Stack of ResCNNBlocks -> global average pool -> Linear head."""

    def __init__(self, hidden_sizes: list[int], input_channels: int = 1, output_dim: int = 1):
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
        x = x.mean(dim=[2, 3])  # global average pooling -> [B, C]
        return self.head(x)

    # ── prunable protocol ─────────────────────────────────────────────────

    def n_prunable_layers(self) -> int:
        return len(self.blocks)

    def prunable_layer(self, idx: int) -> nn.Module:
        return self.blocks[idx].branch[0]

    def prunable_bn(self, idx: int) -> nn.Module:
        return self.blocks[idx].branch[1]

    def prune_layer(self, idx: int, indices_to_remove: list[int]) -> "ResCNN":
        """Prune the intermediate channels of ResCNNBlock `idx`.

        branch[0] (Conv2d) and branch[1] (BN) lose output channels;
        branch[3] (Conv2d) loses input channels. The block's output channels
        and shortcut are completely unchanged."""
        pruned = copy.deepcopy(self)
        block = pruned.blocks[idx]
        conv1, bn1, conv2 = block.branch[0], block.branch[1], block.branch[3]
        device = conv1.weight.device
        keep = sorted(set(range(conv1.out_channels)) - set(indices_to_remove))
        n_keep = len(keep)

        new_conv1 = nn.Conv2d(conv1.in_channels, n_keep, conv1.kernel_size,
                              padding=conv1.padding, bias=False).to(device)
        new_conv1.weight.data = conv1.weight.data[keep].clone()

        new_bn1 = nn.BatchNorm2d(n_keep).to(device)
        new_bn1.weight.data = bn1.weight.data[keep].clone()
        new_bn1.bias.data = bn1.bias.data[keep].clone()
        new_bn1.running_mean.data = bn1.running_mean.data[keep].clone()
        new_bn1.running_var.data = bn1.running_var.data[keep].clone()
        new_bn1.num_batches_tracked = bn1.num_batches_tracked.clone()

        new_conv2 = nn.Conv2d(n_keep, conv2.out_channels, conv2.kernel_size,
                              padding=conv2.padding, bias=False).to(device)
        new_conv2.weight.data = conv2.weight.data[:, keep].clone()

        block.branch[0] = new_conv1
        block.branch[1] = new_bn1
        block.branch[3] = new_conv2
        return pruned


@register_model("rescnn")
def build_rescnn(
    bundle: DatasetBundle, hidden_sizes: list[int] = [32, 64, 128], input_channels: int | None = None
) -> ResCNN:
    if input_channels is None:
        input_channels = bundle.input_shape[0] if len(bundle.input_shape) == 3 else 1
    return ResCNN(hidden_sizes, input_channels=input_channels, output_dim=bundle.output_dim)
