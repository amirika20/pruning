from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.data.registry import DatasetBundle
from src.models.registry import MergeOp, PrunableModel, register_model


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


class ResMLP(PrunableModel):
    """ResNet over linear layers: stack of ResidualBlocks + Linear head."""

    def __init__(self, hidden_sizes: list[int], input_dim: int = 1, output_dim: int = 1):
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

    # ── prunable protocol ─────────────────────────────────────────────────
    # The prunable dimension is each block's branch-intermediate width
    # (branch[0]'s output); the block's output size and shortcut never change.

    def n_prunable_layers(self) -> int:
        return len(self.blocks)

    def prunable_layer(self, idx: int) -> nn.Module:
        return self.blocks[idx].branch[0]

    def outgoing_weights(self, idx: int) -> torch.Tensor:
        return self.blocks[idx].branch[2].weight.data.t()  # [H, out]

    def merge_outgoing(self, idx: int, merges: list[MergeOp]) -> "ResMLP":
        merged = copy.deepcopy(self)
        W = merged.blocks[idx].branch[2].weight.data  # [out, H]
        for op in merges:
            W[:, op.survivor] += op.scale * W[:, op.removed]
        return merged

    def prune_layer(self, idx: int, indices_to_remove: list[int]) -> "ResMLP":
        """Shrink the branch's intermediate hidden dimension in block `idx`."""
        branch_in = self.blocks[idx].branch[0]   # Linear(in_size, hidden_dim)
        branch_out = self.blocks[idx].branch[2]  # Linear(hidden_dim, out_size)
        device = branch_in.weight.device
        keep = [i for i in range(branch_in.out_features) if i not in set(indices_to_remove)]

        new_branch_in = nn.Linear(branch_in.in_features, len(keep)).to(device)
        new_branch_in.weight.data = branch_in.weight.data[keep].clone()
        new_branch_in.bias.data = branch_in.bias.data[keep].clone()

        new_branch_out = nn.Linear(len(keep), branch_out.out_features).to(device)
        new_branch_out.weight.data = branch_out.weight.data[:, keep].clone()
        new_branch_out.bias.data = branch_out.bias.data.clone()

        pruned = copy.deepcopy(self)
        pruned.blocks[idx].branch[0] = new_branch_in
        pruned.blocks[idx].branch[2] = new_branch_out
        return pruned


@register_model("resmlp")
def build_resmlp(bundle: DatasetBundle, hidden_sizes: list[int] = [64, 64, 64]) -> ResMLP:
    return ResMLP(hidden_sizes, input_dim=bundle.input_dim, output_dim=bundle.output_dim)
