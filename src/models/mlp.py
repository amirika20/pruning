from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.data.registry import DatasetBundle
from src.models.registry import MergeOp, PrunableModel, register_model


class MLP(PrunableModel):
    def __init__(self, hidden_sizes: list[int], input_dim: int = 1, output_dim: int = 1):
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

    # ── prunable protocol ─────────────────────────────────────────────────
    # net = [Linear, ReLU] * n_hidden + [Linear], so hidden Linear i sits at 2i.

    def n_prunable_layers(self) -> int:
        return (len(self.net) - 1) // 2

    def prunable_layer(self, idx: int) -> nn.Module:
        return self.net[2 * idx]

    def outgoing_module(self, idx: int) -> nn.Module:
        return self.net[2 * (idx + 1)]

    def outgoing_weights(self, idx: int) -> torch.Tensor:
        return self.net[2 * (idx + 1)].weight.data.t()  # [H, out]

    def merge_outgoing(self, idx: int, merges: list[MergeOp]) -> "MLP":
        merged = copy.deepcopy(self)
        W = merged.net[2 * (idx + 1)].weight.data  # [out, H]
        for op in merges:
            W[:, op.survivor] += op.scale * W[:, op.removed]
        return merged

    def prune_layer(self, idx: int, indices_to_remove: list[int]) -> "MLP":
        """Remove output neurons from hidden layer `idx` and fix the next layer's inputs."""
        this_linear = self.net[2 * idx]
        next_linear = self.net[2 * (idx + 1)]
        device = this_linear.weight.device
        keep = [i for i in range(this_linear.out_features) if i not in set(indices_to_remove)]

        new_this = nn.Linear(this_linear.in_features, len(keep)).to(device)
        new_this.weight.data = this_linear.weight.data[keep].clone()
        new_this.bias.data = this_linear.bias.data[keep].clone()

        new_next = nn.Linear(len(keep), next_linear.out_features).to(device)
        new_next.weight.data = next_linear.weight.data[:, keep].clone()
        new_next.bias.data = next_linear.bias.data.clone()

        pruned = copy.deepcopy(self)
        pruned.net[2 * idx] = new_this
        pruned.net[2 * (idx + 1)] = new_next
        return pruned


@register_model("mlp")
def build_mlp(bundle: DatasetBundle, hidden_sizes: list[int] = [64, 64, 64]) -> MLP:
    return MLP(hidden_sizes, input_dim=bundle.input_dim, output_dim=bundle.output_dim)
