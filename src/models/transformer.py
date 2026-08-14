from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.data.registry import DatasetBundle
from src.models.registry import MergeOp, PrunableModel, register_model


class TransformerFFN(nn.Module):
    """Two-layer feed-forward network inside a transformer block (fc1 -> ReLU -> fc2)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = TransformerFFN(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class ModularTransformer(PrunableModel):
    """Light transformer for modular arithmetic.

    Input : [B, seq_len] int64 tokens
    Output: [B, n_classes] logits classified from the last token position.
    Vocabulary: {0..p-1} + {op=p, eq=p+1} -> vocab_size = p+2.
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, n_classes: int, seq_len: int):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            h = layer(h)
        return self.head(h[:, -1, :])  # classify from last ("=") token

    def make_dummy_input(self, device: torch.device) -> torch.Tensor:
        seq_len = self.pos_embed.num_embeddings
        return torch.zeros(1, seq_len, dtype=torch.long, device=device)

    # ── prunable protocol ─────────────────────────────────────────────────
    # The prunable dimension is each layer's FFN intermediate width (fc1's
    # output, fc2's input); attention and d_model are untouched.

    def n_prunable_layers(self) -> int:
        return len(self.layers)

    def prunable_layer(self, idx: int) -> nn.Module:
        return self.layers[idx].ffn.fc1

    def outgoing_module(self, idx: int) -> nn.Module:
        return self.layers[idx].ffn.fc2

    def outgoing_weights(self, idx: int) -> torch.Tensor:
        return self.layers[idx].ffn.fc2.weight.data.t()  # [H, d_model]

    def merge_outgoing(self, idx: int, merges: list[MergeOp]) -> "ModularTransformer":
        merged = copy.deepcopy(self)
        W = merged.layers[idx].ffn.fc2.weight.data  # [d_model, H]
        for op in merges:
            W[:, op.survivor] += op.scale * W[:, op.removed]
        return merged

    def prune_layer(self, idx: int, indices_to_remove: list[int]) -> "ModularTransformer":
        """Remove hidden neurons from the FFN of transformer layer `idx`."""
        ffn = self.layers[idx].ffn
        fc1, fc2 = ffn.fc1, ffn.fc2
        device = fc1.weight.device
        keep = [i for i in range(fc1.out_features) if i not in set(indices_to_remove)]

        new_fc1 = nn.Linear(fc1.in_features, len(keep)).to(device)
        new_fc1.weight.data = fc1.weight.data[keep].clone()
        new_fc1.bias.data = fc1.bias.data[keep].clone()

        new_fc2 = nn.Linear(len(keep), fc2.out_features).to(device)
        new_fc2.weight.data = fc2.weight.data[:, keep].clone()
        new_fc2.bias.data = fc2.bias.data.clone()

        pruned = copy.deepcopy(self)
        pruned.layers[idx].ffn.fc1 = new_fc1
        pruned.layers[idx].ffn.fc2 = new_fc2
        return pruned


@register_model("transformer")
def build_transformer(
    bundle: DatasetBundle,
    d_model: int = 128,
    n_heads: int = 4,
    d_ff: int = 256,
    n_layers: int = 2,
) -> ModularTransformer:
    return ModularTransformer(
        vocab_size=bundle.extra["vocab_size"],
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        n_classes=bundle.output_dim,
        seq_len=bundle.extra["seq_len"],
    )
