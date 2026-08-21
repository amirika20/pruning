"""OPT (Zhang et al., 2022) via HuggingFace transformers -- pretrained ONLY.

The OSSCAR paper's language setting: each decoder layer's FFN is
fc1 (d_model -> 4*d_model) -> ReLU -> fc2 (4*d_model -> d_model), and the
prunable dimension is fc1's output neurons, consumed by the nn.Linear fc2 --
the exact FC structure the official OSSCAR code prunes, so all FC-path
methods apply unchanged. Attention, embeddings, and LayerNorms are untouched.

There is no scratch-training path: the builder refuses `pretrained: false`,
and the trainer raises on `task == "causal_lm"` with epochs > 0 -- configure
`training: {epochs: 0}` and `finetune: {epochs: 0}` (the one-shot protocol).
Evaluation reports token-level cross-entropy; perplexity = exp(loss), logged
by the runner as val_ppl/test_ppl.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.data.registry import DatasetBundle
from src.models.registry import PrunableModel, register_model

OPT_SIZES = {"125m": "facebook/opt-125m", "350m": "facebook/opt-350m", "1.3b": "facebook/opt-1.3b"}


class OPT(PrunableModel):
    """PrunableModel adapter around transformers.OPTForCausalLM."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(input_ids=x).logits  # [B, T, vocab]

    def make_dummy_input(self, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.net.config.vocab_size, (1, 8), device=device)

    # ── prunable protocol ─────────────────────────────────────────────────

    def _layer(self, idx: int):
        return self.net.model.decoder.layers[idx]

    def n_prunable_layers(self) -> int:
        return len(self.net.model.decoder.layers)

    def prunable_layer(self, idx: int) -> nn.Module:
        return self._layer(idx).fc1

    def outgoing_module(self, idx: int) -> nn.Module:
        return self._layer(idx).fc2

    def outgoing_weights(self, idx: int) -> torch.Tensor:
        return self._layer(idx).fc2.weight.detach().t()  # [ffn_dim, d_model]

    def prune_layer(self, idx: int, indices_to_remove: list[int]) -> "OPT":
        """Remove FFN hidden neurons of decoder layer `idx`: fc1 loses output
        neurons, fc2 loses input columns. d_model and attention unchanged."""
        pruned = copy.deepcopy(self)
        layer = pruned._layer(idx)
        fc1, fc2 = layer.fc1, layer.fc2
        device = fc1.weight.device
        keep = sorted(set(range(fc1.out_features)) - set(indices_to_remove))

        new_fc1 = nn.Linear(fc1.in_features, len(keep)).to(device).to(fc1.weight.dtype)
        new_fc1.weight.data = fc1.weight.data[keep].clone()
        new_fc1.bias.data = fc1.bias.data[keep].clone()

        new_fc2 = nn.Linear(len(keep), fc2.out_features).to(device).to(fc2.weight.dtype)
        new_fc2.weight.data = fc2.weight.data[:, keep].clone()
        new_fc2.bias.data = fc2.bias.data.clone()

        layer.fc1, layer.fc2 = new_fc1, new_fc2
        return pruned


@register_model("opt")
def build_opt(bundle: DatasetBundle, size: str = "125m", pretrained: bool = True) -> OPT:
    """OPT-{125m, 350m, 1.3b}, pretrained only (~250MB/650MB/2.6GB download,
    cached under ~/.cache/huggingface)."""
    if not pretrained:
        raise ValueError(
            "opt is pretrained-only: this repo has no causal-LM training loop. "
            "Use pretrained: true with training: {epochs: 0}."
        )
    if size not in OPT_SIZES:
        raise ValueError(f"opt size must be one of {sorted(OPT_SIZES)}, got {size!r}")
    from transformers import OPTForCausalLM

    net = OPTForCausalLM.from_pretrained(OPT_SIZES[size], dtype=torch.float32)
    # OPT's embedding matrix is padded past the tokenizer's true vocab
    # (50265 tokens vs 50272 rows for 125m), so ids must merely fit.
    if bundle.output_dim > net.config.vocab_size:
        raise ValueError(
            f"dataset vocab ({bundle.output_dim}) exceeds OPT vocab ({net.config.vocab_size}); "
            "tokenize with the matching tokenizer (data.params.tokenizer)"
        )
    net.eval()
    return OPT(net)
