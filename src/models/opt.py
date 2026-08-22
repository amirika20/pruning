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
import os

# transformers probes for TensorFlow and Flax backends when it resolves a
# checkpoint, and importing TensorFlow costs seconds of startup, a few hundred MB,
# and a page of absl/oneDNN logging that buries the real output. This project is
# torch-only, so switch the other backends off BEFORE transformers is imported --
# these must be set before the first import to have any effect. TF_CPP_* silences
# TensorFlow's C++ logger in case something else pulls it in anyway.
for _k, _v in (("TRANSFORMERS_NO_TF", "1"), ("TRANSFORMERS_NO_FLAX", "1"),
               ("USE_TF", "0"), ("USE_FLAX", "0"),
               ("TF_CPP_MIN_LOG_LEVEL", "3"), ("TF_ENABLE_ONEDNN_OPTS", "0")):
    os.environ.setdefault(_k, _v)

import torch
import torch.nn as nn

from src.data.registry import DatasetBundle
from src.models.registry import MergeOp, PrunableModel, register_model

# Sizes the paper's OPT table sweeps. Weights download to ~/.cache/huggingface;
# float32 footprint is ~4 bytes/param, so 6.7b needs ~27GB and 13b ~52GB of
# device memory before activations -- size the SLURM request accordingly.
OPT_SIZES = {
    "125m": "facebook/opt-125m",
    "350m": "facebook/opt-350m",
    "1.3b": "facebook/opt-1.3b",
    "2.7b": "facebook/opt-2.7b",
    "6.7b": "facebook/opt-6.7b",
    "13b": "facebook/opt-13b",
}


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

    def merge_outgoing(self, idx: int, merges: list[MergeOp]) -> "OPT":
        """Fold each removed FFN neuron's outgoing weights into its survivor.

        fc2 reads the FFN hidden dimension linearly, so a neuron's outgoing
        weights are one column of fc2.weight -- the same transfer as any
        fully-connected layer. Needed by the merge-with-transfer methods
        (data_free_merge, leo_pp); without it they refuse the model.
        """
        merged = copy.deepcopy(self)
        W = merged._layer(idx).fc2.weight.data          # [d_model, ffn_dim]
        for op in merges:
            W[:, op.survivor] += op.scale * W[:, op.removed]
        return merged

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


_DTYPES = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16}


@register_model("opt")
def build_opt(bundle: DatasetBundle, size: str = "125m", pretrained: bool = True,
              dtype: str = "float32") -> OPT:
    """OPT, pretrained only (see OPT_SIZES); weights cache under
    ~/.cache/huggingface on first use."""
    if not pretrained:
        raise ValueError(
            "opt is pretrained-only: this repo has no causal-LM training loop. "
            "Use pretrained: true with training: {epochs: 0}."
        )
    if size not in OPT_SIZES:
        raise ValueError(f"opt size must be one of {sorted(OPT_SIZES)}, got {size!r}")
    from transformers import OPTForCausalLM

    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
    # The released OPT checkpoints are fp16, so float32 is an upcast that doubles
    # resident memory for no extra information -- 13b is 48.4 GiB in fp32 against
    # 24.2 in fp16. float32 stays the default because it is what the rest of this
    # repo uses and the small sizes cost nothing; the big ones set dtype
    # explicitly (see configs/benchmark/suite.yaml).
    #
    # CAVEAT for merge-based pruning at reduced precision: extract_units upcasts
    # to float64 so the pruning arithmetic is unaffected, but the realized
    # weights are cast back to the layer dtype, so a synthesized hyperplane is
    # quantized to fp16's 10-bit mantissa. Deletion-only arms (saturated,
    # magnitude, random, medoid dictionaries) are unaffected; merged units carry
    # that rounding, and the dense baseline shifts too, so the dtype belongs in
    # the reported protocol rather than being silently varied across rows.
    net = OPTForCausalLM.from_pretrained(OPT_SIZES[size], dtype=_DTYPES[dtype])
    # OPT's embedding matrix is padded past the tokenizer's true vocab
    # (50265 tokens vs 50272 rows for 125m), so ids must merely fit.
    if bundle.output_dim > net.config.vocab_size:
        raise ValueError(
            f"dataset vocab ({bundle.output_dim}) exceeds OPT vocab ({net.config.vocab_size}); "
            "tokenize with the matching tokenizer (data.params.tokenizer)"
        )
    net.eval()
    return OPT(net)
