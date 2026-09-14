"""Pythia (Biderman et al., 2023; GPT-NeoX architecture) via HuggingFace
transformers -- pretrained ONLY.

Each decoder layer's MLP is
    dense_h_to_4h (d_model -> 4*d_model) -> GELU -> dense_4h_to_h (4*d_model -> d_model)
and the prunable dimension is dense_h_to_4h's output neurons, consumed
linearly by the nn.Linear dense_4h_to_h -- the same single-producer /
single-consumer FFN as OPT (src/models/opt.py), so every column-transfer and
delete-and-repair method applies unchanged. Attention, embeddings, LayerNorms
and the parallel residual are untouched.

THE ACTIVATION IS GELU, NOT ReLU. This adapter therefore overrides
`activation()` to hand back the layer's own act module, which switches the
MASH stack into its activation-aware ("functional") path: units are scored
and repaired on their TRUE post-GELU responses over the calibration tokens,
and everything that needs ReLU specifically -- the closed-form Gaussian
moments, the hyperplane merge dictionary, the cylinder score and its
sup-norm certificate -- is refused with a message rather than silently
computed as if the network were rectified. Supported on Pythia: score
delta_f, measure empirical, dictionary medoid (or drop), repair none / sum /
empirical; plus the deletion baselines with the empirical repair.

There is no scratch-training path: the builder refuses `pretrained: false`.
Configure `training: {epochs: 0}` and `finetune: {epochs: 0}`; evaluation
reports token-level cross-entropy, perplexity = exp(loss).
"""

from __future__ import annotations

import copy
import os

# See src/models/opt.py: switch the TF/Flax backends off before transformers is
# imported, or the import costs seconds and a page of logging.
for _k, _v in (("TRANSFORMERS_NO_TF", "1"), ("TRANSFORMERS_NO_FLAX", "1"),
               ("USE_TF", "0"), ("USE_FLAX", "0"),
               ("TF_CPP_MIN_LOG_LEVEL", "3"), ("TF_ENABLE_ONEDNN_OPTS", "0")):
    os.environ.setdefault(_k, _v)

import torch
import torch.nn as nn

from src.data.registry import DatasetBundle
from src.models.registry import MergeOp, PrunableModel, register_model

# The Pythia scaling suite. `deduped: true` selects the -deduped twins, trained
# on the deduplicated Pile; same architecture and tokenizer. Weights download
# to $HF_HOME; fp32 footprint ~4 bytes/param (6.9b ~28GB, 12b ~48GB) -- the big
# ones load in float16, the dtype they were trained in.
PYTHIA_SIZES = {
    "70m": "EleutherAI/pythia-70m",
    "160m": "EleutherAI/pythia-160m",
    "410m": "EleutherAI/pythia-410m",
    "1b": "EleutherAI/pythia-1b",
    "1.4b": "EleutherAI/pythia-1.4b",
    "2.8b": "EleutherAI/pythia-2.8b",
    "6.9b": "EleutherAI/pythia-6.9b",
    "12b": "EleutherAI/pythia-12b",
}


def pythia_repo(size: str, deduped: bool = False) -> str:
    if size not in PYTHIA_SIZES:
        raise ValueError(f"pythia size must be one of {sorted(PYTHIA_SIZES)}, got {size!r}")
    return PYTHIA_SIZES[size] + ("-deduped" if deduped else "")


class Pythia(PrunableModel):
    """PrunableModel adapter around transformers.GPTNeoXForCausalLM."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(input_ids=x).logits  # [B, T, vocab]

    def make_dummy_input(self, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.net.config.vocab_size, (1, 8), device=device)

    # ── prunable protocol ─────────────────────────────────────────────────

    def _layer(self, idx: int):
        return self.net.gpt_neox.layers[idx]

    def n_prunable_layers(self) -> int:
        return len(self.net.gpt_neox.layers)

    def prunable_layer(self, idx: int) -> nn.Module:
        return self._layer(idx).mlp.dense_h_to_4h

    def outgoing_module(self, idx: int) -> nn.Module:
        return self._layer(idx).mlp.dense_4h_to_h

    def outgoing_weights(self, idx: int) -> torch.Tensor:
        return self._layer(idx).mlp.dense_4h_to_h.weight.detach().t()  # [ffn, d_model]

    def activation(self, idx: int):
        """The MLP's own act module (nn.GELU for every released Pythia), so the
        functional path evaluates exactly the nonlinearity the network runs."""
        return self._layer(idx).mlp.act

    def merge_outgoing(self, idx: int, merges: list[MergeOp]) -> "Pythia":
        """Fold each removed FFN neuron's outgoing column into its survivor:
        dense_4h_to_h reads the hidden dimension linearly, so this is one
        column transfer, as in any fully-connected layer."""
        merged = copy.deepcopy(self)
        W = merged._layer(idx).mlp.dense_4h_to_h.weight.data   # [d_model, ffn]
        for op in merges:
            W[:, op.survivor] += op.scale * W[:, op.removed]
        return merged

    def prune_layer(self, idx: int, indices_to_remove: list[int]) -> "Pythia":
        """Remove FFN hidden neurons of layer `idx`: dense_h_to_4h loses output
        rows, dense_4h_to_h loses input columns. d_model and attention are
        unchanged. Biases are carried when present (they are, in GPT-NeoX)."""
        pruned = copy.deepcopy(self)
        mlp = pruned._layer(idx).mlp
        fc1, fc2 = mlp.dense_h_to_4h, mlp.dense_4h_to_h
        device, dtype = fc1.weight.device, fc1.weight.dtype
        keep = sorted(set(range(fc1.out_features)) - set(indices_to_remove))

        new_fc1 = nn.Linear(fc1.in_features, len(keep),
                            bias=fc1.bias is not None).to(device).to(dtype)
        new_fc1.weight.data = fc1.weight.data[keep].clone()
        if fc1.bias is not None:
            new_fc1.bias.data = fc1.bias.data[keep].clone()

        new_fc2 = nn.Linear(len(keep), fc2.out_features,
                            bias=fc2.bias is not None).to(device).to(dtype)
        new_fc2.weight.data = fc2.weight.data[:, keep].clone()
        if fc2.bias is not None:
            new_fc2.bias.data = fc2.bias.data.clone()

        mlp.dense_h_to_4h, mlp.dense_4h_to_h = new_fc1, new_fc2
        return pruned


_DTYPES = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16}


@register_model("pythia")
def build_pythia(bundle: DatasetBundle, size: str = "160m", pretrained: bool = True,
                 dtype: str = "float32", deduped: bool = False) -> Pythia:
    """Pythia, pretrained only (see PYTHIA_SIZES); weights cache under $HF_HOME
    on first use. Tokenize the data with the SAME repo's tokenizer (every Pythia
    size shares one, so `EleutherAI/pythia-70m` serves all of them)."""
    if not pretrained:
        raise ValueError(
            "pythia is pretrained-only: this repo has no causal-LM training loop. "
            "Use pretrained: true with training: {epochs: 0}."
        )
    repo = pythia_repo(size, deduped)
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
    from transformers import GPTNeoXForCausalLM

    # Pythia was trained in fp16; float32 is an upcast. The same dtype caveat
    # as OPT applies (src/models/opt.py) -- except that the merge dictionary is
    # refused on GELU, so no synthesized hyperplane is ever rounded here.
    net = GPTNeoXForCausalLM.from_pretrained(repo, dtype=_DTYPES[dtype])
    # The embedding matrix is padded past the tokenizer's vocabulary (50304 rows
    # against 50277 tokens), so ids must merely fit.
    if bundle.output_dim > net.config.vocab_size:
        raise ValueError(
            f"dataset vocab ({bundle.output_dim}) exceeds Pythia vocab "
            f"({net.config.vocab_size}); tokenize with the matching tokenizer "
            "(data.params.tokenizer)"
        )
    net.eval()
    return Pythia(net)
