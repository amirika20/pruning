"""Gated-MLP decoder LMs (Llama 2/3, Qwen 2/2.5, Mistral, ...) via HuggingFace
transformers -- pretrained ONLY.

Each decoder layer's MLP is

    down_proj( act_fn(gate_proj(x)) * up_proj(x) )        (SwiGLU when act_fn = SiLU)

and the prunable dimension is the hidden channel: channel i has TWO producer
rows (gate_proj[i], up_proj[i]) and ONE consumer column down_proj[:, i], which
down_proj reads linearly. Its response

    phi_i(x) = act_fn(g_i^T x) * (u_i^T x)

is a scalar per token exactly as a plain unit's act(w^T x + b) is, so every
method that only reads responses and consumer columns applies unchanged: the
functional MASH path (RMS gauge, delta_f from sample Grams, medoid dictionary,
sum / empirical / ridge repair, the drop ablation), OSSCAR (Hessian of the
consumer's input) and the deletion baselines. What is refused is what already
was on Pythia: the closed-form Gaussian moments, the merge dictionary, the
cylinder score and its certificate -- all of them are pre-activation
hyperplane geometry, and a gated channel has no single hyperplane.

HOW THE GATE IS DECLARED. The response is homogeneous in the up row (scaling
u_i scales phi_i), so `prunable_layer` is up_proj: its rows carry the RMS gauge
as a plain unit's rows do, and `activation()` returns a GatedActivation
(src.models.registry) that holds gate_proj and act_fn. The functional helpers
form responses through `act.gate(idx)(pre, z)` = act_fn(z g_idx^T) * pre.

Surgery: `prune_layer` removes the channel from gate_proj and up_proj (rows)
and from down_proj (columns); `merge_outgoing` and `set_outgoing_weights` act
on down_proj's columns only. Attention, embeddings, norms are untouched.

Weights are gated on the Hub for Llama (accept the license, `huggingface-cli
login` on the login node); Qwen and Mistral download without a token. There is
no training path: the builder refuses `pretrained: false`.
"""

from __future__ import annotations

import copy
import os

for _k, _v in (("TRANSFORMERS_NO_TF", "1"), ("TRANSFORMERS_NO_FLAX", "1"),
               ("USE_TF", "0"), ("USE_FLAX", "0"),
               ("TF_CPP_MIN_LOG_LEVEL", "3"), ("TF_ENABLE_ONEDNN_OPTS", "0")):
    os.environ.setdefault(_k, _v)

import torch
import torch.nn as nn

from src.data.registry import DatasetBundle
from src.models.registry import GatedActivation, MergeOp, PrunableModel, register_model

# Short names -> Hub repos. Hidden widths (layers x intermediate): Llama-3.2-1B
# 16x8192, Llama-3.2-3B 28x8192, Llama-3.1-8B 32x14336, Qwen2.5-3B 36x11008,
# Qwen2.5-7B 28x18944, Mistral-7B 32x14336. bf16 is the training dtype of all
# of them; fp32 doubles the footprint (8B: 32 GB).
GATED_LM_REPOS = {
    "llama3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b": "mistralai/Mistral-7B-v0.3",
}


def gated_lm_repo(name: str) -> str:
    """A short name from GATED_LM_REPOS, or a Hub repo id passed through."""
    return GATED_LM_REPOS.get(name, name)


class GatedLM(PrunableModel):
    """PrunableModel adapter around any transformers causal LM whose decoder
    layers expose `.mlp.gate_proj / up_proj / down_proj / act_fn`."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        layers = self._layers()
        mlp = layers[0].mlp
        for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
            if not hasattr(mlp, name):
                raise TypeError(f"{type(net).__name__}: layer MLP has no `{name}`; "
                                "GatedLM needs a gate/up/down block")

    # ── module access ─────────────────────────────────────────────────────

    def _layers(self):
        # LlamaForCausalLM / Qwen2ForCausalLM / MistralForCausalLM all keep the
        # decoder at .model.layers
        return self.net.model.layers

    def _mlp(self, idx: int):
        return self._layers()[idx].mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(input_ids=x).logits  # [B, T, vocab]

    def make_dummy_input(self, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.net.config.vocab_size, (1, 8), device=device)

    # ── prunable protocol ─────────────────────────────────────────────────

    def n_prunable_layers(self) -> int:
        return len(self._layers())

    def prunable_layer(self, idx: int) -> nn.Module:
        """up_proj: the row the response is homogeneous in (module docstring)."""
        return self._mlp(idx).up_proj

    def outgoing_module(self, idx: int) -> nn.Module:
        return self._mlp(idx).down_proj

    def outgoing_weights(self, idx: int) -> torch.Tensor:
        return self._mlp(idx).down_proj.weight.detach().t()  # [ffn, d_model]

    def activation(self, idx: int) -> GatedActivation:
        mlp = self._mlp(idx)
        return GatedActivation(mlp.gate_proj.weight, mlp.gate_proj.bias, mlp.act_fn)

    def merge_outgoing(self, idx: int, merges: list[MergeOp]) -> "GatedLM":
        """Fold each removed channel's down_proj column into its survivor's."""
        merged = copy.deepcopy(self)
        W = merged._mlp(idx).down_proj.weight.data   # [d_model, ffn]
        for op in merges:
            W[:, op.survivor] += op.scale * W[:, op.removed]
        return merged

    def prune_layer(self, idx: int, indices_to_remove: list[int]) -> "GatedLM":
        """Remove hidden channels of layer `idx`: gate_proj and up_proj lose
        output rows, down_proj loses input columns."""
        pruned = copy.deepcopy(self)
        mlp = pruned._mlp(idx)
        H = mlp.up_proj.out_features
        keep = sorted(set(range(H)) - set(int(i) for i in indices_to_remove))

        def rows(lin: nn.Linear) -> nn.Linear:
            new = nn.Linear(lin.in_features, len(keep), bias=lin.bias is not None)
            new = new.to(lin.weight.device).to(lin.weight.dtype)
            new.weight.data = lin.weight.data[keep].clone()
            if lin.bias is not None:
                new.bias.data = lin.bias.data[keep].clone()
            return new

        down = mlp.down_proj
        new_down = nn.Linear(len(keep), down.out_features, bias=down.bias is not None)
        new_down = new_down.to(down.weight.device).to(down.weight.dtype)
        new_down.weight.data = down.weight.data[:, keep].clone()
        if down.bias is not None:
            new_down.bias.data = down.bias.data.clone()

        mlp.gate_proj, mlp.up_proj, mlp.down_proj = rows(mlp.gate_proj), rows(mlp.up_proj), new_down
        if hasattr(mlp, "intermediate_size"):
            mlp.intermediate_size = len(keep)
        return pruned


_DTYPES = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16}


@register_model("gated_lm")
def build_gated_lm(bundle: DatasetBundle, repo: str = "qwen2.5-0.5b", pretrained: bool = True,
                   dtype: str = "bfloat16") -> GatedLM:
    """A gated-MLP causal LM (GATED_LM_REPOS or any Hub id), pretrained only;
    weights cache under $HF_HOME. Tokenize the data with the SAME repo's
    tokenizer (data.params.tokenizer)."""
    if not pretrained:
        raise ValueError("gated_lm is pretrained-only: this repo has no causal-LM "
                         "training loop. Use pretrained: true with training: {epochs: 0}.")
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
    from transformers import AutoModelForCausalLM

    net = AutoModelForCausalLM.from_pretrained(gated_lm_repo(repo), dtype=_DTYPES[dtype])
    if bundle.output_dim > net.config.vocab_size:
        raise ValueError(
            f"dataset vocab ({bundle.output_dim}) exceeds the model's vocab "
            f"({net.config.vocab_size}); tokenize with the matching tokenizer "
            "(data.params.tokenizer)")
    net.eval()
    return GatedLM(net)


def tiny_gated_lm(vocab: int = 512, d: int = 64, ffn: int = 96, layers: int = 2,
                  seed: int = 0) -> GatedLM:
    """A random Llama-architecture LM for tests -- no network, no weights."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=d, intermediate_size=ffn,
                      num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=128)
    net = LlamaForCausalLM(cfg).eval()
    return GatedLM(net)
