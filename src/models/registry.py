"""Model registry + the prunable-model protocol.

A model builder has signature (bundle: DatasetBundle, **params) -> nn.Module
and reads input/output sizes (and any task-specific info, e.g. vocab_size for
the transformer) from the bundle, so configs never repeat what the dataset
already knows.

Every registered model implements `PrunableModel`, which is the only
interface `src.pruning` sees: pruning methods score neurons through
`prunable_layer`/`prunable_bn`, and the structural surgery is the model's own
`prune_layer`. Adding a new architecture means implementing these four
methods in one new file -- nothing in src/pruning changes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from src.data.registry import DatasetBundle

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


@dataclass
class MergeOp:
    """One outgoing-weight transfer: before neuron `removed` is deleted, its
    contribution is folded into `survivor` (a_survivor += scale * a_removed on
    the next layer's input weights). Ops are applied in list order, so chains
    (j merged into i, i later merged into k) compose correctly."""

    removed: int
    survivor: int
    scale: float = 1.0


class PrunableModel(nn.Module, abc.ABC):
    """A model whose hidden layers expose prunable output neurons/filters."""

    @abc.abstractmethod
    def n_prunable_layers(self) -> int:
        """Number of prunable layers/blocks."""

    @abc.abstractmethod
    def prunable_layer(self, idx: int) -> nn.Module:
        """The Linear/Conv2d whose OUTPUT neurons/filters are prunable at `idx`."""

    def prunable_bn(self, idx: int) -> nn.Module | None:
        """BatchNorm paired with the prunable layer (its output is the
        pre-ReLU activation), or None when the layer feeds ReLU directly."""
        return None

    @abc.abstractmethod
    def prune_layer(self, idx: int, indices_to_remove: list[int]) -> "PrunableModel":
        """Return a copy of this model with the given output neurons/filters
        of prunable layer `idx` removed (and downstream consumers fixed)."""

    def outgoing_module(self, idx: int) -> nn.Module:
        """The module that consumes the activations of prunable layer `idx`:
        an nn.Linear whose input columns correspond one-to-one to the
        prunable neurons, or an nn.Conv2d whose input channels do (conv
        models override outgoing_weights/set_outgoing_weights to use the
        unfolded [Cin*kH*kW, Cout] matrix view). Basis for activation capture
        and weight surgery in reconstruction-based methods like OSSCAR."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose its consumer module (reconstruction-"
            "based pruning needs it; implemented by mlp, resmlp, transformer, resnet)"
        )

    def set_outgoing_weights(self, idx: int, new_weights: torch.Tensor) -> "PrunableModel":
        """Return a copy where the consumer's weights are replaced by
        `new_weights` ([H, fan_out], same orientation as outgoing_weights) --
        the surgery step of reconstruction-based methods, which re-solve the
        surviving weights instead of transferring columns. Neurons are NOT
        removed here; call prune_layer afterwards."""
        import copy

        clone = copy.deepcopy(self)
        module = clone.outgoing_module(idx)
        module.weight.data = new_weights.t().to(module.weight.dtype).to(module.weight.device).contiguous()
        return clone

    def add_outgoing_bias(self, idx: int, bias_delta: torch.Tensor) -> "PrunableModel":
        """Return a copy where `bias_delta` ([fan_out]) is added to the
        consumer's bias -- the constant-absorption step of folding-based
        methods (e.g. LEO++), which replace a removed neuron's affine
        contribution with merges plus a constant. Neurons are NOT removed
        here; call prune_layer afterwards."""
        import copy

        clone = copy.deepcopy(self)
        module = clone.outgoing_module(idx)
        module.bias.data += bias_delta.to(module.bias.dtype).to(module.bias.device)
        return clone

    def outgoing_weights(self, idx: int) -> torch.Tensor:
        """[H, fan_out]: for each prunable neuron of layer `idx`, the weights
        through which its activation feeds the next layer. Only meaningful
        where the consumer reads the activation linearly -- conv models with
        BatchNorm between the prunable layer and its consumer don't qualify,
        and leave this unimplemented."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose outgoing weights (merge-based "
            "pruning supports fully-connected-style layers only: mlp, resmlp, transformer)"
        )

    def merge_outgoing(self, idx: int, merges: list[MergeOp]) -> "PrunableModel":
        """Return a copy where, for each MergeOp in order, the removed
        neuron's outgoing weights are added (scaled) onto its survivor's --
        the 'surgery' step of merge-based pruning. Neurons are NOT removed
        here; call prune_layer afterwards."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support outgoing-weight merging (supported "
            "for fully-connected-style layers only: mlp, resmlp, transformer)"
        )


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    def decorator(builder: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"model kind {name!r} already registered")
        MODEL_REGISTRY[name] = builder
        return builder

    return decorator


def build_model(kind: str, bundle: DatasetBundle, **params) -> nn.Module:
    if kind not in MODEL_REGISTRY:
        raise KeyError(f"unknown model kind {kind!r}. available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[kind](bundle, **params)
