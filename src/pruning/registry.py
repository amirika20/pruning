"""Pruning-method registry: maps a config's `pruning.methods[].kind` string
to a PruningMethod class.

A pruning method only *selects* which neurons/filters to remove -- the
structural surgery is the model's own `prune_layer` (see
src.models.registry.PrunableModel), so methods work on every registered
architecture automatically.

To add a new method:
  1. create src/pruning/methods/<name>.py with a PruningMethod subclass
     decorated with @register_pruning_method("<name>")
  2. import it in src/pruning/methods/__init__.py
  3. reference it from YAML:  pruning: {methods: [{kind: <name>, params: {...}}]}
Nothing in the runner or the models changes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import torch

from src.data.registry import DatasetBundle
from src.models.registry import MergeOp, PrunableModel

PRUNING_METHOD_REGISTRY: dict[str, type["PruningMethod"]] = {}


@dataclass
class PruneDecision:
    """A selection that also carries surgery beyond plain removal.

    `remove` are the indices to delete. `merges` (optional, applied in list
    order BEFORE removal via PrunableModel.merge_outgoing) fold each removed
    neuron's outgoing weights into a survivor's, so near-duplicate neurons
    can be eliminated with (almost) no change to the network's function.
    `new_outgoing` (optional, [H, fan_out], applied BEFORE removal via
    PrunableModel.set_outgoing_weights) replaces the consumer's weights
    outright -- for reconstruction-based methods (e.g. OSSCAR) that re-solve
    the surviving weights rather than transfer columns; rows of removed
    neurons should be zero. `bias_delta` (optional, [fan_out], applied after
    merges via PrunableModel.add_outgoing_bias) is added to the consumer's
    bias -- for folding-based methods (e.g. LEO++) that absorb a removed
    neuron's constant contribution into the next layer.
    `diagnostics` (optional) is free-form per-unit bookkeeping for analysis --
    it never affects surgery. Convention: every key maps to a sequence of
    length H (the layer's PRE-prune width), one entry per original unit, except
    the reserved key "_scalars" whose value is a dict of layer-level numbers.
    This is what lets an ablation say which units were removed, which were
    absorbed into which survivor, and at what cost -- see
    src.analysis.pruning_detail.
    `new_incoming` (optional, (weight, bias) for the PRUNABLE layer itself,
    applied BEFORE removal) rewrites surviving units' own hyperplanes -- for
    methods (e.g. HOPE) whose merge synthesizes a new parent unit that no
    original member realizes, rather than only transferring outgoing columns.
    Rows belonging to removed units are ignored. Methods that only
    drop neurons keep returning a plain list[int].
    """

    remove: list[int]
    merges: list[MergeOp] = field(default_factory=list)
    new_outgoing: "torch.Tensor | None" = None
    bias_delta: "torch.Tensor | None" = None
    new_incoming: "tuple[torch.Tensor, torch.Tensor] | None" = None
    diagnostics: "dict[str, Any] | None" = None


@dataclass
class PruneContext:
    """Everything a selection criterion may need beyond the model itself.

    already_selected holds the indices chosen by earlier methods in the
    config's list for the layer currently being pruned -- methods must not
    re-select them (the runner also filters as a safety net).
    """

    train_inputs: torch.Tensor  # full training inputs, on the model's device
    bundle: DatasetBundle
    device: torch.device
    already_selected: set[int] = field(default_factory=set)


class PruningMethod(abc.ABC):
    """A selection criterion: which output neurons/filters of one prunable
    layer should be removed. Constructor kwargs come straight from the YAML
    entry's `params`."""

    @abc.abstractmethod
    def select(self, model: PrunableModel, layer_idx: int, ctx: PruneContext) -> "list[int] | PruneDecision":
        """Indices to remove from prunable layer `layer_idx` of `model` --
        either a plain list, or a PruneDecision when the method also wants
        outgoing-weight merges applied before removal."""


def register_pruning_method(name: str):
    def decorator(cls: type[PruningMethod]) -> type[PruningMethod]:
        if name in PRUNING_METHOD_REGISTRY:
            raise ValueError(f"pruning method {name!r} already registered")
        PRUNING_METHOD_REGISTRY[name] = cls
        return cls

    return decorator


def build_pruning_method(kind: str, **params: Any) -> PruningMethod:
    if kind not in PRUNING_METHOD_REGISTRY:
        raise KeyError(f"unknown pruning method {kind!r}. available: {list(PRUNING_METHOD_REGISTRY)}")
    return PRUNING_METHOD_REGISTRY[kind](**params)
