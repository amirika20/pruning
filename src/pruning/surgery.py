"""Apply a config's pruning methods to every prunable layer of a model.

The per-architecture surgery lives on the models themselves (see
src.models.registry.PrunableModel.prune_layer); this module owns the
method-agnostic loop: run each configured selection method in order per
layer (later methods never re-select earlier picks), remove the union, and
record per-layer/per-method counts.
"""

from __future__ import annotations

import copy
import logging

import torch

from src.config import PruningConfig
from src.data.registry import DatasetBundle
from src.models.registry import PrunableModel
from src.pruning.geometry import layer_width
from src.pruning.registry import PruneContext, PruneDecision, build_pruning_method


def apply_decision(
    model: PrunableModel,
    layer_idx: int,
    decision: "list[int] | PruneDecision",
    already_removed: "set[int] | frozenset[int]" = frozenset(),
) -> "tuple[PrunableModel, list[int]]":
    """Apply one selection's surgery to `layer_idx` and return (model, selected).

    Neurons are NOT removed here -- that is a single prune_layer call once every
    method has had its say. What happens here is the weight surgery a method asks
    for, and the ORDER matters:

      merges        first, on the still-full-width layer, so later methods score
                    the merged weights
      new_incoming  the method synthesized hyperplanes for surviving units, so
                    rewrite the layer's own rows/bias (in place, on this model)
      bias_delta    after merges, since the deltas come from the removed units'
                    own outgoing columns, which merges never touch
      new_outgoing  last: a re-solved consumer matrix supersedes any transfer

    Factored out of prune_model because the width-sweep harness has to apply
    decisions without re-running selection, and a second copy of this ordering
    is how subtle surgery bugs get in.
    """
    selected: list[int]
    if not isinstance(decision, PruneDecision):
        return model, [i for i in decision if i not in already_removed]

    selected = [i for i in decision.remove if i not in already_removed]
    selected_set = set(selected)
    # Ops touching neurons another method already claimed are dropped: their
    # transfer target or source is gone.
    ops = [op for op in decision.merges
           if op.removed in selected_set and op.survivor not in already_removed]
    if len(ops) < len(decision.merges):
        logging.warning(
            f"  Layer {layer_idx}: dropped {len(decision.merges) - len(ops)} "
            "merge op(s) that overlapped a previous method's selection")
    if ops:
        model = model.merge_outgoing(layer_idx, ops)
    if decision.new_incoming is not None:
        w_new, b_new = decision.new_incoming
        lin = model.prunable_layer(layer_idx)
        lin.weight.data.copy_(
            w_new.to(lin.weight.dtype).to(lin.weight.device).view_as(lin.weight))
        if lin.bias is not None:
            lin.bias.data.copy_(
                b_new.to(lin.bias.dtype).to(lin.bias.device).view_as(lin.bias))
    if decision.bias_delta is not None:
        model = model.add_outgoing_bias(layer_idx, decision.bias_delta)
    if decision.new_outgoing is not None:
        model = model.set_outgoing_weights(layer_idx, decision.new_outgoing)
    return model, selected


def prune_model(
    model: PrunableModel,
    pruning: PruningConfig,
    bundle: DatasetBundle,
    device: torch.device,
) -> tuple[PrunableModel, list[dict]]:
    """Prune all prunable layers of `model` sequentially.

    Returns (pruned_model, per_layer_report) where each report entry holds
    the layer index, widths before/after, and per-method removal counts.
    """
    methods = [(m.kind, build_pruning_method(m.kind, **m.params)) for m in pruning.methods]
    train_inputs = bundle.train_ds.tensors[0].to(device)

    # A deep copy up front: the new_incoming path writes rows into the prunable
    # layer IN PLACE, so without this the caller's model is silently mutated on
    # the first layer -- which corrupts every later comparison against it (and
    # every other method run from the same checkpoint).
    current = copy.deepcopy(model)
    report: list[dict] = []

    for layer_idx in range(model.n_prunable_layers()):
        n_before = layer_width(current, layer_idx)
        to_remove: set[int] = set()
        per_method: dict[str, int] = {}
        removed_by: dict[str, list[int]] = {}
        merge_ops: dict[str, list[dict]] = {}
        diagnostics: dict[str, dict] = {}

        for kind, method in methods:
            ctx = PruneContext(
                train_inputs=train_inputs,
                bundle=bundle,
                device=device,
                already_selected=set(to_remove),
            )
            decision = method.select(current, layer_idx, ctx)
            current, selected = apply_decision(current, layer_idx, decision,
                                               already_removed=to_remove)
            per_method[kind] = len(selected)
            removed_by[kind] = sorted(int(i) for i in selected)
            if isinstance(decision, PruneDecision):
                if decision.merges:
                    merge_ops[kind] = [
                        {"removed": int(op.removed), "survivor": int(op.survivor),
                         "scale": float(op.scale)} for op in ops
                    ]
                if decision.diagnostics:
                    diagnostics[kind] = decision.diagnostics
            to_remove.update(selected)

        report.append({
            "layer": layer_idx,
            "neurons_before": n_before,
            "removed_per_method": per_method,
            "total_removed": len(to_remove),
            "neurons_after": n_before - len(to_remove),
            # Analysis payload: WHICH units went, per method, plus whatever
            # per-unit bookkeeping the method chose to expose.
            "removed_indices": sorted(int(i) for i in to_remove),
            "removed_indices_per_method": removed_by,
            "merge_ops": merge_ops,
            "diagnostics": diagnostics,
        })
        method_str = "  ".join(f"{k}={n}" for k, n in per_method.items())
        logging.info(
            f"  Layer {layer_idx}: {method_str}"
            f"  | removed={len(to_remove)}  remaining={n_before - len(to_remove)}"
        )
        if to_remove:
            current = current.prune_layer(layer_idx, list(to_remove))

    logging.info(f"  Total removed: {sum(r['total_removed'] for r in report)}")
    return current, report
