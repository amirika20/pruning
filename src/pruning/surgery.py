"""Apply a config's pruning methods to every prunable layer of a model.

The per-architecture surgery lives on the models themselves (see
src.models.registry.PrunableModel.prune_layer); this module owns the
method-agnostic loop: run each configured selection method in order per
layer (later methods never re-select earlier picks), remove the union, and
record per-layer/per-method counts.
"""

from __future__ import annotations

import logging

import torch

from src.config import PruningConfig
from src.data.registry import DatasetBundle
from src.models.registry import PrunableModel
from src.pruning.geometry import layer_width
from src.pruning.registry import PruneContext, PruneDecision, build_pruning_method


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

    current = model
    report: list[dict] = []

    for layer_idx in range(model.n_prunable_layers()):
        n_before = layer_width(current, layer_idx)
        to_remove: set[int] = set()
        per_method: dict[str, int] = {}

        for kind, method in methods:
            ctx = PruneContext(
                train_inputs=train_inputs,
                bundle=bundle,
                device=device,
                already_selected=set(to_remove),
            )
            decision = method.select(current, layer_idx, ctx)
            if isinstance(decision, PruneDecision):
                selected = [i for i in decision.remove if i not in to_remove]
                selected_set = set(selected)
                # Merge surgery happens immediately (on the still-full-width
                # layer), so later methods score the merged weights; the
                # actual removal is deferred to the single prune_layer call
                # below. Ops touching neurons another method already claimed
                # are dropped -- their transfer target/source is gone.
                ops = [
                    op for op in decision.merges
                    if op.removed in selected_set and op.survivor not in to_remove
                ]
                if len(ops) < len(decision.merges):
                    logging.warning(
                        f"  Layer {layer_idx}: dropped {len(decision.merges) - len(ops)} merge op(s) "
                        "that overlapped a previous method's selection"
                    )
                if ops:
                    current = current.merge_outgoing(layer_idx, ops)
                # Constant absorption (e.g. LEO++): fold the removed neurons'
                # constant contribution into the consumer's bias. Applied
                # after merges -- deltas are computed from the removed
                # neurons' own outgoing columns, which merges never touch.
                if decision.bias_delta is not None:
                    current = current.add_outgoing_bias(layer_idx, decision.bias_delta)
                # Reconstruction surgery (e.g. OSSCAR): replace the consumer's
                # weights with the method's re-solved matrix before removal.
                if decision.new_outgoing is not None:
                    current = current.set_outgoing_weights(layer_idx, decision.new_outgoing)
            else:
                selected = [i for i in decision if i not in to_remove]
            per_method[kind] = len(selected)
            to_remove.update(selected)

        report.append({
            "layer": layer_idx,
            "neurons_before": n_before,
            "removed_per_method": per_method,
            "total_removed": len(to_remove),
            "neurons_after": n_before - len(to_remove),
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
