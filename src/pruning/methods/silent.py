"""Silent-neuron pruning: remove units that never activate on training data."""

from __future__ import annotations

import torch

from src.models.registry import PrunableModel
from src.pruning.registry import PruneContext, PruningMethod, register_pruning_method


@register_pruning_method("silent")
class SilentPruning(PruningMethod):
    """Pass all training inputs through the model and record which neurons/
    filters ever produce a positive pre-ReLU activation.

    For conv models: hooks the paired BatchNorm output (post-BN, pre-ReLU)
    and checks whether any spatial position, in any sample, is positive.
    For linear models: hooks the Linear output [N, H] (or [N, T, H] for
    token sequences).

    Selects indices that are always silent (never activate on any input).
    """

    def select(self, model: PrunableModel, layer_idx: int, ctx: PruneContext) -> list[int]:
        bn = model.prunable_bn(layer_idx)
        hook_layer = bn if bn is not None else model.prunable_layer(layer_idx)

        captured = []

        def hook(_module, _input, output):
            captured.append(output.detach() > 0)

        handle = hook_layer.register_forward_hook(hook)
        model.eval()
        with torch.no_grad():
            model(ctx.train_inputs)
        handle.remove()

        stacked = torch.cat(captured, dim=0)  # [N, H], [N, T, H], or [N, C, H, W]
        if stacked.dim() == 4:
            ever_active = stacked.any(dim=0).any(dim=-1).any(dim=-1)  # [C]
        elif stacked.dim() == 3:
            ever_active = stacked.any(dim=0).any(dim=0)               # [H] (over N and T)
        else:
            ever_active = stacked.any(dim=0)                          # [H]

        return [j for j, active in enumerate(ever_active.tolist()) if not active]
