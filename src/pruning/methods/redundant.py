"""Redundant-neuron pruning: remove units whose hyperplanes nearly coincide."""

from __future__ import annotations

import numpy as np

from src.models.registry import PrunableModel
from src.pruning.geometry import compute_hyperplane_params
from src.pruning.registry import PruneContext, PruningMethod, register_pruning_method


@register_pruning_method("redundant")
class RedundantPruning(PruningMethod):
    """Greedy: two neurons are redundant if their hyperplanes nearly coincide, i.e.
      1 - |dot(w_i/||w_i||, w_j/||w_j||)| < orientation_threshold  (parallel normals)
      |b_i/||w_i|| - b_j/||w_j||| < magnitude_threshold             (same offset)
    Removes the later neuron in each redundant pair.
    """

    def __init__(self, orientation_threshold: float = 0.01, magnitude_threshold: float = 0.01):
        self.orientation_threshold = orientation_threshold
        self.magnitude_threshold = magnitude_threshold

    def select(self, model: PrunableModel, layer_idx: int, ctx: PruneContext) -> list[int]:
        orientations, magnitudes = compute_hyperplane_params(model, layer_idx)
        valid = ~(np.any(np.isnan(orientations), axis=1) | np.isnan(magnitudes))

        kept_orientations: list = []
        kept_magnitudes: list = []
        to_remove: list[int] = []

        for j in range(len(orientations)):
            if not valid[j]:
                continue
            o, m = orientations[j], magnitudes[j]
            redundant = any(
                (1 - abs(np.dot(o, ko))) < self.orientation_threshold
                and abs(m - km) < self.magnitude_threshold
                for ko, km in zip(kept_orientations, kept_magnitudes)
            )
            if redundant:
                to_remove.append(j)
            else:
                kept_orientations.append(o)
                kept_magnitudes.append(m)

        return [j for j in to_remove if j not in ctx.already_selected]
