"""Hyperplane characterisation of a prunable layer's neurons/filters."""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from src.models.registry import PrunableModel


def compute_hyperplane_params(model: PrunableModel, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """For each prunable neuron/filter of layer `layer_idx`, compute:
      - orientation: w / ||w||   shape [H, d]
      - magnitude:   b / ||w||   shape [H]

    For conv layers, w is flattened [in_ch * kH * kW] and b is taken from the
    paired BatchNorm's beta (bias) parameter.
    Rows where ||w|| < 1e-8 are marked NaN.
    """
    layer = model.prunable_layer(layer_idx)

    if isinstance(layer, nn.Conv2d):
        W = layer.weight.data.cpu().numpy()  # [out_ch, in_ch, kH, kW]
        W = W.reshape(W.shape[0], -1)        # [out_ch, fan_in]
        bn = model.prunable_bn(layer_idx)
        b = bn.bias.data.cpu().numpy() if bn is not None else np.zeros(W.shape[0])
    else:
        W = layer.weight.data.cpu().numpy()  # [H, d]
        b = layer.bias.data.cpu().numpy()    # [H]

    norms = np.linalg.norm(W, axis=1)
    safe_norms = np.where(norms < 1e-8, np.nan, norms)
    orientations = W / safe_norms[:, None]
    magnitudes = b / safe_norms
    return orientations, magnitudes


def layer_width(model: PrunableModel, layer_idx: int) -> int:
    """Number of prunable neurons/filters in the given layer/block."""
    layer = model.prunable_layer(layer_idx)
    if isinstance(layer, nn.Conv2d):
        return layer.out_channels
    return layer.out_features
