"""Per-neuron measurement: activation behaviour over the training set plus a
ladder of geometric scores, from fully data-free to data-dependent.

A neuron's pre-activation z = w.x + b is read where the ReLU reads it: the
prunable layer's output for plain Linear/Conv layers, or the paired
BatchNorm's output when the model exposes one (matching
compute_hyperplane_params, which takes b from the BN bias in that case).

Columns per neuron:
    act_freq    fraction of training points with z > 0
                (conv: fraction of sample x spatial-position pairs)
    beta        b / ||w||            (compute_hyperplane_params; hyperplane
    abs_beta    |b| / ||w||           distance from the ORIGIN)
    dist_mean   |w.mu + b| / ||w||   hyperplane distance from the layer-input
                                     CENTROID mu = mean of inputs to the layer
    z_mean, z_std                    pre-activation moments over the train set
    z_margin    |z_mean| / z_std     standardized margin: how many stds the
                                     pre-activation mean sits from zero
    cos_w_mu    cosine(w, mu)        alignment of the weight vector with the
                                     mean input direction
    frac_pos_w  fraction of positive weights (deep inputs are nonnegative, so
                                     sign structure controls w.x)
    w_norm, b   raw ingredients
    ibp_lo, ibp_hi                   interval-bound-propagation bounds on z
                                     over the input box (MLP only)
    ibp_dead, ibp_always             data-free certificates: z provably <= 0
                                     (resp. >= 0) for EVERY input in the box
    ibp_score   max(ibp_lo, -ibp_hi) / ||w||  -- > 0 iff certified; the
                                     continuous version of the certificate

Input-dependent columns (dist_mean, cos_w_mu) are NaN for conv/BN layers,
where the hooked module's input is not the prunable layer's input.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.registry import DatasetBundle
from src.models.mlp import MLP
from src.models.registry import PrunableModel
from src.pruning.geometry import compute_hyperplane_params


# ── one pass over the training set: activation counts + moments ─────────────

def forward_stats(
    model: PrunableModel,
    bundle: DatasetBundle,
    device: torch.device,
    batch_size: int = 512,
) -> list[dict]:
    """Per prunable layer: act_freq [H], z_mean [H], z_std [H], and (when the
    hooked module's input is the layer input, i.e. no BN) mu_in [d]."""
    n_layers = model.n_prunable_layers()
    acc = [
        {"active": None, "n_pos": 0, "z_sum": None, "z2_sum": None, "n_z": 0,
         "in_sum": None, "n_in": 0}
        for _ in range(n_layers)
    ]

    def make_hook(i: int, capture_input: bool):
        def hook(module, inputs, out):
            z = out.detach()
            a = acc[i]
            if z.dim() == 4:  # conv: [B, C, H, W] -> stats over B, H, W
                dims, n = (0, 2, 3), z.shape[0] * z.shape[2] * z.shape[3]
            else:  # linear: [B, H] -> stats over B
                dims, n = (0,), z.shape[0]
            active = (z > 0).sum(dim=dims).double().cpu()
            z_sum = z.double().sum(dim=dims).cpu()
            z2_sum = (z.double() ** 2).sum(dim=dims).cpu()
            a["active"] = active if a["active"] is None else a["active"] + active
            a["z_sum"] = z_sum if a["z_sum"] is None else a["z_sum"] + z_sum
            a["z2_sum"] = z2_sum if a["z2_sum"] is None else a["z2_sum"] + z2_sum
            a["n_pos"] += n
            a["n_z"] += n
            if capture_input and z.dim() == 2:
                x = inputs[0].detach().double().sum(dim=0).cpu()
                a["in_sum"] = x if a["in_sum"] is None else a["in_sum"] + x
                a["n_in"] += inputs[0].shape[0]
        return hook

    handles = []
    for i in range(n_layers):
        bn = model.prunable_bn(i)
        target = bn if bn is not None else model.prunable_layer(i)
        # With BN, the hooked module's input is NOT the layer input -> skip it.
        handles.append(target.register_forward_hook(make_hook(i, capture_input=bn is None)))

    loader = DataLoader(bundle.train_ds, batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        for x_batch, _ in loader:
            model(x_batch.to(device))
    for h in handles:
        h.remove()

    stats = []
    for a in acc:
        z_mean = (a["z_sum"] / a["n_z"]).numpy()
        z_var = np.maximum((a["z2_sum"] / a["n_z"]).numpy() - z_mean ** 2, 0.0)
        stats.append({
            "act_freq": (a["active"] / a["n_pos"]).numpy(),
            "z_mean": z_mean,
            "z_std": np.sqrt(z_var),
            "mu_in": (a["in_sum"] / a["n_in"]).numpy() if a["in_sum"] is not None else None,
        })
    return stats


# ── data-free certificates: interval bound propagation ──────────────────────

def ibp_bounds(model: PrunableModel, bundle: DatasetBundle) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """For plain MLPs: per layer, elementwise bounds (z_lo, z_hi) on the
    pre-activation over the whole input box [min(x), max(x)] (per feature,
    from the training inputs -- for images this is essentially the pixel
    range, known a priori). z_hi <= 0 certifies a dead neuron for ANY input
    in the box; z_lo >= 0 certifies an always-on one. Bounds loosen with
    depth -- that decay is part of what this study measures.

    Returns [None]*n_layers for architectures where layer i's input is not
    simply relu(layer i-1's output) (resmlp residuals, conv pooling, ...).
    """
    n_layers = model.n_prunable_layers()
    if not isinstance(model, MLP):
        return [None] * n_layers

    x = bundle.train_ds.tensors[0].double()
    lo = x.amin(dim=0)
    hi = x.amax(dim=0)

    out = []
    for i in range(n_layers):
        layer = model.prunable_layer(i)
        W = layer.weight.data.double().cpu()
        b = layer.bias.data.double().cpu()
        W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)
        z_lo = W_pos @ lo + W_neg @ hi + b
        z_hi = W_pos @ hi + W_neg @ lo + b
        out.append((z_lo.numpy(), z_hi.numpy()))
        lo, hi = z_lo.clamp(min=0), z_hi.clamp(min=0)
    return out


# ── assembly ─────────────────────────────────────────────────────────────────

def collect_neuron_stats(
    model: PrunableModel,
    bundle: DatasetBundle,
    device: torch.device,
    seed: int,
) -> list[dict]:
    """One record per neuron across all prunable layers (see module docstring)."""
    stats = forward_stats(model, bundle, device)
    ibp = ibp_bounds(model, bundle)

    records = []
    for layer_idx in range(model.n_prunable_layers()):
        _, magnitudes = compute_hyperplane_params(model, layer_idx)  # beta = b / ||w||

        layer = model.prunable_layer(layer_idx)
        W = layer.weight.data.cpu().numpy().reshape(layer.weight.shape[0], -1)
        w_norms = np.linalg.norm(W, axis=1)
        safe_norms = np.where(w_norms < 1e-8, np.nan, w_norms)
        bn = model.prunable_bn(layer_idx)
        b = (bn.bias if bn is not None else layer.bias).data.cpu().numpy()

        s = stats[layer_idx]
        z_margin = np.abs(s["z_mean"]) / np.where(s["z_std"] < 1e-12, np.nan, s["z_std"])

        if s["mu_in"] is not None:
            w_dot_mu = W @ s["mu_in"]
            dist_mean = np.abs(w_dot_mu + b) / safe_norms
            mu_norm = np.linalg.norm(s["mu_in"])
            cos_w_mu = w_dot_mu / (safe_norms * (mu_norm if mu_norm > 1e-12 else np.nan))
        else:
            dist_mean = np.full(len(b), np.nan)
            cos_w_mu = np.full(len(b), np.nan)

        if ibp[layer_idx] is not None:
            z_lo, z_hi = ibp[layer_idx]
            ibp_score = np.maximum(z_lo, -z_hi) / safe_norms
        else:
            z_lo = z_hi = ibp_score = np.full(len(b), np.nan)

        for j in range(len(b)):
            records.append({
                "seed": seed,
                "layer": layer_idx,
                "neuron": j,
                "act_freq": float(s["act_freq"][j]),
                "beta": float(magnitudes[j]),
                "abs_beta": float(abs(magnitudes[j])),
                "dist_mean": float(dist_mean[j]),
                "z_mean": float(s["z_mean"][j]),
                "z_std": float(s["z_std"][j]),
                "z_margin": float(z_margin[j]),
                "cos_w_mu": float(cos_w_mu[j]),
                "frac_pos_w": float((W[j] > 0).mean()),
                "w_norm": float(w_norms[j]),
                "b": float(b[j]),
                "ibp_lo": float(z_lo[j]),
                "ibp_hi": float(z_hi[j]),
                "ibp_dead": bool(z_hi[j] <= 0) if not np.isnan(z_hi[j]) else None,
                "ibp_always": bool(z_lo[j] >= 0) if not np.isnan(z_lo[j]) else None,
                "ibp_score": float(ibp_score[j]),
            })
    return records
