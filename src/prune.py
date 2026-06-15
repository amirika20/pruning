import copy
import numpy as np
import torch
import torch.nn as nn
from model import MLP, ResNet


def _get_prunable_linear(model: nn.Module, layer_idx: int) -> nn.Linear:
    """
    MLP:    the linear layer of hidden layer `layer_idx` (before its ReLU).
    ResNet: the first linear of block `layer_idx`'s branch (before the branch's ReLU).
            Pruning the branch's intermediate dimension leaves the block output size
            and the shortcut connection untouched.
    """
    if isinstance(model, MLP):
        return model.net[2 * layer_idx]
    elif isinstance(model, ResNet):
        return model.blocks[layer_idx].branch[0]
    raise ValueError(f"Unsupported model type: {type(model).__name__}")


def layer_width(model: nn.Module, layer_idx: int) -> int:
    """Number of prunable neurons in the given layer / block."""
    return _get_prunable_linear(model, layer_idx).out_features


def compute_hyperplane_params(model: nn.Module, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    For each prunable neuron, compute:
      - orientation: w / ||w||   shape [H, d]
      - magnitude:   b / ||w||   shape [H]
    Rows where ||w|| < 1e-8 are marked NaN.
    """
    linear = _get_prunable_linear(model, layer_idx)
    W = linear.weight.data.numpy()   # [H, d]
    b = linear.bias.data.numpy()     # [H]

    norms = np.linalg.norm(W, axis=1)
    safe_norms = np.where(norms < 1e-8, np.nan, norms)
    orientations = W / safe_norms[:, None]
    magnitudes = b / safe_norms
    return orientations, magnitudes


def find_redundant_neurons(
    model: nn.Module,
    layer_idx: int,
    orientation_threshold: float = 0.01,
    magnitude_threshold: float = 0.01,
) -> list:
    """
    Greedy: two neurons are redundant if their hyperplanes nearly coincide, i.e.
      1 - |dot(w_i/||w_i||, w_j/||w_j||)| < orientation_threshold  (parallel normals)
      |b_i/||w_i|| - b_j/||w_j||| < magnitude_threshold             (same offset)
    Removes the later neuron in each redundant pair.
    """
    orientations, magnitudes = compute_hyperplane_params(model, layer_idx)
    valid = ~(np.any(np.isnan(orientations), axis=1) | np.isnan(magnitudes))

    kept_orientations: list = []
    kept_magnitudes: list = []
    to_remove: list = []

    for j in range(len(orientations)):
        if not valid[j]:
            continue
        o, m = orientations[j], magnitudes[j]
        redundant = any(
            (1 - abs(np.dot(o, ko))) < orientation_threshold
            and abs(m - km) < magnitude_threshold
            for ko, km in zip(kept_orientations, kept_magnitudes)
        )
        if redundant:
            to_remove.append(j)
        else:
            kept_orientations.append(o)
            kept_magnitudes.append(m)

    return to_remove


def find_silent_neurons(
    model: nn.Module,
    layer_idx: int,
    train_inputs: torch.Tensor,
) -> list:
    """
    Pass all training inputs through the model and record which prunable neurons
    ever produce a positive pre-ReLU activation.
    Returns indices of neurons that are always silent (never activate on any input).
    """
    linear = _get_prunable_linear(model, layer_idx)
    captured = []

    def hook(_module, _input, output):
        captured.append(output.detach() > 0)   # [batch, H] bool

    handle = linear.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model(train_inputs)
    handle.remove()

    ever_active = torch.cat(captured, dim=0).any(dim=0)  # [H] bool
    return [j for j, active in enumerate(ever_active.tolist()) if not active]


def prune_layer(model: nn.Module, layer_idx: int, indices_to_remove: list) -> nn.Module:
    if isinstance(model, MLP):
        return _prune_mlp_layer(model, layer_idx, indices_to_remove)
    elif isinstance(model, ResNet):
        return _prune_resnet_block(model, layer_idx, indices_to_remove)
    raise ValueError(f"Unsupported model type: {type(model).__name__}")


def _prune_mlp_layer(model: MLP, layer_idx: int, indices_to_remove: list) -> MLP:
    """Remove output neurons from hidden layer `layer_idx` and fix the next layer's inputs."""
    this_linear = model.net[2 * layer_idx]
    next_linear = model.net[2 * (layer_idx + 1)]
    keep = [i for i in range(this_linear.out_features) if i not in set(indices_to_remove)]

    new_this = nn.Linear(this_linear.in_features, len(keep))
    new_this.weight.data = this_linear.weight.data[keep].clone()
    new_this.bias.data = this_linear.bias.data[keep].clone()

    new_next = nn.Linear(len(keep), next_linear.out_features)
    new_next.weight.data = next_linear.weight.data[:, keep].clone()
    new_next.bias.data = next_linear.bias.data.clone()

    pruned = copy.deepcopy(model)
    pruned.net[2 * layer_idx] = new_this
    pruned.net[2 * (layer_idx + 1)] = new_next
    return pruned


def _prune_resnet_block(model: ResNet, block_idx: int, indices_to_remove: list) -> ResNet:
    """
    Shrink the branch's intermediate hidden dimension in block `block_idx`.
    The block's output size and shortcut are unchanged.
    """
    branch_in  = model.blocks[block_idx].branch[0]   # Linear(in_size, hidden_dim)
    branch_out = model.blocks[block_idx].branch[2]   # Linear(hidden_dim, out_size)
    keep = [i for i in range(branch_in.out_features) if i not in set(indices_to_remove)]

    new_branch_in = nn.Linear(branch_in.in_features, len(keep))
    new_branch_in.weight.data = branch_in.weight.data[keep].clone()
    new_branch_in.bias.data   = branch_in.bias.data[keep].clone()

    new_branch_out = nn.Linear(len(keep), branch_out.out_features)
    new_branch_out.weight.data = branch_out.weight.data[:, keep].clone()
    new_branch_out.bias.data   = branch_out.bias.data.clone()

    pruned = copy.deepcopy(model)
    pruned.blocks[block_idx].branch[0] = new_branch_in
    pruned.blocks[block_idx].branch[2] = new_branch_out
    return pruned
