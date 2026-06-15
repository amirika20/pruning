import copy
import numpy as np
import torch
import torch.nn as nn
from model import MLP, ResNet, CNN, ResCNN


# ── layer accessors ───────────────────────────────────────────────────────────

def _get_prunable_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    """
    Returns the primary weight layer whose OUTPUT channels/neurons are prunable.

    MLP    → hidden Linear before its ReLU
    ResNet → branch[0] Linear (branch intermediate dim; block output unchanged)
    CNN    → block.conv  Conv2d (output channels = filters)
    ResCNN → block.branch[0] Conv2d (intermediate channels; block output unchanged)
    """
    if isinstance(model, MLP):
        return model.net[2 * layer_idx]
    elif isinstance(model, ResNet):
        return model.blocks[layer_idx].branch[0]
    elif isinstance(model, CNN):
        return model.blocks[layer_idx].conv
    elif isinstance(model, ResCNN):
        return model.blocks[layer_idx].branch[0]
    raise ValueError(f"Unsupported model type: {type(model).__name__}")


def _get_prunable_bn(model: nn.Module, layer_idx: int):
    """BatchNorm paired with the prunable conv layer, or None for Linear models."""
    if isinstance(model, CNN):
        return model.blocks[layer_idx].bn
    if isinstance(model, ResCNN):
        return model.blocks[layer_idx].branch[1]
    return None


def layer_width(model: nn.Module, layer_idx: int) -> int:
    """Number of prunable neurons / filters in the given layer / block."""
    layer = _get_prunable_layer(model, layer_idx)
    if isinstance(layer, nn.Conv2d):
        return layer.out_channels
    return layer.out_features


# ── hyperplane characterisation ───────────────────────────────────────────────

def compute_hyperplane_params(
    model: nn.Module, layer_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each prunable neuron / filter, compute:
      - orientation: w / ||w||   shape [H, d]
      - magnitude:   b / ||w||   shape [H]

    For conv layers, w is flattened [in_ch * kH * kW] and b is taken from the
    paired BatchNorm's beta (bias) parameter.
    Rows where ||w|| < 1e-8 are marked NaN.
    """
    layer = _get_prunable_layer(model, layer_idx)

    if isinstance(layer, nn.Conv2d):
        W = layer.weight.data.cpu().numpy()            # [out_ch, in_ch, kH, kW]
        W = W.reshape(W.shape[0], -1)                   # [out_ch, fan_in]
        bn = _get_prunable_bn(model, layer_idx)
        b = bn.bias.data.cpu().numpy() if bn is not None else np.zeros(W.shape[0])
    else:
        W = layer.weight.data.cpu().numpy()             # [H, d]
        b = layer.bias.data.cpu().numpy()               # [H]

    norms = np.linalg.norm(W, axis=1)
    safe_norms = np.where(norms < 1e-8, np.nan, norms)
    orientations = W / safe_norms[:, None]
    magnitudes   = b / safe_norms
    return orientations, magnitudes


# ── redundancy detection ──────────────────────────────────────────────────────

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


# ── silence detection ─────────────────────────────────────────────────────────

def find_silent_neurons(
    model: nn.Module,
    layer_idx: int,
    train_inputs: torch.Tensor,
) -> list:
    """
    Pass all training inputs through the model and record which neurons / filters
    ever produce a positive pre-ReLU activation.

    For conv models (CNN / ResCNN): hooks the paired BatchNorm output (post-BN,
    pre-ReLU) and checks whether any spatial position, in any sample, is positive.
    Output shape [N, C, H, W]; ever_active per filter: any over (N, H, W).

    For linear models (MLP / ResNet): hooks the Linear output [N, H].

    Returns indices that are always silent (never activate on any input).
    """
    bn = _get_prunable_bn(model, layer_idx)
    hook_layer = bn if bn is not None else _get_prunable_layer(model, layer_idx)

    captured = []

    def hook(_module, _input, output):
        captured.append(output.detach() > 0)

    handle = hook_layer.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model(train_inputs)
    handle.remove()

    stacked = torch.cat(captured, dim=0)   # [N, H] or [N, C, H, W]
    if stacked.dim() == 4:
        ever_active = stacked.any(dim=0).any(dim=-1).any(dim=-1)   # [C]
    else:
        ever_active = stacked.any(dim=0)   # [H]

    return [j for j, active in enumerate(ever_active.tolist()) if not active]


# ── pruning dispatch ──────────────────────────────────────────────────────────

def prune_layer(model: nn.Module, layer_idx: int, indices_to_remove: list) -> nn.Module:
    if isinstance(model, MLP):
        return _prune_mlp_layer(model, layer_idx, indices_to_remove)
    elif isinstance(model, ResNet):
        return _prune_resnet_block(model, layer_idx, indices_to_remove)
    elif isinstance(model, CNN):
        return _prune_cnn_block(model, layer_idx, indices_to_remove)
    elif isinstance(model, ResCNN):
        return _prune_rescnn_block(model, layer_idx, indices_to_remove)
    raise ValueError(f"Unsupported model type: {type(model).__name__}")


# ── MLP pruning ───────────────────────────────────────────────────────────────

def _prune_mlp_layer(model: MLP, layer_idx: int, indices_to_remove: list) -> MLP:
    """Remove output neurons from hidden layer `layer_idx` and fix the next layer's inputs."""
    this_linear = model.net[2 * layer_idx]
    next_linear = model.net[2 * (layer_idx + 1)]
    device = this_linear.weight.device
    keep = [i for i in range(this_linear.out_features) if i not in set(indices_to_remove)]

    new_this = nn.Linear(this_linear.in_features, len(keep)).to(device)
    new_this.weight.data = this_linear.weight.data[keep].clone()
    new_this.bias.data   = this_linear.bias.data[keep].clone()

    new_next = nn.Linear(len(keep), next_linear.out_features).to(device)
    new_next.weight.data = next_linear.weight.data[:, keep].clone()
    new_next.bias.data   = next_linear.bias.data.clone()

    pruned = copy.deepcopy(model)
    pruned.net[2 * layer_idx]       = new_this
    pruned.net[2 * (layer_idx + 1)] = new_next
    return pruned


# ── ResNet pruning ────────────────────────────────────────────────────────────

def _prune_resnet_block(model: ResNet, block_idx: int, indices_to_remove: list) -> ResNet:
    """
    Shrink the branch's intermediate hidden dimension in block `block_idx`.
    The block's output size and shortcut are unchanged.
    """
    branch_in  = model.blocks[block_idx].branch[0]   # Linear(in_size, hidden_dim)
    branch_out = model.blocks[block_idx].branch[2]   # Linear(hidden_dim, out_size)
    device = branch_in.weight.device
    keep = [i for i in range(branch_in.out_features) if i not in set(indices_to_remove)]

    new_branch_in = nn.Linear(branch_in.in_features, len(keep)).to(device)
    new_branch_in.weight.data = branch_in.weight.data[keep].clone()
    new_branch_in.bias.data   = branch_in.bias.data[keep].clone()

    new_branch_out = nn.Linear(len(keep), branch_out.out_features).to(device)
    new_branch_out.weight.data = branch_out.weight.data[:, keep].clone()
    new_branch_out.bias.data   = branch_out.bias.data.clone()

    pruned = copy.deepcopy(model)
    pruned.blocks[block_idx].branch[0] = new_branch_in
    pruned.blocks[block_idx].branch[2] = new_branch_out
    return pruned


# ── CNN pruning ───────────────────────────────────────────────────────────────

def _prune_cnn_block(model: CNN, block_idx: int, indices_to_remove: list) -> CNN:
    """
    Remove output filters from CNNBlock `block_idx`.

    Shrinks block.conv output channels + block.bn, then fixes the next consumer:
      - next CNNBlock's conv in_channels, or
      - head Linear's in_features (last block).
    """
    pruned = copy.deepcopy(model)
    block  = pruned.blocks[block_idx]
    conv, bn = block.conv, block.bn
    device = conv.weight.device
    keep   = sorted(set(range(conv.out_channels)) - set(indices_to_remove))
    n_keep = len(keep)

    new_conv = nn.Conv2d(conv.in_channels, n_keep, conv.kernel_size,
                         padding=conv.padding, bias=False).to(device)
    new_conv.weight.data = conv.weight.data[keep].clone()

    new_bn = nn.BatchNorm2d(n_keep).to(device)
    new_bn.weight.data        = bn.weight.data[keep].clone()
    new_bn.bias.data          = bn.bias.data[keep].clone()
    new_bn.running_mean.data  = bn.running_mean.data[keep].clone()
    new_bn.running_var.data   = bn.running_var.data[keep].clone()
    new_bn.num_batches_tracked = bn.num_batches_tracked.clone()

    block.conv = new_conv
    block.bn   = new_bn

    if block_idx < len(pruned.blocks) - 1:
        nc = pruned.blocks[block_idx + 1].conv
        new_nc = nn.Conv2d(n_keep, nc.out_channels, nc.kernel_size,
                           padding=nc.padding, bias=False).to(device)
        new_nc.weight.data = nc.weight.data[:, keep].clone()
        pruned.blocks[block_idx + 1].conv = new_nc
    else:
        h = pruned.head
        new_h = nn.Linear(n_keep, h.out_features).to(device)
        new_h.weight.data = h.weight.data[:, keep].clone()
        new_h.bias.data   = h.bias.data.clone()
        pruned.head = new_h

    return pruned


# ── ResCNN pruning ────────────────────────────────────────────────────────────

def _prune_rescnn_block(model: ResCNN, block_idx: int, indices_to_remove: list) -> ResCNN:
    """
    Prune the intermediate channels of ResCNNBlock `block_idx`.

    branch[0] (Conv2d) and branch[1] (BN) lose output channels.
    branch[3] (Conv2d) loses input channels.
    The block's output channels and shortcut are completely unchanged.
    """
    pruned = copy.deepcopy(model)
    block  = pruned.blocks[block_idx]
    conv1, bn1, conv2 = block.branch[0], block.branch[1], block.branch[3]
    device = conv1.weight.device
    keep   = sorted(set(range(conv1.out_channels)) - set(indices_to_remove))
    n_keep = len(keep)

    # Shrink conv1
    new_conv1 = nn.Conv2d(conv1.in_channels, n_keep, conv1.kernel_size,
                          padding=conv1.padding, bias=False).to(device)
    new_conv1.weight.data = conv1.weight.data[keep].clone()

    # Shrink bn1
    new_bn1 = nn.BatchNorm2d(n_keep).to(device)
    new_bn1.weight.data        = bn1.weight.data[keep].clone()
    new_bn1.bias.data          = bn1.bias.data[keep].clone()
    new_bn1.running_mean.data  = bn1.running_mean.data[keep].clone()
    new_bn1.running_var.data   = bn1.running_var.data[keep].clone()
    new_bn1.num_batches_tracked = bn1.num_batches_tracked.clone()

    # Fix conv2 input channels (output channels and bn2 unchanged)
    new_conv2 = nn.Conv2d(n_keep, conv2.out_channels, conv2.kernel_size,
                          padding=conv2.padding, bias=False).to(device)
    new_conv2.weight.data = conv2.weight.data[:, keep].clone()

    block.branch[0] = new_conv1
    block.branch[1] = new_bn1
    block.branch[3] = new_conv2
    return pruned
