"""Standard ResNets (He et al., CVPR 2016) in both classic families:

- CIFAR-style  (`resnet_cifar`):    3x3 stem, 3 stages of n BasicBlocks each,
  depth = 6n+2 -> resnet20/32/44/56/110 with 16/32/64 base channels.
- ImageNet-style (`resnet_imagenet`): 7x7/2 stem + maxpool, 4 stages ->
  resnet18/34 (BasicBlock) and resnet50/101/152 (Bottleneck).

Prunable dimensions are the channels *inside* each residual block -- the ones
not tied to the shortcut: BasicBlock exposes one slot (conv1's output),
Bottleneck exposes two (conv1's and conv2's outputs). Block output channels
and shortcuts are untouched during pruning, exactly like ResCNN/ResMLP.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.data.registry import DatasetBundle
from src.models.registry import PrunableModel, register_model


class BasicBlock(nn.Module):
    """conv3x3 -> BN -> ReLU -> conv3x3 -> BN, plus shortcut. One prunable
    slot: conv1's output channels (consumed by conv2)."""

    expansion = 1
    n_slots = 1

    def __init__(self, in_channels: int, planes: int, stride: int = 1):
        super().__init__()
        out_channels = planes * self.expansion
        self.conv1 = nn.Conv2d(in_channels, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.downsample(x))

    def slot_modules(self, slot: int) -> tuple[str, str, str]:
        """(prunable conv, its BN, consumer conv) attribute names for `slot`."""
        assert slot == 0
        return "conv1", "bn1", "conv2"


class Bottleneck(nn.Module):
    """conv1x1 -> BN -> ReLU -> conv3x3 -> BN -> ReLU -> conv1x1 -> BN, plus
    shortcut. Two prunable slots: conv1's output (consumed by conv2) and
    conv2's output (consumed by conv3)."""

    expansion = 4
    n_slots = 2

    def __init__(self, in_channels: int, planes: int, stride: int = 1):
        super().__init__()
        out_channels = planes * self.expansion
        self.conv1 = nn.Conv2d(in_channels, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + self.downsample(x))

    def slot_modules(self, slot: int) -> tuple[str, str, str]:
        assert slot in (0, 1)
        return ("conv1", "bn1", "conv2") if slot == 0 else ("conv2", "bn2", "conv3")


class ResNet(PrunableModel):
    """Stem -> flat list of residual blocks -> global average pool -> Linear."""

    def __init__(
        self,
        block_cls: type,
        stage_blocks: list[int],
        stage_planes: list[int],
        input_channels: int = 3,
        output_dim: int = 10,
        imagenet_stem: bool = False,
    ):
        super().__init__()
        stem_out = stage_planes[0] if not imagenet_stem else 64
        if imagenet_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(input_channels, stem_out, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(stem_out),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(input_channels, stem_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(stem_out),
                nn.ReLU(),
            )

        blocks: list[nn.Module] = []
        in_channels = stem_out
        for stage, (n_blocks, planes) in enumerate(zip(stage_blocks, stage_planes)):
            for b in range(n_blocks):
                stride = 2 if (stage > 0 and b == 0) else 1
                blocks.append(block_cls(in_channels, planes, stride=stride))
                in_channels = planes * block_cls.expansion
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(in_channels, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=[2, 3])
        return self.head(x)

    # ── prunable protocol ─────────────────────────────────────────────────
    # Prunable layer `idx` enumerates (block, slot) pairs in network order.

    def _slot(self, idx: int) -> tuple[nn.Module, int]:
        for block in self.blocks:
            if idx < block.n_slots:
                return block, idx
            idx -= block.n_slots
        raise IndexError(f"prunable layer index {idx} out of range")

    def n_prunable_layers(self) -> int:
        return sum(block.n_slots for block in self.blocks)

    def prunable_layer(self, idx: int) -> nn.Module:
        block, slot = self._slot(idx)
        return getattr(block, block.slot_modules(slot)[0])

    def prunable_bn(self, idx: int) -> nn.Module:
        block, slot = self._slot(idx)
        return getattr(block, block.slot_modules(slot)[1])

    def outgoing_module(self, idx: int) -> nn.Module:
        """The conv that consumes prunable layer `idx`'s activations (after
        the block-internal BN+ReLU) -- the reconstruction target of
        activation-matching methods like OSSCAR."""
        block, slot = self._slot(idx)
        return getattr(block, block.slot_modules(slot)[2])

    def outgoing_weights(self, idx: int) -> torch.Tensor:
        """[Cin*kH*kW, Cout]: the consumer conv's weights as a matrix over
        unfolded input patches, rows grouped by input channel (kH*kW rows per
        prunable channel, matching F.unfold's channel-major column order)."""
        conv = self.outgoing_module(idx)
        return conv.weight.detach().reshape(conv.out_channels, -1).t()

    def set_outgoing_weights(self, idx: int, new_weights: torch.Tensor) -> "ResNet":
        import copy as _copy

        clone = _copy.deepcopy(self)
        conv = clone.outgoing_module(idx)
        clone_weight = new_weights.t().reshape(conv.weight.shape)
        conv.weight.data = clone_weight.to(conv.weight.dtype).to(conv.weight.device).contiguous()
        return clone

    def prune_layer(self, idx: int, indices_to_remove: list[int]) -> "ResNet":
        """Remove channels inside a residual block: the prunable conv and its
        BN lose output channels, the consumer conv loses input channels. The
        block's output channels and shortcut are completely unchanged."""
        pruned = copy.deepcopy(self)
        block, slot = pruned._slot(idx)
        conv_name, bn_name, next_name = block.slot_modules(slot)
        conv: nn.Conv2d = getattr(block, conv_name)
        bn: nn.BatchNorm2d = getattr(block, bn_name)
        next_conv: nn.Conv2d = getattr(block, next_name)
        device = conv.weight.device
        keep = sorted(set(range(conv.out_channels)) - set(indices_to_remove))
        n_keep = len(keep)

        new_conv = nn.Conv2d(conv.in_channels, n_keep, conv.kernel_size,
                             stride=conv.stride, padding=conv.padding, bias=False).to(device)
        new_conv.weight.data = conv.weight.data[keep].clone()

        new_bn = nn.BatchNorm2d(n_keep).to(device)
        # A freshly constructed BatchNorm starts in TRAIN mode, and assigning
        # it into an eval-mode model leaves that submodule normalizing by BATCH
        # statistics -- so a pruned model silently computes something else, and
        # a forward pass also overwrites the running stats we just copied.
        new_bn.train(bn.training)
        new_bn.weight.data = bn.weight.data[keep].clone()
        new_bn.bias.data = bn.bias.data[keep].clone()
        new_bn.running_mean.data = bn.running_mean.data[keep].clone()
        new_bn.running_var.data = bn.running_var.data[keep].clone()
        new_bn.num_batches_tracked = bn.num_batches_tracked.clone()

        new_next = nn.Conv2d(n_keep, next_conv.out_channels, next_conv.kernel_size,
                             stride=next_conv.stride, padding=next_conv.padding, bias=False).to(device)
        new_next.weight.data = next_conv.weight.data[:, keep].clone()

        setattr(block, conv_name, new_conv)
        setattr(block, bn_name, new_bn)
        setattr(block, next_name, new_next)
        return pruned


def _input_channels(bundle: DatasetBundle) -> int:
    return bundle.input_shape[0] if len(bundle.input_shape) == 3 else 3


# ── pretrained checkpoints ────────────────────────────────────────────────
# Both checkpoint sources use the torchvision-style layout (conv1/bn1,
# layer{s}.{b}.*, fc); ours is stem.0/stem.1, blocks.{i}.*, head. Block-
# internal names (conv1/bn1/.../downsample.0/downsample.1) match exactly.

# ImageNet-1k class index of each WNID we know about -- lets a pretrained
# 1000-way head be sliced down to a subset dataset's classes (zero-training
# evaluation). Currently the 10 Imagenette classes; extend as needed.
IMAGENET1K_CLASS_INDEX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def _rename_torchvision_keys(state: dict, stage_blocks: list[int]) -> dict:
    offsets = [0]
    for n_blocks in stage_blocks:
        offsets.append(offsets[-1] + n_blocks)
    renamed = {}
    for key, value in state.items():
        if key.startswith("conv1."):
            new_key = "stem.0." + key[len("conv1."):]
        elif key.startswith("bn1."):
            new_key = "stem.1." + key[len("bn1."):]
        elif key.startswith("fc."):
            new_key = "head." + key[len("fc."):]
        elif key.startswith("layer"):
            stage_str, block_str, rest = key.split(".", 2)
            block_idx = offsets[int(stage_str[len("layer"):]) - 1] + int(block_str)
            new_key = f"blocks.{block_idx}.{rest}"
        else:
            raise KeyError(f"unrecognized checkpoint key {key!r}")
        renamed[new_key] = value
    return renamed


def _load_pretrained(model: ResNet, state: dict, stage_blocks: list[int], output_dim: int,
                     head_classes: int, bundle: DatasetBundle) -> None:
    """Load a torchvision-layout checkpoint into `model` (in place). If the
    dataset has fewer classes than the checkpoint's `head_classes`-way head,
    the head is sliced to the dataset's classes when their WNIDs are known
    (see IMAGENET1K_CLASS_INDEX), otherwise left randomly initialized (train
    or fine-tune it before measuring accuracy)."""
    renamed = _rename_torchvision_keys(state, stage_blocks)
    if output_dim != head_classes:
        wnids = bundle.extra.get("class_names", [])
        rows = [IMAGENET1K_CLASS_INDEX[w] for w in wnids] if (
            head_classes == 1000 and len(wnids) == output_dim
            and all(w in IMAGENET1K_CLASS_INDEX for w in wnids)
        ) else None
        if rows is not None:
            renamed["head.weight"] = renamed["head.weight"][rows]
            renamed["head.bias"] = renamed["head.bias"][rows]
        else:
            del renamed["head.weight"], renamed["head.bias"]
            print(f"[resnet] pretrained head is {head_classes}-way but dataset has "
                  f"{output_dim} classes with no known ImageNet-1k mapping -- head left "
                  "randomly initialized; train or fine-tune before measuring accuracy")
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    assert not unexpected, f"unexpected checkpoint keys: {unexpected}"
    assert all(k.startswith("head.") for k in missing), f"missing checkpoint keys: {missing}"


@register_model("resnet_cifar")
def build_resnet_cifar(
    bundle: DatasetBundle, depth: int = 20, base_width: int = 16, pretrained: bool = False
) -> ResNet:
    """CIFAR ResNet-(6n+2): depth in {20, 32, 44, 56, 110, ...}.

    `pretrained=True` downloads CIFAR-10/CIFAR-100 weights (selected by the
    dataset's class count) from chenyaofo/pytorch-cifar-models via torch.hub
    (depths 20/32/44/56, base_width 16 only)."""
    if (depth - 2) % 6 != 0:
        raise ValueError(f"CIFAR ResNet depth must be 6n+2 (20, 32, 44, 56, 110, ...), got {depth}")
    n = (depth - 2) // 6
    stage_blocks = [n, n, n]
    model = ResNet(
        BasicBlock,
        stage_blocks=stage_blocks,
        stage_planes=[base_width, 2 * base_width, 4 * base_width],
        input_channels=_input_channels(bundle),
        output_dim=bundle.output_dim,
    )
    if pretrained:
        if depth not in (20, 32, 44, 56) or base_width != 16:
            raise ValueError(
                f"pretrained CIFAR ResNets exist for depth 20/32/44/56 at base_width 16 "
                f"(chenyaofo/pytorch-cifar-models), got depth={depth}, base_width={base_width}"
            )
        dataset = {10: "cifar10", 100: "cifar100"}.get(bundle.output_dim)
        if dataset is None:
            raise ValueError(
                f"pretrained CIFAR ResNets are trained for 10 or 100 classes, "
                f"dataset has {bundle.output_dim}"
            )
        hub_model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", f"{dataset}_resnet{depth}",
            pretrained=True, verbose=False, trust_repo=True,
        )
        _load_pretrained(model, hub_model.state_dict(), stage_blocks,
                         bundle.output_dim, bundle.output_dim, bundle)
    return model


@register_model("resnet_imagenet")
def build_resnet_imagenet(
    bundle: DatasetBundle, depth: int = 18, pretrained: bool = False,
    weights: str = "IMAGENET1K_V1",
) -> ResNet:
    """ImageNet ResNet: depth in {18, 34, 50, 101, 152}.

    `pretrained=True` downloads torchvision's ImageNet-1k checkpoint
    (`weights` selects the recipe; V1 exists for every depth). On a dataset
    whose classes are a known ImageNet-1k subset (e.g. Imagenette), the
    1000-way head is sliced to those classes, giving a ready-to-evaluate
    classifier with no training."""
    configs = {
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
        152: (Bottleneck, [3, 8, 36, 3]),
    }
    if depth not in configs:
        raise ValueError(f"ImageNet ResNet depth must be one of {sorted(configs)}, got {depth}")
    block_cls, stage_blocks = configs[depth]
    model = ResNet(
        block_cls,
        stage_blocks=stage_blocks,
        stage_planes=[64, 128, 256, 512],
        input_channels=_input_channels(bundle),
        output_dim=bundle.output_dim,
        imagenet_stem=True,
    )
    if pretrained:
        import torchvision.models as tvm

        state = tvm.get_model_weights(f"resnet{depth}")[weights].get_state_dict(progress=True)
        _load_pretrained(model, state, stage_blocks, bundle.output_dim, 1000, bundle)
    return model
