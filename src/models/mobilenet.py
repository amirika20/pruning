"""MobileNetV2 (Sandler et al., CVPR 2018), mirroring torchvision's module
layout exactly so the official ImageNet-1k checkpoint loads key-for-key.

Each inverted-residual block runs expand 1x1 -> BN -> ReLU6 -> depthwise 3x3
-> BN -> ReLU6 -> project 1x1 -> BN (+ identity shortcut when stride 1 and
in==out). The prunable dimension is the EXPANDED channel count: the depthwise
conv is per-channel, so expanded channel c maps one-to-one onto input channel
c of the project conv, and removing it touches nothing outside the block --
the same internal-channels-only policy as ResNet/ResMLP. The first block
(expand ratio 1) has no expand conv (its width is tied to the stem) and is
not prunable, as are the stem and the final 1x1 conv.

`pretrained=True` downloads torchvision's checkpoint (width_mult 1.0 only).
On a dataset whose classes are a known ImageNet-1k subset (e.g. Imagenette)
the 1000-way classifier is sliced to those classes -- see
src.models.resnet.IMAGENET1K_CLASS_INDEX.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.data.registry import DatasetBundle
from src.models.registry import PrunableModel, register_model
from src.models.resnet import IMAGENET1K_CLASS_INDEX


def _make_divisible(v: float, divisor: int = 8) -> int:
    """torchvision's channel rounding: nearest multiple of `divisor`, never
    more than 10% below `v`."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _conv_bn_relu6(inp: int, oup: int, kernel: int, stride: int = 1, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding=kernel // 2, groups=groups, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        self.has_expand = expand_ratio != 1
        layers: list[nn.Module] = []
        if self.has_expand:
            layers.append(_conv_bn_relu6(inp, hidden, 1))           # conv.0: expand
        layers.append(_conv_bn_relu6(hidden, hidden, 3, stride, groups=hidden))  # depthwise
        layers.append(nn.Conv2d(hidden, oup, 1, bias=False))        # project
        layers.append(nn.BatchNorm2d(oup))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class MobileNetV2(PrunableModel):
    # (expand_ratio t, output channels c, repeats n, first stride s)
    SETTING = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(
        self,
        width_mult: float = 1.0,
        input_channels: int = 3,
        output_dim: int = 1000,
        dropout: float = 0.2,
    ):
        super().__init__()
        stem_out = _make_divisible(32 * width_mult)
        last_out = _make_divisible(1280 * max(1.0, width_mult))
        features: list[nn.Module] = [_conv_bn_relu6(input_channels, stem_out, 3, stride=2)]
        inp = stem_out
        for t, c, n, s in self.SETTING:
            oup = _make_divisible(c * width_mult)
            for i in range(n):
                features.append(InvertedResidual(inp, oup, s if i == 0 else 1, t))
                inp = oup
        features.append(_conv_bn_relu6(inp, last_out, 1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(last_out, output_dim))
        # Indices into self.features of the prunable (expand-conv) blocks.
        self._prunable = [
            i for i, m in enumerate(self.features)
            if isinstance(m, InvertedResidual) and m.has_expand
        ]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)

    # ── prunable protocol ─────────────────────────────────────────────────
    # One prunable layer per expand block: the expanded channels.

    def _block(self, idx: int) -> InvertedResidual:
        return self.features[self._prunable[idx]]

    def n_prunable_layers(self) -> int:
        return len(self._prunable)

    def prunable_layer(self, idx: int) -> nn.Module:
        return self._block(idx).conv[0][0]  # expand conv

    def prunable_bn(self, idx: int) -> nn.Module:
        return self._block(idx).conv[0][1]

    def outgoing_module(self, idx: int) -> nn.Module:
        return self._block(idx).conv[2]  # project 1x1 conv

    def outgoing_weights(self, idx: int) -> torch.Tensor:
        conv = self.outgoing_module(idx)
        return conv.weight.reshape(conv.out_channels, -1).t()  # [hidden, oup] (1x1 -> kk=1)

    def set_outgoing_weights(self, idx: int, new_weights: torch.Tensor) -> "MobileNetV2":
        clone = copy.deepcopy(self)
        conv = clone.outgoing_module(idx)
        w = new_weights.t().reshape(conv.weight.shape)
        conv.weight.data = w.to(conv.weight.dtype).to(conv.weight.device).contiguous()
        return clone

    def prune_layer(self, idx: int, indices_to_remove: list[int]) -> "MobileNetV2":
        """Remove expanded channels of block `idx`: the expand conv and its BN
        lose output channels, the depthwise conv (per-channel: groups shrink
        with it) and its BN lose channels, the project conv loses input
        channels. The block's output channels and shortcut are unchanged."""
        pruned = copy.deepcopy(self)
        block = pruned._block(idx)
        expand, dw = block.conv[0], block.conv[1]
        project = block.conv[2]
        conv_e, bn_e = expand[0], expand[1]
        conv_d, bn_d = dw[0], dw[1]
        device = conv_e.weight.device
        keep = sorted(set(range(conv_e.out_channels)) - set(indices_to_remove))
        n_keep = len(keep)

        new_e = nn.Conv2d(conv_e.in_channels, n_keep, 1, bias=False).to(device)
        new_e.weight.data = conv_e.weight.data[keep].clone()

        new_d = nn.Conv2d(n_keep, n_keep, conv_d.kernel_size, conv_d.stride,
                          padding=conv_d.padding, groups=n_keep, bias=False).to(device)
        new_d.weight.data = conv_d.weight.data[keep].clone()  # [C, 1, kH, kW]

        new_p = nn.Conv2d(n_keep, project.out_channels, 1, bias=False).to(device)
        new_p.weight.data = project.weight.data[:, keep].clone()

        def slice_bn(bn: nn.BatchNorm2d) -> nn.BatchNorm2d:
            new_bn = nn.BatchNorm2d(n_keep).to(device)
            new_bn.weight.data = bn.weight.data[keep].clone()
            new_bn.bias.data = bn.bias.data[keep].clone()
            new_bn.running_mean.data = bn.running_mean.data[keep].clone()
            new_bn.running_var.data = bn.running_var.data[keep].clone()
            new_bn.num_batches_tracked = bn.num_batches_tracked.clone()
            return new_bn

        expand[0], expand[1] = new_e, slice_bn(bn_e)
        dw[0], dw[1] = new_d, slice_bn(bn_d)
        block.conv[2] = new_p
        return pruned


@register_model("mobilenet_v2")
def build_mobilenet_v2(
    bundle: DatasetBundle,
    width_mult: float = 1.0,
    dropout: float = 0.2,
    pretrained: bool = False,
    weights: str = "IMAGENET1K_V1",
) -> MobileNetV2:
    """MobileNetV2; sizes via `width_mult` (0.5/0.75/1.0/1.4). `pretrained`
    loads torchvision's ImageNet-1k checkpoint (width_mult 1.0 only; V1
    71.88% / V2 72.15% top-1), slicing the classifier to the dataset's
    classes when they are a known ImageNet-1k subset."""
    input_channels = bundle.input_shape[0] if len(bundle.input_shape) == 3 else 3
    model = MobileNetV2(
        width_mult=width_mult,
        input_channels=input_channels,
        output_dim=bundle.output_dim,
        dropout=dropout,
    )
    if pretrained:
        if width_mult != 1.0 or input_channels != 3:
            raise ValueError(
                "pretrained MobileNetV2 exists for width_mult=1.0 on 3-channel input only, "
                f"got width_mult={width_mult}, input_channels={input_channels}"
            )
        import torchvision.models as tvm

        state = dict(tvm.get_model_weights("mobilenet_v2")[weights].get_state_dict(progress=True))
        if bundle.output_dim != 1000:
            wnids = bundle.extra.get("class_names", [])
            if len(wnids) == bundle.output_dim and all(w in IMAGENET1K_CLASS_INDEX for w in wnids):
                rows = [IMAGENET1K_CLASS_INDEX[w] for w in wnids]
                state["classifier.1.weight"] = state["classifier.1.weight"][rows]
                state["classifier.1.bias"] = state["classifier.1.bias"][rows]
            else:
                del state["classifier.1.weight"], state["classifier.1.bias"]
                print("[mobilenet_v2] pretrained classifier is 1000-way but dataset has "
                      f"{bundle.output_dim} classes with no known ImageNet-1k mapping -- head left "
                      "randomly initialized; train or fine-tune before measuring accuracy")
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, f"unexpected checkpoint keys: {unexpected}"
        assert all(k.startswith("classifier.") for k in missing), f"missing checkpoint keys: {missing}"
    return model
