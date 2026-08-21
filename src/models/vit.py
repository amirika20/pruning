"""Vision Transformer (Dosovitskiy et al., ICLR 2021), wrapping torchvision's
VisionTransformer so the official ImageNet-1k checkpoints load key-for-key
and the attention/positional-embedding implementation is exactly theirs.

The prunable dimension is each encoder block's MLP hidden width (Linear
d_model -> mlp_dim -> GELU -> Linear mlp_dim -> d_model): the output neurons
of mlp[0], consumed by the nn.Linear mlp[3]. That is the same structure
OSSCAR prunes in language transformers (their OPT experiments), so all
FC-path pruning methods apply unchanged; attention heads, embeddings, and
LayerNorms are untouched. One prunable layer per encoder block.

`pretrained=True` downloads torchvision's IMAGENET1K_V1 checkpoint for the
named variant (b_16 81.07% / b_32 75.91% / l_16 79.66% / l_32 76.97% top-1;
requires image_size 224 and 3-channel input). On a dataset whose classes are
a known ImageNet-1k subset (e.g. Imagenette) the 1000-way head is sliced to
those classes -- see src.models.resnet.IMAGENET1K_CLASS_INDEX.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.data.registry import DatasetBundle
from src.models.registry import MergeOp, PrunableModel, register_model
from src.models.resnet import IMAGENET1K_CLASS_INDEX

# (patch_size, num_layers, num_heads, hidden_dim, mlp_dim)
VIT_VARIANTS = {
    "b_16": (16, 12, 12, 768, 3072),
    "b_32": (32, 12, 12, 768, 3072),
    "l_16": (16, 24, 16, 1024, 4096),
    "l_32": (32, 24, 16, 1024, 4096),
    "h_14": (14, 32, 16, 1280, 5120),
}
PRETRAINED_VARIANTS = ("b_16", "b_32", "l_16", "l_32")  # IMAGENET1K_V1 exists


class ViT(PrunableModel):
    """PrunableModel adapter around torchvision.models.VisionTransformer."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ── prunable protocol ─────────────────────────────────────────────────

    def _mlp(self, idx: int) -> nn.Module:
        return self.net.encoder.layers[idx].mlp

    def n_prunable_layers(self) -> int:
        return len(self.net.encoder.layers)

    def prunable_layer(self, idx: int) -> nn.Module:
        return self._mlp(idx)[0]

    def outgoing_module(self, idx: int) -> nn.Module:
        return self._mlp(idx)[3]

    def outgoing_weights(self, idx: int) -> torch.Tensor:
        return self._mlp(idx)[3].weight.detach().t()  # [mlp_dim, d_model]

    def merge_outgoing(self, idx: int, merges: list[MergeOp]) -> "ViT":
        """Fold each removed MLP neuron's outgoing weights into its survivor --
        mlp[3] reads the hidden dimension linearly, so this is one column
        transfer, exactly as in any fully-connected layer."""
        merged = copy.deepcopy(self)
        W = merged._mlp(idx)[3].weight.data            # [d_model, mlp_dim]
        for op in merges:
            W[:, op.survivor] += op.scale * W[:, op.removed]
        return merged

    def prune_layer(self, idx: int, indices_to_remove: list[int]) -> "ViT":
        """Remove hidden neurons of encoder block `idx`'s MLP: mlp[0] loses
        output neurons, mlp[3] loses input columns. d_model, attention, and
        the residual stream are unchanged."""
        pruned = copy.deepcopy(self)
        mlp = pruned._mlp(idx)
        fc1, fc2 = mlp[0], mlp[3]
        device = fc1.weight.device
        keep = sorted(set(range(fc1.out_features)) - set(indices_to_remove))

        new_fc1 = nn.Linear(fc1.in_features, len(keep)).to(device)
        new_fc1.weight.data = fc1.weight.data[keep].clone()
        new_fc1.bias.data = fc1.bias.data[keep].clone()

        new_fc2 = nn.Linear(len(keep), fc2.out_features).to(device)
        new_fc2.weight.data = fc2.weight.data[:, keep].clone()
        new_fc2.bias.data = fc2.bias.data.clone()

        mlp[0], mlp[3] = new_fc1, new_fc2
        return pruned


@register_model("vit")
def build_vit(
    bundle: DatasetBundle,
    variant: str = "b_16",
    image_size: int | None = None,
    dropout: float = 0.0,
    attention_dropout: float = 0.0,
    pretrained: bool = False,
    weights: str = "IMAGENET1K_V1",
    patch_size: int | None = None,
    num_layers: int | None = None,
    num_heads: int | None = None,
    hidden_dim: int | None = None,
    mlp_dim: int | None = None,
) -> ViT:
    """ViT; sizes via `variant` (b_16/b_32/l_16/l_32/h_14), or override any
    of patch_size/num_layers/num_heads/hidden_dim/mlp_dim for custom scratch
    models (e.g. a small ViT on CIFAR). `image_size` defaults to the
    dataset's spatial size. `pretrained` needs a stock 224px 3-channel
    variant with IMAGENET1K_V1 weights (b_16, b_32, l_16, l_32)."""
    import torchvision.models as tvm

    if variant not in VIT_VARIANTS:
        raise ValueError(f"vit variant must be one of {sorted(VIT_VARIANTS)}, got {variant!r}")
    p, nl, nh, hd, md = VIT_VARIANTS[variant]
    p = patch_size if patch_size is not None else p
    nl = num_layers if num_layers is not None else nl
    nh = num_heads if num_heads is not None else nh
    hd = hidden_dim if hidden_dim is not None else hd
    md = mlp_dim if mlp_dim is not None else md
    if image_size is None:
        image_size = bundle.input_shape[-1] if len(bundle.input_shape) == 3 else 224
    in_channels = bundle.input_shape[0] if len(bundle.input_shape) == 3 else 3
    if in_channels != 3:
        raise ValueError(f"torchvision's ViT stem takes 3-channel input, dataset has {in_channels}")

    net = tvm.VisionTransformer(
        image_size=image_size, patch_size=p, num_layers=nl, num_heads=nh,
        hidden_dim=hd, mlp_dim=md, dropout=dropout,
        attention_dropout=attention_dropout, num_classes=bundle.output_dim,
    )

    if pretrained:
        custom = any(v is not None for v in (patch_size, num_layers, num_heads, hidden_dim, mlp_dim))
        if variant not in PRETRAINED_VARIANTS or custom or image_size != 224:
            raise ValueError(
                "pretrained ViT needs a stock variant in "
                f"{PRETRAINED_VARIANTS} at image_size 224 with no dimension overrides"
            )
        state = dict(tvm.get_model_weights(f"vit_{variant}")[weights].get_state_dict(progress=True))
        if bundle.output_dim != 1000:
            wnids = bundle.extra.get("class_names", [])
            if len(wnids) == bundle.output_dim and all(w in IMAGENET1K_CLASS_INDEX for w in wnids):
                rows = [IMAGENET1K_CLASS_INDEX[w] for w in wnids]
                state["heads.head.weight"] = state["heads.head.weight"][rows]
                state["heads.head.bias"] = state["heads.head.bias"][rows]
            else:
                del state["heads.head.weight"], state["heads.head.bias"]
                print(f"[vit] pretrained head is 1000-way but dataset has {bundle.output_dim} "
                      "classes with no known ImageNet-1k mapping -- head left randomly "
                      "initialized; train or fine-tune before measuring accuracy")
        missing, unexpected = net.load_state_dict(state, strict=False)
        assert not unexpected, f"unexpected checkpoint keys: {unexpected}"
        assert all(k.startswith("heads.") for k in missing), f"missing checkpoint keys: {missing}"

    return ViT(net)
