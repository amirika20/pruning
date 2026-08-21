"""LeNet-5 (LeCun et al., 1998), with ReLU in place of the original tanh.

    conv(6, 5x5) -> ReLU -> maxpool2 -> conv(16, 5x5) -> ReLU -> maxpool2
      -> flatten -> fc(120) -> ReLU -> fc(84) -> ReLU -> fc(classes)

Four prunable slots -- the two conv feature maps and the two hidden fully
connected layers -- which makes this the one entry in the suite that exercises
the conv path and the FC path in a single model.

NO BATCHNORM, deliberately, and that is the point of carrying it. BatchNorm
standardizes the pre-activation, so no channel sits entirely on one side of
zero and the saturation detector finds nothing on a BN network; without it,
saturation reappears (the earlier no-BN ablation found ~29% of a deep block's
filters strictly dead). LeNet is therefore where `saturated` has something to
do, and the BatchNorm entries are where it correctly reports nothing.

Spatial arithmetic on 28x28 MNIST with padding=2 on the first conv reproduces
the classic 16x5x5 = 400 flattened features; the flatten width is measured with
a dry forward pass so other input sizes work too.

Run `python -c "from src.models.lenet import _selftest; _selftest()"` for the
prunable-protocol self-tests.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.data.registry import DatasetBundle
from src.models.registry import MergeOp, PrunableModel, register_model


class LeNet(PrunableModel):
    """LeNet-5. Prunable slots: 0 conv1, 1 conv2, 2 fc1, 3 fc2."""

    def __init__(self, input_channels: int = 1, output_dim: int = 10,
                 image_size: int = 28, conv_channels: tuple[int, int] = (6, 16),
                 hidden_sizes: tuple[int, int] = (120, 84)):
        super().__init__()
        c1, c2 = conv_channels
        h1, h2 = hidden_sizes
        self.conv1 = nn.Conv2d(input_channels, c1, 5, padding=2)
        self.conv2 = nn.Conv2d(c1, c2, 5)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        with torch.no_grad():
            probe = torch.zeros(1, input_channels, image_size, image_size)
            flat = self._features(probe).flatten(1).shape[1]
        # Spatial elements per channel after the second pool. A flattened
        # [N, C, H, W] indexes as c*H*W + h*W + w, so each channel owns a
        # CONTIGUOUS block of `spatial` columns of fc1 -- which is what makes
        # the outgoing-weight view below a plain transpose.
        self.spatial = flat // c2
        self.fc1 = nn.Linear(flat, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, output_dim)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        return self.pool(self.relu(self.conv2(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features(x).flatten(1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    # ── prunable protocol ─────────────────────────────────────────────────

    def n_prunable_layers(self) -> int:
        return 4

    def prunable_layer(self, idx: int) -> nn.Module:
        return [self.conv1, self.conv2, self.fc1, self.fc2][idx]

    def prunable_bn(self, idx: int) -> nn.Module | None:
        return None                       # LeNet has no BatchNorm, by design

    def outgoing_module(self, idx: int) -> nn.Module:
        return [self.conv2, self.fc1, self.fc2, self.fc3][idx]

    def outgoing_weights(self, idx: int) -> torch.Tensor:
        """[H*k, fan_out], rows grouped per prunable unit.

        k is the number of columns each unit owns in its consumer: kH*kW for the
        conv1 -> conv2 hop, `spatial` for conv2 -> fc1 (the flatten), and 1 for
        the fully connected hops.
        """
        consumer = self.outgoing_module(idx)
        if isinstance(consumer, nn.Conv2d):
            return consumer.weight.detach().reshape(consumer.out_channels, -1).t()
        return consumer.weight.detach().t()

    def set_outgoing_weights(self, idx: int, new_weights: torch.Tensor) -> "LeNet":
        clone = copy.deepcopy(self)
        consumer = clone.outgoing_module(idx)
        w = new_weights.t()
        if isinstance(consumer, nn.Conv2d):
            w = w.reshape(consumer.weight.shape)
        consumer.weight.data = w.to(consumer.weight.dtype).to(
            consumer.weight.device).contiguous()
        return clone

    def add_outgoing_bias(self, idx: int, bias_delta: torch.Tensor) -> "LeNet":
        clone = copy.deepcopy(self)
        consumer = clone.outgoing_module(idx)
        consumer.bias.data += bias_delta.to(consumer.bias.dtype).to(
            consumer.bias.device)
        return clone

    def merge_outgoing(self, idx: int, merges: list[MergeOp]) -> "LeNet":
        """Fold each removed unit's outgoing weights into its survivor.

        A unit owns a contiguous block of its consumer's input columns (one for
        the FC hops, kH*kW or `spatial` for the conv ones), so the transfer is a
        block-wise add.
        """
        merged = copy.deepcopy(self)
        consumer = merged.outgoing_module(idx)
        W = consumer.weight.data
        block = W.shape[1] // merged.prunable_layer(idx).weight.shape[0] \
            if isinstance(consumer, nn.Linear) else 1
        for op in merges:
            if isinstance(consumer, nn.Conv2d):
                W[:, op.survivor] += op.scale * W[:, op.removed]
            else:
                s, r = op.survivor * block, op.removed * block
                W[:, s:s + block] += op.scale * W[:, r:r + block]
        return merged

    def prune_layer(self, idx: int, indices_to_remove: list[int]) -> "LeNet":
        """Remove units from slot `idx` and fix the consumer's inputs."""
        pruned = copy.deepcopy(self)
        layer = pruned.prunable_layer(idx)
        consumer = pruned.outgoing_module(idx)
        n_out = layer.weight.shape[0]
        keep = sorted(set(range(n_out)) - set(indices_to_remove))
        dev, dt = layer.weight.device, layer.weight.dtype

        if isinstance(layer, nn.Conv2d):
            new = nn.Conv2d(layer.in_channels, len(keep), layer.kernel_size,
                            padding=layer.padding).to(dev).to(dt)
        else:
            new = nn.Linear(layer.in_features, len(keep)).to(dev).to(dt)
        new.weight.data = layer.weight.data[keep].clone()
        new.bias.data = layer.bias.data[keep].clone()

        if isinstance(consumer, nn.Conv2d):
            new_c = nn.Conv2d(len(keep), consumer.out_channels,
                              consumer.kernel_size,
                              padding=consumer.padding).to(dev).to(dt)
            new_c.weight.data = consumer.weight.data[:, keep].clone()
        else:
            # Each surviving unit keeps its whole block of consumer columns --
            # `spatial` of them across the flatten, one otherwise.
            block = consumer.weight.shape[1] // n_out
            cols = [k * block + s for k in keep for s in range(block)]
            new_c = nn.Linear(len(cols), consumer.out_features).to(dev).to(dt)
            new_c.weight.data = consumer.weight.data[:, cols].clone()
        new_c.bias.data = consumer.bias.data.clone()

        for attr, mod in (([("conv1", new), ("conv2", new_c)],
                           [("conv2", new), ("fc1", new_c)],
                           [("fc1", new), ("fc2", new_c)],
                           [("fc2", new), ("fc3", new_c)])[idx]):
            setattr(pruned, attr, mod)
        if idx == 1:
            pruned.spatial = self.spatial          # unchanged by channel removal
        return pruned


@register_model("lenet")
def build_lenet(bundle: DatasetBundle, conv_channels: tuple[int, int] = (6, 16),
                hidden_sizes: tuple[int, int] = (120, 84),
                width_mult: float = 1.0) -> LeNet:
    """LeNet-5. `width_mult` scales every prunable width, for a version wide
    enough that pruning has something to remove (the stock 6/16/120/84 is
    already close to minimal, so capacity numbers on it are pessimistic)."""
    shape = bundle.input_shape
    channels, size = (shape[0], shape[-1]) if len(shape) == 3 else (1, 28)
    m = max(1, int(round(width_mult)))
    return LeNet(input_channels=channels, output_dim=bundle.output_dim,
                 image_size=size,
                 conv_channels=tuple(c * m for c in conv_channels),
                 hidden_sizes=tuple(h * m for h in hidden_sizes))


def _selftest() -> None:  # pragma: no cover
    from torch.utils.data import TensorDataset

    from src.config import PruningConfig
    from src.pruning.surgery import prune_model

    torch.manual_seed(0)
    X = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    bundle = DatasetBundle(train_ds=TensorDataset(X, y),
                           val_ds=TensorDataset(X[:4], y[:4]),
                           input_shape=(1, 28, 28), output_dim=10,
                           task="multiclass")
    net = build_lenet(bundle, width_mult=4).eval()
    widths = [net.prunable_layer(i).weight.shape[0] for i in range(4)]
    assert widths == [24, 64, 480, 336], widths
    # the classic flatten arithmetic: 28 -> pad2 conv5 -> 28 -> pool 14
    #                                    -> conv5 -> 10 -> pool 5, so 5x5 = 25
    assert net.spatial == 25, net.spatial
    with torch.no_grad():
        ref = net(X).double()

    # 1. outgoing_weights round-trips through set_outgoing_weights
    for i in range(4):
        ow = net.outgoing_weights(i)
        back = net.set_outgoing_weights(i, ow)
        with torch.no_grad():
            assert (back(X).double() - ref).abs().max() < 1e-9, f"slot {i} round-trip"

    # 2. every method runs on every slot, and the widths shrink as asked
    arms = [("random", dict(fraction=0.25, seed=0)),
            ("random", dict(fraction=0.25, seed=0, repair="empirical")),
            ("magnitude", dict(fraction=0.25, norm="mass")),
            ("magnitude", dict(fraction=0.25, norm="mass", repair="empirical")),
            ("saturated", dict(mode="dead", criterion="empirical")),
            ("saturated", dict(mode="dead", criterion="margin")),
            ("mash", dict(fraction=0.25, score="cylinder", repair="sum")),
            ("mash", dict(fraction=0.25, score="delta_f", repair="empirical")),
            ("mash", dict(fraction=0.25, dictionary="medoid", repair="empirical")),
            ("mash", dict(fraction=0.25, score="exact_damage", repair="sum")),
            ("mash_certified", dict(tol=0.05)),
            ("osscar", dict(n_remove=4)),
            ("hope", dict(n_remove=4))]
    for kind, params in arms:
        class _M:
            pass
        _M.kind, _M.params = kind, params
        out, _rep = prune_model(net, PruningConfig(methods=[_M()]), bundle,
                                torch.device("cpu"))
        with torch.no_grad():
            out(X)                        # the pruned net must still be wired up
        w = [out.prunable_layer(i).weight.shape[0] for i in range(4)]
        assert all(a <= b for a, b in zip(w, widths)), f"{kind}: widths grew {w}"
        if params.get("fraction") == 0.25:
            assert w == [18, 48, 360, 252], f"{kind}: expected 25% off, got {w}"

    # 3. removing a unit whose outgoing block is zero is exactly free -- this is
    # what checks the flatten block arithmetic, since an off-by-one in the
    # column mapping would corrupt a DIFFERENT channel's weights.
    for slot in range(4):
        probe = copy.deepcopy(net)
        consumer = probe.outgoing_module(slot)
        n_out = probe.prunable_layer(slot).weight.shape[0]
        with torch.no_grad():
            if isinstance(consumer, nn.Conv2d):
                consumer.weight[:, 3] = 0.0
            else:
                block = consumer.weight.shape[1] // n_out
                consumer.weight[:, 3 * block:4 * block] = 0.0
        with torch.no_grad():
            base = probe(X).double()
        gone = probe.prune_layer(slot, [3])
        with torch.no_grad():
            err = (gone(X).double() - base).abs().max().item()
        assert err < 1e-6, f"slot {slot}: zero-fanout removal cost {err:.2e}"

    print("lenet.py self-tests passed:")
    print(f"  LeNet-5 shape {widths}, flatten 5x5 per channel, no BatchNorm")
    print("  outgoing_weights round-trips on all 4 slots (2 conv, 2 fc)")
    print(f"  {len(arms)} method configurations prune all 4 slots correctly")
    print("  zero-fanout removal exact on every slot (flatten blocks correct)")


if __name__ == "__main__":
    _selftest()
