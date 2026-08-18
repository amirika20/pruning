"""Moment-based rule (z_mean + c*z_std < 0) from a small calibration set,
vs the support rule; and how alive the 'false removals' actually are."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, "/home/amirabbas-kazeminia/Projects/pruning")
from src.data import build_dataset
from src.models import build_model

run = sorted(Path("/home/amirabbas-kazeminia/Projects/pruning/studies/beta_saturation/outputs").glob("beta_sat_mnist_mlp_*"))[-1]
bundle = build_dataset("mnist", data_seed=0, flatten=True, n_samples=2000, train_ratio=0.8)
model = build_model("mlp", bundle, hidden_sizes=[256, 128, 64])
model.load_state_dict(torch.load(run / "models" / "seed_0.pt", map_location="cpu"))

X = bundle.train_ds.tensors[0]
W, b = model.net[0].weight.data, model.net[0].bias.data
Z = X @ W.T + b                              # pre-activations, full data
freq = (Z > 0).float().mean(0)
dead_strict = freq == 0
dead_eps = freq <= 0.01
print(f"full data: strictly dead={dead_strict.sum().item()}, eps-dead (<=1%)={dead_eps.sum().item()}")

g = torch.Generator().manual_seed(0)
for n in (100, 200):
    for c in (3.0, 4.0):
        removed, false_strict, false_eps, worst_freq = [], [], [], 0.0
        for trial in range(20):
            idx = torch.randperm(len(X), generator=g)[:n]
            zs = Z[idx]
            sel = zs.mean(0) + c * zs.std(0) < 0
            removed.append(sel.sum().item())
            false_strict.append((sel & ~dead_strict).sum().item())
            false_eps.append((sel & ~dead_eps).sum().item())
            if sel.any():
                worst_freq = max(worst_freq, freq[sel].max().item())
        print(f"n={n:4d} rule mean+{c}std<0: removed={sum(removed)/20:5.1f} avg | "
              f"false vs strict={sum(false_strict)/20:5.2f} | false vs eps-dead={sum(false_eps)/20:.2f} "
              f"| worst true act_freq among removed={worst_freq:.4f}")
