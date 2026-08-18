"""Does the PCA-summary moment rule work at deeper layers (post-ReLU inputs)?"""
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

X = bundle.train_ds.tensors[0].double()
C = 4.0
A = X
for layer_idx in range(3):
    lin = model.net[2 * layer_idx]
    W, b = lin.weight.data.double(), lin.bias.data.double()
    Z = A @ W.T + b
    freq = (Z > 0).float().mean(0)
    dead_eps = freq <= 0.01
    d = A.shape[1]

    mu = A.mean(0)
    Ac = A - mu
    cov = (Ac.T @ Ac) / (len(A) - 1)
    evals, evecs = torch.linalg.eigh(cov)
    evals, evecs = evals.flip(0).clamp(min=0), evecs.flip(1)

    z_mean = W @ mu + b
    sel_exact = z_mean + C * ((W @ cov) * W).sum(1).sqrt() < 0
    fe = (sel_exact & ~dead_eps).sum().item()
    print(f"layer {layer_idx} (input dim {d}): eps-dead={dead_eps.sum().item()}, "
          f"exact moment rule removed={sel_exact.sum().item()} (false={fe})")

    for k in (10, 30):
        Vk, Lk = evecs[:, :k], evals[:k]
        pw = W @ Vk
        sigma2 = evals[k:].mean()
        var = (pw ** 2 * Lk).sum(1) + sigma2 * (W.norm(dim=1) ** 2 - (pw ** 2).sum(1))
        sel = z_mean + C * var.sqrt() < 0
        false = (sel & ~dead_eps).sum().item()
        worst = freq[sel].max().item() if sel.any() else 0.0
        agree = (sel == sel_exact).float().mean().item()
        print(f"   k={k:3d} (+floor, {evals[:k].sum()/evals.sum():.0%} var): removed={sel.sum().item():3d} "
              f"false={false} worst_freq={worst:.4f} agreement with exact={agree:.1%}")
    A = torch.relu(Z)
