"""Moment rule computed from truncated PCA of the data, with and without the
PPCA residual-variance floor, vs exact moments."""
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
W, b = model.net[0].weight.data.double(), model.net[0].bias.data.double()
Z = X @ W.T + b
freq = (Z > 0).float().mean(0)
dead_eps = freq <= 0.01
C = 4.0  # rule: E[z] + C*std(z) < 0

mu = X.mean(0)
Xc = X - mu
cov = (Xc.T @ Xc) / (len(X) - 1)
evals, evecs = torch.linalg.eigh(cov)           # ascending
evals, evecs = evals.flip(0), evecs.flip(1)

z_mean = W @ mu + b
exact_var = ((W @ cov) * W).sum(1)
sel_exact = z_mean + C * exact_var.sqrt() < 0
print(f"exact moments: removed={sel_exact.sum().item()} "
      f"(false vs eps-dead={(sel_exact & ~dead_eps).sum().item()}) | "
      f"variance explained by top-k: " +
      ", ".join(f"k={k}: {evals[:k].sum()/evals.sum():.1%}" for k in (10, 50, 150)))

for k in (10, 50, 150):
    Vk, Lk = evecs[:, :k], evals[:k]
    proj_w = W @ Vk                              # [H, k]
    var_k = (proj_w ** 2 * Lk).sum(1)            # truncated: only top-k directions
    sigma2 = evals[k:].mean()                    # PPCA floor: avg discarded eigenvalue
    var_ppca = var_k + sigma2 * (W.norm(dim=1) ** 2 - (proj_w ** 2).sum(1))
    for label, var in (("trunc", var_k), ("+floor", var_ppca)):
        sel = z_mean + C * var.sqrt() < 0
        false = (sel & ~dead_eps).sum().item()
        worst = freq[sel].max().item() if sel.any() else 0.0
        print(f"k={k:4d} {label:6s}: removed={sel.sum().item():3d} "
              f"false vs eps-dead={false:3d} worst act_freq removed={worst:.4f}")
