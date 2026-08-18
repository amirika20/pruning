"""Compare threshold rules for deciding which layer-0 neurons are saturated,
using the saved MNIST checkpoint (seed 0)."""
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

X = bundle.train_ds.tensors[0]                      # [1600, 784]
W, b = model.net[0].weight.data, model.net[0].bias.data
wn = W.norm(dim=1)
what = W / wn[:, None]
beta = b / wn

# Ground truth on the training set
freq = ((X @ W.T + b) > 0).float().mean(0)
dead_true, always_true = freq == 0, freq == 1

# Rule 1: data radius (a single scalar). |beta| > R => provably saturated.
R = X.norm(dim=1).max()
r1_dead, r1_always = beta < -R, beta > R

# Rule 2: per-feature box (= layer-0 IBP)
l, u = X.amin(0), X.amax(0)
s_hi = what.clamp(min=0) @ u + what.clamp(max=0) @ l   # max of what.x over box
s_lo = what.clamp(min=0) @ l + what.clamp(max=0) @ u   # min of what.x over box
r2_dead, r2_always = beta < -s_hi, beta > -s_lo

# Rule 3: directional support of the actual data
proj = X @ what.T                                      # [N, H]
pmin, pmax = proj.amin(0), proj.amax(0)
r3_dead, r3_always = beta < -pmax, beta > -pmin

# Rule 3 + margin: hyperplane clears the support by kappa projection-stds
pstd = proj.std(0)
for kappa in (0.5, 1.0):
    d = (beta < -pmax - kappa * pstd).sum().item()
    a = (beta > -pmin + kappa * pstd).sum().item()
    print(f"rule3 margin kappa={kappa}: dead={d} always={a}")

print(f"\nH = {len(beta)} neurons | ground truth: dead={dead_true.sum().item()} always={always_true.sum().item()}")
print(f"R = max||x|| = {R:.2f} | |beta| range: {beta.abs().min():.4f} .. {beta.abs().max():.4f}")
print(f"box support width (median): [{s_lo.median():.2f}, {s_hi.median():.2f}]")
print(f"data support width (median): [{pmin.median():.2f}, {pmax.median():.2f}]")
print(f"rule1 radius : dead={r1_dead.sum().item()} always={r1_always.sum().item()}")
print(f"rule2 box    : dead={r2_dead.sum().item()} always={r2_always.sum().item()}")
print(f"rule3 support: dead={r3_dead.sum().item()} always={r3_always.sum().item()} "
      f"(matches truth: {bool((r3_dead == dead_true).all() and (r3_always == always_true).all())})")

# Generalization: do train-set-dead neurons stay dead on the val split?
Xv = bundle.val_ds.tensors[0]
freq_val = ((Xv @ W.T + b) > 0).float().mean(0)
for kappa in (0.0, 0.5):
    sel = beta < -pmax - kappa * pstd
    fired = (freq_val[sel] > 0).sum().item()
    print(f"kappa={kappa}: {sel.sum().item()} train-dead neurons, {fired} fire on val")
