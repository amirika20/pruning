"""LLM-style constraint: can a small random calibration set + margin
reproduce the full-data support decisions safely?"""
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
what = W / W.norm(dim=1, keepdim=True)
beta = b / W.norm(dim=1)
proj = X @ what.T
dead_full = beta < -proj.amax(0)          # ground truth: dead on ALL 1600 points
print(f"ground truth (all {len(X)} points): dead={dead_full.sum().item()}")

g = torch.Generator().manual_seed(0)
for n in (50, 100, 200):
    stats = []
    for trial in range(20):
        idx = torch.randperm(len(X), generator=g)[:n]
        ps = proj[idx]
        pmax, pstd = ps.amax(0), ps.std(0)
        for kappa in (0.0, 0.5):
            sel = beta < -pmax - kappa * pstd
            stats.append((kappa, sel.sum().item(), (sel & ~dead_full).sum().item()))
    for kappa in (0.0, 0.5):
        rows = [(d, f) for k, d, f in stats if k == kappa]
        d_avg = sum(r[0] for r in rows) / len(rows)
        f_avg = sum(r[1] for r in rows) / len(rows)
        f_max = max(r[1] for r in rows)
        print(f"n={n:4d} kappa={kappa}: removed={d_avg:5.1f} avg | false removals avg={f_avg:.2f} worst={f_max}")
