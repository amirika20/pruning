"""Does the directional support survive subsampling to high-norm points?"""
import sys, time
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

t0 = time.perf_counter()
proj = X @ what.T
pmax_full = proj.amax(0)
t1 = time.perf_counter()
dead_full = beta < -pmax_full
print(f"full pass: {X.shape[0]} points, {(t1-t0)*1e3:.1f} ms -> dead={dead_full.sum().item()}")

norms = X.norm(dim=1)
for frac in (0.5, 0.2, 0.1):
    k = int(len(X) * frac)
    idx = norms.topk(k).indices
    pmax_sub = (X[idx] @ what.T).amax(0)
    dead_sub = beta < -pmax_sub
    false_dead = (dead_sub & ~dead_full).sum().item()
    # which fraction of per-neuron argmax points does the subset contain?
    argmax_in = torch.isin(proj.argmax(0), idx).float().mean().item()
    print(f"top {int(frac*100)}% by norm: dead={dead_sub.sum().item()} "
          f"(false removals={false_dead}) | argmax point kept for {argmax_in:.0%} of neurons")
