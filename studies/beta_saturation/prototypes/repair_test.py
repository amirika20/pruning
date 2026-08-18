"""Remove always-on neurons with different repairs; measure error on held-out data.

Repairs compared:
  naive      delete neuron + column, no compensation
  bias-fold  fold a_j * E[h_j] into next bias (constant absorption)
  pairwise   merge each removed neuron into its single best-correlated survivor
  regression joint LS: h_R ~ C^T h_S + c0 from TRAIN moments, fold into A and bias
"""
import copy, sys
from pathlib import Path
import torch
sys.path.insert(0, "/home/amirabbas-kazeminia/Projects/pruning")
from src.data import build_dataset
from src.models import build_model
from src.training.trainer import evaluate

run = sorted(Path("/home/amirabbas-kazeminia/Projects/pruning/studies/beta_saturation/outputs").glob("beta_sat_fashion_mnist_mlp_*"))[-1]
bundle = build_dataset("fashion_mnist", data_seed=0, flatten=True, n_samples=2000, train_ratio=0.8)
model = build_model("mlp", bundle, hidden_sizes=[256, 128, 64])
model.load_state_dict(torch.load(run / "models" / "seed_0.pt", map_location="cpu"))
model.eval()

L = 1                          # prunable layer index (Linear at net[2L]) 
lin, nxt = model.net[2 * L], model.net[2 * (L + 1)]

def acts(X):                   # post-ReLU output of layer L = input to nxt
    with torch.no_grad():
        h = X
        for m in model.net[: 2 * L + 2]:
            h = m(h)
    return h.double()

Xtr = bundle.train_ds.tensors[0]
Xva = bundle.val_ds.tensors[0]
Htr, Hva = acts(Xtr), acts(Xva)
freq = (Htr > 0).float().mean(0)
R = (freq == 1.0).nonzero().flatten()          # strictly always-on on train
S = (freq < 1.0).nonzero().flatten()
print(f"layer {L}: {len(freq)} neurons, always-on={len(R)}, removing them all")

mu = Htr.mean(0)
Hc = Htr - mu
cov = (Hc.T @ Hc) / (len(Htr) - 1)
ridge = 1e-8 * torch.trace(cov) / len(cov)
C = torch.linalg.solve(cov[S][:, S] + ridge * torch.eye(len(S), dtype=torch.float64),
                       cov[S][:, R])           # [|S|, |R|]
c0 = mu[R] - C.T @ mu[S]

A = nxt.weight.data.double()                   # [out, H]
resid_cov = cov[R][:, R] - cov[R][:, S] @ C
pred_mse = torch.einsum("or,rq,oq->", A[:, R], resid_cov, A[:, R]) / A.shape[0]
print(f"predicted next-layer pre-act MSE (train moments): {pred_mse:.6e}")

with torch.no_grad():
    z_full = nxt(Hva.float())

def try_repair(name, dA_S=None, db=None):
    m2 = copy.deepcopy(model)
    n2 = m2.net[2 * (L + 1)]
    if dA_S is not None:
        n2.weight.data[:, S] += dA_S.float()
    if db is not None:
        n2.bias.data += db.float()
    n2.weight.data = n2.weight.data[:, S].clone()   # drop removed columns
    n2.in_features = len(S)
    l2 = m2.net[2 * L]                              # drop removed rows
    l2.weight.data = l2.weight.data[S].clone()
    l2.bias.data = l2.bias.data[S].clone()
    l2.out_features = len(S)
    with torch.no_grad():
        h = Xva
        for mod in m2.net[: 2 * L + 2]:
            h = mod(h)
        z = m2.net[2 * (L + 1)](h)
    mse = ((z - z_full) ** 2).mean().item()
    val_loss, val_acc = evaluate(m2, bundle.loaders(64)[1], bundle.task)
    print(f"{name:11s}: val pre-act MSE={mse:.6e}  val loss={val_loss:.4f}  acc={val_acc:.4f}")
    return mse

base_loss, base_acc = evaluate(model, bundle.loaders(64)[1], bundle.task)
print(f"{'original':11s}: val pre-act MSE={0:.6e}  val loss={base_loss:.4f}  acc={base_acc:.4f}")
try_repair("naive")
try_repair("bias-fold", db=A[:, R] @ mu[R])
# pairwise: each removed -> best-correlated survivor, scale = cov_sj / var_s
dA, db = torch.zeros_like(A[:, S]), torch.zeros(A.shape[0], dtype=torch.float64)
for ji, j in enumerate(R):
    corr = cov[S][:, ji].abs() / (cov[S, S].sqrt() * resid_cov.new_tensor(1))
    s = (cov[S][:, ji].abs() / cov[S, S].clamp(min=1e-12).sqrt()).argmax()
    scale = cov[S[s], j] / cov[S[s], S[s]].clamp(min=1e-12)
    dA[:, s] += scale * A[:, j]
    db += A[:, j] * (mu[j] - scale * mu[S[s]])
try_repair("pairwise", dA_S=dA, db=db)
mse_reg = try_repair("regression", dA_S=A[:, R] @ C.T, db=A[:, R] @ c0)

# optimality spot-check: jitter the regression coefficients -> error must rise
g = torch.Generator().manual_seed(0)
worse = 0
for _ in range(10):
    noise = torch.randn(C.shape, generator=g, dtype=torch.float64) * 0.01
    m = try_repair.__wrapped__ if False else None
    dA_j = A[:, R] @ (C + noise).T
    m2 = copy.deepcopy(model); n2 = m2.net[2 * (L + 1)]
    n2.weight.data[:, S] += dA_j.float(); n2.bias.data += (A[:, R] @ c0).float()
    n2.weight.data = n2.weight.data[:, S].clone()
    l2 = m2.net[2 * L]; l2.weight.data = l2.weight.data[S].clone(); l2.bias.data = l2.bias.data[S].clone()
    with torch.no_grad():
        h = Xva
        for mod in m2.net[: 2 * L + 2]: h = mod(h)
        z = m2.net[2 * (L + 1)](h)
    worse += (((z - z_full) ** 2).mean().item() > mse_reg)
print(f"jittered-coefficient check: {worse}/10 perturbations increased val MSE")
