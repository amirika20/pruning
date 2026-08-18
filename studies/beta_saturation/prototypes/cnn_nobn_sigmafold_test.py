"""SigmaFold on a plain CNN WITHOUT BatchNorm (Conv+bias -> ReLU, GAP head).

Companion to cnn_sigmafold_test.py, which found that BN suppresses saturation
(no filter beyond |t| ~ 1.4). Here the same architecture/recipe minus BN tests
whether saturated filters reappear -- and runs the same surgery ladder:
dead-delete, always-on fold (constant goes into the NEXT CONV'S BIAS, exact in
the interior), and redundancy-ranked fold.

Run:  .ml/bin/python studies/beta_saturation/prototypes/cnn_nobn_sigmafold_test.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.data import build_dataset
from src.training.trainer import evaluate, train
from src.config import TrainingConfig

C_CONF = 4.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = Path(__file__).resolve().parent.parent / "outputs" / "cnn_sigmafold" / "nobn_seed_0.pt"


class PlainCNN(nn.Module):
    """Conv(bias)->ReLU blocks -> GAP -> Linear. Same widths as the BN CNN."""

    def __init__(self, hidden_sizes: list[int], input_channels: int, output_dim: int):
        super().__init__()
        chans = [input_channels] + hidden_sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, bias=True)
            for i in range(len(hidden_sizes))
        ])
        self.head = nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, x):
        for conv in self.convs:
            x = torch.relu(conv(x))
        return self.head(x.mean(dim=[2, 3]))

    def prune(self, li: int, remove: list[int]) -> "PlainCNN":
        m2 = copy.deepcopy(self)
        conv = m2.convs[li]
        keep = [j for j in range(conv.out_channels) if j not in set(remove)]
        new = nn.Conv2d(conv.in_channels, len(keep), 3, padding=1, bias=True).to(conv.weight.device)
        new.weight.data = conv.weight.data[keep].clone()
        new.bias.data = conv.bias.data[keep].clone()
        m2.convs[li] = new
        if li < len(m2.convs) - 1:
            nc = m2.convs[li + 1]
            new_nc = nn.Conv2d(len(keep), nc.out_channels, 3, padding=1, bias=True).to(nc.weight.device)
            new_nc.weight.data = nc.weight.data[:, keep].clone()
            new_nc.bias.data = nc.bias.data.clone()
            m2.convs[li + 1] = new_nc
        else:
            h = m2.head
            new_h = nn.Linear(len(keep), h.out_features).to(h.weight.device)
            new_h.weight.data = h.weight.data[:, keep].clone()
            new_h.bias.data = h.bias.data.clone()
            m2.head = new_h
        return m2


torch.manual_seed(0)
bundle = build_dataset("mnist", data_seed=0, flatten=False, n_samples=2000, train_ratio=0.8)
model = PlainCNN([32, 64, 128], 1, bundle.output_dim).to(DEVICE)

if CKPT.exists():
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    print(f"loaded checkpoint {CKPT}")
else:
    tr, va = bundle.loaders(64)
    cfg = TrainingConfig(optimizer="adam", epochs=150, lr=1e-3, batch_size=64,
                         weight_decay=1e-4, log_every=50, device=str(DEVICE))
    train(model, tr, va, cfg, task=bundle.task, desc="plain cnn train")
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT)
model.eval()

val_loader = bundle.loaders(64)[1]
base_loss, base_acc = evaluate(model, val_loader, bundle.task)
print(f"trained plain CNN (no BN): val loss={base_loss:.4f} acc={base_acc:.4f}\n")

Xtr = bundle.train_ds.tensors[0]
n_blocks = len(model.convs)
d_patch = [model.convs[i].weight[0].numel() for i in range(n_blocks)]
stats = [dict(n=0, mu=torch.zeros(d, dtype=torch.float64, device=DEVICE),
              Sg=torch.zeros(d, d, dtype=torch.float64, device=DEVICE),
              act=None, zsum=None, hn=0, hmu=None, hS=None) for d in d_patch]

with torch.no_grad():
    for i0 in range(0, len(Xtr), 256):
        x = Xtr[i0:i0 + 256].to(DEVICE)
        for li in range(n_blocks):
            st = stats[li]
            P = F.unfold(x, 3, padding=1).double().permute(0, 2, 1).reshape(-1, d_patch[li])
            st["mu"] += P.sum(0); st["Sg"] += P.T @ P; st["n"] += len(P)
            z = model.convs[li](x).double()
            a = (z > 0).sum(dim=(0, 2, 3))
            zs = z.sum(dim=(0, 2, 3))
            st["act"] = a if st["act"] is None else st["act"] + a
            st["zsum"] = zs if st["zsum"] is None else st["zsum"] + zs
            st["hn"] += z.shape[0] * z.shape[2] * z.shape[3]
            h = torch.relu(z).permute(0, 2, 3, 1).reshape(-1, z.shape[1])
            hmu, hS = h.sum(0), h.T @ h
            st["hmu"] = hmu if st["hmu"] is None else st["hmu"] + hmu
            st["hS"] = hS if st["hS"] is None else st["hS"] + hS
            x = torch.relu(model.convs[li](x))

margins, freqs = [], []
for li in range(n_blocks):
    st, conv = stats[li], model.convs[li]
    mu = st["mu"] / st["n"]
    Sig = st["Sg"] / st["n"] - torch.outer(mu, mu)
    w = conv.weight.data.double().reshape(conv.out_channels, -1)
    b = conv.bias.data.double()
    Ez = w @ mu + b
    Vz = ((w @ Sig) * w).sum(1).clamp(min=1e-12)
    t = Ez / Vz.sqrt()
    freq = st["act"].double() / st["hn"]
    err = (Ez - st["zsum"] / st["hn"]).abs().max().item()
    margins.append(t); freqs.append(freq)
    wn = w.norm(dim=1)
    print(f"block {li}: {len(t)} filters | max|E[z] pred - emp|={err:.2e} | "
          f"freq==0: {(freq == 0).sum().item()}, freq<=1%: {(freq <= .01).sum().item()}, "
          f"freq>=99%: {(freq >= .99).sum().item()} | t<=-{C_CONF:g}: {(t <= -C_CONF).sum().item()}, "
          f"t>=+{C_CONF:g}: {(t >= C_CONF).sum().item()}")
    print(f"         freq range [{freq.min():.3f}, {freq.max():.3f}] | t range [{t.min():.2f}, {t.max():.2f}] | "
          f"||w|| range [{wn.min():.4f}, {wn.max():.4f}]")
    for j in torch.where(t <= -C_CONF)[0]:
        assert freq[j] <= 0.01, f"unsafe dead call: block {li} filter {j} freq={freq[j]:.3f}"
    for j in torch.where(t >= C_CONF)[0]:
        assert freq[j] >= 0.99, f"unsafe always call: block {li} filter {j} freq={freq[j]:.3f}"
print("all sigma-margin calls consistent with measured activation frequencies\n")


def fold_and_prune(m0, li, R_idx, S_idx, C_mat, c0):
    m2 = copy.deepcopy(m0)
    dev = next(m2.parameters()).device
    C_mat, c0 = C_mat.to(dev), c0.to(dev)
    if li < len(m2.convs) - 1:
        nxt = m2.convs[li + 1]
        K = nxt.weight.data
        for rj, j in enumerate(R_idx):
            for si, s in enumerate(S_idx):
                if C_mat[si, rj] != 0:
                    K[:, s] += float(C_mat[si, rj]) * K[:, j]
        tap_sum = nxt.weight.data[:, R_idx].sum(dim=(2, 3)).double()
        nxt.bias.data += (tap_sum @ c0).float()          # exact in the interior
    else:
        A = m2.head.weight.data.double()
        m2.head.weight.data[:, S_idx] += (A[:, R_idx] @ C_mat.T).float()
        m2.head.bias.data += (A[:, R_idx] @ c0).float()
    return m2.prune(li, [int(j) for j in R_idx])


def consumer_matrix(m0, li):
    if li < len(m0.convs) - 1:
        K = m0.convs[li + 1].weight.data.double()
        return K.permute(0, 2, 3, 1).reshape(-1, K.shape[1])
    return m0.head.weight.data.double()


results = []
for li in range(n_blocks):
    t, freq, st = margins[li], freqs[li], stats[li]
    hmu = st["hmu"] / st["hn"]
    hS = st["hS"] / st["hn"] - torch.outer(hmu, hmu)
    ridge = 1e-8 * torch.trace(hS) / len(hS)
    hS_r = hS + ridge * torch.eye(len(hS), dtype=torch.float64, device=DEVICE)

    dead = torch.where(t <= -C_CONF)[0]
    on = torch.where(t >= C_CONF)[0]
    if len(dead):
        m2 = model.prune(li, [int(j) for j in dead])
        l2, a2 = evaluate(m2, val_loader, bundle.task)
        results.append((li, "dead-delete", len(dead), l2, a2))
    if len(on):
        S_idx = torch.tensor([j for j in range(len(t)) if j not in set(on.tolist())])
        Cm = torch.linalg.solve(hS_r[S_idx][:, S_idx], hS[S_idx][:, on])
        c0 = hmu[on] - Cm.T @ hmu[S_idx]
        ln, an = evaluate(model.prune(li, [int(j) for j in on]), val_loader, bundle.task)
        lf, af = evaluate(fold_and_prune(model, li, on, S_idx, Cm, c0), val_loader, bundle.task)
        results.append((li, "alwayson-naive", len(on), ln, an))
        results.append((li, "alwayson-FOLD", len(on), lf, af))

    A = consumer_matrix(model, li).to(DEVICE)
    resid_1 = 1.0 / torch.linalg.inv(hS_r).diag()
    cost = resid_1 * (A ** 2).sum(0)
    k = max(1, len(t) // 4)
    R_idx = cost.argsort()[:k].cpu()
    S_idx = torch.tensor([j for j in range(len(t)) if j not in set(R_idx.tolist())])
    Cm = torch.linalg.solve(hS_r[S_idx][:, S_idx], hS[S_idx][:, R_idx])
    c0 = hmu[R_idx] - Cm.T @ hmu[S_idx]
    ln, an = evaluate(model.prune(li, [int(j) for j in R_idx]), val_loader, bundle.task)
    lf, af = evaluate(fold_and_prune(model, li, R_idx, S_idx, Cm, c0), val_loader, bundle.task)
    results.append((li, "redund25%-naive", k, ln, an))
    results.append((li, "redund25%-FOLD", k, lf, af))

print(f"{'block':>5} {'surgery':>15} {'n_removed':>9} {'val_loss':>9} {'val_acc':>8}   (original: {base_loss:.4f} / {base_acc:.4f})")
for li, kind, n, l, a in results:
    print(f"{li:>5} {kind:>15} {n:>9} {l:>9.4f} {a:>8.4f}")
