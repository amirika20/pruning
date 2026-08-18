"""SigmaFold on a CNN (Conv->BN->ReLU blocks, GAP head) on MNIST.

Tests, per conv block:
  [1] BN folding sanity: z = BN(conv(x)) equals w_eff.patch + b_eff
  [2] detection: sigma-margin t from im2col patch moments vs actual per-filter
      activation frequency over (image, position) pairs
  [3] surgery: remove predicted-dead filters (t <= -c) outright; remove
      predicted-always-on filters (t >= +c) with the regression fold
      (channel mixing folded into the next conv's kernels + next BN's
      running_mean, or into the GAP head for the last block), vs naive.

Run:  .ml/bin/python studies/beta_saturation/prototypes/cnn_sigmafold_test.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.data import build_dataset
from src.models import build_model
from src.training.trainer import evaluate, train
from src.config import TrainingConfig

C_CONF = 4.0          # sigma-margin confidence
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = Path(__file__).resolve().parent.parent / "outputs" / "cnn_sigmafold" / "seed_0.pt"

torch.manual_seed(0)
bundle = build_dataset("mnist", data_seed=0, flatten=False, n_samples=2000, train_ratio=0.8)
model = build_model("cnn", bundle, hidden_sizes=[32, 64, 128], input_channels=1).to(DEVICE)

if CKPT.exists():
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    print(f"loaded checkpoint {CKPT}")
else:
    tr, va = bundle.loaders(64)
    cfg = TrainingConfig(optimizer="adam", epochs=150, lr=1e-3, batch_size=64,
                         weight_decay=1e-4, log_every=50, device=str(DEVICE))
    train(model, tr, va, cfg, task=bundle.task, desc="cnn train")
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT)
model.eval()

val_loader = bundle.loaders(64)[1]
base_loss, base_acc = evaluate(model, val_loader, bundle.task)
print(f"trained CNN: val loss={base_loss:.4f} acc={base_acc:.4f}\n")

Xtr = bundle.train_ds.tensors[0]


def folded_params(block):
    """Conv+BN -> effective hyperplane per filter: w_eff [C_out, C_in*k*k], b_eff [C_out]."""
    W = block.conv.weight.data.double()
    bn = block.bn
    g, beta = bn.weight.data.double(), bn.bias.data.double()
    m, v = bn.running_mean.data.double(), bn.running_var.data.double()
    s = g / (v + bn.eps).sqrt()
    return (W * s[:, None, None, None]).reshape(W.shape[0], -1), beta - s * m, s


def block_inputs(x):
    """Yield the input feature map to each block for a batch x."""
    for blk in model.blocks:
        yield x
        x = blk(x)


# ── pass 1: moments + activation stats per block ─────────────────────────────
n_blocks = len(model.blocks)
d_patch = [model.blocks[i].conv.weight[0].numel() for i in range(n_blocks)]
stats = [dict(n=0, mu=torch.zeros(d, dtype=torch.float64, device=DEVICE),
              Sg=torch.zeros(d, d, dtype=torch.float64, device=DEVICE),
              act=None, zsum=None, hn=0,
              hmu=None, hS=None) for d in d_patch]

with torch.no_grad():
    for i0 in range(0, len(Xtr), 256):
        x = Xtr[i0:i0 + 256].to(DEVICE)
        for li, xin in enumerate(block_inputs(x)):
            blk, st = model.blocks[li], stats[li]
            P = F.unfold(xin, 3, padding=1).double()          # [B, d, L]
            P = P.permute(0, 2, 1).reshape(-1, d_patch[li])   # [B*L, d]
            st["mu"] += P.sum(0); st["Sg"] += P.T @ P; st["n"] += len(P)
            z = blk.bn(blk.conv(xin)).double()                # pre-ReLU [B,C,H,W]
            a = (z > 0).sum(dim=(0, 2, 3))
            zs = z.sum(dim=(0, 2, 3))
            st["act"] = a if st["act"] is None else st["act"] + a
            st["zsum"] = zs if st["zsum"] is None else st["zsum"] + zs
            st["hn"] += z.shape[0] * z.shape[2] * z.shape[3]
            h = torch.relu(z).permute(0, 2, 3, 1).reshape(-1, z.shape[1])  # [B*L, C]
            hmu, hS = h.sum(0), h.T @ h
            st["hmu"] = hmu if st["hmu"] is None else st["hmu"] + hmu
            st["hS"] = hS if st["hS"] is None else st["hS"] + hS

margins, freqs = [], []
for li in range(n_blocks):
    st, blk = stats[li], model.blocks[li]
    mu = st["mu"] / st["n"]
    Sig = st["Sg"] / st["n"] - torch.outer(mu, mu)
    w_eff, b_eff, _ = folded_params(blk)
    Ez = w_eff @ mu + b_eff
    Vz = ((w_eff @ Sig) * w_eff).sum(1).clamp(min=1e-12)
    t = Ez / Vz.sqrt()
    freq = (st["act"].double() / st["hn"])
    emp_Ez = st["zsum"] / st["hn"]
    margins.append(t); freqs.append(freq)

    # [1] BN-fold sanity: predicted E[z] vs empirical mean of BN output
    err = (Ez - emp_Ez).abs().max().item()
    n_dead_pred = (t <= -C_CONF).sum().item()
    n_on_pred = (t >= C_CONF).sum().item()
    print(f"block {li}: {len(t)} filters | max|E[z] pred - emp|={err:.2e} | "
          f"freq==0: {(freq == 0).sum().item()}, freq<=1%: {(freq <= .01).sum().item()}, "
          f"freq>=99%: {(freq >= .99).sum().item()} | t<=-{C_CONF:g}: {n_dead_pred}, t>=+{C_CONF:g}: {n_on_pred}")
    print(f"         freq range [{freq.min():.3f}, {freq.max():.3f}] | "
          f"t range [{t.min():.2f}, {t.max():.2f}] | "
          f"BN gamma range [{blk.bn.weight.data.abs().min():.3f}, {blk.bn.weight.data.abs().max():.3f}]")
    for j in torch.where(t <= -C_CONF)[0]:
        assert freq[j] <= 0.01, f"unsafe dead call: block {li} filter {j} freq={freq[j]:.3f}"
    for j in torch.where(t >= C_CONF)[0]:
        assert freq[j] >= 0.99, f"unsafe always call: block {li} filter {j} freq={freq[j]:.3f}"
print("all sigma-margin calls consistent with measured activation frequencies\n")

# ── surgery ──────────────────────────────────────────────────────────────────

def fold_and_prune(m0, li, R_idx, S_idx, C_mat, c0):
    """Fold removed channels R of block li into survivors S, then prune."""
    m2 = copy.deepcopy(m0)
    dev = next(m2.parameters()).device
    C_mat, c0 = C_mat.to(dev), c0.to(dev)
    if li < len(m2.blocks) - 1:
        nxt = m2.blocks[li + 1]
        K = nxt.conv.weight.data                     # [out, in, 3, 3]
        for rj, j in enumerate(R_idx):
            for si, s in enumerate(S_idx):
                if C_mat[si, rj] != 0:
                    K[:, s] += float(C_mat[si, rj]) * K[:, j]
        # constant c0 through next conv (interior-exact) -> next BN running_mean
        tap_sum = nxt.conv.weight.data[:, R_idx].sum(dim=(2, 3)).double()  # [out, |R|]
        delta = tap_sum @ c0                                               # [out]
        nxt.bn.running_mean.data -= delta.float()
    else:
        head = m2.head
        A = head.weight.data.double()
        head.weight.data[:, S_idx] += (A[:, R_idx] @ C_mat.T).float()
        head.bias.data += (A[:, R_idx] @ c0).float()
    return m2.prune_layer(li, [int(j) for j in R_idx])


def consumer_matrix(m0, li):
    """Outgoing weights of block li's channels, flattened: [*, C]."""
    if li < len(m0.blocks) - 1:
        K = m0.blocks[li + 1].conv.weight.data.double()      # [out, C, 3, 3]
        return K.permute(0, 2, 3, 1).reshape(-1, K.shape[1])  # [out*9, C]
    return m0.head.weight.data.double()                       # [10, C]


results = []
for li in range(n_blocks):
    t, freq, st = margins[li], freqs[li], stats[li]
    hmu = st["hmu"] / st["hn"]
    hS = st["hS"] / st["hn"] - torch.outer(hmu, hmu)

    dead = torch.where(t <= -C_CONF)[0]
    on = torch.where(t >= C_CONF)[0]
    if len(dead):
        m2 = model.prune_layer(li, [int(j) for j in dead])
        l2, a2 = evaluate(m2, val_loader, bundle.task)
        results.append((li, "dead-delete", len(dead), l2, a2))
    if len(on):
        S_idx = torch.tensor([j for j in range(len(t)) if j not in set(on.tolist())])
        ridge = 1e-8 * torch.trace(hS) / len(hS)
        Cm = torch.linalg.solve(hS[S_idx][:, S_idx] + ridge * torch.eye(len(S_idx), dtype=torch.float64, device=DEVICE),
                                hS[S_idx][:, on])
        c0 = hmu[on] - Cm.T @ hmu[S_idx]
        m_naive = model.prune_layer(li, [int(j) for j in on])
        ln, an = evaluate(m_naive, val_loader, bundle.task)
        m_fold = fold_and_prune(model, li, on, S_idx, Cm.cpu(), c0.cpu())
        lf, af = evaluate(m_fold, val_loader, bundle.task)
        results.append((li, "alwayson-naive", len(on), ln, an))
        results.append((li, "alwayson-FOLD", len(on), lf, af))

    # Step-4 redundancy: rank by predicted removal cost (residual var x outgoing
    # energy, one-at-a-time approx), remove the cheapest 25% jointly with the fold.
    A = consumer_matrix(model, li).to(DEVICE)
    ridge = 1e-8 * torch.trace(hS) / len(hS)
    hS_r = hS + ridge * torch.eye(len(hS), dtype=torch.float64, device=DEVICE)
    Prec = torch.linalg.inv(hS_r)
    resid_1 = 1.0 / Prec.diag()                        # residual var of j given all others
    cost = resid_1 * (A ** 2).sum(0)
    k = max(1, len(t) // 4)
    R_idx = cost.argsort()[:k].cpu()
    S_idx = torch.tensor([j for j in range(len(t)) if j not in set(R_idx.tolist())])
    Cm = torch.linalg.solve(hS_r[S_idx][:, S_idx], hS[S_idx][:, R_idx])
    c0 = hmu[R_idx] - Cm.T @ hmu[S_idx]
    m_naive = model.prune_layer(li, [int(j) for j in R_idx])
    ln, an = evaluate(m_naive, val_loader, bundle.task)
    m_fold = fold_and_prune(model, li, R_idx, S_idx, Cm.cpu(), c0.cpu())
    lf, af = evaluate(m_fold, val_loader, bundle.task)
    results.append((li, "redund25%-naive", k, ln, an))
    results.append((li, "redund25%-FOLD", k, lf, af))

print(f"{'block':>5} {'surgery':>15} {'n_removed':>9} {'val_loss':>9} {'val_acc':>8}   (original: {base_loss:.4f} / {base_acc:.4f})")
for li, kind, n, l, a in results:
    print(f"{li:>5} {kind:>15} {n:>9} {l:>9.4f} {a:>8.4f}")
