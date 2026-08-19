#!/usr/bin/env python
"""Phase E: the unified merge-or-delete engine (span-redundancy detection).

At each greedy step the engine compares
  * the cheapest MERGE (expected damage of the pair merge; existing
    GaussianMeasureMerge costs), and
  * the cheapest DELETION: expected damage of removing survivor k given
    optimal global repair over the rest,
        s_k = ||c_k||^2 * r_k^2,   r_k^2 = 1/[K^{-1}]_kk,
    with K the closed-form kernel Gram of the CURRENT survivor units under
    the matched Gaussian (the population form of the OBS score),
and performs the cheaper operation. Deletions capture span redundancy
(always-on affine dependence, staircase combinations) that no pairwise or
activation-agreement mechanism can see; merges capture clump structure that
subset deletion cannot represent. Realization is mean+global (phase_b 4F4):
the global repair projects deleted units onto the survivors' span exactly.

Arms (identical realization and data budget, 128 calibration inputs):
    merge_only    the Phase-B pipeline (baseline)
    delete_only   population-OBS at the data-light tier (new method by itself)
    unified       merge-or-delete

Run: python studies/gram_stability/phase_e_unified.py --run-dir <gram_* dirs>
Self-test: python studies/gram_stability/phase_e_unified.py --selftest
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.data import build_dataset
from src.models import build_model

from studies.gram_stability.merge import TINY, LayerUnits
from studies.gram_stability.functional import GaussianMeasureMerge, relu_cross, relu_self
from studies.gram_stability.phase_b import apply_layer, realize_variant
from studies.gram_stability.phase_c2 import N_CALIB, evaluate, make_engine

STUDY_ROOT = Path(__file__).resolve().parent
ARMS = ["merge_only", "delete_only", "unified"]


class UnifiedEngine(GaussianMeasureMerge):
    """Merge-or-delete greedy loop; records ops as ('m', i, j) / ('d', k)."""

    def __init__(self, units, lo, hi, mu, cov, mode: str = "unified"):
        super().__init__(units, lo, hi, mu, cov)
        self.mode = mode
        self.ops: list[tuple] = []

    # -- kernel Gram of current survivor units (unit gain) -------------------

    def _survivor_kernel(self, idx: np.ndarray) -> np.ndarray:
        D = self._Dg[np.ix_(idx, idx)]
        SD = np.sqrt(np.clip(np.diag(D), 0.0, None))
        z = (self.S_rho[idx] - self._dotmu[idx]) / np.maximum(SD, 1e-300)
        s = SD / np.maximum(self._n[idx], TINY)
        corr = np.clip(D / np.maximum(np.outer(SD, SD), 1e-300), -1.0, 1.0)
        K = np.outer(s, s) * relu_cross(z[:, None], z[None, :], corr)
        np.fill_diagonal(K, s * s * relu_self(z))
        return 0.5 * (K + K.T)

    def _deletion_scores(self, idx: np.ndarray) -> np.ndarray:
        K = self._survivor_kernel(idx)
        lam = 1e-8 * max(np.trace(K) / max(len(K), 1), 1e-30)
        Kinv = np.linalg.inv(K + lam * np.eye(len(K)))
        r2 = 1.0 / np.maximum(np.diag(Kinv), 1e-300)     # Schur residual energy
        wgg = self._Wg[idx, idx]                          # ||c_k||^2 of survivors
        return wgg * r2

    # -- one merge, replicating the base bookkeeping --------------------------

    def _do_merge(self, k: int, l: int) -> None:
        self.members[k].extend(self.members[l])
        self.A[k] += self.A[l]
        self.S_u[k] += self.S_u[l]
        self.S_rho[k] += self.S_rho[l]
        self.S_c[k] += self.S_c[l]
        self.active[l] = False
        self._cost[l, :] = np.inf
        self._cost[:, l] = np.inf
        self._metric_update(k, l)
        others = np.flatnonzero(self.active)
        others = others[others != k]
        new_costs = self._pair_costs(k, others)
        self._cost[k, :] = np.inf
        self._cost[:, k] = np.inf
        self._cost[k, others] = new_costs
        self._cost[others, k] = new_costs

    def unified_step(self) -> dict:
        idx = np.flatnonzero(self.active)
        i, j = np.unravel_index(np.argmin(self._cost), self._cost.shape)
        mcost = float(self._cost[i, j])

        dcost, dk = np.inf, -1
        if self.mode in ("unified", "delete_only") and len(idx) > 1:
            sdel = self._deletion_scores(idx)
            a = int(np.argmin(sdel))
            dcost, dk = float(sdel[a]), int(idx[a])

        if self.mode == "delete_only" or (self.mode == "unified" and dcost < mcost):
            self.active[dk] = False
            self._cost[dk, :] = np.inf
            self._cost[:, dk] = np.inf
            self.ops.append(("d", dk))
            return {"op": "d", "cost": dcost}
        self._do_merge(int(min(i, j)), int(max(i, j)))
        self.ops.append(("m", int(min(i, j)), int(max(i, j))))
        return {"op": "m", "cost": mcost}


# ── replay: ops prefix -> (surviving clusters, deleted units) ────────────────

def replay(H: int, ops: list[tuple], k: int) -> tuple[list[list[int]], set[int]]:
    parent = list(range(H))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    del_roots = []
    for op in ops[:k]:
        if op[0] == "m":
            parent[find(op[2])] = find(op[1])
        else:
            del_roots.append(op[1])
    del_roots = {find(r) for r in del_roots}
    clusters: dict[int, list[int]] = {}
    deleted: set[int] = set()
    for x in range(H):
        r = find(x)
        (deleted.add(x) if r in del_roots else
         clusters.setdefault(r, []).append(x))
    return list(clusters.values()), deleted


# ── experiment ────────────────────────────────────────────────────────────────

def build_unified(arm, model, bundle, li):
    eng, idx_map, frozen = make_engine("func_matched", model, bundle, li)
    X = bundle.train_ds.tensors[0][:N_CALIB]
    with torch.no_grad():
        Xl = model.net[: 2 * li](X).double().numpy()
    u = eng.orig
    return UnifiedEngine(LayerUnits(u.u, u.rho, u.alpha, u.C),
                         eng.x0 - eng.R0, eng.x0 + eng.R0,  # lo/hi unused beyond x0/R0
                         Xl.mean(axis=0), np.atleast_2d(np.cov(Xl.T)),
                         mode=arm), idx_map, frozen


def run_seed(config, run_dir: Path, seed: int, n_ckpt: int = 25) -> list[dict]:
    torch.manual_seed(seed)
    bundle = build_dataset(config.data.kind, data_seed=seed, **config.data.params)
    model = build_model(config.model.kind, bundle, **config.model.params)
    model.load_state_dict(torch.load(run_dir / "models" / f"seed_{seed}.pt",
                                     weights_only=True))
    model = model.cpu().eval()
    L = model.n_prunable_layers()
    Hs = [model.prunable_layer(li).out_features for li in range(L)]
    acc0 = evaluate(model, bundle)

    records = []
    for arm in ARMS:
        opss, del_frac = [], []
        for li in range(L):
            eng, idx_map, frozen = build_unified(arm, model, bundle, li)
            while eng.n_active > 1:
                eng.unified_step()
            opss.append((eng.ops, idx_map, frozen))
            del_frac.append(np.mean([o[0] == "d" for o in eng.ops]))
        for f in np.linspace(0.05, 0.97, n_ckpt):
            cur = model
            removed_total = 0
            for li in range(L):
                ops, idx_map, frozen = opss[li]
                k = min(int(round(f * (Hs[li] - 1))), len(ops))
                removed_total += k
                sub_clusters, sub_deleted = replay(len(idx_map), ops, k)
                clusters = [[int(idx_map[i]) for i in cl] for cl in sub_clusters] \
                    + [[int(fz)] for fz in frozen]
                X = bundle.train_ds.tensors[0][:N_CALIB]
                with torch.no_grad():
                    Xl = cur.net[: 2 * li](X).double().numpy()
                mu, Sigma = Xl.mean(axis=0), np.atleast_2d(np.cov(Xl.T))
                W_rows, biases, cols, _ = realize_variant(
                    cur, li, clusters, "mean+global", mu, Sigma)
                cur = apply_layer(cur, li, W_rows, biases, cols)
            records.append({"seed": seed, "arm": arm,
                            "frac_removed": removed_total / sum(Hs),
                            "val_acc": evaluate(cur, bundle), "acc0": acc0,
                            "del_frac": float(np.mean(del_frac))})
        logging.info(f"  seed {seed} {arm}: deletion share per layer "
                     f"{[f'{d:.2f}' for d in del_frac]}")
    return records


def capacity(g, drop):
    ok = g[g.val_acc >= g.acc0.iloc[0] - drop]
    return float(ok.frac_removed.max()) if len(ok) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", nargs="*", default=None)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    if args.selftest:
        _selftest()
        return

    for rd in args.run_dir:
        rd = Path(rd)
        config = ExperimentConfig.from_yaml(rd / "config.yaml")
        out_dir = STUDY_ROOT / "outputs" / \
            f"e_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True)
        records = []
        for seed in config.seeds:
            logging.info(f"[{config.name}] seed {seed}")
            records.extend(run_seed(config, rd, seed))
        df = pd.DataFrame(records)
        df.to_csv(out_dir / "curves.csv", index=False)
        lines = [f"Phase E merge-or-delete: {config.name}", "=" * 60,
                 "joint equal-fraction, mean+global realization, 128-input budget",
                 f"  {'arm':<14}{'cap@-0.5pt':>12}{'cap@-1pt':>12}{'cap@-2pt':>12}{'del share':>11}"]
        for arm in ARMS:
            sub = df[df.arm == arm]
            caps = {d: np.median([capacity(g, d) for _, g in sub.groupby("seed")])
                    for d in (0.005, 0.01, 0.02)}
            lines.append(f"  {arm:<14}{caps[0.005]:>12.3f}{caps[0.01]:>12.3f}"
                         f"{caps[0.02]:>12.3f}{sub.del_frac.median():>11.2f}")
        report = "\n".join(lines)
        (out_dir / "report.txt").write_text(report)
        logging.info("\n" + report)


# ── self-test: span redundancy is detected and deleted ~for free ─────────────

def _selftest() -> None:
    rng = np.random.default_rng(0)
    d, m = 2, 3
    # three always-on units with DISTINCT orientations (affine space in d=2 is
    # 3-dimensional) plus a fourth always-on unit -> exactly span-redundant;
    # plus a genuine duplicate pair -> merge channel should fire on those.
    angles = [0.1, 1.2, 2.3, 0.7]
    U = np.array([[np.cos(t), np.sin(t)] for t in angles])
    rho = np.array([-5.0, -5.5, -6.0, -5.2])       # gamma >> R0: always-on
    W = U * 2.0
    b = -2.0 * rho
    Wdup = rng.normal(size=(2, d)); Wdup[1] = 1.7 * Wdup[0]
    bdup = np.array([0.3, 0.51]); bdup[1] = 1.7 * bdup[0]
    Wall = np.vstack([W, Wdup]); ball = np.concatenate([b, bdup])
    C = rng.normal(size=(6, m))
    alpha = np.linalg.norm(Wall, axis=1)
    units = LayerUnits(Wall / alpha[:, None], -ball / alpha, alpha, C)
    lo, hi = -np.ones(d), np.ones(d)
    mu, cov = np.zeros(d), np.eye(d) / 3.0

    eng = UnifiedEngine(units, lo, hi, mu, cov, mode="unified")
    r1 = eng.unified_step()
    r2 = eng.unified_step()
    ops = [o[0] for o in eng.ops]
    assert "d" in ops, f"span-redundant always-on unit should be deleted, ops={eng.ops}"
    assert min(r1["cost"], r2["cost"]) < 1e-9, "first ops should be ~free"

    # realization check: after 2 ops, mean+global repair must be ~exact
    clusters, deleted = replay(6, eng.ops, 2)
    X = rng.normal(size=(4000, d)) / np.sqrt(3.0)
    F0 = np.maximum(X @ Wall.T + ball, 0.0) @ C
    # kernel realize by hand: survivors as mean units + global LS on samples
    surv = [c for c in clusters]
    a = units.a
    W_rows, biases = [], []
    for mem in surv:
        mem = np.array(mem)
        S_u = (a[mem, None] * units.u[mem]).sum(axis=0)
        n = np.linalg.norm(S_u)
        W_rows.append(S_u / n); biases.append(-float((a[mem] * units.rho[mem]).sum()) / n)
    W_rows, biases = np.array(W_rows), np.array(biases)
    Hk = np.maximum(X @ W_rows.T + biases, 0.0)
    cols, *_ = np.linalg.lstsq(Hk, F0, rcond=None)
    err = np.linalg.norm(Hk @ cols - F0) / np.linalg.norm(F0)
    assert err < 1e-6, f"span deletion + global repair should be ~exact, err={err:.2e}"
    print("phase_e self-tests passed: span-redundant unit deleted ~free; "
          f"repair exact (rel err {err:.1e}); ops={eng.ops}")


if __name__ == "__main__":
    main()
