#!/usr/bin/env python
"""Phase C1: evaluate stopping rules post-hoc on recorded merge trajectories.

Every rule stops a trajectory using only data available at merge time (costs,
certified bound, spectral drift) with ONE global knob theta; the oracle stops
at the last step with layer_rel_err <= tol. Scores per (rule, theta):

    captured   mean over runs of  min(rule_frac / oracle_frac, 1)
    violation  fraction of runs where the rule stopped PAST the oracle
               (i.e. accepted layer error > tol)

The decisive property is transfer: theta* is chosen on the CALIBRATION
datasets (max captured s.t. violation <= 5%) and scored frozen on the TEST
datasets. Sources: Phase A compare_* dirs (rules from cost/bound; both
metrics' arms) and the original gram-stability run dirs (spectral rules
wq_fro / m1_real, ward trajectories).

Usage:
    python studies/gram_stability/stopping_rules.py \
        --calib mnist --test fashion_mnist [--tol 0.05]
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

OUT = Path(__file__).resolve().parent / "outputs"


# ── rules: each returns the STOP STEP index given a trajectory dataframe ─────

def _stop_before(g: pd.DataFrame, signal: np.ndarray, theta: float) -> int:
    bad = np.flatnonzero(signal > theta)
    return int(g["step"].iloc[bad[0] - 1]) if len(bad) and bad[0] > 0 else \
        (int(g["step"].iloc[-1]) if not len(bad) else 0)


def rule_bound(g: pd.DataFrame, theta: float) -> int:
    """2S1: certified bound relative to the layer's output scale (mean
    per-sample ||Z0||, recomputed from checkpoints)."""
    b = g["bound_total"].to_numpy() / g.attrs["z0_scale"]
    return _stop_before(g, b, theta)


def rule_damage(g: pd.DataFrame, theta: float) -> int:
    """2S7: predicted relative damage: sqrt(cumulative selection cost) over
    the layer's output scale."""
    c = np.sqrt(np.maximum(g["cost"].to_numpy(), 0.0).cumsum()) / g.attrs["z0_scale"]
    return _stop_before(g, c, theta)


def rule_elbow(g: pd.DataFrame, kappa: float) -> int:
    """2S4: stop before the first sustained cost jump: cost_t exceeds kappa x
    the running median of positive costs so far (min 5 steps of history)."""
    c = np.maximum(g["cost"].to_numpy(), 0.0)
    hist: list[float] = []
    for i in range(len(c)):
        if len(hist) >= 5 and c[i] > kappa * np.median(hist):
            return int(g["step"].iloc[max(i - 1, 0)])
        if c[i] > 0:
            hist.append(c[i])
    return int(g["step"].iloc[-1])


def rule_frac(g: pd.DataFrame, theta: float) -> int:
    """2S5 baseline: fixed fraction of the layer."""
    n = int(g["step"].iloc[-1])
    return int(round(theta * n))


def rule_drift(col: str):
    """2S2/2S3: relative drift of a tracked property beyond theta."""
    def rule(g: pd.DataFrame, theta: float) -> int:
        x = g[col].to_numpy(dtype=float)
        x0 = x[0]
        if not np.isfinite(x0) or abs(x0) < 1e-30:
            return int(g["step"].iloc[-1])
        drift = np.abs(x / x0 - 1.0)
        bad = np.flatnonzero(drift > theta)
        return int(g["step"].iloc[bad[0] - 1]) if len(bad) and bad[0] > 0 else \
            (int(g["step"].iloc[-1]) if not len(bad) else 0)
    return rule


THETAS = np.geomspace(1e-4, 1.0, 41)
RULES = {
    "bound(2S1)": (rule_bound, np.geomspace(1e-3, 100.0, 51)),
    "cum_damage(2S7)": (rule_damage, THETAS),
    "elbow(2S4)": (rule_elbow, np.geomspace(2, 500, 25)),
    "fixed_frac(2S5)": (rule_frac, np.linspace(0.02, 0.98, 25)),
    "wq_fro(2S2)": (rule_drift("wq_fro"), THETAS),
    "m1_real(2S3)": (rule_drift("m1_real_ratio"), THETAS),
}


# ── evaluation ────────────────────────────────────────────────────────────────

def layer_scales(names: list[str]) -> dict:
    """(dataset, seed, layer) -> mean per-sample ||Z0|| on the val set,
    recomputed from the gram_* checkpoints (cached to json)."""
    import json
    cache = OUT / "layer_scales.json"
    scales = json.loads(cache.read_text()) if cache.exists() else {}
    missing = [n for n in names if not any(k.startswith(f"{n}|") for k in scales)]
    if missing:
        import torch
        from src.config import ExperimentConfig
        from src.data import build_dataset
        from src.models import build_model
        from studies.gram_stability.run_study import make_eval_ctx
        from studies.gram_stability.merge import extract_units
        for name in missing:
            run = sorted(glob.glob(str(OUT / f"gram_{name}_mlp_2026*")))[-1]
            config = ExperimentConfig.from_yaml(Path(run) / "config.yaml")
            for seed in config.seeds:
                torch.manual_seed(seed)
                bundle = build_dataset(config.data.kind, data_seed=seed, **config.data.params)
                model = build_model(config.model.kind, bundle, **config.model.params)
                model.load_state_dict(torch.load(Path(run) / "models" / f"seed_{seed}.pt",
                                                 weights_only=True))
                model = model.cpu().eval()
                for li in range(model.n_prunable_layers()):
                    units, ok = extract_units(model, li)
                    layer = model.prunable_layer(li)
                    W = layer.weight.data.double().numpy()
                    b = layer.bias.data.double().numpy()
                    C = model.outgoing_weights(li).double().numpy()
                    ctx = make_eval_ctx(model, bundle, li, (W[~ok], b[~ok], C[~ok]))
                    scales[f"{name}|{seed}|{li}"] = float(
                        np.linalg.norm(ctx.Z0, axis=1).mean())
        cache.write_text(json.dumps(scales, indent=2))
    return scales


def load_trajectories(names: list[str]) -> list[pd.DataFrame]:
    trajs = []
    scales = layer_scales(names)
    for name in names:
        for pattern, metrics in [
            (f"compare_gram_{name}_mlp_20260818_17*", ["ward", "func_matched"]),
            (f"gram_{name}_mlp_2026*", [None]),  # run_study: spectral cols, ward rule
        ]:
            dirs = sorted(glob.glob(str(OUT / pattern)))
            if not dirs:
                continue
            df = pd.read_csv(Path(dirs[-1]) / "steps.csv")
            if "metric" not in df:
                df["metric"] = "ward_spectral"
                df["cost"] = df["ward_cost"]
            for (m, s, l), g in df.groupby(["metric", "seed", "layer"]):
                if metrics != [None] and m not in metrics:
                    continue
                g = g.sort_values("step").reset_index(drop=True)
                g.attrs.update(dataset=name, source=m,
                               z0_scale=scales[f"{name}|{s}|{l}"])
                trajs.append(g)
    return trajs


def oracle_step(g: pd.DataFrame, tol: float) -> int:
    bad = g.loc[g["layer_rel_err"] > tol, "step"]
    return int(bad.iloc[0]) - 1 if len(bad) else int(g["step"].iloc[-1])


def score(trajs: list[pd.DataFrame], rule, theta: float, tol: float) -> tuple[float, float]:
    captured, violations, n = [], 0, 0
    for g in trajs:
        needed = {"bound_total", "cost"} if rule in (rule_bound, rule_damage) else set()
        o = oracle_step(g, tol)
        if o <= 0:
            continue
        try:
            s = rule(g, theta)
        except KeyError:
            continue
        err_at_s = float(g.loc[g["step"] <= s, "layer_rel_err"].iloc[-1])
        violations += err_at_s > tol
        captured.append(min(s / o, 1.0))
        n += 1
    if n == 0:
        return np.nan, np.nan
    return float(np.mean(captured)), violations / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib", nargs="+", default=["mnist"])
    parser.add_argument("--test", nargs="+", default=["fashion_mnist", "sine", "shape2d"])
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--max-violation", type=float, default=0.05)
    args = parser.parse_args()

    cal = load_trajectories(args.calib)
    tst = load_trajectories(args.test)
    print(f"calibration trajectories: {len(cal)}   test: {len(tst)}   tol={args.tol}\n")
    print(f"{'rule':<18}{'theta*':>10}{'cal capt':>10}{'cal viol':>10}"
          f"{'TEST capt':>11}{'TEST viol':>11}")
    print("-" * 70)
    rows = []
    for name, (rule, thetas) in RULES.items():
        results = [(t, *score(cal, rule, t, args.tol)) for t in thetas]
        ok = [(t, c, v) for t, c, v in results
              if np.isfinite(c) and v <= args.max_violation]
        if not ok:
            print(f"{name:<18}{'--':>10}  (no theta meets violation cap on calibration)")
            continue
        t_star, c_cal, v_cal = max(ok, key=lambda r: r[1])
        c_tst, v_tst = score(tst, rule, t_star, args.tol)
        print(f"{name:<18}{t_star:>10.4g}{c_cal:>10.3f}{v_cal:>10.3f}"
              f"{c_tst:>11.3f}{v_tst:>11.3f}")
        rows.append({"rule": name, "theta": t_star, "cal_captured": c_cal,
                     "cal_violation": v_cal, "test_captured": c_tst,
                     "test_violation": v_tst})
    pd.DataFrame(rows).to_json(OUT / f"stopping_rules_tol{args.tol}.json",
                               orient="records", indent=2)


if __name__ == "__main__":
    main()
