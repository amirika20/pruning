#!/usr/bin/env python
"""Post-activation Ward vs stock delta_f on CIFAR-10 ResNet-20.

Two selection arms, identical realization (medoid dictionary, since the CIFAR
ResNets are BatchNorm-paired) and identical repairs:

    stock      the production MASH: sampled delta_f, cluster response rebuilt
               from the summed covector after each merge (pre-activation
               centroid pushed through the ReLU), pairs rescored by a matvec.
    post_ward  studies/post_ward/post_ward.py: same singleton matrix, then
               Lance--Williams on the mass-weighted Ward increments, i.e. exact
               Ward with the centroid taken in response space. No response is
               recomputed inside the loop.

NOTHING IN src/ IS MODIFIED. The post_ward arm swaps the engine class MASH.plan
looks up at call time (mash.MashEngine) for the study subclass, inside a
context manager, and runs the stock sweep harness otherwise.

    python studies/post_ward/run.py                  # 3 seeds, both repairs
    python studies/post_ward/run.py --seeds 0 --repairs ridge
    python studies/post_ward/run.py --report         # tables from what has run

Outputs: studies/post_ward/outputs/<cell>/seed_<s>/{curve.csv,report.json}
and studies/post_ward/RESULTS.md.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import build_dataset                            # noqa: E402
from src.experiments.sweep import sweep_report, sweep_widths  # noqa: E402
from src.models import build_model                            # noqa: E402
from src.reproducibility import seed_everything               # noqa: E402
from studies.post_ward.post_ward import post_ward_selection   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

ARMS = ("stock", "post_ward", "post_src")
REPAIRS = {"ridge": dict(repair="empirical", ridge=1e-2), "none": dict(repair="none")}
# `post_src` is the promoted knob (2026-09-17): the stock method with
# centroid='post' in its params, no monkeypatch. It must reproduce `post_ward`
# bit for bit; run.py --check asserts that on the curves.


def run_cell(depth: int, arm: str, repair: str, seed: int, fractions, device) -> dict:
    name = f"cifar10_resnet{depth}__{arm}_{repair}"
    out = OUT / name / f"seed_{seed}"
    if (out / "report.json").exists():
        return json.loads((out / "report.json").read_text())
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(seed, deterministic=True)
    # The repo's local CIFAR copy; the builder's default cache holds a corrupt
    # tarball and torchvision would re-download.
    bundle = build_dataset("cifar10", data_seed=seed, n_samples=10000,
                           train_ratio=0.8, n_test=10000, root=str(ROOT / ".data" / "cifar"))
    seed_everything(seed, deterministic=True)
    model = build_model("resnet_cifar", bundle, depth=depth, pretrained=True).to(device).eval()

    params = dict(score="delta_f", dictionary="medoid", **REPAIRS[repair])
    if arm == "post_src":
        params["centroid"] = "post"
    t0 = time.perf_counter()
    ctx = post_ward_selection() if arm == "post_ward" else contextlib.nullcontext()
    with ctx:
        curve = sweep_widths(model, bundle, "mash", params, fractions=fractions,
                             device=device, eval_split="test",
                             seed=seed, name=name, arm="mash")
    rep = sweep_report(curve)
    rep.update(cell=name, depth=depth, arm=arm, repair=repair, seed=seed,
               arm_params=params, wall_seconds=time.perf_counter() - t0)
    curve.to_csv(out / "curve.csv", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    logging.info(f"{name} s{seed}: acc0 {rep['acc0']:.4f} auc {rep['auc']:.4f} "
                 f"cap01 {rep['cap01']:.3f} cap02 {rep['cap02']:.3f} "
                 f"plan {rep['plan_seconds']:.1f}s ({rep['wall_seconds']:.0f}s)")
    return rep


def report() -> str:
    rows = []
    for rj in sorted(OUT.glob("*/seed_*/report.json")):
        r = json.loads(rj.read_text())
        c = pd.read_csv(rj.parent / "curve.csv")
        row = dict(depth=r["depth"], arm=r["arm"], repair=r["repair"], seed=r["seed"],
                   acc0=r["acc0"], auc=r["auc"], cap01=r["cap01"], cap02=r["cap02"],
                   cap005=r["cap005"], plan_seconds=r["plan_seconds"])
        for _, x in c[c.fraction > 0].iterrows():
            row[f"acc@{x.fraction:.2f}"] = x.val_acc
        rows.append(row)
    if not rows:
        return "no results yet"
    df = pd.DataFrame(rows)
    acc_cols = sorted(c for c in df.columns if c.startswith("acc@"))
    lines = ["# Post-activation Ward (Lance--Williams) vs stock delta_f", "",
             "CIFAR-10 ResNet-20 (pretrained, no fine-tune), sampled delta_f, medoid",
             "dictionary, 16-point width grid, test split, mean +- std over seeds.",
             "`stock` rebuilds the merged cluster's response from the summed covector",
             "(pre-activation centroid through the ReLU); `post_ward` is exact Ward on",
             "the response rows via Lance--Williams, no response recomputed in the loop.", ""]
    for depth in sorted(df.depth.unique()):
        for repair in ("none", "ridge"):
            d = df[(df.depth == depth) & (df.repair == repair)]
            if d.empty:
                continue
            lines += [f"## ResNet-{depth}, repair = {repair}", "",
                      "| arm | n | AUC | cap 0.5pt | cap 1pt | cap 2pt | plan s | "
                      + " | ".join(acc_cols[:8]) + " |",
                      "|---|---|---|---|---|---|---|" + "---|" * len(acc_cols[:8])]
            for arm in ARMS:
                x = d[d.arm == arm]
                if x.empty:
                    continue

                def ms(col, fmt="{:.3f}"):
                    v = x[col].astype(float)
                    return (fmt.format(v.mean()) + (f" ± {v.std(ddof=0):.3f}" if len(v) > 1 else ""))
                lines.append(f"| {arm} | {len(x)} | {ms('auc')} | {ms('cap005')} | {ms('cap01')} | "
                             f"{ms('cap02')} | {ms('plan_seconds', '{:.1f}')} | "
                             + " | ".join(ms(c, "{:.4f}") for c in acc_cols[:8]) + " |")
            lines.append("")
            w = d[d.arm == "stock"].set_index("seed")
            for arm in ("post_ward", "post_src"):
                x = d[d.arm == arm].set_index("seed")
                common = w.index.intersection(x.index)
                if len(common):
                    dauc = x.loc[common, "auc"] - w.loc[common, "auc"]
                    dacc = {c: (x.loc[common, c] - w.loc[common, c]).mean() for c in acc_cols[:8]}
                    lines.append(f"- {arm} − stock: ΔAUC {dauc.mean():+.4f} (per seed "
                                 + ", ".join(f"{v:+.4f}" for v in dauc) + "); Δacc "
                                 + ", ".join(f"{k[4:]}:{v:+.4f}" for k, v in dacc.items()))
            lines.append("")
    text = "\n".join(lines)
    (HERE / "RESULTS.md").write_text(text)
    df.to_csv(HERE / "results.csv", index=False)
    return text


def check_promoted() -> str:
    """The promoted knob (post_src) must reproduce the study engine (post_ward)
    bit for bit on every curve that both have run."""
    lines, n = [], 0
    for pj in sorted(OUT.glob("*__post_src_*/seed_*/curve.csv")):
        qj = Path(str(pj).replace("__post_src_", "__post_ward_"))
        if not qj.exists():
            continue
        a = pd.read_csv(pj); b = pd.read_csv(qj)
        cols = ["fraction", "units_after", "val_acc", "val_loss"]
        same = np.array_equal(a[cols].to_numpy(), b[cols].to_numpy())
        n += 1
        lines.append(f"{'OK ' if same else 'DIFF'} {pj.parent.parent.name}/{pj.parent.name}")
        if not same:
            raise AssertionError(f"post_src differs from post_ward at {pj}")
    return f"{n} curve(s) checked, promoted centroid='post' == study engine, bit for bit\n" \
        + "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depths", type=int, nargs="*", default=[20])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--repairs", nargs="*", default=list(REPAIRS))
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--report", action="store_true", help="only rebuild RESULTS.md")
    ap.add_argument("--check", action="store_true",
                    help="assert post_src (promoted knob) == post_ward (study engine)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    if args.check:
        print(check_promoted())
        return
    if not args.report:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fractions = list(np.linspace(0.95 / args.grid, 0.95, args.grid))
        for depth in args.depths:
            for seed in args.seeds:
                for repair in args.repairs:
                    for arm in args.arms:
                        run_cell(depth, arm, repair, seed, fractions, device)
    print(report())


if __name__ == "__main__":
    main()
