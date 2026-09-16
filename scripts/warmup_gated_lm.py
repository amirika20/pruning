#!/usr/bin/env python
"""Gated-MLP LM (Llama / Qwen / Mistral) warm-up: prove the pipeline on a
random tiny model, fetch the weights, then run the real small test -- BEFORE
any benchmark cell is submitted. Mirrors scripts/warmup_pythia.py.

    # 0. no network, no weights: a random tiny Llama through the whole
    #    pipeline (adapter, gated responses, functional MASH, OSSCAR, surgery).
    python scripts/warmup_gated_lm.py --synthetic

    # 1. LOGIN NODE (has network): weights + tokenizer + WikiText for these
    #    models (short names from src.models.gated_lm.GATED_LM_REPOS or Hub ids).
    #    Llama repos are license-gated: accept it on the Hub and
    #    `huggingface-cli login` first. Qwen and Mistral need no token.
    python scripts/warmup_gated_lm.py --download qwen2.5-0.5b qwen2.5-7b
    python scripts/warmup_gated_lm.py --download llama3.1-8b

    # 2. COMPUTE NODE (GPU, offline): the real small test.
    python scripts/warmup_gated_lm.py --test qwen2.5-0.5b

What --test checks, and what a pass means:
  * the responses MASH scores -- act_fn(gate z) * (up z) at the hooked input --
    equal the tensor the network feeds to down_proj, to float tolerance;
  * the ReLU-only arms are refused with a message (not silently rectified);
  * 20% per-layer MASH medoid + empirical repair lowers perplexity relative to
    deleting the same channels unrepaired and to random deletion under the
    same repair; OSSCAR runs and its perplexity is finite;
  * the pruned model forwards at the new width and its widths are as asked.

Caches follow the job scripts: $PRUNING_SCRATCH/cache/{huggingface,torch}.
--test sets HF_HUB_OFFLINE=1 so a cold cache fails loudly instead of hanging.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

for _k, _v in (("TRANSFORMERS_NO_TF", "1"), ("TRANSFORMERS_NO_FLAX", "1"),
               ("USE_TF", "0"), ("USE_FLAX", "0"),
               ("TF_CPP_MIN_LOG_LEVEL", "3"), ("TF_ENABLE_ONEDNN_OPTS", "0")):
    os.environ.setdefault(_k, _v)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCRATCH = os.environ.setdefault(
    "PRUNING_SCRATCH", "/n/netscratch/pehlevan_lab/Lab/akazeminia/pruning")
os.environ.setdefault("TORCH_HOME", f"{SCRATCH}/cache/torch")
os.environ.setdefault("HF_HOME", f"{SCRATCH}/cache/huggingface")


# ── stage 1: download ────────────────────────────────────────────────────────

def download(names: list[str], force: bool) -> int:
    from huggingface_hub import HfApi, snapshot_download

    from scripts.warm_caches import verify_snapshot
    from src.models.gated_lm import gated_lm_repo

    for d in ("TORCH_HOME", "HF_HOME"):
        Path(os.environ[d]).mkdir(parents=True, exist_ok=True)
    print(f"HF_HOME={os.environ['HF_HOME']}")
    bad = 0
    for name in names:
        repo = gated_lm_repo(name)
        print(f"--- {repo}")
        try:
            files = HfApi().list_repo_files(repo)
            fmt = ("*.safetensors" if any(f.endswith(".safetensors") for f in files)
                   else "*.bin")
            # tokenizer files ride along (*.json, *.txt, *.model, tokenizer.*);
            # the original/ folder of Llama repos holds a second copy of the
            # weights in the reference format and is skipped.
            snap = snapshot_download(repo, force_download=force,
                                     allow_patterns=["*.json", "*.txt", "*.model",
                                                     "*.tiktoken", fmt],
                                     ignore_patterns=["original/*"])
            good, detail = verify_snapshot(Path(snap))
            if not good:
                raise RuntimeError(f"snapshot incomplete: {detail}; re-run with --force")
            print(f"    {fmt}: {detail}")
            from src.data import build_dataset
            b = build_dataset("wikitext", data_seed=0, tokenizer=repo, seq_len=512,
                              n_train=8, n_val=4, n_test=8)
            print(f"    tokenizer + WikiText-2 ok: vocab {b.output_dim}, "
                  f"{len(b.train_ds)} train chunks")
        except Exception as exc:                       # noqa: BLE001
            print(f"    FAILED {type(exc).__name__}: {exc}")
            bad += 1
    print(f"\n{'all fetched' if not bad else f'{bad} failed'}")
    return 1 if bad else 0


# ── the test itself ──────────────────────────────────────────────────────────

def _ppl(model, ds, device, bs=4) -> float:
    import torch
    from src.training.trainer import evaluate

    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)
    loss, _ = evaluate(model.to(device), loader, "causal_lm")
    return math.exp(loss)


def _check_responses(model, x, device) -> float:
    """max |MASH's responses - what down_proj actually receives| on layer 0."""
    import numpy as np
    import torch

    from src.pruning.methods.mash import _layer_inputs, _unit_responses, extract_units

    li = 0
    down = model.outgoing_module(li)
    got = []
    h = down.register_forward_hook(lambda m, inp, out: got.append(inp[0].detach()))
    try:
        with torch.no_grad():
            model(x.to(device))
    finally:
        h.remove()
    fed = torch.cat(got).reshape(-1, down.in_features).double().cpu().numpy()   # [N, H]
    Z = _layer_inputs(model, li, x.to(device), max_rows=10**9)
    units, ok = extract_units(model, li, Z=Z)
    Phi = _unit_responses(Z, units.u, -units.rho, units.alpha, model.activation(li),
                          idx=np.arange(len(ok))).cpu().numpy().T * units.alpha
    # RELATIVE gap: the network's tensor is the bf16/fp16 product of two half-
    # precision matmuls, MASH's is float64 from the captured input, and Qwen's
    # responses reach tens -- an absolute 5e-2 there is half-precision rounding.
    return float(np.abs(Phi - fed).max() / max(np.abs(fed).max(), 1e-12))


def pipeline_test(model, bundle, device, fraction: float, label: str,
                  n_calib: int, max_rows: int, trained: bool = True) -> int:
    import torch

    from src.experiments.sweep import prune_at_fraction
    from src.models.registry import GatedActivation
    from src.pruning.methods.mash import MASH
    from src.pruning.registry import PruneContext

    x = bundle.train_ds.tensors[0]
    fails = 0

    def check(cond, msg, graded=True):
        nonlocal fails
        if not graded:
            print(("  info " if cond else "  info (not graded on random weights) ") + msg)
            return
        print(("  PASS " if cond else "  FAIL ") + msg)
        fails += 0 if cond else 1

    gap = _check_responses(model, x[:2], device)
    half = model.prunable_layer(0).weight.dtype in (torch.float16, torch.bfloat16)
    tol = 2e-2 if half else 1e-4      # bf16 keeps ~3 digits; fp32 ~7
    check(gap < tol, f"gated responses match what down_proj receives "
                     f"(max relative gap {gap:.1e}, tol {tol:.0e} for "
                     f"{model.prunable_layer(0).weight.dtype})")
    check(isinstance(model.activation(0), GatedActivation),
          f"activation is gated ({model.activation(0)!r})")

    ctx = PruneContext(train_inputs=x[:n_calib].to(device), bundle=bundle, device=device)
    for bad in (dict(dictionary="merge"), dict(score="cylinder"), dict(measure="gaussian"),
                dict(repair="kernel")):
        try:
            MASH(n_remove=4, n_calib=n_calib, max_rows=max_rows, **bad).select(model, 0, ctx)
        except NotImplementedError as exc:
            check("not ReLU" in str(exc), f"{bad} refused: {str(exc)[:60]}...")
        else:
            check(False, f"{bad} was NOT refused")
    probe = MASH(n_remove=4, n_calib=n_calib, max_rows=max_rows)
    probe.select(model, 0, ctx)
    check((probe.dictionary, probe.repair) == ("medoid", "empirical"),
          f"auto defaults -> medoid + empirical (got {probe.dictionary}, {probe.repair})")

    ds = bundle.val_ds
    t0 = time.time()
    dense = _ppl(model, ds, device)
    print(f"  dense ppl {dense:.3f}   ({time.time() - t0:.1f}s eval, {len(ds)} chunks)")
    arms = {
        "mash_medoid_none":      ("mash", dict(dictionary="medoid", repair="none")),
        "mash_medoid_sum":       ("mash", dict(dictionary="medoid", repair="sum")),
        "mash_medoid_empirical": ("mash", dict(dictionary="medoid", repair="empirical", ridge=1e-2)),
        "random_empirical":      ("random", dict(seed=0, repair="empirical", ridge=1e-2)),
        "magnitude_empirical":   ("magnitude", dict(norm="mass", repair="empirical", ridge=1e-2)),
        "osscar":                ("osscar", dict()),
    }
    ppl, secs = {}, {}
    widths0 = [model.prunable_layer(i).weight.shape[0] for i in range(model.n_prunable_layers())]
    for name, (kind, params) in arms.items():
        if kind != "osscar":                 # OSSCAR reads every calibration token
            params = dict(params, n_calib=n_calib)
        if kind == "mash":
            params["max_rows"] = max_rows
        t0 = time.time()
        pruned, _ = prune_at_fraction(model, bundle, kind, params, fraction=fraction,
                                      device=device, n_calib=n_calib)
        secs[name] = time.time() - t0
        widths1 = [pruned.prunable_layer(i).weight.shape[0]
                   for i in range(pruned.n_prunable_layers())]
        gates1 = [pruned._mlp(i).gate_proj.out_features for i in range(pruned.n_prunable_layers())]
        ppl[name] = _ppl(pruned, ds, device)
        print(f"  {name:24s} ppl {ppl[name]:9.3f}   {secs[name]:7.1f}s   widths {widths1[0]}/{widths0[0]}")
        if name == "mash_medoid_empirical":
            want = [w - max(0, min(int(round(fraction * (w - 1))), w - 1)) for w in widths0]
            check(widths1 == want, f"pruned widths are {fraction:.0%} narrower")
            check(gates1 == widths1, "gate_proj and up_proj were sliced together")
        del pruned
        if device.type == "cuda":
            torch.cuda.empty_cache()
    check(ppl["mash_medoid_empirical"] < ppl["mash_medoid_none"],
          "empirical repair beats deleting the same channels unrepaired", graded=trained)
    check(ppl["mash_medoid_empirical"] < ppl["random_empirical"],
          "MASH selection beats random selection under the same repair", graded=trained)
    check(all(math.isfinite(v) for v in ppl.values()), "every perplexity is finite")
    print(f"\n{label}: {'ALL PASS' if not fails else f'{fails} FAILED'}")
    return fails


# ── stage 0: synthetic ───────────────────────────────────────────────────────

def synthetic(device_str: str | None) -> int:
    import torch
    from torch.utils.data import TensorDataset

    from src.data.registry import DatasetBundle
    from src.models.gated_lm import tiny_gated_lm

    model = tiny_gated_lm(vocab=256, d=64, ffn=256, layers=2)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    T = 32
    xs = torch.randint(0, 256, (40, T))
    bundle = DatasetBundle(train_ds=TensorDataset(xs[:32], xs[:32].clone()),
                           val_ds=TensorDataset(xs[32:], xs[32:].clone()),
                           test_ds=None, input_shape=(T,), output_dim=256,
                           task="causal_lm", extra={"tokenizer": None, "seq_len": T})
    print(f"synthetic Llama: {model.n_prunable_layers()} layers x "
          f"{model.prunable_layer(0).out_features} FFN, act={model.activation(0)!r}, device={device}")
    return pipeline_test(model, bundle, device, fraction=0.25, label="synthetic",
                         n_calib=32, max_rows=20000, trained=False)


# ── stage 2: the real small test ─────────────────────────────────────────────

def real_test(name: str, dtype: str, n_val: int, seq_len: int, fraction: float,
              device_str: str | None) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    import torch

    from src.data import build_dataset
    from src.models import build_model
    from src.models.gated_lm import gated_lm_repo

    repo = gated_lm_repo(name)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()
    bundle = build_dataset("wikitext", data_seed=0, tokenizer=repo, seq_len=seq_len,
                           n_train=128, n_val=n_val, n_test=16)
    print(f"data: {len(bundle.train_ds)} calibration chunks x {seq_len} tokens, "
          f"{len(bundle.val_ds)} val chunks   ({time.time() - t0:.1f}s)")
    t0 = time.time()
    model = build_model("gated_lm", bundle, repo=repo, pretrained=True, dtype=dtype).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {repo} ({n_params / 1e6:.0f}M params, {model.n_prunable_layers()} layers x "
          f"{model.prunable_layer(0).out_features} FFN, {dtype}) on {device}   "
          f"({time.time() - t0:.1f}s load)")
    if device.type == "cuda":
        print(f"       {torch.cuda.memory_allocated() / 2**30:.1f} GiB resident")
    return pipeline_test(model, bundle, device, fraction=fraction, label=repo,
                         n_calib=128, max_rows=20000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--download", nargs="*", metavar="NAME")
    ap.add_argument("--test", metavar="NAME")
    ap.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if not (args.synthetic or args.download is not None or args.test):
        ap.error("pick a stage: --synthetic, --download NAME..., or --test NAME")
    rc = 0
    if args.synthetic:
        rc |= synthetic(args.device)
    if args.download is not None:
        rc |= download(args.download or ["qwen2.5-0.5b"], args.force)
    if args.test:
        rc |= real_test(args.test, args.dtype, args.n_val, args.seq_len, args.fraction,
                        args.device)
    sys.exit(rc)


if __name__ == "__main__":
    main()
