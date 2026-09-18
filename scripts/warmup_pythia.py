#!/usr/bin/env python
"""Pythia warm-up: fetch the weights, then prove the pipeline end to end on a
small size BEFORE any benchmark cell is submitted.

Three stages, each its own flag so the network-bound part runs on a login node
and the GPU-bound part on a compute node:

    # 0. no network, no weights: a random tiny GPT-NeoX through the whole
    #    pipeline (adapter, hooks, functional MASH, surgery). Runs anywhere.
    python scripts/warmup_pythia.py --synthetic

    # 1. LOGIN NODE (has network): download weights + tokenizer + WikiText.
    #    Files are fetched, never instantiated (see warm_caches.py).
    python scripts/warmup_pythia.py --download 160m 410m
    python scripts/warmup_pythia.py --download 1.4b 2.8b 6.9b        # later

    # 2. COMPUTE NODE (GPU, offline): the real small test on pythia-160m.
    #    Dense perplexity, then MASH (medoid + empirical repair, the Pythia
    #    recipe) and the deletion baselines at 20% of every FFN, with timings.
    sbatch --gres=gpu:1 --mem=32G --time=01:00:00 --cpus-per-task=2 \
        --wrap 'module load python; source activate <env>; python scripts/warmup_pythia.py --test 160m'
    # or, on an interactive GPU shell:
    python scripts/warmup_pythia.py --test 160m

What --test checks, and what a pass means:
  * the adapter's `activation()` IS the network's nonlinearity: the responses
    MASH scores (act(W z + b) at the hooked input) equal the tensor the
    network actually feeds to dense_4h_to_h, to float tolerance;
  * the ReLU-only arms are refused with a message (not silently rectified);
  * 20% per-layer MASH medoid + empirical repair lowers perplexity relative to
    deleting the same units unrepaired, and relative to random deletion with
    the same repair -- the two orderings the OPT results showed, so a
    violation here means the functional path is wrong, not that Pythia differs;
  * the pruned model forwards at the new width and its widths are what was asked.

Caches follow the job scripts: $PRUNING_SCRATCH/cache/{huggingface,torch}.
Set PRUNING_SCRATCH to relocate (e.g. on a laptop). --test sets
HF_HUB_OFFLINE=1 so a cold cache fails loudly instead of hanging.
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

# Line-buffered stdout: under sbatch, plain prints sit in a block buffer and a
# job killed at its time limit leaves an empty-looking log.
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCRATCH = os.environ.setdefault(
    "PRUNING_SCRATCH", "/n/netscratch/pehlevan_lab/Lab/akazeminia/pruning")
os.environ.setdefault("TORCH_HOME", f"{SCRATCH}/cache/torch")
os.environ.setdefault("HF_HOME", f"{SCRATCH}/cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", f"{SCRATCH}/cache/huggingface/datasets")

TOKENIZER = "EleutherAI/pythia-70m"     # one tokenizer for the whole family


# ── stage 1: download ────────────────────────────────────────────────────────

def download(sizes: list[str], deduped: bool, force: bool) -> int:
    from huggingface_hub import HfApi, snapshot_download

    sys.path.insert(0, str(ROOT))
    from scripts.warm_caches import verify_snapshot
    from src.models.pythia import pythia_repo

    for d in ("TORCH_HOME", "HF_HOME"):
        Path(os.environ[d]).mkdir(parents=True, exist_ok=True)
    print(f"HF_HOME={os.environ['HF_HOME']}")
    bad = 0
    for size in sizes:
        repo = pythia_repo(size, deduped)
        print(f"--- {repo}")
        try:
            files = HfApi().list_repo_files(repo)
            fmt = ("*.safetensors" if any(f.endswith(".safetensors") for f in files)
                   else "*.bin")
            snap = snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.model", fmt],
                                     force_download=force)
            good, detail = verify_snapshot(Path(snap))
            if not good:
                raise RuntimeError(f"snapshot incomplete: {detail}; re-run with --force")
            print(f"    {fmt}: {detail}")
        except Exception as exc:                       # noqa: BLE001
            print(f"    FAILED {type(exc).__name__}: {exc}")
            bad += 1
    # tokenizer (shared) + WikiText-2 raw, through the project's own builder so
    # the cache layout is exactly what a job will look for
    print(f"--- tokenizer {TOKENIZER} + WikiText-2")
    try:
        from src.data import build_dataset
        b = build_dataset("wikitext", data_seed=0, tokenizer=TOKENIZER, seq_len=512,
                          n_train=8, n_val=4, n_test=8)
        print(f"    ok: vocab {b.output_dim}, {len(b.train_ds)} train chunks")
    except Exception as exc:                           # noqa: BLE001
        print(f"    FAILED {type(exc).__name__}: {exc}")
        bad += 1
    print(f"\n{'all fetched' if not bad else f'{bad} failed'}")
    return 1 if bad else 0


# ── shared: the pipeline test on one Pythia model ────────────────────────────

def _ppl(model, ds, device, bs=4) -> float:
    import torch
    from src.training.trainer import evaluate

    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)
    loss, _ = evaluate(model.to(device), loader, "causal_lm")
    return math.exp(loss)


def _check_activation_hook(model, x, device) -> float:
    """max |act(W z + b) - (what dense_4h_to_h actually receives)| on layer 0."""
    import torch

    from src.pruning.methods.mash import _layer_inputs

    li = 0
    fc1, fc2, act = model.prunable_layer(li), model.outgoing_module(li), model.activation(li)
    got = []
    h = fc2.register_forward_hook(lambda m, inp, out: got.append(inp[0].detach()))
    try:
        with torch.no_grad():
            model(x.to(device))
    finally:
        h.remove()
    fed = torch.cat(got).reshape(-1, fc1.out_features).double()
    Z = torch.as_tensor(_layer_inputs(model, li, x.to(device), max_rows=10**9),
                        device=fed.device)
    with torch.no_grad():
        mine = act(Z @ fc1.weight.double().T + fc1.bias.double())
    return float((mine - fed).abs().max())


def pipeline_test(model, bundle, device, fraction: float, label: str,
                  n_calib: int, max_rows: int, trained: bool = True) -> int:
    """MASH functional recipe vs its controls at one width; 0 on pass.

    `trained=False` (the synthetic stage) turns the perplexity ORDERING checks
    into information: on random weights every arm sits at ppl ~ vocab and the
    order is noise, so only the mechanics are graded there."""
    import torch

    from src.experiments.sweep import prune_at_fraction
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

    # 1. the activation the adapter declares is the one the network runs
    gap = _check_activation_hook(model, x[:2], device)
    check(gap < 1e-4, f"activation() matches the network (max gap {gap:.1e})")
    check(model.activation(0).__class__.__name__.lower().startswith("gelu"),
          f"activation is GELU ({type(model.activation(0)).__name__})")

    # 2. ReLU-only knobs are refused, not silently rectified
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

    # 3. the recipe vs its controls at `fraction` of every FFN
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
    }
    ppl, secs = {}, {}
    widths0 = [model.prunable_layer(i).weight.shape[0] for i in range(model.n_prunable_layers())]
    for name, (kind, params) in arms.items():
        params = dict(params, n_calib=n_calib)
        if kind == "mash":
            params["max_rows"] = max_rows
        t0 = time.time()
        pruned, _ = prune_at_fraction(model, bundle, kind, params, fraction=fraction,
                                      device=device, n_calib=n_calib)
        secs[name] = time.time() - t0
        widths1 = [pruned.prunable_layer(i).weight.shape[0]
                   for i in range(pruned.n_prunable_layers())]
        ppl[name] = _ppl(pruned, ds, device)
        print(f"  {name:24s} ppl {ppl[name]:9.3f}   {secs[name]:7.1f}s   widths {widths1[0]}/{widths0[0]}")
        if name == "mash_medoid_empirical":
            want = [w - max(0, min(int(round(fraction * (w - 1))), w - 1)) for w in widths0]
            check(widths1 == want, f"pruned widths are {fraction:.0%} narrower")
        del pruned
        if device.type == "cuda":
            torch.cuda.empty_cache()
    check(ppl["mash_medoid_empirical"] < ppl["mash_medoid_none"],
          "empirical repair beats deleting the same units unrepaired", graded=trained)
    check(ppl["mash_medoid_empirical"] < ppl["random_empirical"],
          "MASH selection beats random selection under the same repair", graded=trained)
    check(all(math.isfinite(v) for v in ppl.values()), "every perplexity is finite")
    print(f"\n{label}: {'ALL PASS' if not fails else f'{fails} FAILED'}")
    return fails


# ── stage 0: synthetic (no weights) ──────────────────────────────────────────

def synthetic(device_str: str | None) -> int:
    import torch
    from torch.utils.data import TensorDataset
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    from src.data.registry import DatasetBundle
    from src.models.pythia import Pythia

    torch.manual_seed(0)
    cfg = GPTNeoXConfig(vocab_size=256, hidden_size=64, intermediate_size=256,
                        num_hidden_layers=2, num_attention_heads=4,
                        max_position_embeddings=128, rotary_pct=0.25,
                        use_parallel_residual=True, hidden_act="gelu")
    net = GPTNeoXForCausalLM(cfg).eval()
    model = Pythia(net)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    T = 32
    xs = torch.randint(0, 256, (40, T))
    bundle = DatasetBundle(train_ds=TensorDataset(xs[:32], xs[:32].clone()),
                           val_ds=TensorDataset(xs[32:], xs[32:].clone()),
                           test_ds=None, input_shape=(T,), output_dim=256,
                           task="causal_lm", extra={"tokenizer": None, "seq_len": T})
    print(f"synthetic GPT-NeoX: {cfg.num_hidden_layers} layers x {cfg.intermediate_size} FFN, "
          f"act={type(model.activation(0)).__name__}, device={device}")
    return pipeline_test(model, bundle, device, fraction=0.25, label="synthetic",
                         n_calib=32, max_rows=20000, trained=False)


# ── stage 2: the real small test ─────────────────────────────────────────────

def real_test(size: str, deduped: bool, dtype: str, n_val: int, seq_len: int,
              fraction: float, device_str: str | None) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    import torch

    from src.data import build_dataset
    from src.models import build_model

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()
    bundle = build_dataset("wikitext", data_seed=0, tokenizer=TOKENIZER, seq_len=seq_len,
                           n_train=128, n_val=n_val, n_test=16)
    print(f"data: {len(bundle.train_ds)} calibration chunks x {seq_len} tokens, "
          f"{len(bundle.val_ds)} val chunks   ({time.time() - t0:.1f}s)")
    t0 = time.time()
    model = build_model("pythia", bundle, size=size, pretrained=True, dtype=dtype,
                        deduped=deduped).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: pythia-{size}{'-deduped' if deduped else ''} ({n_params / 1e6:.0f}M params, "
          f"{model.n_prunable_layers()} layers x {model.prunable_layer(0).weight.shape[0]} FFN, "
          f"{dtype}) on {device}   ({time.time() - t0:.1f}s load)")
    if device.type == "cuda":
        print(f"       {torch.cuda.memory_allocated() / 2**30:.1f} GiB resident")
    return pipeline_test(model, bundle, device, fraction=fraction,
                         label=f"pythia-{size}", n_calib=128, max_rows=20000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true",
                    help="stage 0: random tiny GPT-NeoX through the pipeline (no network)")
    ap.add_argument("--download", nargs="*", metavar="SIZE",
                    help="stage 1: fetch these Pythia sizes (+ tokenizer + WikiText)")
    ap.add_argument("--test", metavar="SIZE",
                    help="stage 2: the real small test on this size (offline, GPU)")
    ap.add_argument("--deduped", action="store_true", help="use the -deduped checkpoints")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    ap.add_argument("--force", action="store_true", help="re-download")
    ap.add_argument("--n-val", type=int, default=32, help="validation chunks for --test")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--fraction", type=float, default=0.2,
                    help="per-layer removal fraction for the --test comparison")
    ap.add_argument("--device", default=None, help="override cuda/cpu")
    args = ap.parse_args()
    if not (args.synthetic or args.download is not None or args.test):
        ap.error("pick a stage: --synthetic, --download SIZE..., or --test SIZE")

    rc = 0
    if args.synthetic:
        rc |= synthetic(args.device)
    if args.download is not None:
        sizes = args.download or ["160m"]
        rc |= download(sizes, args.deduped, args.force)
    if args.test:
        rc |= real_test(args.test, args.deduped, args.dtype, args.n_val, args.seq_len,
                        args.fraction, args.device)
    sys.exit(rc)


if __name__ == "__main__":
    main()
