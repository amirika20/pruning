#!/usr/bin/env python
"""Download every pretrained checkpoint the benchmark needs, on a login node.

    python scripts/warm_caches.py                       # every pretrained entry
    python scripts/warm_caches.py --entries wikitext_opt13b

    python scripts/warm_caches.py --datasets             # the DATA
    python scripts/warm_caches.py                       # the WEIGHTS

Compute nodes are frequently network-isolated, and the job scripts export
HF_HUB_OFFLINE=1 so a cold cache fails loudly rather than hanging on a blocked
connection. Run BOTH forms once, from a node with network, before submitting:
the datasets download too (MNIST/Fashion to ~/.cache/mnist, CIFAR to
~/.cache/cifar, WikiText through HuggingFace), and a job that has to fetch one
will fail rather than wait.

Weights land in ~/.cache/torch (torchvision, torch.hub) and
~/.cache/huggingface (OPT + tokenizers). Nothing is trained and no dataset is
touched -- this only fetches model weights, so it is safe to run repeatedly.
"""
from __future__ import annotations

import argparse
import os
import sys

# transformers probes for TensorFlow and Flax backends when it resolves a
# checkpoint, and importing TensorFlow costs seconds of startup, a few hundred MB,
# and a page of absl/oneDNN logging that buries the real output. This project is
# torch-only, so switch the other backends off BEFORE transformers is imported --
# these must be set before the first import to have any effect. TF_CPP_* silences
# TensorFlow's C++ logger in case something else pulls it in anyway.
for _k, _v in (("TRANSFORMERS_NO_TF", "1"), ("TRANSFORMERS_NO_FLAX", "1"),
               ("USE_TF", "0"), ("USE_FLAX", "0"),
               ("TF_CPP_MIN_LOG_LEVEL", "3"), ("TF_ENABLE_ONEDNN_OPTS", "0")):
    os.environ.setdefault(_k, _v)

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUITE = ROOT / "configs" / "benchmark" / "suite.yaml"

# Same cache roots the job scripts export, so a login-node warm is what the
# compute nodes then find. Set PRUNING_SCRATCH to relocate.
SCRATCH = os.environ.setdefault(
    "PRUNING_SCRATCH", "/n/netscratch/pehlevan_lab/Lab/akazeminia/pruning")
os.environ.setdefault("TORCH_HOME", f"{SCRATCH}/cache/torch")
os.environ.setdefault("HF_HOME", f"{SCRATCH}/cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", f"{SCRATCH}/cache/huggingface/datasets")
for _d in ("TORCH_HOME", "HF_HOME"):
    # Tolerated: PRUNING_SCRATCH points at cluster storage that does not exist
    # on a laptop, and failing here would break --help and every dry run.
    try:
        Path(os.environ[_d]).mkdir(parents=True, exist_ok=True)
    except OSError as _exc:
        print(f"note: cannot create {os.environ[_d]} ({_exc.strerror}); "
              f"set PRUNING_SCRATCH to a writable path", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entries", nargs="*", default=None)
    ap.add_argument("--datasets", action="store_true",
                    help="materialize each entry's DATASET instead of its model "
                         "checkpoint (MNIST/Fashion/CIFAR via torchvision, "
                         "WikiText via HuggingFace). ImageNet is skipped -- it is "
                         "read from a local copy and never downloaded.")
    args = ap.parse_args()

    suite = yaml.safe_load(SUITE.read_text())
    if args.datasets:
        # One entry per distinct (kind, params) so a dataset shared by several
        # entries is fetched once.
        seen, todo = set(), []
        for e in suite["entries"]:
            if e["data"]["kind"] == "imagenet":
                continue                       # local copy; nothing to fetch
            key = (e["data"]["kind"],
                   repr(sorted((e["data"].get("params") or {}).items())))
            if key not in seen:
                seen.add(key)
                todo.append(e)
    else:
        todo = [e for e in suite["entries"]
                if (e.get("model", {}).get("params") or {}).get("pretrained")]
    if args.entries:
        todo = [e for e in todo if e["name"] in set(args.entries)]

    print(f"caches: TORCH_HOME={os.environ['TORCH_HOME']}")
    print(f"        HF_HOME={os.environ['HF_HOME']}")
    what = "dataset" if args.datasets else "checkpoint"
    print(f"{len(todo)} {what}(s) to warm\n")
    ok = bad = 0
    for e in todo:
        if args.datasets:
            print(f"--- {e['name']}: {e['data']['kind']}")
            try:
                sys.path.insert(0, str(ROOT))
                from src.data import build_dataset
                build_dataset(e["data"]["kind"], data_seed=0,
                              **(e["data"].get("params") or {}))
                print("    ok")
                ok += 1
            except Exception as exc:            # noqa: BLE001
                print(f"    FAILED {type(exc).__name__}: {exc}")
                bad += 1
            continue
        kind = e["model"]["kind"]
        params = dict(e["model"]["params"])
        print(f"--- {e['name']}: {kind} {params}")
        try:
            if kind == "opt":
                from transformers import AutoTokenizer, OPTForCausalLM

                from src.models.opt import OPT_SIZES
                repo = OPT_SIZES[params["size"]]
                OPTForCausalLM.from_pretrained(repo)
                AutoTokenizer.from_pretrained(repo)
            elif kind == "resnet_cifar":
                import torch
                d = params["depth"]
                torch.hub.load("chenyaofo/pytorch-cifar-models",
                               f"cifar10_resnet{d}", pretrained=True,
                               verbose=False, trust_repo=True)
            elif kind == "resnet_imagenet":
                from torchvision.models import get_model
                get_model(f"resnet{params['depth']}", weights="IMAGENET1K_V1")
            elif kind == "mobilenet_v2":
                from torchvision.models import mobilenet_v2
                mobilenet_v2(weights="IMAGENET1K_V1")
            elif kind == "vit":
                from torchvision.models import get_model
                get_model(f"vit_{params.get('variant', 'b_16')}",
                          weights="IMAGENET1K_V1")
            else:
                print(f"    (no known download path for {kind}; skipped)")
                continue
            print("    ok")
            ok += 1
        except Exception as exc:            # noqa: BLE001 -- report, keep going
            print(f"    FAILED {type(exc).__name__}: {exc}")
            bad += 1
    print(f"\n{ok} warmed, {bad} failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
