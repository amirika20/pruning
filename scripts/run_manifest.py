#!/usr/bin/env python
"""Run many benchmark cells from a manifest -- the whole-tier entry point.

    # every small cell, sequentially
    python scripts/run_manifest.py --resources small

    # just the headline arms of a class
    python scripts/run_manifest.py --resources small --tier headline

    # an explicit manifest, or a directory of configs, or single configs
    python scripts/run_manifest.py --manifest configs/benchmark/manifest_small.txt
    python scripts/run_manifest.py --config configs/benchmark/generated/mnist_mlp

    # one shard of n -- the same striding the SLURM array uses
    python scripts/run_manifest.py --resources small --shard 3/16

    python scripts/run_manifest.py --resources small --dry-run   # list, don't run

Each cell runs as a SUBPROCESS, so a segfault or an out-of-memory kill costs one
cell rather than the batch, and a cell that fails does not stop the rest -- the
run ends with a summary and a non-zero exit if anything failed. This is the same
loop `scripts/_slurm_body.sh` drives, so a local run and a cluster task execute
identical code.

Sharding is STRIDED (shard i of n takes manifest lines i, i+n, i+2n, ...) rather
than contiguous, because manifests are ordered by model and contiguous blocks
would pile all the expensive models into one shard.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "configs" / "benchmark"

# Same scratch layout the job scripts use, so a local run writes where the
# cluster run would. Override PRUNING_SCRATCH to relocate.
SCRATCH = os.environ.setdefault(
    "PRUNING_SCRATCH", "/n/netscratch/pehlevan_lab/Lab/akazeminia/pruning")

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



def collect(args) -> list[Path]:
    """The cells to run, in manifest order."""
    cells: list[Path] = []
    if args.manifest:
        for m in args.manifest:
            cells += [ROOT / line for line in
                      Path(m).read_text().split() if line.strip()]
    if args.resources:
        for cls in args.resources:
            suffix = f"_{args.tier}" if args.tier else ""
            man = BENCH / f"manifest_{cls}{suffix}.txt"
            if not man.exists():
                print(f"  no {man.name} (a class can legitimately have no cells "
                      f"in a tier)", file=sys.stderr)
                continue
            cells += [ROOT / line for line in man.read_text().split()
                      if line.strip()]
    for c in args.config or []:
        p = Path(c)
        cells += sorted(p.rglob("*.yaml")) if p.is_dir() else [p]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 1 <= i <= n:
            raise SystemExit(f"--shard i/n needs 1 <= i <= n, got {args.shard}")
        cells = cells[i - 1::n]
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="*", help="manifest file(s)")
    ap.add_argument("--resources", nargs="*",
                    choices=("small", "medium", "large", "xlarge"),
                    help="resource class(es), resolved to their manifest")
    ap.add_argument("--tier", choices=("headline", "ablation"),
                    help="with --resources, use that tier's manifest")
    ap.add_argument("--config", nargs="*", help="config file(s) or directory(ies)")
    ap.add_argument("--shard", help="i/n -- run only this strided share")
    ap.add_argument("--grid", type=int, default=16, help="widths per sweep")
    ap.add_argument("--seed", type=int, default=None, help="one seed only")
    ap.add_argument("--out", default=f"{SCRATCH}/results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cells = collect(args)
    if not cells:
        raise SystemExit("nothing to run: pass --resources, --manifest or --config")

    print(f"{len(cells)} cell(s), grid={args.grid}, out={args.out}")
    if args.dry_run:
        for c in cells:
            print(f"  {c.relative_to(ROOT) if c.is_relative_to(ROOT) else c}")
        return

    # Same preflight the job scripts run: better to hear "12 cells need
    # `datasets`" now than a ModuleNotFoundError partway through the batch.
    check = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_env.py")]
        + sum([["--manifest", m] for m in (args.manifest or [])], []),
        cwd=ROOT)
    if check.returncode != 0:
        raise SystemExit("environment check failed; nothing run")

    failed: list[str] = []
    start = time.perf_counter()
    for i, cfg in enumerate(cells, 1):
        rel = cfg.relative_to(ROOT) if cfg.is_relative_to(ROOT) else cfg
        print(f"\n=== [{i}/{len(cells)}] {rel}  "
              f"(t+{time.perf_counter() - start:.0f}s)", flush=True)
        if not cfg.exists():
            print("  no such config", file=sys.stderr)
            failed.append(f"{rel} (missing)")
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "run_sweep.py"),
               "--config", str(cfg), "--grid", str(args.grid), "--out", args.out]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        # A subprocess per cell: an OOM kill or a segfault then costs one cell
        # instead of the batch, and memory is reclaimed between cells.
        if subprocess.run(cmd, cwd=ROOT).returncode == 0:
            print("  ok", flush=True)
        else:
            print("  FAILED (continuing)", file=sys.stderr)
            failed.append(str(rel))

    dt = time.perf_counter() - start
    print(f"\n{len(cells) - len(failed)}/{len(cells)} cells ok in {dt:.0f}s "
          f"({dt / max(len(cells), 1):.1f}s/cell)")
    if failed:
        print("FAILED:", file=sys.stderr)
        for f in failed:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
