#!/usr/bin/env python
"""Verify this environment can run a benchmark manifest, before anything starts.

    python scripts/check_env.py                                   # all of requirements.txt
    python scripts/check_env.py --manifest configs/benchmark/manifest_medium.txt
    python scripts/check_env.py --resources small medium

A missing optional package is the worst kind of cluster failure: the array
starts, burns its queue slot, and only the cells that happen to need the package
fail -- so `datasets` being absent takes out every OPT cell of a 16-task array
while the CIFAR cells succeed beside them and the run looks half-healthy.

Given a manifest this reports which CELLS would fail and why, so the message is
"12 cells need `datasets`" rather than a ModuleNotFoundError sixteen tasks in.
Exits non-zero when anything required is missing, which is what makes it usable
as a job-script preflight.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "configs" / "benchmark"

# Distribution name -> import name, where they differ.
IMPORT_NAME = {"pyyaml": "yaml", "scikit-learn": "sklearn",
               "pillow": "PIL", "huggingface-hub": "huggingface_hub"}

# What each config ingredient needs on top of the core.
DATA_NEEDS = {"wikitext": ("datasets", "transformers")}
MODEL_NEEDS = {"opt": ("transformers", "huggingface_hub"),
               "vit": ("torchvision",),
               "mobilenet_v2": ("torchvision",),
               "resnet_imagenet": ("torchvision",),
               "resnet_cifar": ("torchvision",)}


def have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def requirement_modules() -> list[str]:
    """Import names for everything requirements.txt asks for."""
    out = []
    req = ROOT / "requirements.txt"
    if not req.exists():
        return out
    for line in req.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        name = line.split("[")[0]
        for sep in (">=", "==", "<=", "~=", ">", "<"):
            name = name.split(sep)[0]
        name = name.strip().lower()
        out.append(IMPORT_NAME.get(name, name.replace("-", "_")))
    return out


def read_manifest(m: str) -> list[Path]:
    """Cells listed in a manifest, or a clean error naming what is missing.

    Resolved against the repo root as well as the cwd: the job scripts cd to
    SLURM_SUBMIT_DIR, but a manifest passed as a repo-relative path should work
    from anywhere. A missing manifest is nearly always a cluster checkout that
    has not been pulled -- say so, rather than raising FileNotFoundError from
    inside pathlib, which is what a preflight exists to avoid.
    """
    for cand in (Path(m), ROOT / m):
        if cand.is_file():
            return [ROOT / x for x in cand.read_text().split() if x.strip()]
    raise SystemExit(
        f"manifest not found: {m}\n"
        f"  looked in: {', '.join(dict.fromkeys( (str(Path(m).resolve()), str(ROOT / m))))}\n"
        f"  if this is a cluster checkout, `git pull`; if it is a generated\n"
        f"  manifest, `python scripts/generate_benchmark_configs.py`")


def cells_from(args) -> list[Path]:
    cells: list[Path] = []
    for m in args.manifest or []:
        cells += read_manifest(m)
    for cls in args.resources or []:
        suffix = f"_{args.tier}" if args.tier else ""
        man = BENCH / f"manifest_{cls}{suffix}.txt"
        if man.exists():
            cells += [ROOT / x for x in man.read_text().split() if x.strip()]
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="*")
    ap.add_argument("--resources", nargs="*",
                    choices=("small", "medium", "large", "xlarge"))
    ap.add_argument("--tier", choices=("headline", "ablation"))
    args = ap.parse_args()

    print(f"python  {sys.version.split()[0]}  ({sys.executable})")

    missing_core = [m for m in ("torch", "numpy", "scipy", "pandas", "yaml")
                    if not have(m)]
    missing_req = sorted({m for m in requirement_modules() if not have(m)})
    for label, mods in (("core", missing_core), ("requirements.txt", missing_req)):
        print(f"{label:<18} {'ok' if not mods else 'MISSING ' + ', '.join(mods)}")

    # Which cells would actually break, and on what
    blockers: dict[str, list[str]] = {}
    cells = cells_from(args)
    for cfg in cells:
        if not cfg.exists():
            continue
        d = yaml.safe_load(cfg.read_text())
        needs = set(DATA_NEEDS.get(d["data"]["kind"], ()))
        needs |= set(MODEL_NEEDS.get(d["model"]["kind"], ()))
        for mod in needs:
            if not have(mod):
                blockers.setdefault(mod, []).append(d["name"])
    if cells:
        print(f"cells inspected    {len(cells)}")
        if blockers:
            for mod, names in sorted(blockers.items()):
                models = sorted({n.split('__')[0] for n in names})
                print(f"  MISSING {mod!r}: {len(names)} cell(s) would fail "
                      f"-- {', '.join(models)}")
        else:
            print("cell requirements  ok")

    bad = missing_core or missing_req or blockers
    if bad:
        need = sorted(set(missing_core) | set(missing_req) | set(blockers))
        print(f"\nfix: mamba activate <env> && pip install -r requirements.txt")
        print(f"     (missing: {', '.join(need)})")
        sys.exit(1)
    print("\nenvironment ok")


if __name__ == "__main__":
    main()
