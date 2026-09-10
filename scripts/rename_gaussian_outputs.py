#!/usr/bin/env python
"""Rename finished MASH cells to the names they mean after the measure flip.

    python scripts/rename_gaussian_outputs.py --root outputs/benchmark --dry-run
    python scripts/rename_gaussian_outputs.py --root /n/netscratch/.../results

On 2026-09-10 the delta_f / exact_damage scores' measure defaulted to EMPIRICAL
(sample Grams) on every layer type. Until then a Linear layer scored from
closed-form Gaussian moments, so every MASH cell on a fully-connected or mixed
entry whose score is not `cylinder` and whose arm did not set `measure` was
Gaussian-scored -- and its directory name is now the name of a SAMPLED arm.
Left alone, `run_manifest` would count those cells as done and the tables would
mix the two measures under one label.

So, under `--root`, for the FC/mixed entries:

    <entry>__mash_<rest>              -> <entry>__mash_gaussian_<rest>
    <entry>__mash_ridge_<rest>        -> <entry>__mash_ridge_gaussian_<rest>
    <entry>__mash_bias_<rest>         -> <entry>__mash_bias_gaussian_<rest>
    <entry>__mash_nogauge_<rest>      -> <entry>__mash_nogauge_gaussian_<rest>
    <entry>__mash_certified           -> <entry>__mash_certified_gaussian
    <entry>__mash_sampled_<rest>      -> <entry>__mash_<rest>          (they WERE sampled)
    <entry>__mash_ridge_sampled_<rest>-> <entry>__mash_ridge_<rest>

`cylinder`-scored arms use no measure and keep their names. Conv entries keep
every name (they always scored empirically). Inside each moved cell the
`cell` field of report.json / removals.json / dendrogram.json is rewritten, and
the dendrogram's plan_key.measure is set to the measure that produced it
(`gaussian` for the moved cells, `empirical` where it was None on conv), so
plan reuse keys match the new arms exactly. A dendrogram written before plan
keys existed (the OPT-1.3b probe) gets one synthesised from its report's
arm_params, so the 10 h Gaussian plan is reusable by mash_gaussian_*.

IDEMPOTENT BY CONTENT, NOT BY NAME. A cell's measure is read from its own
report.json arm_params (an explicit `measure`) or its dendrogram's stamped
plan_key, so a plain-named cell that is genuinely sampled -- e.g. one this
script itself just renamed from mash_sampled_* -- is recognised and left alone
on the next run. Only a plain-named cell with NO measure on record is taken to
be a pre-flip Gaussian cell. The bias/nogauge/certified renames create names no
arm generates; they are kept as the record of those runs rather than deleted.
"""
from __future__ import annotations

import sys

import argparse
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FC_OR_MIXED = ("mnist_mlp", "fashion_mnist_mlp", "modular_transformer", "mnist_lenet",
               "imagenet_vit_b16", "wikitext_opt125m", "wikitext_opt350m",
               "wikitext_opt1.3b", "wikitext_opt2.7b", "wikitext_opt6.7b", "wikitext_opt13b")
RULES = [  # (regex on the arm part, replacement); first match wins
    (re.compile(r"^mash_ridge_sampled_(.+)$"), r"mash_ridge_\1"),
    (re.compile(r"^mash_sampled_(.+)$"), r"mash_\1"),
    (re.compile(r"^mash_ridge_(?!gaussian_)(.+)$"), r"mash_ridge_gaussian_\1"),
    (re.compile(r"^mash_bias_(?!gaussian_)(.+)$"), r"mash_bias_gaussian_\1"),
    (re.compile(r"^mash_nogauge_(?!gaussian_)(.+)$"), r"mash_nogauge_gaussian_\1"),
    (re.compile(r"^mash_certified$"), r"mash_certified_gaussian"),
    (re.compile(r"^mash_(?!gaussian_|ridge_|bias_|nogauge_|radius_|certified)(.+)$"),
     r"mash_gaussian_\1"),
]


def new_arm_name(arm: str) -> str | None:
    if "cylinder" in arm:                      # no measure enters the cylinder score
        return None
    for pat, repl in RULES:
        if pat.match(arm):
            return pat.sub(repl, arm)
    return None


def recorded_measure(cell_dir: Path) -> str | None:
    """The measure this cell's own files say it was scored with, or None."""
    for seed_dir in sorted(cell_dir.glob("seed_*")):
        try:
            rep = json.loads((seed_dir / "report.json").read_text())
            m = (rep.get("arm_params") or {}).get("measure")
            if m in ("gaussian", "empirical"):
                return m
        except (OSError, ValueError):
            pass
        try:
            dj = json.loads((seed_dir / "dendrogram.json").read_text())
            m = (dj.get("plan_key") or {}).get("measure")
            if m in ("gaussian", "empirical"):
                return m
        except (OSError, ValueError):
            pass
    return None


def synth_plan_key(seed_dir: Path, measure: str) -> dict | None:
    """A plan_key for a dendrogram written before plan keys existed, from the
    report's arm_params -- the same method the sweep would build."""
    try:
        rep = json.loads((seed_dir / "report.json").read_text())
        from src.pruning.registry import build_pruning_method
        key = build_pruning_method(rep["arm"], **(rep.get("arm_params") or {})).plan_key()
        key["measure"] = measure
        return key
    except Exception:                                  # noqa: BLE001
        return None


def patch_json(path: Path, cell: str, measure: str | None) -> None:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    changed = False
    if d.get("cell") not in (None, cell):
        d["cell"] = cell; changed = True
    if measure and isinstance(d.get("plan_key"), dict) and d["plan_key"].get("measure") != measure:
        d["plan_key"]["measure"] = measure; changed = True
    if measure and not isinstance(d.get("plan_key"), dict) and path.name == "dendrogram.json":
        key = synth_plan_key(path.parent, measure)
        if key:
            d["plan_key"] = key; changed = True
    if changed:
        path.write_text(json.dumps(d, indent=2 if path.name == "report.json" else None,
                                   default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="results root holding <entry>__<arm> dirs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)

    moves: list[tuple[Path, Path, str]] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or "__" not in d.name:
            continue
        entry, arm = d.name.split("__", 1)
        if entry not in FC_OR_MIXED:
            # conv: names stand; only stamp the measure into old plan keys
            if not args.dry_run:
                for dj in d.glob("seed_*/dendrogram.json"):
                    patch_json(dj, d.name, "empirical")
            continue
        new = new_arm_name(arm)
        if new is None or new == arm:
            # Already-renamed Gaussian cells still need their dendrograms
            # stamped (or given a synthesised key) so mash_gaussian_* reuses them.
            if "gaussian" in arm and "cylinder" not in arm and not args.dry_run:
                for dj in d.glob("seed_*/dendrogram.json"):
                    patch_json(dj, d.name, "gaussian")
            continue
        on_record = recorded_measure(d)
        if "sampled" in arm:
            measure = "empirical"                    # the name says so
        elif on_record == "empirical":
            continue                                 # a plain name that IS sampled: leave it
        elif on_record == "gaussian" and "gaussian" in arm:
            continue                                 # already renamed and stamped
        else:
            measure = "gaussian"                     # pre-flip cell: no measure on record
        moves.append((d, root / f"{entry}__{new}", measure))

    # ORDER MATTERS: on ViT both <x>__mash_medoid_empirical_delta_f (Gaussian)
    # and <x>__mash_sampled_medoid_empirical_delta_f exist, and the second's
    # target IS the first's current name. Move the Gaussian cells out of the
    # way first, then let the sampled ones take the plain names.
    moves.sort(key=lambda t: "sampled" in t[0].name)
    freed = {a for a, _, _ in moves}
    clashes = [(a, b) for a, b, _ in moves if b.exists() and b not in freed]
    if clashes:
        print("REFUSING: target already exists for")
        for a, b in clashes:
            print(f"  {a.name} -> {b.name}")
        raise SystemExit(1)
    for src, dst, measure in moves:
        print(f"{src.name:60s} -> {dst.name}")
        if args.dry_run:
            continue
        src.rename(dst)
        for seed_dir in dst.glob("seed_*"):
            for name in ("report.json", "removals.json"):
                if (seed_dir / name).exists():
                    patch_json(seed_dir / name, dst.name, None)
            if (seed_dir / "dendrogram.json").exists():
                patch_json(seed_dir / "dendrogram.json", dst.name, measure)
    print(f"\n{len(moves)} cell(s) {'would be ' if args.dry_run else ''}renamed")


if __name__ == "__main__":
    main()
