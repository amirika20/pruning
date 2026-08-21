"""Per-unit pruning analysis: who was removed, who absorbed whom, and at what cost.

`prune_model` records, per layer, the removed indices (overall and per method),
the merge operations, and whatever per-unit bookkeeping each method exposed
through `PruneDecision.diagnostics`. This module turns that into tables:

    unit_table     one row per (layer, unit): its fate, role, cluster, mass, ...
    layer_table    one row per layer: widths, counts per method, cluster stats
    cluster_table  one row per merge cluster: survivor, members, size, mass
    overlap_table  DO two methods remove the SAME units? Jaccard and overlap at
                   matched size, against the k/H chance baseline

The overlap table is the one that answers "are these methods finding the same
expendable neurons?". Its chance baseline matters: two methods that each remove
half a layer overlap ~50% by accident, so a raw 0.6 is nearly nothing while a
0.6 at 25% removal (chance 0.25) is a strong signal. `excess` reports the gap.

Comparisons are only valid between runs that started from the same trained
weights, so `compare_runs` refuses to proceed on a fingerprint mismatch rather
than quietly producing a meaningless table (see src.reproducibility).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.reproducibility import check_comparable

# per-unit diagnostic keys we promote to columns when a method supplies them
_UNIT_KEYS = ("role", "cluster", "merge_step", "merge_cost", "mass", "eta",
              "act_freq", "energy", "collapsed")


def _layer_entries(report: Sequence[dict]) -> list[dict]:
    return [e for e in report if "layer" in e]


# ── tables ───────────────────────────────────────────────────────────────────

def unit_table(report: Sequence[dict], **meta: Any) -> pd.DataFrame:
    """One row per original unit of every prunable layer.

    Columns: layer, unit, removed (bool), method (which one claimed it, empty
    if kept), plus any per-unit diagnostics the method exposed. `meta` is
    copied onto every row -- pass method/seed/config labels for later grouping.
    """
    rows: list[dict] = []
    for e in _layer_entries(report):
        H = e["neurons_before"]
        claimed: dict[int, str] = {}
        for kind, idxs in e.get("removed_indices_per_method", {}).items():
            for i in idxs:
                claimed.setdefault(int(i), kind)
        # a unit's survivor, when some method merged it away
        absorbed_into: dict[int, int] = {}
        for kind, ops in e.get("merge_ops", {}).items():
            for op in ops:
                absorbed_into[int(op["removed"])] = int(op["survivor"])

        diags = e.get("diagnostics", {}) or {}
        for unit in range(H):
            row: dict[str, Any] = dict(meta)
            row.update(layer=e["layer"], unit=unit,
                       removed=unit in claimed,
                       removed_by=claimed.get(unit, ""),
                       absorbed_into=absorbed_into.get(unit, -1))
            for kind, d in diags.items():
                for key in _UNIT_KEYS:
                    if key in d and len(d[key]) == H:
                        col = key if len(diags) == 1 else f"{kind}.{key}"
                        row[col] = d[key][unit]
            rows.append(row)
    return pd.DataFrame(rows)


def layer_table(report: Sequence[dict], **meta: Any) -> pd.DataFrame:
    """One row per layer: widths, per-method counts, and any layer-level
    scalars the methods reported (certificate, cluster counts, ...)."""
    rows: list[dict] = []
    for e in _layer_entries(report):
        H = e["neurons_before"]
        row: dict[str, Any] = dict(meta)
        row.update(layer=e["layer"], width_before=H,
                   width_after=e["neurons_after"],
                   removed=e["total_removed"],
                   frac_removed=e["total_removed"] / H if H else 0.0)
        for kind, n in e.get("removed_per_method", {}).items():
            row[f"n_{kind}"] = n
        # A unit can leave a layer two ways, and the distinction is the point
        # of this table. Methods that transfer columns say so with MergeOps;
        # methods that synthesize a new hyperplane (MASH) instead report a
        # per-unit `role`, so count both rather than trusting MergeOps alone.
        n_merged = sum(len(ops) for ops in e.get("merge_ops", {}).values())
        roles: list[str] = []
        for d in (e.get("diagnostics", {}) or {}).values():
            r = d.get("role")
            if r is not None and len(r) == H:
                roles.extend(r)
        n_merged = max(n_merged, sum(1 for r in roles if r == "absorbed"))
        row["n_merged"] = n_merged
        row["n_deleted"] = e["total_removed"] - n_merged
        for kind, d in (e.get("diagnostics", {}) or {}).items():
            for k, v in (d.get("_scalars", {}) or {}).items():
                row[f"{kind}.{k}" if len(e["diagnostics"]) > 1 else k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_table(report: Sequence[dict], **meta: Any) -> pd.DataFrame:
    """One row per merge cluster, for methods that expose a `cluster` column.

    Cluster size is the ablation-relevant quantity: a method that only ever
    merges pairs is doing something qualitatively different from one that
    collapses groups of six, even at the same width.
    """
    rows: list[dict] = []
    for e in _layer_entries(report):
        H = e["neurons_before"]
        for kind, d in (e.get("diagnostics", {}) or {}).items():
            if "cluster" not in d:
                continue
            cl = np.asarray(d["cluster"])
            role = np.asarray(d.get("role", ["?"] * H), dtype=object)
            mass = np.asarray(d.get("mass", [np.nan] * H), dtype=float)
            eta = np.asarray(d.get("eta", [np.nan] * H), dtype=float)
            for cid in sorted(set(cl.tolist()) - {-1}):
                mem = np.flatnonzero(cl == cid)
                surv = [i for i in mem if role[i] == "survivor"]
                row = dict(meta)
                row.update(method=kind, layer=e["layer"], cluster=int(cid),
                           size=len(mem),
                           survivor=int(surv[0]) if surv else int(mem[0]),
                           members=",".join(str(int(i)) for i in mem),
                           mass=float(np.nansum(mass[mem])),
                           eta=(float(np.nanmean(eta[mem]))
                                if np.isfinite(eta[mem]).any() else float("nan")))
                rows.append(row)
    return pd.DataFrame(rows)


# ── cross-method comparison ──────────────────────────────────────────────────

def removal_sets(report: Sequence[dict]) -> dict[int, set[int]]:
    return {e["layer"]: set(e.get("removed_indices", []))
            for e in _layer_entries(report)}


def widths(report: Sequence[dict]) -> dict[int, int]:
    return {e["layer"]: e["neurons_before"] for e in _layer_entries(report)}


def overlap_table(reports: dict[str, Sequence[dict]]) -> pd.DataFrame:
    """Pairwise removal-set agreement per layer, against the chance baseline.

    For sets A, B of a layer of width H:
        jaccard  |A n B| / |A u B|
        overlap  |A n B| / min(|A|, |B|)   -- the "same units?" number
        chance   |A||B| / (H min(|A|,|B|)) -- expected `overlap` for random sets
        excess   overlap - chance
    A high `overlap` with a high `chance` says nothing; `excess` is the signal.
    """
    labels = sorted(reports)
    rows: list[dict] = []
    for i, a in enumerate(labels):
        for bl in labels[i + 1:]:
            sa, sb = removal_sets(reports[a]), removal_sets(reports[bl])
            wa = widths(reports[a])
            for layer in sorted(set(sa) & set(sb)):
                A, B, H = sa[layer], sb[layer], wa[layer]
                inter, union = len(A & B), len(A | B)
                denom = min(len(A), len(B))
                chance = (len(A) * len(B) / (H * denom)) if denom and H else 0.0
                ov = inter / denom if denom else float("nan")
                rows.append({
                    "a": a, "b": bl, "layer": layer, "width": H,
                    "n_a": len(A), "n_b": len(B), "n_shared": inter,
                    "jaccard": inter / union if union else float("nan"),
                    "overlap": ov, "chance": chance,
                    "excess": ov - chance if denom else float("nan"),
                })
    return pd.DataFrame(rows)


# ── run loading ──────────────────────────────────────────────────────────────

def load_seed_result(path: str | Path) -> dict:
    """Read one seed's results.json (accepts the seed dir or the file)."""
    p = Path(path)
    if p.is_dir():
        p = p / "results.json"
    return json.loads(p.read_text())


def compare_runs(runs: dict[str, str | Path], strict: bool = True
                 ) -> dict[str, Any]:
    """Load several seed results and compare their pruning decisions.

    `runs` maps a label (e.g. the method name) to a seed directory. Returns
    {'units', 'layers', 'clusters', 'overlap', 'fingerprints', 'problems'}.

    With `strict` (the default) a fingerprint mismatch raises: if two runs did
    not start from the same trained weights then their removal sets are not
    comparable, and a table that looks fine would be misleading.
    """
    results = {k: load_seed_result(v) for k, v in runs.items()}
    fps = {k: r.get("fingerprints", {}) for k, r in results.items()}
    ok, problems = check_comparable(fps)
    if not ok and strict:
        raise ValueError(
            "runs are not comparable:\n  " + "\n  ".join(problems)
            + "\n(pass strict=False to compare anyway)")

    reports = {k: r.get("pruning_per_layer", []) for k, r in results.items()}
    units = pd.concat([unit_table(rp, method=k) for k, rp in reports.items()],
                      ignore_index=True) if reports else pd.DataFrame()
    layers = pd.concat([layer_table(rp, method=k) for k, rp in reports.items()],
                       ignore_index=True) if reports else pd.DataFrame()
    clusters = [cluster_table(rp, run=k) for k, rp in reports.items()]
    clusters = pd.concat([c for c in clusters if len(c)], ignore_index=True) \
        if any(len(c) for c in clusters) else pd.DataFrame()
    return {"units": units, "layers": layers, "clusters": clusters,
            "overlap": overlap_table(reports), "fingerprints": fps,
            "problems": problems}


# ── human-readable summary ───────────────────────────────────────────────────

def format_pruning_detail(report: Sequence[dict], label: str = "") -> str:
    """Per-layer breakdown as text, for run.log and report.txt."""
    lt = layer_table(report)
    if lt.empty:
        return "no prunable layers\n"
    out = [f"Pruning detail{f' [{label}]' if label else ''}", "=" * 64]
    head = f"{'layer':>5}{'before':>8}{'after':>7}{'removed':>9}{'frac':>7}"
    extra = [c for c in ("n_merged", "n_deleted", "n_clusters",
                         "max_cluster_size", "n_dead", "n_always_on")
             if c in lt.columns]
    out.append(head + "".join(f"{c:>17}" for c in extra))
    for _, r in lt.iterrows():
        line = (f"{int(r.layer):>5}{int(r.width_before):>8}"
                f"{int(r.width_after):>7}{int(r.removed):>9}"
                f"{r.frac_removed:>7.3f}")
        for c in extra:
            v = r[c]
            line += f"{'' if pd.isna(v) else (int(v) if float(v).is_integer() else round(float(v), 4)):>17}"
        out.append(line)
    tot_b, tot_a = int(lt.width_before.sum()), int(lt.width_after.sum())
    out.append("-" * 64)
    out.append(f"{'all':>5}{tot_b:>8}{tot_a:>7}{tot_b - tot_a:>9}"
               f"{(tot_b - tot_a) / max(tot_b, 1):>7.3f}")

    ct = cluster_table(report)
    if len(ct):
        sizes = ct["size"].value_counts().sort_index()
        out.append("")
        out.append("cluster sizes: " + "  ".join(
            f"{int(k)}x{int(v)}" for k, v in sizes.items()))
    return "\n".join(out) + "\n"
