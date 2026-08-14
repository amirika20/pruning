"""Cross-seed aggregation: mean +- std of the scalar outcomes of each seed."""

from __future__ import annotations

import numpy as np


def _agg(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "values": list(map(float, values))}


def aggregate_seed_results(seed_results: list[dict]) -> dict:
    """Aggregate the per-seed result dicts produced by
    `src.experiments.runner.run_single_seed`."""
    first = seed_results[0]
    is_multiclass = first["task"] == "multiclass"

    final = {
        key: _agg([r["final"][key] for r in seed_results])
        for key in first["final"]
    }

    efficiency = {}
    for stage in ("pruned", "finetuned"):
        if first[stage] is None:  # finetuning skipped (finetune.epochs == 0)
            continue
        efficiency[stage] = {
            key: _agg([r[stage][key] for r in seed_results])
            for key in ("params_reduction_pct", "flops_reduction_pct", "inference_speedup_pct")
        }

    total_removed = _agg([
        sum(layer["total_removed"] for layer in r["pruning_per_layer"])
        for r in seed_results
    ])

    return {
        "n_seeds": len(seed_results),
        "seeds": [r["seed"] for r in seed_results],
        "task": first["task"],
        "is_multiclass": is_multiclass,
        "final": final,
        "total_neurons_removed": total_removed,
        "efficiency": efficiency,
    }
