"""Human-readable end-of-run report (report.txt).

One report per seed (seeds/seed_<s>/report.txt) plus a cross-seed one at the
experiment root: pruning per layer, efficiency before/after (params, FLOPs,
model size, inference time), val/test performance at every stage (trained ->
pruned -> fine-tuned), and the performance deltas pruning and fine-tuning
caused. Numbers are shown as `mean +- std` when aggregating multiple seeds.
"""

from __future__ import annotations

import numpy as np

from src.config import ExperimentConfig

_W = 72  # report width


class _Agg:
    """A scalar across seeds: renders plain for one seed, mean +- std for many."""

    def __init__(self, values: list):
        self.values = [v for v in values if v is not None]

    @property
    def mean(self) -> float | None:
        return float(np.mean(self.values)) if self.values else None

    def fmt(self, spec: str = ",.0f", suffix: str = "") -> str:
        if not self.values:
            return "-"
        mean = float(np.mean(self.values))
        if len(self.values) == 1:
            return f"{mean:{spec}}{suffix}"
        # std is a magnitude -- never signed, even when the mean spec is.
        std_spec = spec.replace("+", "")
        return f"{mean:{spec}} +- {float(np.std(self.values)):{std_spec}}{suffix}"


def _get(results: list[dict], *path) -> _Agg:
    def dig(r):
        for key in path:
            if isinstance(r, dict):
                r = r.get(key)
            elif isinstance(r, list) and isinstance(key, int) and key < len(r):
                r = r[key]
            else:
                return None
        return r
    return _Agg([dig(r) for r in results])


def _pct(acc: _Agg) -> str:
    if not acc.values:
        return "-"
    return _Agg([100 * v for v in acc.values]).fmt(".2f", "%")


def _delta_pp(a: _Agg, b: _Agg) -> str:
    """Accuracy change b - a in percentage points (paired per seed)."""
    if not a.values or not b.values:
        return "-"
    diffs = [100 * (y - x) for x, y in zip(a.values, b.values)]
    return _Agg(diffs).fmt("+.2f", " pp")


def _delta_loss(a: _Agg, b: _Agg) -> str:
    if not a.values or not b.values:
        return "-"
    return _Agg([y - x for x, y in zip(a.values, b.values)]).fmt("+.4f")


def format_report(config: ExperimentConfig, seed_results: list[dict], title_suffix: str = "") -> str:
    """The report for one or several seeds of the same experiment."""
    r0 = seed_results[0]
    is_multiclass = r0["task"] == "multiclass"
    has_test = r0["stages"]["trained"] is not None and "test_loss" in r0["stages"]["trained"]
    finetuned = r0["stages"]["finetuned"] is not None

    methods = ", ".join(
        m.kind + (f" {m.params}" if m.params else "") for m in config.pruning.methods
    ) or "none"
    lines: list[str] = []
    add = lines.append

    add("=" * _W)
    add(f" Experiment : {config.name}{title_suffix}")
    add(f" Dataset    : {config.data.kind}   Model: {config.model.kind}   Task: {r0['task']}")
    add(f" Seeds      : {[r['seed'] for r in seed_results]}")
    add(f" Pruning    : {methods}")
    add(f" Fine-tune  : " + (f"{config.finetune.epochs} epochs (lr={config.finetune.lr})"
                             if finetuned else "none (one-shot)"))
    add("=" * _W)

    # ── pruning per layer ────────────────────────────────────────────────
    add("")
    add("PRUNING PER LAYER")
    add(f"  {'layer':>5} | {'before':>7} | {'removed':>9} | {'after':>7} | by method")
    n_layers = len(r0["pruning_per_layer"])
    for li in range(n_layers):
        before = _get(seed_results, "pruning_per_layer", li, "neurons_before")
        removed = _get(seed_results, "pruning_per_layer", li, "total_removed")
        after = _get(seed_results, "pruning_per_layer", li, "neurons_after")
        per_method = "  ".join(
            f"{kind}={_get(seed_results, 'pruning_per_layer', li, 'removed_per_method', kind).fmt('.0f')}"
            for kind in r0["pruning_per_layer"][li]["removed_per_method"]
        )
        add(f"  {li:>5} | {before.fmt('.0f'):>7} | {removed.fmt('.0f'):>9} | {after.fmt('.0f'):>7} | {per_method}")
    total = _Agg([sum(layer["total_removed"] for layer in r["pruning_per_layer"]) for r in seed_results])
    add(f"  total removed: {total.fmt('.0f')}")

    # ── efficiency ───────────────────────────────────────────────────────
    eff = lambda key: _get(seed_results, "pruned", key)  # noqa: E731
    add("")
    add("EFFICIENCY (original -> pruned)")
    add(f"  parameters     : {eff('params_before').fmt()} -> {eff('params_after').fmt()}"
        f"   ({eff('params_reduction_pct').fmt('.1f', '%')} reduction)")
    add(f"  model size     : {eff('model_size_before_kb').fmt(',.1f', ' KB')} -> "
        f"{eff('model_size_after_kb').fmt(',.1f', ' KB')}")
    add(f"  FLOPs / fwd    : {eff('flops_before').fmt()} -> {eff('flops_after').fmt()}"
        f"   ({eff('flops_reduction_pct').fmt('.1f', '%')} reduction)")
    add(f"  inference time : {eff('inference_time_before_us').fmt(',.1f', ' us')} -> "
        f"{eff('inference_time_after_us').fmt(',.1f', ' us')}"
        f"   ({eff('inference_speedup_pct').fmt('.1f', '%')} faster)")

    # ── performance per stage ────────────────────────────────────────────
    stages = [("after training", "trained"), ("after pruning", "pruned")]
    if finetuned:
        stages.append(("after fine-tuning", "finetuned"))

    add("")
    add("PERFORMANCE")
    columns = ["val loss"] + (["val acc"] if is_multiclass else [])
    if has_test:
        columns += ["test loss"] + (["test acc"] if is_multiclass else [])
    add("  " + " | ".join([f"{'stage':<19}"] + [f"{c:>16}" for c in columns]))
    for label, key in stages:
        cells = [f"{label:<19}", f"{_get(seed_results, 'stages', key, 'val_loss').fmt('.4f'):>16}"]
        if is_multiclass:
            cells.append(f"{_pct(_get(seed_results, 'stages', key, 'val_acc')):>16}")
        if has_test:
            cells.append(f"{_get(seed_results, 'stages', key, 'test_loss').fmt('.4f'):>16}")
            if is_multiclass:
                cells.append(f"{_pct(_get(seed_results, 'stages', key, 'test_acc')):>16}")
        add("  " + " | ".join(cells))

    # ── impact of pruning / fine-tuning ──────────────────────────────────
    def stage(key, metric):
        return _get(seed_results, "stages", key, metric)

    add("")
    add("IMPACT")
    if is_multiclass:
        add(f"  pruning: val acc change              : {_delta_pp(stage('trained', 'val_acc'), stage('pruned', 'val_acc'))}")
        if has_test:
            add(f"  pruning: test acc change             : {_delta_pp(stage('trained', 'test_acc'), stage('pruned', 'test_acc'))}")
        if finetuned:
            add(f"  fine-tuning: val acc recovery        : {_delta_pp(stage('pruned', 'val_acc'), stage('finetuned', 'val_acc'))}")
            if has_test:
                add(f"  fine-tuning: test acc recovery       : {_delta_pp(stage('pruned', 'test_acc'), stage('finetuned', 'test_acc'))}")
            add(f"  net: final vs trained (val acc)      : {_delta_pp(stage('trained', 'val_acc'), stage('finetuned', 'val_acc'))}")
            if has_test:
                add(f"  net: final vs trained (test acc)     : {_delta_pp(stage('trained', 'test_acc'), stage('finetuned', 'test_acc'))}")
    else:
        add(f"  pruning: val loss change             : {_delta_loss(stage('trained', 'val_loss'), stage('pruned', 'val_loss'))}")
        if has_test:
            add(f"  pruning: test loss change            : {_delta_loss(stage('trained', 'test_loss'), stage('pruned', 'test_loss'))}")
        if finetuned:
            add(f"  fine-tuning: val loss recovery       : {_delta_loss(stage('pruned', 'val_loss'), stage('finetuned', 'val_loss'))}")
            if has_test:
                add(f"  fine-tuning: test loss recovery      : {_delta_loss(stage('pruned', 'test_loss'), stage('finetuned', 'test_loss'))}")
            add(f"  net: final vs trained (val loss)     : {_delta_loss(stage('trained', 'val_loss'), stage('finetuned', 'val_loss'))}")

    add("")
    return "\n".join(lines)
