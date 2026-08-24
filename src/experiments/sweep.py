"""Width sweeps: the accuracy-versus-sparsity curve a capacity number is read from.

One prune gives one point. Capacity at -delta needs the curve, so this drives a
whole grid of target widths and assembles it, together with the timings that make
the cost comparison honest.

THE ANYTIME PROPERTY IS A MEASUREMENT, NOT A REMARK. A method that builds a merge
trajectory once can be cut at every width for free; a method that solves a
subset-selection problem pays again per width. That asymmetry is a headline claim,
so it is timed rather than asserted:

    plan_seconds   one-off work, amortized over the whole grid (MASH's greedy
                   pass). Zero for methods with no plan step.
    solve_seconds  work repeated at every width -- selection for methods without
                   a plan, realization and repair for methods with one.
    total_seconds  what producing the entire sweep actually cost.

Methods are used through whichever interface they have; nothing is engineered
around a method's absence of a plan step, because that absence is the thing being
measured.

DENDROGRAMS COME FROM THE INTACT MODEL. Layers are pruned in order, so by the
time layer L is realized its input has already shrunk -- but its output units,
and hence a partition over them, have not. Plans are therefore built once on the
unpruned model and the realization is recomputed per width in the current model's
coordinates. The recorded studies measured intact-model partitions as equal or
better than recomputing them on the progressively pruned model, which is what
licenses the whole approach.

EQUAL FRACTIONS ACROSS LAYERS is the protocol: one `fraction` gives every layer
the same relative cut. Absolute counts cannot express it, since layers differ in
width and the count would clamp the narrow ones.
"""

from __future__ import annotations

import copy
import inspect
import logging
import time
from typing import Any, Callable, Sequence

import pandas as pd
import torch

from src.analysis.metrics import curve_report
from src.data.registry import DatasetBundle
from src.models.registry import PrunableModel
from src.pruning.registry import (
    PRUNING_METHOD_REGISTRY, PruneContext, build_pruning_method)
from src.pruning.surgery import apply_decision
from src.training.trainer import evaluate


def _accepted_params(cls: type) -> set[str]:
    """Every keyword `cls(...)` accepts, following the MRO.

    Subclasses here take `**kw` and forward to a base (RandomPruning takes only
    `seed` and passes the rest on), so inspecting the immediate __init__ alone
    reports that `fraction` is unsupported -- which silently leaves a method at
    its default width instead of the one the sweep asked for.
    """
    names: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for name, prm in inspect.signature(init).parameters.items():
            if prm.kind not in (prm.VAR_KEYWORD, prm.VAR_POSITIONAL):
                names.add(name)
    return names - {"self"}


def _width_params(cls: type, params: dict, n_units: int, fraction: float) -> dict:
    """Params for `cls` that express "remove this fraction of the layer".

    Methods spell their budget differently -- `fraction`, `n_remove`, or not at
    all (the tolerance-driven and saturation methods choose their own width). We
    ask the constructor rather than special-casing names.
    """
    sig = _accepted_params(cls)
    out = dict(params)
    if "fraction" in sig:
        out["fraction"] = fraction
    elif "n_remove" in sig:
        out["n_remove"] = int(round(fraction * (n_units - 1)))
    elif "prune_fraction" in sig:
        out["prune_fraction"] = fraction
    return out


# Sequences per forward pass when scoring a language model. Logits dominate
# (batch x seq x vocab), so this bounds evaluation memory independently of how
# many chunks the split holds.
LM_EVAL_BATCH = 4


def _accuracy(model: PrunableModel, bundle: DatasetBundle,
              batch_size: int = 512, split: str = "val") -> tuple[float, float | None]:
    """Loss and accuracy on `split` ("val" or "test").

    A downloaded checkpoint was trained on the whole official train split, so
    the val slice carved out of it is training data as far as that model is
    concerned -- grading it there overstates the dense baseline every capacity
    is measured against. Pretrained cells pass split="test".
    """
    # BOUND THE LM BATCH. The val loader batches the whole split at once, which
    # was fine at the 8 chunks n_val defaults to and OOMs at the ~560 of a full
    # wikitext test split: logits are batch x seq x vocab, so 560 x 512 x 50272
    # fp32 is a single 53.8 GiB allocation. Cap causal_lm regardless of split so
    # memory stays a function of the model, not of how much text was asked for.
    if bundle.task == "causal_lm":
        batch_size = min(batch_size, LM_EVAL_BATCH)
    if split == "test":
        loader = bundle.test_loader(batch_size)
        if loader is None:
            raise ValueError(
                f"eval_split='test' but this dataset has no test split; either "
                f"give its data params an n_test or use eval_split='val'")
    else:
        _, loader = bundle.loaders(batch_size)
    return evaluate(model, loader, bundle.task)


def sweep_widths(
    model: PrunableModel,
    bundle: DatasetBundle,
    kind: str,
    params: dict[str, Any] | None = None,
    fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    device: torch.device | None = None,
    eval_split: str = "val",
    n_calib: int | None = None,
    evaluate_fn: Callable[[PrunableModel], dict] | None = None,
    **meta: Any,
) -> pd.DataFrame:
    """One row per target fraction: width, accuracy, loss and timings.

    `evaluate_fn` overrides the default validation evaluation, for protocols that
    need something else (perplexity, a held-out split, a task subset).
    """
    params = dict(params or {})
    device = device or torch.device("cpu")
    # Take the class from the registry rather than instantiating: some methods
    # (osscar) require a width at construction, which we do not have yet.
    if kind not in PRUNING_METHOD_REGISTRY:
        raise KeyError(f"unknown pruning method {kind!r}")
    cls = PRUNING_METHOD_REGISTRY[kind]
    train_inputs = bundle.train_ds.tensors[0].to(device)
    model = model.to(device).eval()

    def measure(m: PrunableModel) -> dict:
        if evaluate_fn is not None:
            return evaluate_fn(m)
        loss, acc = _accuracy(m, bundle, split=eval_split)
        return {"val_loss": loss, "val_acc": acc}

    widths0 = [model.prunable_layer(i).weight.shape[0]
               for i in range(model.n_prunable_layers())]
    dense = measure(model)
    rows: list[dict] = [dict(meta, fraction=0.0, units_before=sum(widths0),
                             units_after=sum(widths0), plan_seconds=0.0,
                             solve_seconds=0.0, **dense)]

    # One planning pass on the intact model, if the method offers one.
    plans: dict[int, Any] = {}
    plan_seconds = 0.0
    probe = None
    if hasattr(cls, "plan") and hasattr(cls, "emit_at"):
        probe = build_pruning_method(kind, **params)
        t0 = time.perf_counter()
        for li in range(model.n_prunable_layers()):
            ctx = PruneContext(train_inputs=train_inputs, bundle=bundle,
                               device=device)
            plans[li] = probe.plan(model, li, ctx)
        plan_seconds = time.perf_counter() - t0
        logging.info(f"  [{kind}] planned {len(plans)} layer(s) in "
                     f"{plan_seconds:.2f}s -- reused at every width")

    for f in fractions:
        t0 = time.perf_counter()
        # A DEEP COPY PER WIDTH, for the same reason prune_model takes one: the
        # new_incoming path rewrites the prunable layer's rows IN PLACE, and on
        # the first layer `current` would otherwise still be the caller's model.
        # Without this, a merge-emitting method overwrites layer 0 of the dense
        # weights on the first fraction and every later width silently starts
        # from a contaminated model -- so the curve stops being a set of
        # independent prunes of one trained network, which is the only thing
        # that makes its points comparable.
        current = copy.deepcopy(model)
        for li in range(model.n_prunable_layers()):
            n_units = current.prunable_layer(li).weight.shape[0]
            ctx = PruneContext(train_inputs=train_inputs, bundle=bundle,
                              device=device)
            if li in plans:
                k = plans[li].merges_for(f)
                decision = probe.emit_at(current, li, plans[li], k, ctx)
            else:
                wp = _width_params(cls, params, n_units, f)
                if n_calib is not None and "n_calib" in _accepted_params(cls):
                    wp.setdefault("n_calib", n_calib)
                decision = build_pruning_method(kind, **wp).select(current, li, ctx)
            current, selected, _ = apply_decision(current, li, decision)
            if selected:
                current = current.prune_layer(li, sorted(selected))
        solve_seconds = time.perf_counter() - t0
        widths = [current.prunable_layer(i).weight.shape[0]
                  for i in range(current.n_prunable_layers())]
        rows.append(dict(meta, fraction=float(f), units_before=sum(widths0),
                         units_after=sum(widths), plan_seconds=0.0,
                         solve_seconds=solve_seconds, **measure(current)))

    df = pd.DataFrame(rows)
    # The realized fraction is what the curve should be read against: a method
    # may refuse to reach the requested one (a certificate, or a layer that ran
    # out of mergeable units).
    df["removed"] = 1.0 - df.units_after / df.units_before
    df["plan_seconds"] = plan_seconds
    df["total_seconds"] = plan_seconds + df.solve_seconds.sum()
    df["method"] = kind
    return df


def prune_at_fraction(model: PrunableModel, bundle: DatasetBundle, kind: str,
                      params: dict[str, Any] | None = None,
                      fraction: float = 0.5,
                      device: torch.device | None = None,
                      n_calib: int | None = None,
                      ) -> tuple[PrunableModel, list[dict]]:
    """The pruned model at ONE target width, budgeted exactly as sweep_widths.

    The geometry and similarity batteries need a model pruned to a MEANINGFUL
    width. Calling prune_model with a benchmark cell's own pruning spec gives
    the method's default budget instead, which is n_remove=1 -- one unit per
    layer, so 12 of OPT-125m's 36864 units (0.03%). Every before/after
    comparison built on that measures a perturbation indistinguishable from
    none. This routes the same _width_params budget logic the curve uses.
    """
    params = dict(params or {})
    cls = PRUNING_METHOD_REGISTRY[kind]
    current = copy.deepcopy(model)
    reports: list[dict] = []
    train_inputs = bundle.train_ds.tensors[0].to(device) if device else \
        bundle.train_ds.tensors[0]
    for li in range(model.n_prunable_layers()):
        before = current.prunable_layer(li).weight.shape[0]
        ctx = PruneContext(train_inputs=train_inputs, bundle=bundle, device=device)
        if hasattr(cls, "plan") and hasattr(cls, "emit_at"):
            probe = build_pruning_method(kind, **params)
            plan = probe.plan(model, li, ctx)
            decision = probe.emit_at(current, li, plan, plan.merges_for(fraction), ctx)
        else:
            wp = _width_params(cls, params, before, fraction)
            if n_calib is not None and "n_calib" in _accepted_params(cls):
                wp.setdefault("n_calib", n_calib)
            decision = build_pruning_method(kind, **wp).select(current, li, ctx)
        current, selected, applied = apply_decision(current, li, decision)
        if selected:
            current = current.prune_layer(li, sorted(selected))
        after = current.prunable_layer(li).weight.shape[0]
        # SAME SCHEMA AS prune_model's report. geometry_shift and
        # similarity_table read neurons_before/removed_indices off these entries;
        # a near-miss here is swallowed by the caller's except-and-warn and the
        # whole analysis silently produces nothing.
        merge_ops = ([{"removed": int(o.removed), "survivor": int(o.survivor),
                       "scale": float(o.scale)} for o in applied] if applied else [])
        reports.append({
            "layer": li,
            "neurons_before": before,
            "removed_per_method": {kind: len(selected)},
            "total_removed": len(selected),
            "neurons_after": after,
            "removed_indices": sorted(int(i) for i in selected),
            "removed_indices_per_method": {kind: sorted(int(i) for i in selected)},
            "merge_ops": {kind: merge_ops} if merge_ops else {},
            "diagnostics": ({kind: decision.diagnostics}
                            if getattr(decision, "diagnostics", None) else {}),
            "removed": before - after,
        })
    return current, reports


def sweep_report(df: pd.DataFrame, drops: Sequence[float] = (0.005, 0.01, 0.02),
                 sustain: int = 1) -> dict:
    """Capacity readings plus the cost of having produced the curve."""
    acc_col = "val_acc" if df.val_acc.notna().any() else "val_loss"
    sign = 1.0 if acc_col == "val_acc" else -1.0
    dense = float(sign * df.loc[df.fraction == 0.0, acc_col].iloc[0])
    swept = df[df.fraction > 0.0]
    out = curve_report(swept.removed.tolist(),
                       (sign * swept[acc_col]).tolist(), dense,
                       drops=drops, sustain=sustain)
    out.update(metric=acc_col,
               plan_seconds=float(df.plan_seconds.iloc[0]),
               solve_seconds=float(swept.solve_seconds.sum()),
               total_seconds=float(df.total_seconds.iloc[0]),
               seconds_per_width=float(swept.solve_seconds.mean()))
    return out


def format_sweep(reports: dict[str, dict]) -> str:
    """Compact comparison table over {label: sweep_report}."""
    if not reports:
        return "no sweeps\n"
    drops = sorted({k for r in reports.values() for k in r
                    if k.startswith("cap") and not k.endswith("max_passing")})
    out = ["Width sweep: capacity (first crossing) and sweep cost", "=" * 86]
    head = f"{'method':<26}" + "".join(f"{d:>9}" for d in drops) + \
           f"{'AUC':>8}{'plan s':>9}{'solve s':>9}{'total s':>9}"
    out.append(head)
    for label, r in reports.items():
        row = f"{label:<26}"
        for d in drops:
            row += f"{r.get(d, float('nan')):>9.3f}"
        row += (f"{r['auc']:>8.4f}{r['plan_seconds']:>9.2f}"
                f"{r['solve_seconds']:>9.2f}{r['total_seconds']:>9.2f}")
        out.append(row)
    # Where max-passing and first-crossing disagree, the grid or the validation
    # set is too coarse to support a capacity claim at that tolerance.
    gaps = []
    for label, r in reports.items():
        for d in drops:
            mp = r.get(f"{d}_max_passing")
            if mp is not None and mp - r.get(d, 0.0) > 1e-9:
                gaps.append(f"{label} {d}: first-crossing {r[d]:.3f} vs "
                            f"max-passing {mp:.3f}")
    if gaps:
        out.append("")
        out.append("curve oscillates through the tolerance (grid too coarse or "
                   "val set too small to claim capacity here):")
        out.extend("  " + g for g in gaps)
    spacing = {r["grid_spacing"] for r in reports.values()}
    out.append("")
    out.append(f"grid spacing: {', '.join(f'{s:.3f}' for s in sorted(spacing))}"
               "   (capacities are only comparable at equal spacing)")
    return "\n".join(out) + "\n"


# ── self-tests ───────────────────────────────────────────────────────────────

def _selftest() -> None:  # pragma: no cover
    import numpy as np
    import torch.nn as nn
    from torch.utils.data import TensorDataset

    from src.analysis.metrics import capacity_at
    from src.models.mlp import MLP
    from src.pruning.registry import PRUNING_METHOD_REGISTRY

    torch.manual_seed(0)
    d, H = 16, 24
    net = MLP(input_dim=d, output_dim=3, hidden_sizes=[H, 12]).eval()
    X = torch.randn(96, d)

    class _B:
        train_ds = TensorDataset(X, torch.zeros(len(X), dtype=torch.long))
        val_ds = TensorDataset(X, torch.zeros(len(X), dtype=torch.long))
        test_ds = None
        task = "multiclass"

        def loaders(self, batch_size=None, seed=None):
            from torch.utils.data import DataLoader
            return (DataLoader(self.train_ds, batch_size=batch_size or 32),
                    DataLoader(self.val_ds, batch_size=len(self.val_ds)))

    bundle = _B()
    ctx = PruneContext(train_inputs=X, bundle=bundle, device=torch.device("cpu"))

    # 1. THE CORRECTNESS PROPERTY OF THE PLAN SPLIT: cutting a cached full pass
    # must give exactly what a fresh select() would at the same width. If this
    # drifts, every swept number is measuring a different method per width.
    from src.pruning.methods.mash import MASH
    for score in ("delta_f", "cylinder"):
        m = MASH(score=score)
        plan = m.plan(net, 0, ctx)
        for f in (0.1, 0.25, 0.5, 0.75):
            k = plan.merges_for(f)
            via_plan = m.emit_at(net, 0, plan, k, ctx)
            fresh = MASH(score=score, fraction=f).select(net, 0, ctx)
            assert via_plan.remove == fresh.remove, (
                f"{score} f={f}: plan cut {via_plan.remove} != select "
                f"{fresh.remove}")
            for attr in ("new_outgoing", "new_incoming"):
                a, b = getattr(via_plan, attr), getattr(fresh, attr)
                if a is None or b is None:
                    assert a is b, f"{score} f={f}: {attr} presence differs"
                    continue
                pair = zip(a, b) if isinstance(a, tuple) else [(a, b)]
                for ta, tb in pair:
                    assert torch.allclose(ta, tb, atol=1e-10), \
                        f"{score} f={f}: {attr} differs between plan and select"

    # 2. one plan really does serve every width: planning is done once
    calls = {"n": 0}
    orig = MASH.plan

    def counting(self, *a, **kw):
        calls["n"] += 1
        return orig(self, *a, **kw)

    MASH.plan = counting
    try:
        df = sweep_widths(net, bundle, "mash", {}, fractions=(0.2, 0.4, 0.6))
    finally:
        MASH.plan = orig
    assert calls["n"] == net.n_prunable_layers(), \
        f"expected one plan per layer, got {calls['n']}"
    assert len(df) == 4, f"expected dense + 3 widths, got {len(df)}"
    assert df.removed.iloc[0] == 0.0 and df.removed.is_monotonic_increasing

    # 3. every registered method can be given a width by the sweep
    for kind, cls in PRUNING_METHOD_REGISTRY.items():
        names = _accepted_params(cls)
        wp = _width_params(cls, {}, H, 0.5)
        if names & {"fraction", "n_remove", "prune_fraction"}:
            assert wp, f"{kind}: sweep could not express a width"
        # the tolerance-driven and saturation methods legitimately choose their
        # own width, so an empty override is correct for them

    # 4. the metric's two readings differ exactly where the curve oscillates
    f = [0.1, 0.2, 0.3, 0.4, 0.5]
    a = [0.9, 0.9, 0.8, 0.8, 0.9]
    assert capacity_at(f, a, 0.9, 0.01, "first_crossing") == 0.2
    assert capacity_at(f, a, 0.9, 0.01, "max_passing") == 0.5
    rep = sweep_report(df)
    assert "cap01" in rep and rep["grid_spacing"] > 0
    assert rep["total_seconds"] >= rep["plan_seconds"] > 0

    # 5. THE SWEEP MUST NOT TOUCH THE MODEL IT IS GIVEN, and its points must be
    # independent of each other. apply_decision's new_incoming path rewrites the
    # prunable layer's rows in place, so without a copy per width a
    # merge-emitting method overwrites layer 0 of the dense weights on the first
    # fraction and every later width starts from a contaminated model -- which
    # was a live bug, and changed the measured accuracies.
    from src.reproducibility import state_dict_digest
    before = state_dict_digest(net)
    fwd = sweep_widths(net, bundle, "mash",
                       {"score": "delta_f", "repair": "kernel"},
                       fractions=[0.2, 0.4, 0.6])
    assert state_dict_digest(net) == before, \
        "sweep_widths mutated the model it was given"
    rev = sweep_widths(net, bundle, "mash",
                       {"score": "delta_f", "repair": "kernel"},
                       fractions=[0.6, 0.4, 0.2])
    f_acc = fwd.set_index("fraction").val_acc.round(10).to_dict()
    r_acc = rev.set_index("fraction").val_acc.round(10).to_dict()
    assert f_acc == r_acc, \
        f"curve depends on the order widths were swept: {f_acc} vs {r_acc}"

    print("sweep.py self-tests passed:")
    print("  a cut of a cached plan == a fresh select() at the same width")
    print("    (removal sets AND emitted weights, delta_f and cylinder)")
    print(f"  one plan per layer serves the whole grid ({calls['n']} plan calls)")
    print("  every registered method can be given a width by the sweep")
    print("  first-crossing 0.2 vs max-passing 0.5 on an oscillating curve")
    print("  the swept model is left untouched, and the curve is independent")
    print("    of the order the widths were visited")


if __name__ == "__main__":
    _selftest()
