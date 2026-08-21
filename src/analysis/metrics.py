import logging
import time
import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _make_dummy_input(model: nn.Module, device: torch.device) -> torch.Tensor:
    """Create a single-sample dummy input that matches the model's first layer."""
    if hasattr(model, "make_dummy_input"):
        return model.make_dummy_input(device)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return torch.zeros(1, m.in_channels, 8, 8, device=device)
        if isinstance(m, nn.Linear):
            return torch.zeros(1, m.in_features, device=device)
    return torch.zeros(1, 1, device=device)


def count_flops(model: nn.Module) -> int:
    """Approximate FLOPs for one forward pass (MACs × 2), covering Linear and Conv2d."""
    device = next(model.parameters()).device
    x = _make_dummy_input(model, device)
    total = [0]

    hooks = []

    def _linear_hook(m, _inp, _out):
        total[0] += 2 * m.in_features * m.out_features

    def _conv_hook(m, _inp, out):
        # out shape: [1, out_ch, H, W]
        _, out_ch, H, W = out.shape
        in_ch, kH, kW = m.in_channels, m.kernel_size[0], m.kernel_size[1]
        total[0] += 2 * in_ch * kH * kW * out_ch * H * W

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv_hook))

    model.eval()
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()
    return total[0]


def measure_inference_us(model: nn.Module, n_runs: int = 2000) -> float:
    """Mean single-sample inference time in microseconds."""
    device = next(model.parameters()).device
    model.eval()
    x = _make_dummy_input(model, device)
    with torch.no_grad():
        for _ in range(50):
            model(x)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    return float(np.mean(times)) * 1e6


def compute_efficiency_metrics(before: nn.Module, after: nn.Module) -> dict:
    def pct_drop(a, b):
        return round(100 * (1 - b / a), 2) if a > 0 else 0.0

    params_b, params_a = count_parameters(before), count_parameters(after)
    flops_b,  flops_a  = count_flops(before),      count_flops(after)
    size_b  = params_b * 4 / 1024
    size_a  = params_a * 4 / 1024
    time_b  = measure_inference_us(before)
    time_a  = measure_inference_us(after)

    return {
        "params_before":            params_b,
        "params_after":             params_a,
        "params_reduction_pct":     pct_drop(params_b, params_a),
        "model_size_before_kb":     round(size_b, 3),
        "model_size_after_kb":      round(size_a, 3),
        "flops_before":             flops_b,
        "flops_after":              flops_a,
        "flops_reduction_pct":      pct_drop(flops_b, flops_a),
        "inference_time_before_us": round(time_b, 3),
        "inference_time_after_us":  round(time_a, 3),
        "inference_speedup_pct":    pct_drop(time_b, time_a),
    }


def print_efficiency_report(before: nn.Module, after: nn.Module, label: str = "pruned") -> dict:
    m = compute_efficiency_metrics(before, after)
    logging.info(f"=== Efficiency: original vs {label} ===")
    logging.info(f"  Parameters : {m['params_before']:>8,}  →  {m['params_after']:>8,}   ({m['params_reduction_pct']:.1f}% reduction)")
    logging.info(f"  Model size : {m['model_size_before_kb']:>8.2f} KB  →  {m['model_size_after_kb']:>8.2f} KB")
    logging.info(f"  FLOPs      : {m['flops_before']:>8,}  →  {m['flops_after']:>8,}   ({m['flops_reduction_pct']:.1f}% reduction)")
    logging.info(f"  Infer time : {m['inference_time_before_us']:>8.2f} µs  →  {m['inference_time_after_us']:>8.2f} µs   ({m['inference_speedup_pct']:.1f}% faster)")
    return m


# ── capacity: reading a width sweep ──────────────────────────────────────────
#
# "Capacity at -delta" is the largest fraction of prunable units removable while
# validation accuracy stays within delta of the dense model. Read off a noisy
# accuracy-versus-sparsity curve, that sentence admits two answers, and they are
# not interchangeable.

def capacity_at(fractions, accs, acc0: float, drop: float,
                mode: str = "first_crossing", sustain: int = 1) -> float:
    """Largest removable fraction meeting an accuracy tolerance.

    mode='first_crossing' (default) is the largest tested fraction such that the
    curve had not yet fallen through the tolerance -- the honest reading, since
    claiming a fraction is removable implies the model works there AND at every
    less aggressive width.

    mode='max_passing' is the largest fraction that happens to pass anywhere on
    the grid. It is a MAXIMUM OVER NOISY SAMPLES, so its expected value grows
    with the number of grid points: every extra width evaluated is another
    chance at a lucky pass near the tolerance. That makes it depend on grid
    density rather than on the method -- the recorded studies saw 0.772 against
    0.484 for one identical configuration measured on a 25-point and a 12-point
    grid. It is kept only so that bias can be shown rather than asserted; do not
    report it as capacity.

    `sustain` tolerates isolated dips: the curve must fail on `sustain`
    consecutive grid points to count as having crossed. Capacity never exceeds
    the largest fraction that itself passed, so a single failure at the end of
    the grid cannot be read as a pass.
    """
    if mode not in ("first_crossing", "max_passing"):
        raise ValueError("mode must be 'first_crossing' or 'max_passing'")
    if sustain < 1:
        raise ValueError(f"sustain must be >= 1, got {sustain}")
    order = sorted(range(len(fractions)), key=lambda i: fractions[i])
    f = [float(fractions[i]) for i in order]
    passing = [float(accs[i]) >= acc0 - drop for i in order]
    if not f:
        return 0.0
    if mode == "max_passing":
        hits = [fi for fi, ok in zip(f, passing) if ok]
        return max(hits) if hits else 0.0
    best, run = 0.0, 0
    for fi, ok in zip(f, passing):
        if ok:
            run, best = 0, fi
        else:
            run += 1
            if run >= sustain:
                break
    return best


def grid_spacing(fractions) -> float:
    """Largest gap between consecutive tested widths -- the resolution any
    capacity number is quoted at, and the number two tables must share before
    their capacities can be compared."""
    f = sorted(float(x) for x in fractions)
    return max((b - a for a, b in zip(f, f[1:])), default=0.0)


def curve_auc(fractions, accs) -> float:
    """Mean accuracy over the swept range (trapezoidal, normalized by the range).

    Tolerance-free, so it does not inherit the grid-density problem at all, and
    it uses the whole curve rather than one crossing. Worth reporting beside
    capacity: two methods with equal capacity can have very different curves.
    """
    order = sorted(range(len(fractions)), key=lambda i: fractions[i])
    f = [float(fractions[i]) for i in order]
    a = [float(accs[i]) for i in order]
    if len(f) < 2:
        return a[0] if a else float("nan")
    span = f[-1] - f[0]
    if span <= 0:
        return float(sum(a) / len(a))
    area = sum((f[i + 1] - f[i]) * (a[i + 1] + a[i]) / 2.0 for i in range(len(f) - 1))
    return area / span


def curve_report(fractions, accs, acc0: float,
                 drops=(0.005, 0.01, 0.02), sustain: int = 1) -> dict:
    """Every capacity reading of one curve, both modes, plus AUC and spacing.

    The `*_max_passing` entries are diagnostics: a large gap against the
    first-crossing value means the curve oscillates through the tolerance, so
    the grid is too coarse or the validation set too small to support a
    capacity claim at that tolerance.
    """
    out = {"acc0": float(acc0), "n_grid": len(fractions),
           "grid_spacing": grid_spacing(fractions),
           "auc": curve_auc(fractions, accs)}
    for d in drops:
        tag = f"{d:g}".replace("0.", "")
        out[f"cap{tag}"] = capacity_at(fractions, accs, acc0, d,
                                       "first_crossing", sustain)
        out[f"cap{tag}_max_passing"] = capacity_at(fractions, accs, acc0, d,
                                                   "max_passing")
    return out
