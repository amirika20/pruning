import logging
import time
import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _make_dummy_input(model: nn.Module, device: torch.device) -> torch.Tensor:
    """Create a single-sample dummy input that matches the model's first layer."""
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
