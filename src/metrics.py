import time
import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: nn.Module) -> int:
    """FLOPs for one forward pass: 2 * d_in * d_out per Linear layer (MACs × 2)."""
    flops = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            flops += 2 * module.in_features * module.out_features
    return flops


def measure_inference_us(model: nn.Module, n_runs: int = 2000) -> float:
    """Mean single-sample inference time in microseconds."""
    model.eval()
    x = torch.zeros(1, 1)
    with torch.no_grad():
        for _ in range(50):           # warm-up
            model(x)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    return float(np.mean(times)) * 1e6


def print_efficiency_report(before: nn.Module, after: nn.Module, label: str = "pruned") -> None:
    def pct_drop(a, b):
        return 100 * (1 - b / a) if a > 0 else 0.0

    params_b, params_a = count_parameters(before), count_parameters(after)
    flops_b,  flops_a  = count_flops(before),      count_flops(after)
    size_b  = params_b * 4 / 1024   # float32 → KB
    size_a  = params_a * 4 / 1024
    time_b  = measure_inference_us(before)
    time_a  = measure_inference_us(after)

    print(f"\n=== Efficiency: original vs {label} ===")
    print(f"  Parameters : {params_b:>8,}  →  {params_a:>8,}   ({pct_drop(params_b, params_a):.1f}% reduction)")
    print(f"  Model size : {size_b:>8.2f} KB  →  {size_a:>8.2f} KB")
    print(f"  FLOPs      : {flops_b:>8,}  →  {flops_a:>8,}   ({pct_drop(flops_b, flops_a):.1f}% reduction)")
    print(f"  Infer time : {time_b:>8.2f} µs  →  {time_a:>8.2f} µs   ({pct_drop(time_b, time_a):.1f}% faster)")
