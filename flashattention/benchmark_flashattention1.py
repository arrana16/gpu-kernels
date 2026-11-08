"""Compare local FlashAttention with explicit, quadratic PyTorch attention.

Run from the repository root: python -m flashattention.benchmark_flashattention1
"""

import argparse
import csv
import math
from pathlib import Path
import platform
import statistics
import time

import torch
import triton

from flashattention.flashattention1 import flash_attention_forward
from flashattention.flashattention1_opt import flash_attention_forward as optimized_attention
from flashattention.stress_flashattention1 import positive_int


def naive_attention(q, k, v):
    """Materialize scores and probabilities; matmuls use the input dtype."""
    scores = (q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5
    # FP32 softmax accumulation, with probabilities stored in the input dtype.
    probabilities = torch.softmax(scores, dim=-1)
    return probabilities @ v


def reference_attention(q, k, v):
    scores = (q.float() @ k.float().transpose(-2, -1)) * q.shape[-1] ** -0.5
    return (torch.softmax(scores, dim=-1) @ v.float()).to(q.dtype)


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        output = fn()
        del output
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    gpu_ms, wall_ms = [], []
    for _ in range(repeats):
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1000)
        gpu_ms.append(start.elapsed_time(end))
        del output

    # Separate memory pass excludes timing events and correctness intermediates.
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    extra_mib = (torch.cuda.max_memory_allocated() - baseline) / 2**20
    del output
    return {
        "median_ms": statistics.median(gpu_ms),
        "p95_ms": sorted(gpu_ms)[math.ceil(0.95 * repeats) - 1],
        "wall_ms": statistics.median(wall_ms),
        "extra_mib": extra_mib,
    }


@torch.inference_mode()
def run_case(args, n, heads):
    shape = (args.batch, heads, n, args.head_dim)
    q, k, v = [torch.randn(shape, device="cuda", dtype=getattr(torch, args.dtype))
               for _ in range(3)]
    expected = reference_attention(q, k, v)
    methods = {
        "naive": lambda: naive_attention(q, k, v),
        "flash": lambda: flash_attention_forward(q, k, v, M=args.memory_budget),
        "flash_opt": lambda: optimized_attention(q, k, v, M=args.memory_budget),
    }
    errors = {}
    tolerance = 2e-2 if q.dtype == torch.bfloat16 else 2e-3
    for name, fn in methods.items():
        output = fn()
        try:
            torch.testing.assert_close(output, expected, atol=tolerance, rtol=tolerance)
        except AssertionError as error:
            raise AssertionError(f"{name} failed correctness: {error}") from error
        errors[name] = (output.float() - expected.float()).abs().max().item()
        del output
    del expected
    results = {}
    for name, fn in methods.items():
        result = measure(fn, args.warmup, args.repeats)
        result.update(batch=args.batch, heads=heads, n=n, head_dim=args.head_dim,
                      dtype=args.dtype, memory_budget=args.memory_budget,
                      method=name, max_abs_error=errors[name])
        # Leading-order QK^T and PV work; excludes softmax and initialization.
        result["tflops"] = 4 * args.batch * heads * n * n * args.head_dim / (result["median_ms"] * 1e9)
        results[name] = result
    for name, result in results.items():
        speedup = results["naive"]["median_ms"] / result["median_ms"]
        result["speedup_vs_naive"] = speedup
        print(f"{n:6d} {heads:5d} {name:>9} {result['median_ms']:10.3f} "
              f"{result['p95_ms']:10.3f} {result['wall_ms']:10.3f} "
              f"{result['extra_mib']:10.2f} {result['tflops']:9.3f} "
              f"{result['max_abs_error']:10.3g} {speedup:8.3f}x", flush=True)
    return list(results.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=positive_int, default=[128, 256, 512, 1024])
    parser.add_argument("--heads", nargs="+", type=positive_int, default=[1])
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--head-dim", type=positive_int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--memory-budget", type=positive_int, default=32768,
                        help="M passed to the local kernel, in elements")
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--repeats", type=positive_int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, help="save per-method measurements for later comparisons")
    args = parser.parse_args()
    if args.head_dim < 16 or args.head_dim & (args.head_dim - 1):
        parser.error("--head-dim must be a power of two >= 16")
    if not torch.cuda.is_available():
        parser.exit(1, "This benchmark requires a CUDA GPU.\n")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("this GPU does not support bfloat16")
    if args.memory_budget <= 60 * args.head_dim:
        parser.error("--memory-budget is too small for tiles of at least 16")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    print(f"GPU: {torch.cuda.get_device_name()} | Python {platform.python_version()} | "
          f"PyTorch {torch.__version__} | Triton {triton.__version__}")
    print(f"B={args.batch}, D={args.head_dim}, dtype={args.dtype}, M={args.memory_budget}, "
          f"seed={args.seed}, warmup={args.warmup}, repeats={args.repeats}")
    print("Methods: naive, flash (original), flash_opt (Q-tile parallelism and local state).")
    print("Forward only, noncausal, no dropout. All outputs checked against FP32 attention.")
    print("Times include the complete forward wrapper; exclude compilation and correctness checks.")
    print("Extra MiB: peak live PyTorch allocations above inputs, including output; excludes allocator cache.")
    print("Speedup = naive median / method median; >1 means this method is faster than naive.")
    print(f"{'N':>6} {'Heads':>5} {'Method':>9} {'GPU ms':>10} {'P95 ms':>10} "
          f"{'Wall ms':>10} {'Extra MiB':>10} {'TFLOP/s':>9} {'Max error':>10} {'Speedup':>9}")
    rows = []
    try:
        for heads in args.heads:
            for n in args.sizes:
                print(f"Checking and measuring N={n}, heads={heads}...", flush=True)
                rows.extend(run_case(args, n, heads))
    except (torch.cuda.OutOfMemoryError, AssertionError) as error:
        parser.exit(1, f"Failed at N={n}, heads={heads}: {error}\n")
    finally:
        if args.csv and rows:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved measurements to {args.csv}")


if __name__ == "__main__":
    main()
