"""Stress the local Triton kernel: python -m flashattention.stress_flashattention1."""

import argparse
import math
import statistics
import time

import torch

from flashattention.flashattention1 import flash_attention_forward


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return seconds


def check_output(q, k, v, output, reference_rows):
    """Check selected query rows against all keys, without an N x N matrix."""
    if output.shape != q.shape or output.dtype != q.dtype or output.device != q.device:
        raise AssertionError("Incorrect output shape, dtype, or device")
    if not torch.isfinite(output).all().item():
        raise AssertionError("Output contains NaN or infinity")
    n = q.shape[-2]
    rows = torch.linspace(0, n - 1, min(reference_rows, n),
                          device=q.device).long()
    scores = (q[:, :, rows].float() @ k.float().transpose(-2, -1))
    scores *= q.shape[-1] ** -0.5
    expected = (torch.softmax(scores, dim=-1) @ v.float()).to(q.dtype)
    tolerance = 2e-2 if q.dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(output[:, :, rows], expected,
                               atol=tolerance, rtol=tolerance)


@torch.inference_mode()
def run_case(args, n, heads, dtype):
    shape = (args.batch, heads, n, args.head_dim)
    q, k, v = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3)]
    # Compilation, warmup, and input generation are outside latency samples.
    for _ in range(args.warmup):
        output = flash_attention_forward(q, k, v)
        check_output(q, k, v, output, args.reference_rows)
        del output
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    torch.cuda.reset_peak_memory_stats()
    flash_peak = torch.cuda.memory_allocated()
    samples = []
    deadline = time.perf_counter() + args.seconds
    while not samples or time.perf_counter() < deadline:
        # Fresh inputs exercise state initialization on every forward pass.
        for tensor in (q, k, v):
            tensor.normal_()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start.record()
        output = flash_attention_forward(q, k, v)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        # Capture before allocating reference buffers, which are excluded.
        flash_peak = max(flash_peak, torch.cuda.max_memory_allocated())
        check_output(q, k, v, output, args.reference_rows)
        del output
    ordered = sorted(samples)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    print(f"{n:7d} {heads:5d} {len(samples):8d} "
          f"{statistics.median(samples):12.3f} {p95:11.3f} "
          f"{flash_peak / 2**20:12.1f}  PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=positive_int, nargs="+",
                        default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--heads", type=positive_int, nargs="+", default=[1])
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--head-dim", type=positive_int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--seconds", type=positive_seconds, default=3,
                        help="minimum wall time per shape after warmup, including checks")
    parser.add_argument("--warmup", type=positive_int, default=2)
    parser.add_argument("--reference-rows", type=positive_int, default=32,
                        help="query rows checked against all keys on every iteration")
    args = parser.parse_args()
    if args.head_dim < 16 or args.head_dim & (args.head_dim - 1):
        parser.error("--head-dim must be a power of two >= 16")
    if not torch.cuda.is_available():
        parser.exit(1, "This stress test requires a CUDA GPU.\n")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("this GPU does not support bfloat16")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"GPU: {torch.cuda.get_device_name()} | local FlashAttention v1 kernel")
    print(f"B={args.batch}, D={args.head_dim}, dtype={args.dtype}, "
          f"{args.seconds:g}s per shape after warmup")
    print(f"Every iteration: fresh inputs, finite output, up to {args.reference_rows} "
          "query rows checked against all keys in float32.")
    print("CUDA-event latency excludes compilation, input generation, and checks.")
    print("Peak allocated memory includes inputs and forward buffers; excludes reference.")
    print(f"{'N':>7} {'Heads':>5} {'Runs':>8} {'Median ms':>12} "
          f"{'P95 ms':>11} {'Peak MiB':>12}  Status", flush=True)
    try:
        for heads in args.heads:
            for n in args.sizes:
                print(f"Running N={n}, heads={heads}...", flush=True)
                run_case(args, n, heads, getattr(torch, args.dtype))
    except torch.cuda.OutOfMemoryError:
        parser.exit(1, f"FAIL: GPU out of memory at N={n}, heads={heads}.\n")
    except AssertionError as error:
        parser.exit(1, f"FAIL: correctness check at N={n}, heads={heads}: {error}\n")
    print("All stress cases passed.")


if __name__ == "__main__":
    main()
