# GPU kernels

Triton and PyTorch CUDA development on WSL.

## Set up the Python environment

Use Python 3.14 and a CUDA-capable NVIDIA GPU. From this directory:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.lock.txt
source activate-shell.bash
```

The pinned package versions are in `requirements.lock.txt`. The existing environment at `~/.venvs/triton-gpu` is separate and remains usable; the commands above create a fresh environment inside this repo. The `.venv` directory is excluded from Git.

## FlashAttention correctness

From the repository root with your environment activated:

```bash
python -m unittest flashattention.test_flashattention1 -v
```

## FlashAttention stress test

```bash
python -m flashattention.stress_flashattention1
python -m flashattention.stress_flashattention1 --sizes 1024 2048 4096 8192 --heads 1 --seconds 10
```

This exercises the local Triton kernel with one attention head by default,
sweeping sequence lengths from 512 to 8192. Each shape runs for at least three
seconds after warmup (including correctness checks). Every iteration generates
fresh inputs, checks that all outputs are finite, and compares 32 evenly spaced
query rows against a float32 reference using all keys. This avoids allocating a
full quadratic reference matrix; it checks sampled rows, not every output row.
Use `--reference-rows 8192` to check all rows for sequence lengths up to 8192.

Each row reports run count, median and p95 forward latency, and peak PyTorch
allocated memory including inputs and forward buffers but excluding reference
buffers. Timing excludes compilation, input generation, and correctness checks.
An incorrect result or GPU out-of-memory error exits with a failure status.
Use `--heads 1 2 8` for a head-count sweep or `--dtype bfloat16` for BF16 inputs.
The duration is checked between iterations; compilation and individual forward
passes can make a case run longer than requested.

## FlashAttention versus naive attention benchmark

```bash
python -m flashattention.benchmark_flashattention1
python -m flashattention.benchmark_flashattention1 --sizes 128 256 512 1024 2048 --heads 1 8 --repeats 50 --csv /tmp/attention-baseline.csv
python -m flashattention.benchmark_flashattention1 --dtype bfloat16 --memory-budget 16384
```

The naive baseline explicitly materializes `softmax(Q @ K.T / sqrt(D)) @ V`
using the input dtype for matmuls and stored probabilities. Both implementations
are checked over every output element against a separate FP32 reference before
measurement. This measures forward attention without masking, dropout, or autograd.
The full FP32 reference uses quadratic memory, so start with the default small
sizes and increase gradually. A correctness failure or GPU OOM stops the run;
CSV export preserves any fully measured earlier shapes.

Each method reports median and p95 CUDA-event latency, median synchronized wall
latency, additional peak allocated MiB above the inputs, approximate TFLOP/s, and
maximum absolute error. The speedup column is **naive time / flash time**:
`0.25x` means FlashAttention takes four times as long. Compilation and warmup are
excluded, while allocation, initialization, output conversion, and all per-head
launches in the forward wrapper are included. CUDA-event intervals can include
GPU idle time while Python queues work; wall latency also includes host overhead
and synchronization. Memory counts live PyTorch allocations including output,
not allocator-reserved memory or total device usage. TFLOP/s counts only the two
main matrix products (`4 * B * H * N² * D`), so it is an approximate work rate.

For optimization comparisons, save a baseline CSV and repeat the same command
with a different output path after each change, keeping shapes, dtype, seed, and
GPU workload fixed. Stop GPU monitoring and other GPU jobs during measurements.
The current kernel uses one program per head and Python loops to launch heads;
first try distributing query tiles and heads across programs. Then keep each
query tile's running softmax state and output in registers while iterating over
K/V tiles, avoiding repeated global-memory reads and writes. Use the head-count
sweep to assess launch overhead and sequence-length sweep to assess scaling.
`--memory-budget` exposes the existing tile-sizing parameter for controlled
experiments without editing the kernel. Recheck correctness for every change.
