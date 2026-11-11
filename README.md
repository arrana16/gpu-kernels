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

## GNA on Modal

Install the [Modal CLI](https://modal.com/docs/guide) and authenticate with
`modal setup`. The [GNA image](gna/modal_app.py) uses Python 3.12,
PyTorch 2.12.0 with CUDA 13.0, and the matching prebuilt NATTEN 0.21.7 wheel.
[NATTEN installation instructions](https://natten.org/install/)

From the repository root, build the image on CPU and check that PyTorch, the
native NATTEN extension, and the profiler load.

```bash
modal run gna/modal_app.py::build_image
```

Run the synthetic smoke test to verify a 3D GNA forward pass on a B200.

```bash
modal run gna/modal_app.py::check_b200
```

To rerun the paper's 4K FLUX attention shapes on B200, use the dense baseline
and the three published strides. The profiler supplies synthetic Q/K/V tensors.

```bash
modal run gna/modal_app.py::benchmark_flux_attention --repeats 3
```

See the [FLUX reproduction notes](gna/flux-reproduction.md) for the measured
attention timings and the setup for the end-to-end image comparison.

### FLUX 3 Image Klein

Accept access on the [Hugging Face model page](https://huggingface.co/black-forest-labs/flux-3-image-klein)
with the account used by `HF_TOKEN` in the Modal `huggingface-secret`, then run:

```bash
modal run -m gna.flux3_model::reproduce_flux3
```

This pilot generates one native 4096 × 4096 image per case on a B200 with the
same seed and full scene caption. The GNA case keeps the first 5 of 16 steps
dense, then uses an 80 × 80 window and 16 × 16 stride for image queries. Joint
attention keeps text keys available to every image query and text queries dense.
The command reports dense and GNA pipeline times and their speedup, excluding
checkpoint loading and file writes. Both cases use tiled VAE decoding. The
default caption is supplied directly so the separate prompt upsampler is unnecessary.

The Modal image reuses the existing PyTorch and NATTEN layers. The first run
downloads the new checkpoint components into the `gna-flux` volume. Look in
`flux3-runs/<run-id>` on that volume for the full-size PNGs, dashboard previews,
and `timings.json`.

### Profiling

For profiling, open a shell with the same B200 image.

```bash
modal shell gna/modal_app.py::check_b200
```

Inside the shell, list the compatible tile configurations for the example token
grid.

```bash
python -m natten.profiler --dry-run --backend blackwell-fna \
  --dtype bf16 -n 24 -d 128 -i 8 16 24 -w 4 8 16 -s 1 4 8
```

Remove `--dry-run` to collect timings. Exit the shell when finished to release
the GPU. See the [NATTEN profiler guide](https://natten.org/profiler/) for the
tile and baseline options.

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
using the input dtype for matmuls and stored probabilities. Each shape runs
`naive`, `flash` (original), and `flash_opt` (optimized). All three implementations
are checked over every output element against a separate FP32 reference before
measurement. This measures forward attention without masking, dropout, or autograd.
The full FP32 reference uses quadratic memory, so start with the default small
sizes and increase gradually. A correctness failure or GPU OOM stops the run;
CSV export preserves any fully measured earlier shapes.

Each method reports median and p95 CUDA-event latency, median synchronized wall
latency, additional peak allocated MiB above the inputs, approximate TFLOP/s, and
maximum absolute error. The speedup column is **naive time / method time**:
`0.25x` means that method takes four times as long as naive; naive itself is `1x`.
CSV rows store this ratio as `speedup_vs_naive`, replacing `flash_speedup`.
Compilation and warmup are
excluded, while allocation, initialization, output conversion, and all per-head
launches in the forward wrapper are included. CUDA-event intervals can include
GPU idle time while Python queues work; wall latency also includes host overhead
and synchronization. Memory counts live PyTorch allocations including output,
not allocator-reserved memory or total device usage. TFLOP/s counts only the two
main matrix products (`4 * B * H * N² * D`), so it is an approximate work rate.

For optimization comparisons, save a baseline CSV and repeat the same command
with a different output path after each change, keeping shapes, dtype, seed, and
GPU workload fixed. Stop GPU monitoring and other GPU jobs during measurements.
The original kernel uses one program per head. The optimized kernel distributes
query tiles across programs and keeps each tile's running softmax state and
output local while iterating over K/V tiles, avoiding repeated global-memory
reads and writes. Both wrappers still use Python loops to launch heads. Use the head-count
sweep to assess launch overhead and sequence-length sweep to assess scaling.
`--memory-budget` exposes the existing tile-sizing parameter for controlled
experiments without editing the kernel. Recheck correctness for every change.
