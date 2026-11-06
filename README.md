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
