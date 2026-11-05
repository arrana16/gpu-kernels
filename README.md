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
