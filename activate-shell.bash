#!/usr/bin/env bash
# Source this file from any directory: source /path/to/gpu-kernels/activate-shell.bash
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$repo_dir/.venv/bin/activate" ]]; then
    printf 'Missing %s/.venv; see README.md for setup.\n' "$repo_dir" >&2
    return 1
fi
source "$repo_dir/.venv/bin/activate"
unset repo_dir
