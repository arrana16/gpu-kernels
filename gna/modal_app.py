"""Modal image and environment checks for GNA on an NVIDIA B200."""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.12.0+cu130",
        index_url="https://download.pytorch.org/whl/cu130",
        extra_options="--only-binary=:all:",
    )
    .pip_install(
        "natten==0.21.7+torch2120cu130",
        "numpy>=2,<3",
        find_links="https://whl.natten.org",
        extra_options="--only-binary=:all:",
    )
    .run_commands(
        'python -c "import torch, natten; assert torch.version.cuda == \'13.0\'; '
        'assert natten.HAS_LIBNATTEN; print(torch.__version__, natten.__version__)"',
        "python -m natten.profiler --help > /dev/null",
    )
)

app = modal.App("gpu-kernels-gna", image=image)


@app.local_entrypoint()
def build_image():
    """Build and validate the image on CPU."""
    image.build(app)
    print(f"Built GNA image {image.object_id}")


@app.function(gpu="B200", timeout=300)
def check_b200():
    """Run a small 3D GNA operation through the Blackwell backend."""
    import natten
    import torch

    device = torch.cuda.get_device_name(0)
    if "B200" not in device:
        raise RuntimeError(f"Expected an NVIDIA B200, got {device}")

    shape = (1, 8, 16, 24, 1, 128)
    q, k, v = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    with torch.inference_mode():
        output = natten.na3d(
            q,
            k,
            v,
            kernel_size=(4, 8, 16),
            stride=(1, 4, 8),
            backend="blackwell-fna",
        )

    if output.shape != q.shape or not torch.isfinite(output).all().item():
        raise RuntimeError("GNA returned an unexpected shape or non-finite output")
    print(f"{device}, PyTorch {torch.__version__}, NATTEN {natten.__version__}")
    print(f"Blackwell FNA smoke test passed with output shape {tuple(output.shape)}")


@app.function(gpu="B200", timeout=300)
def benchmark_flux_attention(repeats: int = 3):
    """Profile the 4K FLUX attention shape and the three strides in Table 4."""
    import subprocess
    import sys

    if repeats < 1:
        raise ValueError("repeats must be positive")

    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        check=True,
    )
    base = [
        sys.executable, "-m", "natten.profiler",
        "--dtype", "fp16", "-b", "1", "-n", "24", "-d", "128",
        "-i", "256", "256", "--warmup-steps", "10",
    ]
    sparse = [
        "--backend", "blackwell-fna", "--q-tile", "16", "16",
        "--kv-tile", "16", "8", "-w", "80", "80",
    ]
    cases = [
        ("Dense cuDNN", ["--backend", "cudnn"]),
        ("Stride 1 x 1", sparse + ["-s", "1", "1"]),
        ("Stride 16 x 1", sparse + ["-s", "16", "1"]),
        ("Stride 16 x 16", sparse + ["-s", "16", "16"]),
    ]
    for repeat in range(1, repeats + 1):
        for label, options in cases:
            print(f"\nFLUX attention, trial {repeat}/{repeats}, {label}", flush=True)
            subprocess.run(base + options, check=True)
