"""Reproduce the FlashLight AlphaFold2 measurements on Modal."""

from pathlib import Path

import modal

ARTIFACT_REVISION = "e092d604e63c87806caa61b094fd72039b8a546c"
ARTIFACT_DIR = "/opt/flashlight-artifact"
OPENFOLD_REVISION = "e938c184a291bf053af3b14c1e3e8bb29aee57e2"
OPENFOLD_DIR = "/opt/openfold"
OPENFOLD_ARTIFACT_REVISION = "ff1942b58a717af613454f05617035aad96c83c2"
OPENFOLD_ARTIFACT_DIR = "/opt/flashlight-openfold"
RESULTS_VOLUME = "flashlight-alphafold-results"

app = modal.App("gpu-kernels-flashlight-alphafold")
openfold_app = modal.App("gpu-kernels-flashlight-openfold")
results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)


def with_artifact_torch(image):
    return image.pip_install(
        "torch==2.5.0+cu121", index_url="https://download.pytorch.org/whl/cu121"
    ).pip_install("numpy<2.3", "tabulate", "nvidia-ml-py")


torch_image = with_artifact_torch(
    modal.Image.debian_slim(python_version="3.12").apt_install("git", "gcc", "g++")
)
artifact_image = torch_image.run_commands(
    f"git init {ARTIFACT_DIR}",
    f"git -C {ARTIFACT_DIR} fetch --depth 1 https://github.com/bozhiyou/flashlight.git {ARTIFACT_REVISION}",
    f"git -C {ARTIFACT_DIR} checkout --detach FETCH_HEAD",
)
openfold_image = (
    with_artifact_torch(
        modal.Image.from_registry(
            "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.12"
        ).apt_install("git", "gcc", "g++")
    )
    .pip_install(
        "biopython==1.85",
        "dm-tree==0.1.9",
        "ml-collections==1.0.0",
        "modelcif==0.7",
        "ninja",
        "scipy==1.14.1",
    )
    .run_commands(
        f"git init {OPENFOLD_DIR}",
        f"git -C {OPENFOLD_DIR} fetch --depth 1 https://github.com/aqlaboratory/openfold.git {OPENFOLD_REVISION}",
        f"git -C {OPENFOLD_DIR} checkout --detach FETCH_HEAD",
        # Build the stock extra-MSA CUDA extension for the H100 architecture only.
        f'sed -i \'/^extra_cuda_flags += cc_flag/i cc_flag = ["-gencode", "arch=compute_90,code=sm_90"]\' {OPENFOLD_DIR}/setup.py',
        f"cd {OPENFOLD_DIR} && MAX_JOBS=2 python setup.py build_ext --inplace",
        f"git init {OPENFOLD_ARTIFACT_DIR}",
        f"git -C {OPENFOLD_ARTIFACT_DIR} fetch --depth 1 https://github.com/bozhiyou/flashlight.git {OPENFOLD_ARTIFACT_REVISION}",
        f"git -C {OPENFOLD_ARTIFACT_DIR} checkout --detach FETCH_HEAD",
        f"cp {OPENFOLD_ARTIFACT_DIR}/benchmarks/benchmark_openfold.py /opt/benchmark_openfold_upstream.py",
    )
    .add_local_file(Path(__file__).with_name("benchmark_openfold.py"), "/opt/benchmark_openfold.py")
)


@app.function(image=artifact_image, gpu="H100!", volumes={"/results": results}, timeout=1200)
def evoformer():
    """Run the published Figure 4 Evoformer sweep over batch and head sizes."""
    import os
    import shutil
    import subprocess
    import sys
    from uuid import uuid4

    import torch

    gpu = torch.cuda.get_device_name(0)
    if "H100" not in gpu:
        raise RuntimeError(f"Expected NVIDIA H100, got {gpu}")

    run_id = uuid4().hex[:12]
    run_dir = Path("/results") / f"evoformer-h100-{run_id}"
    run_dir.mkdir()
    env = dict(os.environ, PYTHONPATH=ARTIFACT_DIR, FL_GPU_CLOCK_FREQ_MHZ="1290")
    subprocess.run(
        [sys.executable, "benchmarks/run_fig4_evoformer.py"],
        cwd=ARTIFACT_DIR,
        env=env,
        check=True,
    )
    csv_path = run_dir / "evo_attn.csv"
    shutil.copyfile(Path(ARTIFACT_DIR) / "benchmarks/results/evo_attn.csv", csv_path)
    results.commit()
    print(f"Saved {RESULTS_VOLUME}:/{run_dir.name}/{csv_path.name}", flush=True)
    return csv_path.read_text()


@openfold_app.function(
    image=openfold_image, gpu="H100!", volumes={"/results": results}, timeout=1800
)
def openfold(batch_size: int = 1, warmups: int = 2, iterations: int = 10):
    """Time identical synthetic OpenFold inputs in three isolated processes."""
    import json
    import os
    import subprocess
    import sys
    from uuid import uuid4

    if batch_size not in (1, 2, 4, 8, 16, 32) or warmups < 1 or iterations < 1:
        raise ValueError("Use a paper batch size and positive warmup and iteration counts")

    run_dir = Path("/results") / f"openfold-h100-b{batch_size}-{uuid4().hex[:12]}"
    run_dir.mkdir()
    import_paths = (OPENFOLD_DIR, OPENFOLD_ARTIFACT_DIR, f"{OPENFOLD_ARTIFACT_DIR}/benchmarks")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(import_paths))
    rows = []
    for mode in ("eager", "inductor", "flashlight"):
        output = run_dir / f"{mode}.json"
        subprocess.run(
            [
                sys.executable,
                "/opt/benchmark_openfold.py",
                mode,
                "--batch-size",
                str(batch_size),
                "--warmups",
                str(warmups),
                "--iterations",
                str(iterations),
                "--output",
                str(output),
            ],
            cwd=OPENFOLD_DIR,
            env=dict(env, TORCHINDUCTOR_CACHE_DIR=f"/tmp/openfold-{run_dir.name}-{mode}"),
            check=True,
        )
        rows.append(json.loads(output.read_text()))
        results.commit()

    summary = {
        "openfold_revision": OPENFOLD_REVISION,
        "flashlight_revision": OPENFOLD_ARTIFACT_REVISION,
        "runs": rows,
    }
    text = json.dumps(summary, indent=2) + "\n"
    (run_dir / "summary.json").write_text(text)
    results.commit()
    print(f"Saved {RESULTS_VOLUME}:/{run_dir.name}/summary.json", flush=True)
    return text
