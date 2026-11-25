"""Capture one Cosmos-Predict2 attention call on a Modal H100."""

from pathlib import Path

import modal

COSMOS_REVISION = "661da4774b0ca41d082a0ecbeb47550bcf07e03f"
MODEL = "nvidia/Cosmos-Predict2-2B-Video2World"
MODEL_REVISION = "f50c09f5d8ab133a90cac3f4886a6471e9ba3f18"
COSMOS_DIR = "/opt/cosmos-predict2"
CHECKPOINT_DIR = Path("/data/checkpoints")
READY = CHECKPOINT_DIR / f"video2world-480p-10fps-{COSMOS_REVISION}.ready"
VOLUME_NAME = "flashlight-cosmos"

checkpoint_app = modal.App("gpu-kernels-flashlight-cosmos-checkpoints")
app = modal.App("gpu-kernels-flashlight-cosmos")
data = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

download_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "huggingface_hub==0.32.4"
)

cosmos_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04", add_python="3.10")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("uv==0.8.12")
    .env({"UV_LINK_MODE": "copy", "COSMOS_PREDICT2_ARGS": f"--checkpoints {CHECKPOINT_DIR}"})
    .run_commands(
        f"git init {COSMOS_DIR}",
        f"git -C {COSMOS_DIR} fetch --depth 1 https://github.com/nvidia-cosmos/cosmos-predict2.git {COSMOS_REVISION}",
        f"git -C {COSMOS_DIR} checkout --detach FETCH_HEAD",
        f"cd {COSMOS_DIR} && uv sync --locked --extra cu126",
    )
    .add_local_file(
        Path(__file__).with_name("capture_cosmos_attention.py"),
        remote_path="/opt/capture_cosmos_attention.py",
    )
)


@checkpoint_app.function(
    image=download_image,
    volumes={"/data": data},
    secrets=[hf_secret],
    timeout=3600,
)
def prepare_checkpoints(check_only: bool = False):
    """Check access and cache the exact Cosmos weights on CPU before capture."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url, snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    # Revisions used by the pinned Cosmos checkout's download script.
    checkpoints = (
        (
            MODEL,
            MODEL_REVISION,
            "model-480p-10fps.pt",
            {"allow_patterns": ["model-480p-10fps.pt", "tokenizer/tokenizer.pth"]},
        ),
        (
            "nvidia/Cosmos-Guardrail1",
            "d6d4bfa899a71454a700907664f3e88f503950cf",
            "video_content_safety_filter/safety_filter.pt",
            {},
        ),
        (
            "meta-llama/Llama-Guard-3-8B",
            "7327bd9f6efbbe6101dc6cc4736302b3cbb6e425",
            "config.json",
            {"ignore_patterns": ["original/*"]},
        ),
        (
            "google-t5/t5-11b",
            "90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3",
            "pytorch_model.bin",
            {"ignore_patterns": ["tf_model.h5"]},
        ),
    )
    unavailable = []
    for repo_id, revision, filename, _ in checkpoints:
        try:
            get_hf_file_metadata(hf_hub_url(repo_id, filename, revision=revision), token=True)
        except HfHubHTTPError as error:
            if error.response.status_code not in (401, 403):
                raise
            unavailable.append(repo_id)
    if unavailable:
        raise RuntimeError("Hugging Face token needs access to " + ", ".join(unavailable))

    print("Hugging Face access verified for all four checkpoint repositories", flush=True)
    if check_only:
        return

    for repo_id, revision, _, options in checkpoints:
        print(f"Caching {repo_id} at {revision}", flush=True)
        snapshot_download(
            repo_id,
            revision=revision,
            local_dir=CHECKPOINT_DIR / repo_id,
            token=True,
            **options,
        )
    READY.touch()
    data.commit()


@app.function(image=cosmos_image, gpu="H100!", memory=65536, volumes={"/data": data}, timeout=1800)
def capture(seed: int = 0):
    """Run stock attention once and save the inputs and reference output."""
    import subprocess
    from uuid import uuid4

    if not READY.is_file():
        raise RuntimeError("Run prepare_checkpoints before capturing on H100")

    run_dir = Path("/data/captures") / uuid4().hex[:12]
    subprocess.run(
        [
            f"{COSMOS_DIR}/.venv/bin/python",
            "/opt/capture_cosmos_attention.py",
            "--output-dir",
            str(run_dir),
            "--seed",
            str(seed),
            "--cosmos-revision",
            COSMOS_REVISION,
            "--model-revision",
            MODEL_REVISION,
        ],
        cwd=COSMOS_DIR,
        check=True,
    )
    data.commit()
    volume_path = f"/captures/{run_dir.name}"
    print(f"Saved {VOLUME_NAME}:{volume_path}", flush=True)
    return volume_path
