"""Compare stock attention and FlashLight on a saved Cosmos capture."""

from pathlib import Path

import modal

from flashlight.alphafold_modal import ARTIFACT_DIR, ARTIFACT_REVISION, artifact_image
from flashlight.modal_app import VOLUME_NAME, data

app = modal.App("gpu-kernels-flashlight-cosmos-replay")
image = artifact_image.pip_install("safetensors==0.5.3").add_local_file(
    Path(__file__).with_name("benchmark_cosmos.py"), "/opt/benchmark_cosmos.py"
)


@app.function(image=image, gpu="H100!", volumes={"/data": data}, timeout=1800)
def compare(capture_id: str, warmups: int = 10, iterations: int = 20):
    """Replay every backend in an isolated process on the same H100."""
    import json
    import os
    import re
    import subprocess
    import sys
    from tempfile import TemporaryDirectory
    from uuid import uuid4

    if not re.fullmatch(r"[a-f0-9]{12}", capture_id) or warmups < 1 or iterations < 1:
        raise ValueError("Use a 12-character capture ID and positive iteration and warmup counts")

    capture_dir = Path("/data/captures") / capture_id
    metadata = json.loads((capture_dir / "metadata.json").read_text())
    run_dir = Path("/data/replays") / f"{capture_id}-{uuid4().hex[:12]}"
    run_dir.mkdir(parents=True)
    rows = []
    with TemporaryDirectory(prefix="cosmos-replay-") as cache_dir:
        for mode in ("cudnn_sdpa", "inductor", "flashlight"):
            output = run_dir / f"{mode}.json"
            env = dict(
                os.environ,
                PYTHONPATH=ARTIFACT_DIR,
                TORCHINDUCTOR_CACHE_DIR=f"{cache_dir}/{mode}",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "/opt/benchmark_cosmos.py",
                    mode,
                    str(capture_dir),
                    str(output),
                    "--warmups",
                    str(warmups),
                    "--iterations",
                    str(iterations),
                ],
                env=env,
                check=False,
            )
            if result.returncode:
                failure = {"mode": mode, "status": "failed", "exit_code": result.returncode}
                output.write_text(json.dumps(failure, indent=2) + "\n")
            rows.append(json.loads(output.read_text()))
            data.commit()

    summary = {
        "capture_id": capture_id,
        "capture": metadata,
        "flashlight_revision": ARTIFACT_REVISION,
        "runs": rows,
    }
    report = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (run_dir / "summary.json").write_text(report)
    data.commit()
    print(f"Saved {VOLUME_NAME}:/replays/{run_dir.name}/summary.json", flush=True)
    return report
