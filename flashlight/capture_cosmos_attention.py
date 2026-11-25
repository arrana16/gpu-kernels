"""Record the first dense self-attention call from stock Cosmos inference."""

import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

MODEL_SIZE = "2B"
RESOLUTION = "480"
FPS = 10
ASPECT_RATIO = "16:9"
SAMPLE_IMAGE = Path("assets/video2world/input0.jpg")
PROMPT = "At a city bus terminal at night, the center bus slowly pulls forward and stops."
NEGATIVE_PROMPT = "Blurry, low-quality video."


class AttentionCaptured(Exception):
    """Stop denoising once the capture and its reference output are recorded."""


def capture_attention(args):
    import torch
    from cosmos_predict2.configs.base.config_video2world import (
        get_cosmos_predict2_video2world_pipeline,
    )
    from cosmos_predict2.module.attention import FLASH_ATTN_3_AVAILABLE
    from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
    from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
    from imaginaire.utils import misc
    from safetensors.torch import save_file

    gpu = torch.cuda.get_device_name(0)
    if "H100" not in gpu:
        raise RuntimeError(f"Expected NVIDIA H100, got {gpu}")

    misc.set_random_seed(seed=args.seed, by_rank=True)
    config = get_cosmos_predict2_video2world_pipeline(
        model_size=MODEL_SIZE, resolution=RESOLUTION, fps=FPS
    )
    checkpoint = get_cosmos_predict2_video2world_checkpoint(
        model_size=MODEL_SIZE, resolution=RESOLUTION, fps=FPS
    )
    pipe = Video2WorldPipeline.from_config(
        config,
        dit_path=checkpoint,
        offload_text_encoder=True,
        downcast_text_encoder=True,
        load_prompt_refiner=False,
    )
    if pipe.text_guardrail_runner is None or pipe.video_guardrail_runner is None:
        raise RuntimeError("Expected the stock Cosmos guardrails to be enabled")

    attention = pipe.dit.blocks[0].self_attn.attn_op
    stock_attention = attention.local_attn

    def record(query, key, value):
        if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
            raise RuntimeError("Expected matching heads-last Q, K and V for dense self-attention")
        output = stock_attention(query, key, value)
        tensors = {"q": query, "k": key, "v": value, "output": output}
        if any(
            tensor.shape != query.shape or tensor.dtype != torch.bfloat16
            for tensor in tensors.values()
        ):
            raise RuntimeError(
                "Expected all captured tensors to have the same shape and BF16 dtype"
            )
        if not all(torch.isfinite(tensor).all().item() for tensor in tensors.values()):
            raise RuntimeError("Captured attention contains non-finite values")

        args.output_dir.mkdir(parents=True, exist_ok=False)
        save_file(
            {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()},
            args.output_dir / "attention.safetensors",
        )
        metadata = {
            "cosmos_revision": args.cosmos_revision,
            "model": f"nvidia/Cosmos-Predict2-{MODEL_SIZE}-Video2World",
            "model_revision": args.model_revision,
            "checkpoint": Path(checkpoint).name,
            "input_image": SAMPLE_IMAGE.as_posix(),
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": args.seed,
            "resolution": RESOLUTION,
            "fps": FPS,
            "aspect_ratio": ASPECT_RATIO,
            "block": 0,
            "denoising_step": 0,
            "conditioning": "conditional",
            "layout": "batch, tokens, heads, head_dim",
            "dtype": "bfloat16",
            "shape": list(query.shape),
            "original_strides": {name: list(tensor.stride()) for name, tensor in tensors.items()},
            "scale": 1 / math.sqrt(query.shape[-1]),
            "is_causal": False,
            "dropout_p": 0.0,
            "stock_dispatch": "flash_attention_3" if FLASH_ATTN_3_AVAILABLE else "torch_sdpa",
            "gpu": gpu,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Captured stock Cosmos attention {tuple(query.shape)} on {gpu}", flush=True)
        raise AttentionCaptured

    try:
        with patch.object(attention, "local_attn", record):
            pipe(
                input_path=str(SAMPLE_IMAGE),
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                aspect_ratio=ASPECT_RATIO,
                seed=args.seed,
            )
    except AttentionCaptured:
        return
    raise RuntimeError("Cosmos completed or rejected the sample before reaching the capture hook")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cosmos-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    capture_attention(parser.parse_args())
