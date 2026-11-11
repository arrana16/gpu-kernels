"""Compare dense and GNA 4K FLUX 3 Image Klein generation on Modal."""

import modal

from gna.modal_app import image as gna_image

MODEL = "black-forest-labs/flux-3-image-klein"
MODEL_REVISION = "09561231f44e41381c461caf53d5452005a60ac9"
DIFFUSERS_REVISION = "0905b667fe3f37f43b6ef9c6e05ad20636206938"
SIZE = 4096
GRID = (SIZE // 16, SIZE // 16)
WINDOW = (80, 80)
STRIDE = (16, 16)
STEPS = 16
DENSE_STEPS = 5
CAPTION = (
    "Wide landscape photograph from the rocky shore of an alpine lake just after sunrise. "
    "A small wooden sailboat with a single cream-colored triangular sail crosses the lake "
    "slightly to the left of center, leaving a narrow V-shaped wake. The lower foreground "
    "shows smooth wet stones and dark green reeds along the waterline. A band of evergreen "
    "trees stretches across the far shore below a range of jagged mountains, whose ridges "
    "are lightly dusted with snow. Warm light grazes the peaks from the right while the "
    "lake and foreground sit in cool blue shadow. Thin mist hovers over the water behind "
    "the boat. The mast, sail, and mountains have soft, slightly rippled reflections. "
    "Natural colors, clear detail in the boat and rocks, and a quiet morning atmosphere."
)

model_image = (
    gna_image.apt_install("git")
    .pip_install(
        "torchvision==0.27.0+cu130", index_url="https://download.pytorch.org/whl/cu130"
    )
    .pip_install(
        f"diffusers @ git+https://github.com/huggingface/diffusers.git@{DIFFUSERS_REVISION}",
        "transformers==5.8.1",
        "accelerate==1.13.0",
    )
    .run_commands(
        'python -c "import torch, natten, torchvision; '
        'from diffusers import Flux3Transformer2DModel, ModularPipeline; '
        'from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor; '
        "assert torch.__version__ == '2.12.0+cu130'; assert natten.HAS_LIBNATTEN\""
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

app = modal.App("gpu-kernels-flux3-gna")
cache = modal.Volume.from_name("gna-flux", create_if_missing=True)


def _image_attention(query, key, value, *, grid, window, text_tokens, dense, **kwargs):
    """Apply local image attention with global text context to heads-last Q/K/V."""
    import natten
    import torch

    def image_grid(tensor):
        return tensor[:, text_tokens:].unflatten(1, grid).contiguous()

    text_context = {}
    if text_tokens:
        text_context = {
            "additional_keys": key[:, :text_tokens].contiguous(),
            "additional_values": value[:, :text_tokens].contiguous(),
        }

    image_output = natten.na2d(
        image_grid(query),
        image_grid(key),
        image_grid(value),
        kernel_size=window,
        stride=STRIDE,
        scale=kwargs.get("scale"),
        backend="blackwell-fna",
        q_tile_shape=(16, 16),
        kv_tile_shape=(16, 8),
        **text_context,
    ).flatten(1, 2)
    if not text_tokens:
        return image_output

    text_output = dense(query[:, :text_tokens], key, value, **kwargs)
    return torch.cat((text_output, image_output), dim=1)


@app.function(
    image=model_image.add_local_python_source("gna"),
    gpu="B200",
    timeout=3600,
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def reproduce_flux3(caption: str = CAPTION, seed: int = 0):
    """Generate a matched pair and report synchronized pipeline latency."""
    import json
    from pathlib import Path
    from time import perf_counter
    from unittest.mock import patch
    from uuid import uuid4

    import torch
    from diffusers import ModularPipeline
    from diffusers.models.transformers import transformer_flux3
    from huggingface_hub import hf_hub_download

    # Surface gated-repository errors directly, before Diffusers loads the model.
    hf_hub_download(MODEL, "modular_model_index.json", revision=MODEL_REVISION, token=True)
    blocks = ModularPipeline.from_pretrained(MODEL, revision=MODEL_REVISION).blocks
    blocks.sub_blocks.pop("prompt_upsampler")
    pipe = blocks.init_pipeline(MODEL)
    pipe.load_components(revision=MODEL_REVISION, dtype={"vae": torch.float32, "default": torch.bfloat16})
    pipe.vae.enable_tiling()
    pipe.to("cuda")
    cache.commit()

    image_tokens = GRID[0] * GRID[1]
    sparse_layers = len(pipe.transformer.mode_blocks["image"]) + len(pipe.transformer.single_transformer_blocks)
    dense = transformer_flux3.dispatch_attention_fn

    def generate(use_gna):
        step = 0
        sparse_calls = 0
        text_tokens = 0

        def start_step(_module, _args, kwargs):
            nonlocal step, text_tokens
            step += 1
            text_tokens = kwargs["encoder_hidden_states"].shape[1]
            if kwargs["hidden_states"].shape[1] != image_tokens or kwargs.get("image_cond_hidden_states") is not None:
                raise RuntimeError("Expected a 256 × 256 image-token grid and no reference image")

        def attention(query, key, value, **kwargs):
            nonlocal sparse_calls
            tokens = query.shape[1]
            if not use_gna or step <= DENSE_STEPS or tokens == text_tokens:
                return dense(query, key, value, **kwargs)
            if (
                tokens not in (image_tokens, image_tokens + text_tokens)
                or query.shape != key.shape or query.shape != value.shape
                or kwargs.get("attn_mask") is not None or kwargs.get("parallel_config") is not None
            ):
                raise RuntimeError("Unexpected FLUX 3 attention inputs for GNA")
            sparse_calls += 1
            return _image_attention(
                query, key, value, grid=GRID, window=WINDOW,
                text_tokens=tokens - image_tokens, dense=dense, **kwargs,
            )

        generator = torch.Generator().manual_seed(seed)
        with (
            torch.inference_mode(),
            pipe.transformer.register_forward_pre_hook(start_step, with_kwargs=True),
            patch.object(transformer_flux3, "dispatch_attention_fn", attention),
        ):
            torch.cuda.synchronize()
            started = perf_counter()
            images = pipe(
                prompt=caption, generator=generator, width=SIZE, height=SIZE,
                num_inference_steps=STEPS, output="images",
            )
            torch.cuda.synchronize()
            seconds = perf_counter() - started

        expected_calls = (STEPS - DENSE_STEPS) * sparse_layers if use_gna else 0
        if step != STEPS or sparse_calls != expected_calls or images[0].size != (SIZE, SIZE):
            raise RuntimeError(f"Unexpected run: {step} steps, {sparse_calls} GNA calls, {images[0].size} image")
        return images[0], {"seconds": seconds, "gna_attention_calls": sparse_calls}

    run_dir = Path("/cache/flux3-runs") / uuid4().hex[:12]
    run_dir.mkdir(parents=True)
    results = {
        "model": MODEL, "revision": MODEL_REVISION, "gpu": torch.cuda.get_device_name(),
        "caption": caption, "seed": seed, "image_size": [SIZE, SIZE], "steps": STEPS,
        "dense_steps_before_gna": DENSE_STEPS, "window": WINDOW, "stride": STRIDE,
        "output_dir": str(run_dir.relative_to("/cache")),
    }
    for name, use_gna in (("dense", False), ("gna-16x16", True)):
        print(f"Generating {name} image", flush=True)
        image, results[name] = generate(use_gna)
        image.save(run_dir / f"{name}.png")
        image.thumbnail((2048, 2048))
        image.save(run_dir / f"{name}-preview.jpg", quality=90)
        print(f"{name} took {results[name]['seconds']:.2f}s", flush=True)

    results["speedup"] = results["dense"]["seconds"] / results["gna-16x16"]["seconds"]
    (run_dir / "timings.json").write_text(json.dumps(results, indent=2) + "\n")
    cache.commit()
    print(f"Pipeline speedup: {results['speedup']:.3f}x. Outputs: gna-flux:/{results['output_dir']}")
    return results
