"""Generate paired 4K FLUX.1-dev + URAE images on a Modal B200."""

import modal

from gna.modal_app import image as gna_image

FLUX = "black-forest-labs/FLUX.1-dev"
FLUX_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
URAE = "Huage001/URAE"
URAE_REVISION = "52dc41fca55392333bcf3f029ae42ce1327ba906"
URAE_CODE = "f636a2fd27eb3dad860d0f66f404416f5461cd8b"
GRID = (256, 256)
WINDOW = (80, 80)
STEPS = 28
DENSE_STEPS = 9
STRIDES = {"dense": None, "1x1": (1, 1), "16x1": (16, 1), "16x16": (16, 16)}

model_image = (
    gna_image.apt_install("git")
    .pip_install(
        "diffusers==0.31.0",
        "transformers==4.49.0",
        "accelerate==1.6.0",
        "peft==0.14.0",
        "patch-conv==0.0.1b0",
        "sentencepiece",
        "protobuf",
    )
    .run_commands(
        "git init /opt/urae",
        f"git -C /opt/urae fetch --depth=1 https://github.com/Huage001/URAE.git {URAE_CODE}",
        "git -C /opt/urae checkout --detach FETCH_HEAD",
        'PYTHONPATH=/opt/urae python -c "from pipeline_flux import FluxPipeline; '
        'from transformer_flux import FluxTransformer2DModel; from patch_conv import convert_model"',
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

app = modal.App("gpu-kernels-flux-gna")
cache = modal.Volume.from_name("gna-flux", create_if_missing=True)


def _joint_attention(query, key, value, *, grid, window, stride, scale):
    """Run local image attention with global text keys, then restore FLUX order."""
    import natten
    import torch
    from torch.nn.functional import scaled_dot_product_attention

    batch, heads, tokens, dim = query.shape
    height, width = grid
    text_tokens = tokens - height * width
    if text_tokens <= 0 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("Expected joint FLUX Q/K/V with text followed by the image grid")

    text_output = scaled_dot_product_attention(
        query[:, :, :text_tokens], key, value, scale=scale,
    )

    def image_grid(tensor):
        return tensor[:, :, text_tokens:].transpose(1, 2).contiguous().view(
            batch, height, width, heads, dim
        )

    image_output = natten.na2d(
        image_grid(query),
        image_grid(key),
        image_grid(value),
        kernel_size=window,
        stride=stride,
        scale=scale,
        additional_keys=key[:, :, :text_tokens].transpose(1, 2).contiguous(),
        additional_values=value[:, :, :text_tokens].transpose(1, 2).contiguous(),
        backend="blackwell-fna",
        q_tile_shape=(16, 16),
        kv_tile_shape=(16, 8),
    )
    image_output = image_output.reshape(batch, height * width, heads, dim).transpose(1, 2)
    return torch.cat((text_output, image_output), dim=2)


@app.function(
    image=model_image.add_local_python_source("gna"),
    gpu="B200",
    timeout=1800,
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def reproduce_flux(
    cases: str = "dense,16x16",
    prompt: str = "A sailboat crossing a mountain lake in the morning light",
    seed: int = 0,
):
    """Generate one image per case and record synchronized generation times."""
    import json
    import sys
    from pathlib import Path
    from time import perf_counter
    from types import SimpleNamespace
    from unittest.mock import patch
    from uuid import uuid4

    import torch
    from patch_conv import convert_model
    from torch.nn.functional import scaled_dot_product_attention

    names = [case.strip() for case in cases.split(",")]
    if any(name not in STRIDES for name in names) or len(names) != len(set(names)):
        raise ValueError("Choose unique cases from dense, 1x1, 16x1, and 16x16")

    sys.path.insert(0, "/opt/urae")
    import attention_processor as urae_attention
    from pipeline_flux import FluxPipeline
    from transformer_flux import FluxTransformer2DModel

    dtype = torch.bfloat16
    transformer = FluxTransformer2DModel.from_pretrained(
        FLUX, subfolder="transformer", revision=FLUX_REVISION, torch_dtype=dtype
    )
    pipe = FluxPipeline.from_pretrained(
        FLUX, transformer=transformer, revision=FLUX_REVISION, torch_dtype=dtype
    )
    pipe.scheduler.register_to_config(use_dynamic_shifting=False, time_shift=10)
    pipe.load_lora_weights(
        URAE,
        revision=URAE_REVISION,
        weight_name="urae_4k_adapter_lora_conversion_dev.safetensors",
    )
    pipe.vae = convert_model(pipe.vae, splits=4)
    pipe.to("cuda")
    cache.commit()

    run_dir = Path("/cache/runs") / uuid4().hex[:12]
    run_dir.mkdir(parents=True)
    results = {}
    for name in names:
        stride = STRIDES[name]
        step = 0
        sparse_calls = 0

        def attention(query, key, value, **kwargs):
            nonlocal sparse_calls
            if stride is None or step < DENSE_STEPS:
                return scaled_dot_product_attention(query, key, value, **kwargs)
            if query.shape[-2] != 512 + GRID[0] * GRID[1]:
                raise RuntimeError(f"Unexpected FLUX sequence length {query.shape[-2]}")
            sparse_calls += 1
            return _joint_attention(
                query, key, value, grid=GRID, window=WINDOW, stride=stride,
                scale=kwargs.get("scale"),
            )

        def finish_step(_pipe, index, _timestep, kwargs):
            nonlocal step
            step = index + 1
            return kwargs

        # Only the URAE attention processor sees this replacement.
        with patch.object(
            urae_attention, "F", SimpleNamespace(scaled_dot_product_attention=attention)
        ), torch.inference_mode():
            torch.cuda.synchronize()
            start = perf_counter()
            output = pipe(
                prompt,
                height=4096,
                width=4096,
                guidance_scale=3.5,
                num_inference_steps=STEPS,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(seed),
                ntk_factor=10,
                proportional_attention=True,
                callback_on_step_end=finish_step,
            ).images[0]
            torch.cuda.synchronize()
            elapsed = perf_counter() - start
        blocks = len(transformer.transformer_blocks) + len(transformer.single_transformer_blocks)
        expected_calls = 0 if stride is None else (STEPS - DENSE_STEPS) * blocks
        if step != STEPS or sparse_calls != expected_calls:
            raise RuntimeError(f"Unexpected schedule: {step} steps, {sparse_calls} GNA calls")
        output.save(run_dir / f"{name}.png")
        results[name] = {"seconds": elapsed, "gna_calls": sparse_calls}
        print(f"{name}: {elapsed:.2f} s, {sparse_calls} GNA calls", flush=True)

    if "dense" in results:
        baseline = results["dense"]["seconds"]
        for result in results.values():
            result["speedup_vs_dense"] = baseline / result["seconds"]
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "flux_revision": FLUX_REVISION,
        "urae_revision": URAE_REVISION,
        "prompt": prompt,
        "seed": seed,
        "steps": STEPS,
        "dense_steps_for_gna": DENSE_STEPS,
        "image_token_grid": GRID,
        "window": WINDOW,
        "cases": results,
    }
    (run_dir / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
    cache.commit()
    print(f"Saved images and timings to Modal volume gna-flux at /runs/{run_dir.name}")
