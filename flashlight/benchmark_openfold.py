"""Time one OpenFold model-forward variant on the authors' synthetic features."""

import argparse
import json
import random
from pathlib import Path
from statistics import mean, median
from time import perf_counter

N_RES = 256
N_MSA = 256
N_TEMPLATES = 4
# model_1 appends one MSA row per template before the Evoformer stack.
N_INPUT_MSA = N_MSA - N_TEMPLATES
N_EXTRA_MSA = 1024
CHUNK_SIZE = 64
SEED = 0


def use_compiled_core(layer, core, *, contiguous_bias=False):
    def forward(q_x, kv_x, biases=None, **options):
        if any(options.values()):
            raise RuntimeError("The compiled benchmark expects OpenFold's dense attention path")
        q, k, v = layer._prep_qkv(q_x, kv_x)
        if q.shape[-2] != N_RES:
            raise RuntimeError(f"Expected a 256-position attention axis, got {q.shape[-2]}")
        biases = biases if biases is not None else []
        if contiguous_bias:
            # Normalize the column mask layout for both compiler paths.
            biases = [bias.contiguous() for bias in biases]
        attended = core(q, k, v, biases)
        return layer._wrap_up(attended.transpose(-2, -3), q_x)

    layer.forward = forward


def main(args):
    import numpy as np
    import torch
    import torch._dynamo.config
    from benchmark_openfold_upstream import generate_random_batch
    from openfold.config import model_config
    from openfold.model.model import AlphaFold
    from openfold.model.primitives import _attention
    from openfold.utils.tensor_utils import tensor_tree_map

    gpu = torch.cuda.get_device_name(0)
    if "H100" not in gpu:
        raise RuntimeError(f"Expected NVIDIA H100, got {gpu}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    config = model_config("model_1")
    config.globals.chunk_size = CHUNK_SIZE
    config.model.evoformer_stack.blocks_per_ckpt = None
    config.model.evoformer_stack.tune_chunk_size = False
    config.model.extra_msa.extra_msa_stack.tune_chunk_size = False
    config.model.template.template_pair_stack.tune_chunk_size = False
    model = AlphaFold(config).cuda().eval()

    if args.mode == "flashlight":
        from monkeypatch import disable_flashattention_replacement
        from monkeypatch.fusion import (  # noqa: F401
            block_reduction,
            dependent_reduction_fusion,
            reduction_kernel_fusion,
        )

        disable_flashattention_replacement()

    if args.mode != "eager":
        torch._dynamo.config.cache_size_limit = 65536
        core = torch.compile(_attention, dynamic=False, fullgraph=True)
        for block in model.evoformer.blocks:
            use_compiled_core(block.msa_att_row.mha, core)
            use_compiled_core(block.msa_att_col._msa_att.mha, core, contiguous_bias=True)

    samples = [
        generate_random_batch(config, N_INPUT_MSA, N_TEMPLATES, N_RES, N_EXTRA_MSA)
        for _ in range(args.batch_size)
    ]
    batch = {name: torch.stack([sample[name] for sample in samples]) for name in samples[0]}
    batch = tensor_tree_map(lambda tensor: tensor.cuda(), batch)

    with torch.no_grad():
        torch.cuda.synchronize()
        start = perf_counter()
        for _ in range(args.warmups):
            model(batch)
            torch.cuda.synchronize()
        warmup_s = perf_counter() - start
        print(f"{args.mode}: warmup finished in {warmup_s:.2f} s", flush=True)
        torch.cuda.reset_peak_memory_stats()

        samples_ms = []
        for iteration in range(args.iterations):
            start = perf_counter()
            output = model(batch)
            torch.cuda.synchronize()
            elapsed_ms = (perf_counter() - start) * 1000
            samples_ms.append(elapsed_ms)
            if not torch.isfinite(output["plddt"]).all().item():
                raise RuntimeError("Non-finite model output")
            del output
            print(f"{args.mode}: {iteration + 1}/{args.iterations} {elapsed_ms:.2f} ms", flush=True)

    result = {
        "mode": args.mode,
        "gpu": gpu,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "model": "model_1",
        "dtype": str(next(model.parameters()).dtype),
        "seed": SEED,
        "batch_size": args.batch_size,
        "residues": N_RES,
        "msa_sequences": N_MSA,
        "input_msa_sequences": N_INPUT_MSA,
        "templates": N_TEMPLATES,
        "extra_msa_sequences": N_EXTRA_MSA,
        "evoformer_blocks": len(model.evoformer.blocks),
        "chunk_size": config.globals.chunk_size,
        "passes": batch["aatype"].shape[-1],
        "warmups": args.warmups,
        "warmup_s": warmup_s,
        "samples_ms": samples_ms,
        "mean_ms": mean(samples_ms),
        "median_ms": median(samples_ms),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("eager", "inductor", "flashlight"))
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
