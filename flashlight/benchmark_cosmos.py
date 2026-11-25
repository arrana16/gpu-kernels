"""Replay one Cosmos self-attention backend and record GPU timings and accuracy."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from statistics import mean, median
from time import perf_counter


def main(args):
    import torch
    from attention_variants.vanilla import attention_pytorch_nogqa
    from safetensors.torch import load_file
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    gpu = torch.cuda.get_device_name(0)
    if "H100" not in gpu:
        raise RuntimeError(f"Expected NVIDIA H100, got {gpu}")

    metadata = json.loads((args.capture / "metadata.json").read_text())
    if metadata["is_causal"] or metadata["dropout_p"] or metadata["dtype"] != "bfloat16":
        raise ValueError("Expected an unmasked, dropout-free BF16 capture")
    tensors = load_file(args.capture / "attention.safetensors")
    if any(
        list(tensor.shape) != metadata["shape"] or tensor.dtype != torch.bfloat16
        for tensor in tensors.values()
    ):
        raise ValueError("Captured tensors do not match the declared shape and BF16 dtype")
    query, key, value = (tensors[name].cuda().transpose(1, 2) for name in ("q", "k", "v"))
    scale = metadata["scale"]

    if args.mode == "flashlight":
        from monkeypatch import disable_flashattention_replacement
        from monkeypatch.fusion import (  # noqa: F401
            block_reduction,
            dependent_reduction_fusion,
            reduction_kernel_fusion,
        )

        disable_flashattention_replacement()

    # Both compiler modes take the same explicit attention expression and settings.
    attention = (
        scaled_dot_product_attention
        if args.mode == "cudnn_sdpa"
        else torch.compile(attention_pytorch_nogqa, dynamic=False, fullgraph=True)
    )
    backend = (
        sdpa_kernel(SDPBackend.CUDNN_ATTENTION) if args.mode == "cudnn_sdpa" else nullcontext()
    )

    def forward():
        return attention(query, key, value, scale=scale)

    print(f"{args.mode}: running {metadata['shape']} on {gpu}", flush=True)
    with torch.inference_mode(), backend:
        torch.cuda.synchronize()
        start = perf_counter()
        actual = forward()
        torch.cuda.synchronize()
        first_call_s = perf_counter() - start
        print(f"{args.mode}: first call finished in {first_call_s:.2f} s", flush=True)

        reference = tensors["output"].cuda().transpose(1, 2).float()
        actual = actual.float()
        if actual.shape != reference.shape or not torch.isfinite(actual).all().item():
            raise RuntimeError("Replay produced an invalid output")
        delta = actual - reference
        error = delta.abs()
        atol, rtol = 0.02, 0.01
        accuracy = {
            "max_absolute_error": error.max().item(),
            "mean_absolute_error": error.mean().item(),
            "relative_l2_error": (
                torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference)
            ).item(),
            "atol": atol,
            "rtol": rtol,
            "elements_exceeding_tolerance": (error > atol + rtol * reference.abs()).sum().item(),
            "elements_compared": reference.numel(),
        }
        del actual, reference, delta, error

        for _ in range(args.warmups):
            forward()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        torch.cuda.synchronize()
        resident_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for start_event, end_event in zip(starts, ends, strict=True):
            start_event.record()
            forward()
            end_event.record()
        torch.cuda.synchronize()
        times = [a.elapsed_time(b) for a, b in zip(starts, ends, strict=True)]
        peak_bytes = torch.cuda.max_memory_allocated()

    result = {
        "mode": args.mode,
        "status": "ok",
        "gpu": gpu,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "warmups": args.warmups,
        "first_call_s": first_call_s,
        "samples_ms": times,
        "median_ms": median(times),
        "mean_ms": mean(times),
        "resident_allocated_bytes": resident_bytes,
        "peak_allocated_bytes": peak_bytes,
        "accuracy": accuracy,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{args.mode}: {result['median_ms']:.3f} ms median, accuracy {accuracy}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cudnn_sdpa", "inductor", "flashlight"))
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    main(parser.parse_args())
