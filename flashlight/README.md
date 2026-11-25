# FlashLight benchmarks on Modal

## AlphaFold2

The [Figure 4 artifact](https://github.com/bozhiyou/flashlight/blob/mlsys26-ae/benchmarks/run_fig4_evoformer.py)
times Evoformer's attention calculation at sequence length 256, four heads,
head dimensions 64 and 128, and batches 1, 2, 4, 8, 16, and 32. Its output
labels `ipa` and `ipa_compiled` refer to the unfused reference and the
FlashLight-compiled Evoformer calculation for this sweep. `--stock-inductor`
runs the same benchmark with FlashLight's patch activation omitted. In that
control, `ipa_compiled` measures stock Inductor.

The separate OpenFold run adapts the [authors' public synthetic-input harness](https://github.com/bozhiyou/flashlight/blob/openfold/benchmarks/benchmark_openfold.py)
to match [§4.4](https://arxiv.org/html/2511.02043v4#S4.SS4). It uses 48 Evoformer
blocks, 256 residues and 256 MSA rows, and compiles the row and column attention
modules in the compiled variants. Eager, stock Inductor, and FlashLight run
in separate processes with matched seed, model initialization, and synthetic
features. The test times model forwards after warmup and excludes input
preparation and compilation. It uses stock OpenFold chunk size with chunk-size
autotuning disabled so each implementation follows the same schedule.

Both jobs use one H100, public source, and generated benchmark inputs. They
require no model weights or Hugging Face token. Run from the repository root.

```bash
modal run --write-result /tmp/evoformer-h100.csv flashlight/alphafold_modal.py::evoformer
modal run --write-result /tmp/openfold-h100-b1.json flashlight/alphafold_modal.py::openfold
```

Use `--batch-size` on `openfold` for another paper batch size. The paper reports
both H100 and A100. These jobs target the H100, with the upstream best-effort
warmup for the Figure 4 sweep. The OpenFold job records individual timings and
peak allocated GPU memory. Both write results to the `flashlight-alphafold-results`
Modal Volume and print the path.

## Cosmos attention capture

This captures the first dense self-attention call in block 0 of NVIDIA's
Cosmos-Predict2-2B-Video2World 480p, 10 fps checkpoint. It uses the pinned
NVIDIA runtime and its sample image on a single Modal H100. Cosmos runs its
prompt guardrail and executes the original attention operation once. The run
stops at that point, before generating a video.

From the repository root, use the existing `huggingface-secret` Modal secret
with an `HF_TOKEN` that can read [Cosmos-Predict2-2B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World),
[Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1), and
[Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B).

```bash
modal run flashlight/modal_app.py::prepare_checkpoints --check-only
modal run flashlight/modal_app.py::prepare_checkpoints
modal run flashlight/modal_app.py::capture
```

The first command checks access before downloading weights. The second caches
the checkpoints on the `flashlight-cosmos` Volume using CPU. The third requests
`H100!`, which disables automatic upgrades to H200.

To build the CUDA image ahead of the capture without requesting a GPU, run
`modal run flashlight/modal_app.py::build_capture_image`.

The capture prints a unique `/captures/<run-id>` path. Download it with:

```bash
modal volume get flashlight-cosmos /captures/<run-id> ./cosmos-capture
```

`attention.safetensors` holds `q`, `k`, `v`, and the stock attention `output`,
all contiguous BF16 in `[batch, tokens, heads, head_dim]` order. `metadata.json`
records the source strides, scale, model and code revisions, and the stock
dispatch path. The Q and K tensors already include normalization and RoPE.
The output is recorded before the output projection. The capture is the input
fixture for a separate FlashLight replay on PyTorch 2.5.0.
