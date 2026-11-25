# Cosmos attention capture for FlashLight

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
