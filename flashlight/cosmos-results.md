# FlashLight replay of Cosmos attention on H100

FlashLight replayed the captured Cosmos attention call in **7.133 ms**. It
was **3.70× faster** than Inductor with automatic attention replacement
disabled and used **98.2% less peak allocated GPU memory**. Direct cuDNN SDPA
was the fastest path at **5.626 ms**. Default Inductor also lowered the
attention to cuDNN and measured **5.783 ms**.

| Replay path | Median GPU time (ms) | Mean GPU time (ms) | Peak allocated (GiB) | First call (s) |
|---|---:|---:|---:|---:|
| cuDNN SDPA | 5.626 | 5.619 | 0.320 | 0.652 |
| Default `torch.compile` | 5.783 | 5.758 | 0.558 | 3.255 |
| Inductor with attention replacement disabled | 26.404 | 26.479 | 25.900 | 2.041 |
| FlashLight | 7.133 | 7.146 | 0.479 | 7.110 |

The four paths ran in separate processes on the same Modal NVIDIA H100
80GB using PyTorch 2.5.0 and CUDA 12.1. Each path used ten warmups and twenty
timed CUDA-event samples. Timings cover one attention call on already
resident tensors. Peak memory is PyTorch-tracked allocation and includes
the resident Q, K, and V tensors. First-call wall time includes any
compilation or backend initialization.

The capture comes from block 0 at denoising step 0 of
Cosmos-Predict2-2B-Video2World at 480p and 10 fps. The BF16 tensor shape is
`[1, 20736, 16, 128]` in batch, token, head, and head-dimension order. All four
paths passed the elementwise check against the saved Cosmos PyTorch 2.6
output at `atol=0.02` and `rtol=0.01`. FlashLight's maximum absolute error
was 0.001953 and its relative L2 error was 0.0000425.

[Raw timings and metadata](results/cosmos-h100-dba206d5f72e.json) · [Modal run](https://modal.com/apps/arrana16/main/ap-lbb9dwDzpRW4zZTJwVObvO)
