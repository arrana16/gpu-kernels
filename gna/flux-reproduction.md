# GNA on FLUX.1-dev

The [GNA paper's image experiment](https://arxiv.org/html/2504.16922v1#S4.SS3)
uses FLUX.1-dev with the URAE 4K adapter. It generates 4096 × 4096 images with a
256 × 256 image token grid and uses an 80 × 80 window. Table 4 reports 1.37×,
1.40×, and 1.45× end-to-end speedups for strides 1 × 1, 16 × 1, and 16 × 16
when the first 9 of 28 denoising steps stay dense.

## Measured attention timings

On November 11, 2026, the [Modal B200 run](https://modal.com/apps/arrana16/main/ap-PLQ5PJE7HjGSvvIRKBHXEt)
profiled batch size 1, 24 heads of dimension 128, FP16, and a 256 × 256 image
token grid. The sparse cases use a 16 × 16 Q tile and a 16 × 8 KV tile. Each
case was profiled three times with ten warmup calls before each profile.

| Case | Attention kernel (ms) | Profiler total (ms) | Dense / total |
| --- | ---: | ---: | ---: |
| Dense cuDNN | 39.654 | 39.654 | 1.00× |
| Stride 1 × 1 | 8.769 | 9.310 | 4.26× |
| Stride 16 × 1 | 7.102 | 7.651 | 5.18× |
| Stride 16 × 16 | 4.489 | 5.047 | 7.86× |

Values are medians. The profiler total includes token permutation. This is the
synthetic image attention comparison, which isolates the stride effect with
fixed window and tile shapes. The paper measures generation separately.
[NATTEN profiler example](https://natten.org/profiler/#2-d-use-case-flux)

## End-to-end target

Use the public [URAE 4K notebook and adapter](https://github.com/Huage001/URAE#4k-resolution-or-higher)
with the gated [FLUX.1-dev checkpoint](https://huggingface.co/black-forest-labs/FLUX.1-dev).
Run URAE with dense attention to establish the baseline. In the GNA run, switch
image queries to NATTEN after step 9 and retain text keys as extra context. Keep
text queries dense. Use the same checkpoint, adapter, prompt, seed, precision,
and denoising schedule for each case.

Time generation after model loading, including VAE decoding. Save paired
images and report the latency ratio separately from the attention benchmark.
Use the authors' evaluation prompts and metrics to reproduce the quality
evaluation.
