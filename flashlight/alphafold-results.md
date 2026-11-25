# FlashLight AlphaFold2 results on Modal H100

## Evoformer attention

FlashLight was **4.87–8.20× faster than the artifact's eager implementation**
and **1.85–2.82× faster than stock Inductor**. The stock control used the same
artifact and maximum autotuning with the FlashLight patches omitted. Eager
timings across the two jobs differed by at most 2.5%.

The artifact uses BF16, four attention heads, sequence length 256 on both
axes, and CUDA-event timings after warmup. Both jobs observed 1980 MHz during
warmup. The paper reports capping the SM clock at 1290 MHz. The table uses the
eager timings from the FlashLight job. [Paper §4](https://arxiv.org/html/2511.02043v4#S4)

| Head dimension | Batch | Eager (ms) | Stock Inductor (ms) | FlashLight (ms) | Speedup vs. Inductor |
|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 0.971 | 0.347 | 0.140 | 2.48× |
| 64 | 2 | 1.899 | 0.654 | 0.258 | 2.54× |
| 64 | 4 | 3.748 | 1.247 | 0.483 | 2.58× |
| 64 | 8 | 7.468 | 2.457 | 0.942 | 2.61× |
| 64 | 16 | 14.923 | 4.864 | 1.844 | 2.64× |
| 64 | 32 | 30.093 | 10.330 | 3.669 | 2.82× |
| 128 | 1 | 1.189 | 0.454 | 0.244 | 1.86× |
| 128 | 2 | 2.346 | 0.866 | 0.469 | 1.85× |
| 128 | 4 | 4.633 | 1.689 | 0.904 | 1.87× |
| 128 | 8 | 9.237 | 3.333 | 1.787 | 1.87× |
| 128 | 16 | 18.493 | 6.630 | 3.557 | 1.86× |
| 128 | 32 | 37.108 | 13.496 | 7.057 | 1.91× |

The raw CSVs are in [results](results/). The Modal logs record compilation and
warmup for the [FlashLight run](https://modal.com/apps/arrana16/main/ap-69a44jIbocJaesPkrR4OrT)
and the [stock Inductor control](https://modal.com/apps/arrana16/main/ap-KQLZI9YKuceREjvkeI83Tv).
