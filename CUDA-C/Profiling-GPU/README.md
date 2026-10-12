# Profiling notes

## Measurements

Two naive GEMM launch styles on the same problem size, timed with
`cudaEventElapsedTime`:

| Style | Rough ms (measured locally) | Idea |
|---|---|---|
| Nested grid-stride loops | ~33 ms | Threads may cover multiple cells via stride |
| One cell per thread + bounds `if` | ~10 ms | Simpler mapping, less loop overhead here |

Numbers depend on GPU, problem size, and whether output buffers were zeroed
before accumulation. This is a personal experiment log, not a universal ranking.

## Timing method used later

`GEMM-GPU/_gemm_tiled.cu` compares naive vs tiled GEMM with:

1. Warmup launches
2. `cudaEvent` around several iterations
3. Mean elapsed time
4. `cudaMemset` on accumulation buffers before `+=` kernels

## Open questions

- Bottleneck breakdown in Nsight Compute (beyond wall-clock ms)
- Gap vs cuBLAS `sgemm` on the same size
- Whether a given kernel is memory-bound or compute-bound
