# Status

## In place
- Threads, blocks, warps (working model)
- Grid-stride loops
- Shared-memory reduction and barrier placement
- Coalesced transpose (LeetGPU-Sols)
- Naive GEMM and tiled shared-memory GEMM
- Softmax (LeetGPU-Sols)

## Planned experiments
- Local timing tables for naive vs tiled / coalesced kernels
- One cuBLAS `sgemm` comparison
- Optional pybind of a single kernel into RawTorch

## Not in scope yet
- CUTLASS / Tensor Cores
- Fused attention-style kernels
- Production profiler-driven optimization
