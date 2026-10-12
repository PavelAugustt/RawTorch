# LeetGPU CUDA Solutions

CUDA kernel implementations written while studying GPU programming, with
matching walkthroughs on YouTube.

Focus areas: thread indexing, memory coalescing, shared memory, tiled GEMM,
and a numerically stable softmax. Educational work — not a production kernel
library.

## Implementations

| Kernel | Focus | File |
|---|---|---|
| Vector Addition | Grid-stride loops, coalesced loads | `Vector-Sum/naive_impl.cu` |
| Matrix Transpose (naive) | 2D indexing, uncoalesced stores | `Mat-Trans/naive_impl.cu` |
| Matrix Transpose (coalesced) | Shared tile, bank-conflict padding | `Mat-Trans/mem_coalesced.cu` |
| Matrix Multiply (naive) | One thread per output element | `Mat-Mult/naive_impl.cu` |
| Matrix Multiply (tiled) | Shared-memory tiling | `Mat-Mult/tiled_impl.cu` |
| Softmax (row-wise) | Max subtraction + block reduction | `Softmax/naive_impl.cu` |
| ReLU | Element-wise activation | `ReLU/naive_impl.cu` |
| Leaky ReLU | Element-wise kernel with branching | `Leaky-ReLU/naive_impl.cu` |
| Reverse Array | Index mapping | `Reverse-Array/naive_impl.cu` |

## Design notes

- **Coalesced transpose** loads a tile into shared memory, then writes with
  swapped block indices so both loads and stores hit contiguous addresses.
  `tile[BLOCK][BLOCK+1]` padding reduces shared-memory bank conflicts.
- **Tiled GEMM** reuses `TILE×TILE` panels of A and B in shared memory instead
  of streaming the full reduction dimension from global memory each time.
- **Softmax** subtracts the row max before `exp` to avoid overflow, then
  normalizes with a shared-memory sum reduction.

These are standard textbook patterns, not cuBLAS / Tensor Core / fused-attention
implementations.

## Timing

`CUDA-C/GEMM-GPU/_gemm_tiled.cu` in the related CUDA-C repo compares naive vs
tiled GEMM with warmup and averaged `cudaEvent` timings. Transpose naive vs
coalesced can be timed the same way on a local GPU.

## Video explanations

https://www.youtube.com/@pavelaugust

- Writing a first CUDA kernel (vector addition)
- Matrix transpose and memory coalescing
- ReLU, Leaky ReLU, and warp divergence

## Related work

- CUDA learning lab: https://github.com/PavelAugustt/CUDA-C
- Tensor / autograd study: https://github.com/PavelAugustt/RawTorch

## Notes

Kernels use the LeetGPU `solve()` entry point and expect a platform `solve.h`.
They can be adapted into a standalone `main` if needed.
