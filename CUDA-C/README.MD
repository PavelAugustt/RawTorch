# CUDA-C

CUDA C experiments written while learning GPU programming: execution model,
memory behavior, reductions, and matrix multiplication.

Educational / exploratory — standalone programs with CPU checks where useful,
not a packaged kernel library.

## Highlights

| Folder | Contents |
|---|---|
| `Hello-GPU`, `DeviceProp-GPU` | First kernel launch, device limits |
| `Parallel-VS-GPU` | Indexing, multi-block launches, grid-stride loops |
| `SharedMemory-GPU` | Shared-memory dot product / tree reduction |
| `Transpose-2D-GPU` | 2D matrix transpose |
| `GEMM-GPU/_gemm.cu` | Naive GEMM with CPU verification |
| `GEMM-GPU/_gemm_tiled.cu` | Shared-memory tiled GEMM + `cudaEvent` timing |
| `Profiling-GPU` | Early timing notes on two naive GEMM launch styles |

Other folders (`Repetition-GPU`, older variants) are practice rewrites kept for
reference.

## Topics

- Thread, block, and grid organization
- Grid-stride loops for arbitrary problem sizes
- Device allocation and host/device transfers
- Shared-memory reductions and `__syncthreads` placement
- Naive and tiled GEMM
- Basic kernel timing with warmup and averaged `cudaEvent` runs

## Build

```bash
nvcc -o shared SharedMemory-GPU/main.cu
nvcc -o gemm GEMM-GPU/_gemm.cu
nvcc -o gemm_tiled GEMM-GPU/_gemm_tiled.cu
```

Optional: set `-arch` for your GPU (see `compilationCMD.txt`).

## Related work

- Compact kernels + videos: https://github.com/PavelAugustt/LeetGPU-Sols
- Tensor / autograd study: https://github.com/PavelAugustt/RawTorch
- YouTube: https://www.youtube.com/@pavelaugust

## Status

Covered: fundamentals through tiled GEMM and simple timing.
Next experiments of interest: cuBLAS baselines, deeper Nsight use, and wiring
a kernel into RawTorch via pybind — not claimed as done here.
