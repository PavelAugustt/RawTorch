# LeetGPU CUDA Solutions

Included in [RawTorch](https://github.com/bozo10n/RawTorch). Original repository:
https://github.com/PavelAugustt/LeetGPU-Sols

A collection of CUDA kernel implementations and optimization experiments
developed alongside my technical explanations on GPU programming.

The goal of this repository is to understand GPU execution by implementing
operations directly, starting from simple kernels and progressively exploring
performance considerations such as memory access patterns and warp behavior.

## Implementations

| Kernel | Focus |
|---|---|
| Vector Addition | Basic CUDA execution and indexing |
| Matrix Transpose | Memory access patterns and coalescing |
| Matrix Multiplication | Parallel matrix computation / GEMM fundamentals |
| ReLU | Parallel element-wise activation |
| Leaky ReLU | Element-wise kernels and branching |
| Reverse Array | Index mapping and parallel memory operations |

## Video Explanations

I explain several of these implementations on YouTube:

https://www.youtube.com/@pavelaugust

Topics include:

- Writing a first CUDA kernel using vector addition
- Optimizing matrix transpose through memory coalescing
- ReLU, Leaky ReLU, and warp divergence

## Related Work

For my work on tensor abstractions, automatic differentiation, neural-network
primitives, and experimental GPU integration:

https://github.com/PavelAugustt/RawTorch