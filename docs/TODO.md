# Project notes

Educational systems study — not a PyTorch replacement.

## Done
- Tensor + reverse-mode autograd (NumPy)
- Linear, activations, MSE, XOR demo
- Standalone CUDA fundamentals (CUDA-C / LeetGPU-Sols)
- Tiled GEMM, coalesced transpose, softmax kernels

## Next
- Local timing tables (naive vs tiled / coalesced)
- Optional: pybind one CUDA kernel for a thin Python wrapper
- More autograd tests (broadcast, matmul shapes)

## Out of scope for now
- MoE, world models, HSTU, full RL stacks
- Matching PyTorch / JAX feature sets
- Production kernel stacks (cuBLAS / CUTLASS / Tensor Cores)
