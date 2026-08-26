# CUDA-C

Included in [RawTorch](https://github.com/PavelAugustt/RawTorch). Original repository:
https://github.com/PavelAugustt/CUDA-C

A collection of CUDA C experiments documenting my hands-on study of GPU
programming, parallel algorithms, memory behavior, and matrix computation.

This repository is intentionally exploratory. It contains implementations I
wrote and repeatedly rebuilt while learning CUDA fundamentals and experimenting
with different approaches to correctness and performance.

## Topics Explored

### CUDA execution fundamentals
- Thread, block, and grid organization
- Global thread indexing
- Grid-stride loops
- Device memory allocation and transfers

### Parallel operations
- Vector addition
- Vector dot product
- Shared-memory reduction
- Thread divergence and branching behavior

### Matrix operations
- CPU and GPU matrix transpose
- Row-major and column-major indexing
- Naive GEMM
- GEMM for square and non-square matrices
- Repeated GEMM implementations to reinforce indexing and execution mechanics

### Performance experimentation
- Shared memory
- Reduction techniques
- Memory access patterns
- Kernel launch configuration
- Profiling CUDA implementations
- Comparing alternative stride-loop implementations
- Experiments with GEMM scaling and performance

## Related Work

### RawTorch
A small deep-learning framework built from scratch to explore tensors,
automatic differentiation, neural-network layers, and the connection between
framework-level operations and lower-level numerical computation.

https://github.com/PavelAugustt/RawTorch

### LeetGPU Solutions
Smaller CUDA kernel exercises including memory-coalesced transpose and
element-wise activation kernels.

https://github.com/PavelAugustt/LeetGPU-Sols

### Video Explanations
I make technical explanations of several GPU concepts and implementations:

https://www.youtube.com/@pavelaugust