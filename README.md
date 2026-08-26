# RawTorch

RawTorch is a from-scratch study of deep learning systems: a NumPy-based
tensor library with reverse-mode automatic differentiation, paired with CUDA
kernels for the numerical operations underneath those abstractions.

The CUDA components are standalone experiments, not a Tensor execution
backend. They sit alongside the framework to connect high-level ML ops with
how those ops run on the GPU.

## Why I Built This

Libraries like PyTorch hide tensors, autograd, and GPU execution behind a
single API. This project separates those layers so each can be inspected
directly:

```
Framework   Tensor → graph → gradients → layers → training
GPU         Numerical ops → CUDA kernels → memory access
```

## Project Map

```
RawTorch/
├── legacy/src/tentorch/     Tensor, autograd, nn primitives
├── legacy/src/xor_demo.py   XOR training example
├── CUDA-C/                  CUDA programs, CPU references, profiling
├── LeetGPU-Sols/            Compact kernel exercises
└── src/                     Earlier CUDA sketches
```

**[`legacy/src/tentorch/`](legacy/src/tentorch/)** is the framework: `Tensor`,
elementwise ops and matmul with reverse-mode autograd, `Linear`, `sigmoid` /
`tanh`, `MSELoss`, and a small sequential `NN`.

**[`CUDA-C/`](CUDA-C/)** is the longer CUDA track: first kernels, grid-stride
loops, shared-memory reductions, transpose, naive GEMM, and timing.

**[`LeetGPU-Sols/`](LeetGPU-Sols/)** is a smaller set of kernels — vector add,
transpose (naive and coalesced), GEMM, ReLU, Leaky ReLU, reverse array.

Both CUDA directories started as their own repos. They are here so the
framework-level and GPU-level work stay in one place.

## Framework Internals

[`tensor.py`](legacy/src/tentorch/tensor.py) wraps a NumPy array with
`requires_grad`, `grad`, a parent list (`_prev`), and a local `_backward`
closure. Supported ops: `__add__`, `__sub__`, `__mul__`, and `matmul`
(dot, vector-matrix, matrix-vector, matrix-matrix).

Each op records how to push gradients into its inputs (`+=` for accumulation).
`backward()` seeds the output with ones and walks parents, applying those
closures — reverse-mode autodiff on the recorded graph.

| Primitive | Role | Location |
|---|---|---|
| `Linear` | `y = x @ W + b` | [`nn/layers.py`](legacy/src/tentorch/nn/layers.py) |
| `sigmoid`, `tanh` | Activations | [`nn/activations.py`](legacy/src/tentorch/nn/activations.py) |
| `MSELoss` | Mean squared error | [`nn/loss.py`](legacy/src/tentorch/nn/loss.py) |
| `NN` | Sequential container | [`nn/nn.py`](legacy/src/tentorch/nn/nn.py) |

Training is manual SGD in [`xor_demo.py`](legacy/src/xor_demo.py):
`p.data -= lr * p.grad`. [`vanillaNN.py`](legacy/src/vanillaNN.py) compares a
two-layer net against PyTorch on one batch.

```python
from tentorch.tensor import Tensor

a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
c = a * b
c.backward()
print(a.grad)  # [4. 5. 6.]
print(b.grad)  # [1. 2. 3.]
```

`PYTHONPATH` needs to include `legacy/src`. Full XOR training:
[`legacy/src/xor_demo.py`](legacy/src/xor_demo.py).

## CUDA / GPU Work

Standalone programs, compiled one at a time. They are not called from
`tentorch.Tensor`.

### Execution model

[`Hello-GPU`](CUDA-C/Hello-GPU/hello.cu), [`Add-GPU`](CUDA-C/Add-GPU/add.cu),
and [`Vector-Sum-GPU`](CUDA-C/Vector-Sum-GPU/main.cu) cover kernel launches,
`threadIdx` / `blockIdx` indexing, and `cudaMalloc` / `cudaMemcpy`.
[`deviceProp.cu`](CUDA-C/DeviceProp-GPU/deviceProp.cu) prints device limits
(shared memory, warp size, max grid).

When one block cannot cover `N`, threads use a **grid-stride loop**: start at
a linearized global id, then step by `blockDim.x * gridDim.x`. See
[`ArbVector.cu`](CUDA-C/Parallel-VS-GPU/ArbVector.cu) and
[`LeetGPU-Sols/Vector-Sum/naive_impl.cu`](LeetGPU-Sols/Vector-Sum/naive_impl.cu).
Several kernels have a sequential CPU version next to them.

### Parallel ops

- **Vector add** — one thread per element, then grid-stride variants.
- **Dot product** — grid-stride partial products, then a shared-memory tree
  reduction per block. The host sums one partial per block
  ([`SharedMemory-GPU/main.cu`](CUDA-C/SharedMemory-GPU/main.cu)).
- **ReLU / Leaky ReLU** — element-wise. Leaky ReLU branches on sign
  (`val > 0 ? val : 0.01f * val`), the pattern that can diverge a warp.
- **Reverse array** — swap `tid` with `N - tid - 1`.

### Matrix ops

Naive **transpose** uses 2D indices and row-major layout
(`row * cols + col`). The write `output[col * rows + row] = input[...]`
scatters consecutive threads across non-contiguous addresses:
[`cpu.cu`](CUDA-C/Transpose-2D-GPU/cpu.cu),
[`naiv_impl.cu`](LeetGPU-Sols/Mat-Trans/naiv_impl.cu).

The **coalesced** version
([`mem_coalesced.cu`](LeetGPU-Sols/Mat-Trans/mem_coalesced.cu)) loads a tile
into shared memory (`BLOCK_SIZE+1` padding), then writes with swapped
coordinates so both load and store hit contiguous global memory. Same
arithmetic, different access pattern.

**GEMM** is the naive `C[i,j] += A[i,k] * B[k,j]` kernel on a 2D grid, plus
grid-stride variants and `alpha * AB + beta * C`
([`Profiling-GPU/_gemm.cu`](CUDA-C/Profiling-GPU/_gemm.cu)). GPU results are
checked against a CPU triple loop. Kernels take `M, N, K`;
[`LeetGPU-Sols/Mat-Mult/naive_impl.cu`](LeetGPU-Sols/Mat-Mult/naive_impl.cu)
uses those dimensions independently.

[`Repetition-GPU/`](CUDA-C/Repetition-GPU/) and [`GEMM-GPU/`](CUDA-C/GEMM-GPU/)
are re-writes of the same indexing while I was locking it in.

### Memory and performance

- **Global memory** is the default (vector add, naive GEMM, naive transpose).
- **Shared memory** in the dot product keeps the reduction on-chip; only
  one value per block goes back to global memory. `__syncthreads()` sits
  outside the reduction predicate so the whole block hits the barrier.
- **Coalescing** is the tiled transpose: consecutive threads, consecutive
  addresses. Naive transpose is the control.
- **Launch config** moves from `<<<1, N>>>` to many blocks to grid-stride
  with a small grid.
- **Profiling** in [`Profiling-GPU/`](CUDA-C/Profiling-GPU/) times two GEMM
  launch styles with `cudaEventElapsedTime` (bounds-checked 2D indexing vs
  nested stride loops). Notes in that folder, plus nsys artifacts under
  `src/core/GPU/`.

## Selected GPU Implementations

| Implementation | Concepts | Location |
|---|---|---|
| Vector addition | Indexing, grid-stride loops | [`Vector-Sum-GPU/main.cu`](CUDA-C/Vector-Sum-GPU/main.cu), [`ArbVector.cu`](CUDA-C/Parallel-VS-GPU/ArbVector.cu), [`naive_impl.cu`](LeetGPU-Sols/Vector-Sum/naive_impl.cu) |
| Vector dot product | Reduction, shared memory | [`SharedMemory-GPU/main.cu`](CUDA-C/SharedMemory-GPU/main.cu) |
| Matrix transpose | 2D indexing, row-major layout | [`cpu.cu`](CUDA-C/Transpose-2D-GPU/cpu.cu), [`naiv_impl.cu`](LeetGPU-Sols/Mat-Trans/naiv_impl.cu) |
| Coalesced transpose | Shared-memory tile, coalesced load/store | [`mem_coalesced.cu`](LeetGPU-Sols/Mat-Trans/mem_coalesced.cu) |
| GEMM | Matrix indexing, CPU check | [`_gemm.cu`](CUDA-C/GEMM-GPU/_gemm.cu), [`naive_impl.cu`](LeetGPU-Sols/Mat-Mult/naive_impl.cu) |
| GEMM timing | `cudaEvent`, two launch styles | [`_gemm.cu`](CUDA-C/Profiling-GPU/_gemm.cu), [`_gemmF.cu`](CUDA-C/Profiling-GPU/_gemmF.cu) |
| ReLU | Element-wise parallelism | [`optim_impl.cu`](LeetGPU-Sols/ReLU/optim_impl.cu) |
| Leaky ReLU | Element-wise kernel, branching | [`optim_impl.cu`](LeetGPU-Sols/Leaky-ReLU/optim_impl.cu) |
| Reverse array | Index mapping | [`naive_impl.cu`](LeetGPU-Sols/Reverse-Array/naive_impl.cu) |

## From Framework Operations to GPU Kernels

Framework:

```
Tensor op → graph (_prev, _backward) → reverse-mode gradients → SGD
```

Hardware:

```
Tensor op → vector/matrix work → threads/blocks → memory movement → kernel
```

A `Linear` layer is matmul plus bias — GEMM and a vector add on the GPU.
Autograd through matmul is more GEMM. Activations are element-wise kernels.
A loss mean or a dot product is the reduction in `SharedMemory-GPU`.

That mapping is why both halves are in this repo. Today, `tentorch` still
runs on NumPy; the CUDA programs have their own `main`.

## Technical Explanations / YouTube

I explain several of these kernels at
[youtube.com/@pavelaugust](https://www.youtube.com/@pavelaugust):

- [Write your first GPU kernel with CUDA (Vector Sum)](https://www.youtube.com/watch?v=KEVAsNIKiPo)
- [Optimised Matrix Transpose in CUDA — Memory Coalescing](https://www.youtube.com/watch?v=z91hf3FG8fI)
- [Parallel ReLU, Leaky ReLU, and Warp Divergence](https://www.youtube.com/watch?v=Tak5Zwhro1k)

The point is to be able to say why indexing, memory access, and execution
behavior matter, not only how the kernel is written.

## Running the Framework

```bash
git clone https://github.com/PavelAugustt/RawTorch.git
cd RawTorch
pip install numpy
```

```bash
# Linux / macOS
PYTHONPATH=legacy/src python3 legacy/src/xor_demo.py

# Windows (PowerShell)
$env:PYTHONPATH="legacy/src"
python legacy/src/xor_demo.py
```

## Building CUDA Experiments

Programs under `CUDA-C/` compile individually with `nvcc`. NVIDIA GPU + CUDA
Toolkit required.

```bash
nvcc -o hello CUDA-C/Hello-GPU/hello.cu
nvcc -o vadd CUDA-C/Parallel-VS-GPU/ArbVector.cu
nvcc -o gemm CUDA-C/GEMM-GPU/_gemm.cu
```

[`compilationCMD.txt`](CUDA-C/compilationCMD.txt) uses `-arch=sm_86`; match
`-arch` to your GPU if you set it.

Kernels in `LeetGPU-Sols/` use the LeetGPU `solve()` interface.

## Project Status

Educational / experimental systems project, not a production framework.

- NumPy-backed `Tensor` with a small operator set
- CUDA kernels are independent experiments, not a Tensor backend
- Emphasis is indexing, memory, and CPU reference checks — not production
  kernel performance
- Limited testing and no systematic benchmark harness

## Related Work

- CUDA-C: [github.com/PavelAugustt/CUDA-C](https://github.com/PavelAugustt/CUDA-C)
- LeetGPU-Sols: [github.com/PavelAugustt/LeetGPU-Sols](https://github.com/PavelAugustt/LeetGPU-Sols)
- YouTube: [youtube.com/@pavelaugust](https://www.youtube.com/@pavelaugust)
