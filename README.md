# RawTorch

RawTorch is a from-first-principles exploration of deep-learning systems,
spanning tensor abstractions and reverse-mode automatic differentiation in
Python/NumPy through hands-on CUDA kernel development and GPU execution
experiments.

The CUDA work currently lives alongside the framework rather than serving as a
complete RawTorch execution backend. It is the next layer of the same learning
goal: understanding how numerical operations used by ML frameworks map onto GPU
execution.

## Why I Built This

Frameworks such as PyTorch make tensor math, autograd, and GPU execution look
like a single API. This repository is how I unpacked that stack instead of
treating it as a black box.

The progression I followed:

```
tensor -> computation graph -> gradients -> neural-network primitives -> training
                                              |
                                              v
                         numerical ops -> CUDA execution -> memory access -> kernels
```

The Python side is a small NumPy-backed library (`tentorch` in `legacy/`). The
CUDA side is a set of standalone experiments (`CUDA-C/`, `LeetGPU-Sols/`, and
overlapping sketches in `src/`). They are the same problem at two levels of
abstraction, not a fused product.

## Project Map

```
RawTorch/
├── legacy/src/tentorch/     Tensor, reverse-mode autograd, nn primitives
├── legacy/src/xor_demo.py   XOR training example
├── CUDA-C/                  CUDA learning programs (host + device, CPU refs, profiling)
├── LeetGPU-Sols/            Compact CUDA kernels in LeetGPU `solve()` form
├── src/                     Additional CUDA sketches and a pybind11 CPU add experiment
├── docs/TODO.md             Future ideas (not implemented features)
└── pyproject.toml           Poetry metadata; NumPy is the framework dependency
```

**`legacy/src/tentorch/`** is the framework: a `Tensor` type, elementwise and
matmul ops that record local backward functions, a `Linear` layer, `sigmoid` /
`tanh`, `MSELoss`, and a tiny sequential `NN` container.

**`CUDA-C/`** originated as a separate repository while I was learning CUDA. It
contains self-contained programs: first kernels, grid-stride vector ops, shared-
memory reductions, transpose, naive GEMM, and timing experiments.

**`LeetGPU-Sols/`** originated as a second CUDA repository of smaller kernel
exercises (vector add, transpose, GEMM, ReLU, Leaky ReLU, reverse array),
including a shared-memory coalesced transpose next to a naive one.

**`src/`** holds earlier / overlapping CUDA files (some duplicated from the
CUDA-C experiments), empty placeholders, Nsight report artifacts, and a
pybind11 binding that exposes a CPU `add(int, int)`. It is not a Tensor CUDA
backend.

## Framework Internals

Implemented in [`legacy/src/tentorch/`](legacy/src/tentorch/).

**Tensor.** [`tensor.py`](legacy/src/tentorch/tensor.py) wraps a NumPy `float64`
array with `requires_grad`, `grad`, a predecessor list (`_prev`), and an optional
local `_backward` closure.

**Ops.** Elementwise `__add__`, `__sub__`, `__mul__`, and `matmul` (vector
dot, vector-matrix, matrix-vector, and matrix-matrix, with shape checks).
Elementwise ops call a `broadcast` helper when shapes differ; that helper is an
early attempt at broadcasting, not a complete NumPy broadcast implementation.

**Reverse-mode autograd.** Each op builds an output tensor, attaches a
`_backward` that writes into input `.grad` buffers with `+=` (gradient
accumulation), and stores parents in `_prev`. `Tensor.backward()` seeds
`grad = 1` on the output and walks predecessors, invoking those closures. This
is reverse-mode autodiff on a recorded graph, not a full topological-sort engine.

**Neural-network primitives.**

| Piece | What it is | Location |
|---|---|---|
| `Linear` | `y = x @ W + b` with custom backward for weight, bias, and input | [`nn/layers.py`](legacy/src/tentorch/nn/layers.py) |
| `sigmoid`, `tanh` | Activations with chain-rule backward | [`nn/activations.py`](legacy/src/tentorch/nn/activations.py) |
| `MSELoss` | Mean squared error | [`nn/loss.py`](legacy/src/tentorch/nn/loss.py) |
| `NN` | Applies a sequence of callables | [`nn/nn.py`](legacy/src/tentorch/nn/nn.py) |

There is no optimizer class. Training in [`xor_demo.py`](legacy/src/xor_demo.py)
does manual SGD: `p.data -= lr * p.grad`, then zeros `p.grad`.

[`vanillaNN.py`](legacy/src/vanillaNN.py) compares a two-layer network against
PyTorch on a random batch (forward loss, gradients, one SGD step). That script
depends on `torch`, `matplotlib`, and `networkx` in addition to NumPy. It is a
check against PyTorch's public API, not a reimplementation of PyTorch internals.

### Example

This matches the real `Tensor` API in `legacy/src/tentorch/tensor.py`. Run it
with `PYTHONPATH` pointed at `legacy/src` (see [Running the Framework](#running-the-framework)).

```python
import numpy as np
from tentorch.tensor import Tensor

a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
c = a * b
c.backward()
print(a.grad)  # [4. 5. 6.]
print(b.grad)  # [1. 2. 3.]
```

A full training loop (XOR, `Linear` + `tanh` + `MSELoss`, 10k steps of manual
SGD) is in [`legacy/src/xor_demo.py`](legacy/src/xor_demo.py).

## CUDA / GPU Work

The CUDA directories document a progression from introductory kernels toward
parallel numerical operations and performance reasoning. Programs are mostly
self-contained. They are **not** dispatched from `tentorch.Tensor`.

### CUDA Execution Fundamentals

Early programs in [`CUDA-C/Hello-GPU/`](CUDA-C/Hello-GPU/hello.cu),
[`CUDA-C/Add-GPU/`](CUDA-C/Add-GPU/add.cu), and
[`CUDA-C/Vector-Sum-GPU/`](CUDA-C/Vector-Sum-GPU/main.cu) cover:

- `__global__` kernels and `<<<grid, block>>>` launches
- `threadIdx` / `blockIdx` / `blockDim` indexing
- `cudaMalloc`, `cudaMemcpy` (host-to-device and device-to-host), `cudaFree`
- device queries in [`CUDA-C/DeviceProp-GPU/deviceProp.cu`](CUDA-C/DeviceProp-GPU/deviceProp.cu)

Grid-stride loops appear once a single block is no longer enough to cover `N`.
[`CUDA-C/Parallel-VS-GPU/ArbVector.cu`](CUDA-C/Parallel-VS-GPU/ArbVector.cu)
linearizes a global id, then advances by `blockDim.x * gridDim.x` so an
arbitrary launch configuration can still cover the vector. The same pattern is
in [`LeetGPU-Sols/Vector-Sum/naive_impl.cu`](LeetGPU-Sols/Vector-Sum/naive_impl.cu).
CPU counterparts (sequential host loops) live next to several of these kernels
for comparison.

### Parallel Operations

- **Vector addition.** One thread per element, then grid-stride variants, plus
  host-side sequential versions.
- **Dot product / reduction.** Elementwise products in global memory, then a
  shared-memory tree reduction per block, with the host summing partial block
  results ([`CUDA-C/SharedMemory-GPU/main.cu`](CUDA-C/SharedMemory-GPU/main.cu)).
- **ReLU / Leaky ReLU.** Element-wise kernels in LeetGPU-Sols. ReLU is
  `max(0, x)`. Leaky ReLU uses a per-element sign check
  (`val > 0 ? val : val * 0.01f`), which is the kind of branch that can split a
  warp when neighboring threads take different paths. The files are named
  `optim_impl.cu`; the kernels themselves are straightforward element-wise
  implementations, not a second optimized variant of a naive kernel.
- **Reverse array.** Pairwise swaps using `tid` and `N - tid - 1`.

### Matrix Operations

- **Transpose.** CPU reference plus a GPU kernel with 2D thread indices and
  row-major flattening (`row * cols + col`) in
  [`CUDA-C/Transpose-2D-GPU/cpu.cu`](CUDA-C/Transpose-2D-GPU/cpu.cu) and
  [`LeetGPU-Sols/Mat-Trans/naiv_impl.cu`](LeetGPU-Sols/Mat-Trans/naiv_impl.cu).
  The naive GPU write is `output[col * rows + row] = input[row * cols + col]`,
  which tends to make consecutive threads write non-contiguous addresses.
- **Coalesced transpose.** [`LeetGPU-Sols/Mat-Trans/mem_coalesced.cu`](LeetGPU-Sols/Mat-Trans/mem_coalesced.cu)
  stages a `BLOCK_SIZE x BLOCK_SIZE` tile in shared memory (with `+1` padding
  on the inner dimension), then writes with swapped tile coordinates so both
  the load and the store can use contiguous global-memory addresses. That is
  an actual change relative to the naive kernel: it is investigating how
  memory access order, not arithmetic, affects the transpose.
- **GEMM.** Naive one-output-per-thread kernels
  (`C[i,j] += A[i,k] * B[k,j]`) with 2D grids, plus nested grid-stride versions
  and a variant that applies `alpha * (AB) + beta * C`
  ([`CUDA-C/Profiling-GPU/_gemm.cu`](CUDA-C/Profiling-GPU/_gemm.cu)). Several
  files compare GPU output to a CPU triple loop (`memcmp` / element checks).
  Kernels take `M, N, K` (or `rows, cols`); most CUDA-C host programs I
  inspected still launch square cases. The LeetGPU matmul kernel accepts
  independent `M, N, K`. There is no tiled/shared-memory GEMM and no CUTLASS
  (or cuBLAS) usage in this repo. Comments in the GEMM files discuss tiling as
  a next idea, not as an implementation.

Repeated files under [`CUDA-C/Repetition-GPU/`](CUDA-C/Repetition-GPU/) and
[`CUDA-C/GEMM-GPU/`](CUDA-C/GEMM-GPU/) are re-derivations of the same
indexing, not a library of production kernels.

### Memory and Performance Experiments

What was actually explored, and why it matters:

- **Global memory** is the default path (vector add, naive GEMM, naive
  transpose). It is simple and correctness-first; it also makes memory traffic
  obvious when you later change the access pattern.
- **Shared memory** is used in the dot-product reduction so each block
  combines partial products locally instead of writing every multiply back to
  global memory. `__syncthreads()` is placed outside the reduction predicate
  so all threads in the block participate in the barrier (the comments in
  `SharedMemory-GPU/main.cu` are specifically about avoiding a hang from
  divergent `__syncthreads()`).
- **Memory coalescing** is the point of the tiled transpose: consecutive
  threads should touch consecutive addresses. The naive transpose is the
  control; the shared-memory version is the experiment.
- **Reductions** turn an all-to-one sum into a tree in shared memory, then a
  small host loop over one partial per block.
- **Launch configuration** is varied explicitly (fixed `<<<1, N>>>`, then
  many blocks, then grid-stride with a small grid).
- **Profiling.** [`CUDA-C/Profiling-GPU/`](CUDA-C/Profiling-GPU/) times two
  GEMM launch styles with `cudaEventElapsedTime` (if-guarded 2D indexing vs
  nested grid-stride loops). The notes in that folder record those timings and
  question whether the slower variant is measuring a fair/correct comparison.
  Nsight Systems artifacts also exist under `src/core/GPU/`. This is
  experimental timing, not a published benchmark suite.

## Selected GPU Implementations

| Implementation | Concepts explored | Location |
|---|---|---|
| Hello / scalar add | Kernel launch, device allocation, host/device copy | [`CUDA-C/Hello-GPU/hello.cu`](CUDA-C/Hello-GPU/hello.cu), [`CUDA-C/Add-GPU/add.cu`](CUDA-C/Add-GPU/add.cu) |
| Vector addition | Thread indexing, then grid-stride loops | [`CUDA-C/Vector-Sum-GPU/main.cu`](CUDA-C/Vector-Sum-GPU/main.cu), [`CUDA-C/Parallel-VS-GPU/ArbVector.cu`](CUDA-C/Parallel-VS-GPU/ArbVector.cu), [`LeetGPU-Sols/Vector-Sum/naive_impl.cu`](LeetGPU-Sols/Vector-Sum/naive_impl.cu) |
| Device properties | Compute capability, shared mem, warp size, grid limits | [`CUDA-C/DeviceProp-GPU/deviceProp.cu`](CUDA-C/DeviceProp-GPU/deviceProp.cu) |
| Vector dot product | Grid-stride partials, shared-memory reduction | [`CUDA-C/SharedMemory-GPU/main.cu`](CUDA-C/SharedMemory-GPU/main.cu), [`CUDA-C/Repetition-GPU/_vdotSD.cu`](CUDA-C/Repetition-GPU/_vdotSD.cu) |
| Matrix transpose (naive) | 2D indexing, row-major layout | [`CUDA-C/Transpose-2D-GPU/cpu.cu`](CUDA-C/Transpose-2D-GPU/cpu.cu), [`LeetGPU-Sols/Mat-Trans/naiv_impl.cu`](LeetGPU-Sols/Mat-Trans/naiv_impl.cu) |
| Matrix transpose (tiled) | Shared memory, coalesced load/store, bank-conflict padding | [`LeetGPU-Sols/Mat-Trans/mem_coalesced.cu`](LeetGPU-Sols/Mat-Trans/mem_coalesced.cu) |
| GEMM (naive) | 2D grid, `M/N/K` indexing, CPU reference check | [`CUDA-C/GEMM-GPU/_gemm.cu`](CUDA-C/GEMM-GPU/_gemm.cu), [`LeetGPU-Sols/Mat-Mult/naive_impl.cu`](LeetGPU-Sols/Mat-Mult/naive_impl.cu) |
| GEMM (timing) | `cudaEvent` timing, `alpha/beta` GEMM, two launch styles | [`CUDA-C/Profiling-GPU/_gemm.cu`](CUDA-C/Profiling-GPU/_gemm.cu), [`CUDA-C/Profiling-GPU/_gemmF.cu`](CUDA-C/Profiling-GPU/_gemmF.cu) |
| ReLU | Element-wise parallelism | [`LeetGPU-Sols/ReLU/optim_impl.cu`](LeetGPU-Sols/ReLU/optim_impl.cu) |
| Leaky ReLU | Element-wise kernel with a per-thread branch | [`LeetGPU-Sols/Leaky-ReLU/optim_impl.cu`](LeetGPU-Sols/Leaky-ReLU/optim_impl.cu) |
| Reverse array | Index mapping, in-place swap | [`LeetGPU-Sols/Reverse-Array/naive_impl.cu`](LeetGPU-Sols/Reverse-Array/naive_impl.cu) |

## From Framework Operations to GPU Kernels

At the framework level:

```
Tensor op -> recorded graph (_prev, _backward) -> reverse-mode gradients -> SGD update
```

At the numerical level, the same math is a different problem:

```
Tensor op -> vector/matrix arithmetic -> thread/block decomposition -> memory movement -> kernel
```

A `Linear` layer is `matmul` plus bias. On the GPU that is GEMM plus a vector
add. Autograd through matmul is more GEMM (transposed factors). A `tanh` or a
future ReLU is an element-wise kernel. A reduction (loss mean, dot product) is
the shared-memory pattern in `SharedMemory-GPU`.

That mapping is the point of keeping both halves in one repository. It is **not**
how execution works today: `tentorch` calls NumPy; CUDA programs have their own
`main` (or a LeetGPU `solve()`). Wiring Tensor ops to these kernels is future
work, not an existing backend.

## Technical Explanations / YouTube

I record technical explanations at
[youtube.com/@pavelaugust](https://www.youtube.com/@pavelaugust) as a way to
test whether I actually understand the implementation.

Videos that correspond to code in this repo:

- Write your first GPU kernel with CUDA (Vector Sum)
- Optimised Matrix Transpose in CUDA — Memory Coalescing
- Parallel ReLU, Leaky ReLU, and Warp Divergence

The explanations force me to articulate not only how a kernel is implemented,
but why indexing, memory access, and execution behavior affect the result.

## Running the Framework

The framework is the `tentorch` package under `legacy/src/`. There is no
install step that registers it as a site package from the repo root; `pyproject.toml`
lists NumPy (and pytest as a dev dependency) but does not package `legacy/`
for `pip install .`.

```bash
git clone https://github.com/bozo10n/RawTorch.git
cd RawTorch
pip install numpy
```

Then point `PYTHONPATH` at `legacy/src`:

```bash
# Linux / macOS
PYTHONPATH=legacy/src python3 legacy/src/xor_demo.py

# Windows (PowerShell)
$env:PYTHONPATH="legacy/src"
python legacy/src/xor_demo.py
```

[`legacy/src/autograd_tests.py`](legacy/src/autograd_tests.py) is a `unittest`
script for tensor ops. It uses the same `PYTHONPATH`. Other scripts under
`legacy/src/` (notably `layer_tests.py` and `vanillaNN.py`) assume extra
imports or APIs that do not all match the current package.

## Building CUDA Experiments

CUDA experiments are mostly self-contained and compiled individually with
`nvcc`. There is no unified build system. An NVIDIA GPU and the CUDA Toolkit
are required.

`CUDA-C` programs include `main` and can be compiled as-is. Example:

```bash
nvcc -o hello CUDA-C/Hello-GPU/hello.cu
nvcc -o vadd CUDA-C/Parallel-VS-GPU/ArbVector.cu
nvcc -o gemm CUDA-C/GEMM-GPU/_gemm.cu
```

A local note in [`CUDA-C/compilationCMD.txt`](CUDA-C/compilationCMD.txt) uses
`-arch=sm_86`; pass an architecture that matches your GPU if you set `-arch`.

LeetGPU kernels (`LeetGPU-Sols/**/*.cu`) are written against a platform header
`solve.h` that is **not** in this tree. They are the kernel bodies and
`solve()` launch wrappers, not standalone executables.

[`CUDA-C/Fractal-GPU/main.cu`](CUDA-C/Fractal-GPU/main.cu) maps a 2D grid to
Julia-set pixels but includes `../common/cpu_bitmap.h`, which is not present
here.

The pybind11 sketch in [`src/binding/`](src/binding/add.cpp) compiles a CPU
integer add, not a CUDA tensor op. The command recorded in
[`compilationCMD.txt`](compilationCMD.txt) is:

```bash
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) add.cpp -o add$(python3-config --extension-suffix)
```

## Project Status / Limitations

RawTorch is an educational and experimental systems project, not a production
deep-learning framework.

- The framework is NumPy-backed (`float64`) with a small operator set
  (add/sub/mul/matmul, two activations, MSE, Linear).
- Autograd is reverse-mode via per-op closures; it is not a general, robust
  autograd engine (no topological sort, incomplete broadcasting).
- There is no optimizer module, no convolution / RNN / transformer layers, and
  no CUDA dispatch from `Tensor`.
- CUDA kernels are independent experiments. They emphasize indexing, memory,
  and correctness checks against CPU loops rather than production performance.
- Filenames such as `optim_impl.cu` should not be read as “this kernel is
  optimized”; only the tiled transpose is a clear optimization relative to a
  simpler implementation in-tree.
- Testing is a handful of scripts, not a CI suite. Profiling is ad hoc
  (`cudaEvent`, nsys artifacts), not a reproducible benchmark harness.
- `src/` contains duplicates, empty `.cu` files, and binaries; treat
  `CUDA-C/` and `LeetGPU-Sols/` as the CUDA sources to read.

Items in [`docs/TODO.md`](docs/TODO.md) and [`CUDA-C/todo.txt`](CUDA-C/todo.txt)
are plans (Conv, CUTLASS, a CUDA tensor library, etc.). They are not
implemented.

## Related Work

`CUDA-C/` and `LeetGPU-Sols/` started as separate repositories while I was
learning CUDA. They are included here so the framework-level and GPU-level
work live in one tree. Nested `.git` directories were removed so this
repository is a single git history going forward; the original repos still
have the earlier commit history:

- CUDA-C: [github.com/PavelAugustt/CUDA-C](https://github.com/PavelAugustt/CUDA-C)
- LeetGPU-Sols: [github.com/PavelAugustt/LeetGPU-Sols](https://github.com/PavelAugustt/LeetGPU-Sols)
- YouTube: [youtube.com/@pavelaugust](https://www.youtube.com/@pavelaugust)
