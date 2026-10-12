# RawTorch

From-scratch study of deep learning systems: a NumPy tensor library with
reverse-mode automatic differentiation, alongside CUDA experiments that show
how the numerical ops under those abstractions look on a GPU.

Educational project — not a PyTorch replacement and not a production kernel
library.

The CUDA programs are standalone. They are not a `Tensor` execution backend
yet; they live here so framework-level and GPU-level work stay connected.

## Why this exists

PyTorch hides tensors, autograd, and GPU execution behind one API. This repo
separates those layers:

```
Framework   Tensor → graph → gradients → layers → training
GPU         Indexing → shared memory → coalescing → tiled GEMM → softmax
```

## Layout

```
RawTorch/
├── legacy/src/tentorch/     Tensor, autograd, nn primitives
├── legacy/src/xor_demo.py   XOR training example
├── CUDA-C/                  Longer CUDA experiments
├── LeetGPU-Sols/            Compact kernels (tiled GEMM, softmax, …)
├── docs/                    Project notes
└── src/                     Early sketches + small pybind example
```

`CUDA-C/` and `LeetGPU-Sols/` also exist as standalone repos; copies are kept
here so both tracks stay in one place.

## Framework

[`tensor.py`](legacy/src/tentorch/tensor.py) wraps a NumPy array with
`requires_grad`, `grad`, parents (`_prev`), and a local `_backward` closure.
Ops: `__add__`, `__sub__`, `__mul__`, `matmul`.

`backward()` seeds the output with ones, builds a topological order, then
applies the recorded closures.

| Primitive | Role | Location |
|---|---|---|
| `Linear` | `y = x @ W + b` | [`nn/layers.py`](legacy/src/tentorch/nn/layers.py) |
| `sigmoid`, `tanh` | Activations | [`nn/activations.py`](legacy/src/tentorch/nn/activations.py) |
| `MSELoss` | Mean squared error | [`nn/loss.py`](legacy/src/tentorch/nn/loss.py) |
| `NN` | Sequential container | [`nn/nn.py`](legacy/src/tentorch/nn/nn.py) |

Training uses manual SGD in [`xor_demo.py`](legacy/src/xor_demo.py).

```python
from tentorch.tensor import Tensor

a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
c = a * b
c.backward()
print(a.grad)  # [4. 5. 6.]
print(b.grad)  # [1. 2. 3.]
```

```bash
# Windows PowerShell
$env:PYTHONPATH="legacy/src"
python legacy/src/xor_demo.py
```

## CUDA work

| Implementation | Concepts | Location |
|---|---|---|
| Grid-stride vector add | Indexing, coalesced loads | `CUDA-C/Parallel-VS-GPU`, `LeetGPU-Sols/Vector-Sum` |
| Shared-memory dot | Reduction, `__syncthreads` | `CUDA-C/SharedMemory-GPU/main.cu` |
| Transpose naive vs coalesced | Access patterns, bank padding | `LeetGPU-Sols/Mat-Trans/` |
| GEMM naive vs tiled | Global K-stream vs shared tiles | `LeetGPU-Sols/Mat-Mult/`, `CUDA-C/GEMM-GPU/_gemm_tiled.cu` |
| Softmax | Max trick + block reduction | `LeetGPU-Sols/Softmax/naive_impl.cu` |
| ReLU / Leaky ReLU | Element-wise kernels | `LeetGPU-Sols/ReLU`, `Leaky-ReLU` |

### Framework op → kernel shape

| Framework idea | GPU shape |
|---|---|
| `Linear` / `matmul` | GEMM (+ bias add) |
| Activations | Element-wise kernels |
| Softmax | Row reduction + normalize |
| Dot / loss reductions | Shared-memory reduction |

`tentorch` still runs on NumPy; CUDA entry points are separate `main` /
`solve()` programs.

## YouTube

[youtube.com/@pavelaugust](https://www.youtube.com/@pavelaugust)

- First CUDA kernel (vector add)
- Matrix transpose and memory coalescing
- ReLU / Leaky ReLU and warp divergence

## Building CUDA programs

```bash
nvcc -o shared CUDA-C/SharedMemory-GPU/main.cu
nvcc -o gemm CUDA-C/GEMM-GPU/_gemm.cu
nvcc -o gemm_tiled CUDA-C/GEMM-GPU/_gemm_tiled.cu
```

LeetGPU kernels use the platform `solve()` interface.

## Status

- Educational systems project
- NumPy `Tensor` with a small operator set
- CUDA coverage: fundamentals, coalesced transpose, tiled GEMM, softmax
- Small CPU pybind example under `src/binding/`; CUDA not wired into `Tensor` yet

## Related work

- [CUDA-C](https://github.com/PavelAugustt/CUDA-C)
- [LeetGPU-Sols](https://github.com/PavelAugustt/LeetGPU-Sols)
- [YouTube](https://www.youtube.com/@pavelaugust)
