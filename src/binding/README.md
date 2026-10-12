# Bindings

`add.cpp` is a minimal pybind11 example over the CPU `add(int, int)` helper.
It shows the extension shape; it is not a GPU Tensor backend.

## Build sketch

```bash
# from repo root, after installing pybind11
c++ -O3 -Wall -shared -std=c++11 -fPIC \
  $(python -m pybind11 --includes) \
  src/binding/add.cpp src/core/CPU/_add.cpp \
  -o add$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
```

Windows toolchains need different flags; treat this as a small example.

## Possible extension

The same pattern can expose a CUDA kernel (vector add or tiled matmul) behind
a thin Python function, with NumPy remaining the default path until that is
stable.
