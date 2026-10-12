#include <pybind11/pybind11.h>
#include "../core/CPU/_add.h"

namespace py = pybind11;

// Minimal pybind11 example over a tiny CPU helper.
// CUDA kernels currently live as standalone programs under CUDA-C/ and LeetGPU-Sols/.
PYBIND11_MODULE(add, m) {
    m.doc() = "Minimal RawTorch C++ extension example";
    m.def("add", &add, "Add two ints (CPU smoke-test binding)");
}
