#include "solve.h"
#include <cuda_runtime.h>

/*
 * Element-wise ReLU: y = max(0, x).
 *
 * Uses fmaxf. Named naive because there is no vectorized float4 path —
 * element-wise kernels are usually bandwidth-bound anyway.
 */
__global__ void relu_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = fmaxf(0.0f, input[tid]);
    }
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
