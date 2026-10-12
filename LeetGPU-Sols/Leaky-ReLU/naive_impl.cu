#include "solve.h"
#include <cuda_runtime.h>

/*
 * Leaky ReLU: y = x if x > 0 else 0.01 * x
 *
 * Threads in a warp that take different branches (positive vs negative)
 * serialize — warp divergence. For random signs this can matter; for mostly
 * positive activations it usually does not. Good mental model, not a claim
 * that this kernel is "optimized."
 */
__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        float val = input[tid];
        output[tid] = (val > 0.0f) ? val : val * 0.01f;
    }
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
