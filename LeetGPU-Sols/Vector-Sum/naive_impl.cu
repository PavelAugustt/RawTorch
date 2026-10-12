#include "solve.h"
#include <cuda_runtime.h>

/*
 * Vector addition with a grid-stride loop.
 *
 * Contiguous threads touch contiguous addresses → naturally coalesced.
 * The stride (blockDim.x * gridDim.x) lets a fixed launch config cover any N.
 */
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < N) {
        C[tid] = A[tid] + B[tid];
        tid += blockDim.x * gridDim.x;
    }
}

// A, B, C are device pointers
void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
