// THis naive implementation is memory coalesced, further optimizations are mostly uneccessary
#include "solve.h"
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    while(tid < N)
    {
        C[tid] = A[tid] + B[tid];
        tid += blockDim.x * gridDim.x;

    }

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
