#include "solve.h"
#include <cuda_runtime.h>

/*
 * Naive GEMM: one thread owns one output element C[i,j].
 *
 * A: M x N, B: N x K, C: M x K
 * Inner loop streams K (here named N in LeetGPU dims) from global memory.
 *
 * Correct and easy to reason about. Next step: tiled_impl.cu (shared memory).
 */
__global__ void matrix_multiplication_kernel(const float* A, const float* B,
                                             float* C, int M, int N, int K) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < M && j < K) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[i * N + k] * B[k * K + j];
        }
        C[i * K + j] = sum;
    }
}

// A, B, C are device pointers
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                     N, K);
    cudaDeviceSynchronize();
}
