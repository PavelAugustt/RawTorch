#include "solve.h"
#include <cuda_runtime.h>

/*
 * Tiled shared-memory GEMM.
 *
 * Layout (matches naive_impl.cu / LeetGPU):
 *   A: M x N
 *   B: N x K
 *   C: M x K
 *   C[i,j] = sum_k A[i,k] * B[k,j]
 *
 * Naive kernel: each thread streams the full reduction dimension from global
 * memory. Correct, but the same A/B elements are re-fetched by many threads.
 *
 * Tiling: a TILE x TILE block cooperatively loads panels of A and B into
 * shared memory, reuses them for TILE inner products, then advances.
 * No register blocking, vectorized loads, or tensor cores — plain tiling.
 */
#define TILE 16

__global__ void matrix_multiplication_kernel(const float* A, const float* B,
                                             float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Walk the shared dimension N in chunks of TILE.
    int numTiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < N && col < K) ? B[b_row * K + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Need a barrier before the next tile overwrites As/Bs.
        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

// A, B, C are device pointers
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((K + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                     N, K);
    cudaDeviceSynchronize();
}
