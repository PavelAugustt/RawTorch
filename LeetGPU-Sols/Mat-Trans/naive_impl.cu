#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

/*
 * Naive transpose: correct indexing, poor write pattern.
 * Consecutive threads write with stride `rows` → uncoalesced stores.
 * Compare with mem_coalesced.cu.
 */
__global__ void matrix_transpose_kernel(const float* input, float* output,
                                        int rows, int cols) {
    int colId = threadIdx.x + blockDim.x * blockIdx.x;
    int rowId = threadIdx.y + blockDim.y * blockIdx.y;

    if (rowId < rows && colId < cols) {
        output[colId * rows + rowId] = input[rowId * cols + colId];
    }
}

// input, output are device pointers
void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output,
                                                                rows, cols);
    cudaDeviceSynchronize();
}
