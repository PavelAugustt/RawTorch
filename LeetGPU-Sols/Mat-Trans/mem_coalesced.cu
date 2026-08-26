#include "solve.h"
#include <cuda_runtime.h>

# define BLOCK_SIZE 32
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE+1];

    int x = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    int y = threadIdx.y + BLOCK_SIZE * blockIdx.y;

    if(x < cols && y < rows)
    {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    int new_x = threadIdx.x + BLOCK_SIZE * blockIdx.y;
    int new_y = threadIdx.y + BLOCK_SIZE * blockIdx.x;

    int new_cols = rows;
    int new_rows = cols;
    if(new_x < new_cols && new_y < new_rows)
    {
        output[new_y * new_cols + new_x] = tile[threadIdx.x][threadIdx.y];
    }

    

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
