#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

/*
 * Coalesced matrix transpose via a shared-memory tile.
 *
 * Naive transpose (see naiv_impl.cu):
 *   threads in a warp read contiguous input (good), but write to
 *   output[col * rows + row], which strides by `rows` — uncoalesced stores.
 *
 * This kernel:
 *   1. Cooperative load of a BLOCK_SIZE x BLOCK_SIZE tile (coalesced reads)
 *   2. Transpose the tile in shared memory (cheap on-chip)
 *   3. Cooperative store with blockIdx.x/y swapped (coalesced writes)
 *
 * tile[...][BLOCK_SIZE + 1]:
 *   Shared memory is banked (typically 32 banks). A square tile without
 *   padding means column reads hit the same bank → bank conflicts.
 *   The +1 padding shifts columns so consecutive threads map to different
 *   banks. Standard shared-memory bank-conflict padding.
 */
__global__ void matrix_transpose_kernel(const float* input, float* output,
                                        int rows, int cols) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];

    int x = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    int y = threadIdx.y + BLOCK_SIZE * blockIdx.y;

    // Coalesced load: threads in a warp have consecutive x (same row).
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // Swap block indices so the write walks the transposed layout contiguously.
    int new_x = threadIdx.x + BLOCK_SIZE * blockIdx.y;
    int new_y = threadIdx.y + BLOCK_SIZE * blockIdx.x;

    int new_cols = rows;
    int new_rows = cols;
    if (new_x < new_cols && new_y < new_rows) {
        // Read tile[tx][ty] — the on-chip transpose.
        output[new_y * new_cols + new_x] = tile[threadIdx.x][threadIdx.y];
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
