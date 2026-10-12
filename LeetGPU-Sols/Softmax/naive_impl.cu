#include "solve.h"
#include <cuda_runtime.h>
#include <math.h>

/*
 * Row-wise Softmax (numerically stable).
 *
 * For each row x:
 *   m = max(x)
 *   Softmax(x)_i = exp(x_i - m) / sum_j exp(x_j - m)
 *
 * Subtracting m keeps exp() from overflowing. Without it, large logits
 * can produce Inf and the row becomes NaN.
 *
 * One block per row: threads stride across columns, then a shared-memory
 * reduction finds the max and the sum. Not an online / fused attention softmax.
 */
#define THREADS 256

__global__ void softmax_kernel(const float* input, float* output, int rows,
                               int cols) {
    __shared__ float reduce_buf[THREADS];

    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // ---- pass 1: row max ----
    float local_max = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        local_max = fmaxf(local_max, row_in[c]);
    }
    reduce_buf[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce_buf[threadIdx.x] =
                fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float row_max = reduce_buf[0];
    __syncthreads();

    // ---- pass 2: sum of exp(x - max) ----
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        local_sum += expf(row_in[c] - row_max);
    }
    reduce_buf[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float row_sum = reduce_buf[0];
    __syncthreads();

    // ---- pass 3: normalize ----
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        row_out[c] = expf(row_in[c] - row_max) / row_sum;
    }
}

// input, output are device pointers (rows x cols, row-major)
void solve(const float* input, float* output, int rows, int cols) {
    softmax_kernel<<<rows, THREADS>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
