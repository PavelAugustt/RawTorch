#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

/*
 * Shared-memory dot product (learning demo).
 *
 * Each thread:
 *   1. Grid-strides over the vectors, accumulating a partial product sum
 *   2. Writes that partial into shared memory (one slot per thread)
 *   3. Tree-reduces within the block
 *   4. Thread 0 writes one partial per block to global memory
 *
 * The host sums the per-block partials.
 *
 * Notes:
 *   - __syncthreads() must run for the whole block — never only inside
 *     `if (threadIdx.x < i)`, or idle threads skip the barrier and deadlock
 *   - Tree reduction halves the active window each step so each value is
 *     combined once (no double-counting)
 */

#define THREADS_PER_BLOCK 256
#define N 1000

__global__ void dot(const int* a, const int* b, float* partial, int n) {
    __shared__ float cache[THREADS_PER_BLOCK];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    float temp = 0.0f;

    while (tid < n) {
        temp += static_cast<float>(a[tid]) * static_cast<float>(b[tid]);
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        // Barrier outside the predicate — all threads must participate.
        __syncthreads();
    }

    if (cacheIndex == 0) {
        partial[blockIdx.x] = cache[0];
    }
}

int main() {
    const int blocksPerGrid = 32;

    int h_a[N];
    int h_b[N];
    float h_partial[blocksPerGrid];

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    int* d_a = nullptr;
    int* d_b = nullptr;
    float* d_partial = nullptr;

    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_partial, blocksPerGrid * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    dot<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_a, d_b, d_partial, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_partial, d_partial, blocksPerGrid * sizeof(float),
               cudaMemcpyDeviceToHost);

    float result = 0.0f;
    for (int i = 0; i < blocksPerGrid; ++i) {
        result += h_partial[i];
    }

    // CPU reference: sum i*i for i in [0, N)
    float expected = 0.0f;
    for (int i = 0; i < N; ++i) {
        expected += static_cast<float>(i) * static_cast<float>(i);
    }

    float rel = (expected == 0.0f) ? 0.0f : fabsf(result - expected) / expected;
    printf("gpu result: %f\n", result);
    printf("cpu expect: %f\n", expected);
    printf("rel err: %g\n", rel);
    // float32 cannot hold large integer sums exactly — allow a small relative error
    bool ok = rel < 1e-5f;
    printf("%s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial);
    return ok ? 0 : 1;
}
