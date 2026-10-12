#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/*
 * Tiled GEMM (float32) with cudaEvent timing vs a naive baseline.
 *
 *   - naive: each output re-reads a full row/col from global memory
 *   - tiled: a block reuses TILE x TILE panels from shared memory
 *
 * Plain shared-memory tiling — not a cuBLAS-level implementation.
 */

#define N 512
#define TILE 16

__global__ void gemm_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void gemm_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < n; t += TILE) {
        As[threadIdx.y][threadIdx.x] =
            (row < n && t + threadIdx.x < n) ? A[row * n + t + threadIdx.x]
                                             : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (t + threadIdx.y < n && col < n) ? B[(t + threadIdx.y) * n + col]
                                             : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

static float time_kernel(void (*launch)(const float*, const float*, float*, int,
                                        dim3, dim3),
                         const float* dA, const float* dB, float* dC, int n,
                         dim3 grid, dim3 block, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        launch(dA, dB, dC, n, grid, block);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        launch(dA, dB, dC, n, grid, block);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / static_cast<float>(iters);
}

static void launch_naive(const float* A, const float* B, float* C, int n,
                         dim3 grid, dim3 block) {
    gemm_naive<<<grid, block>>>(A, B, C, n);
}

static void launch_tiled(const float* A, const float* B, float* C, int n,
                         dim3 grid, dim3 block) {
    gemm_tiled<<<grid, block>>>(A, B, C, n);
}

int main() {
    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);
    float* hA = static_cast<float*>(malloc(bytes));
    float* hB = static_cast<float*>(malloc(bytes));
    float* hC = static_cast<float*>(malloc(bytes));
    float* hRef = static_cast<float*>(malloc(bytes));

    for (int i = 0; i < N * N; ++i) {
        hA[i] = static_cast<float>((i % 13) + 1);
        hB[i] = static_cast<float>((i % 11) + 1);
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // Correctness: tiled vs naive
    launch_naive(dA, dB, dC, N, grid, block);
    cudaMemcpy(hRef, dC, bytes, cudaMemcpyDeviceToHost);
    launch_tiled(dA, dB, dC, N, grid, block);
    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < N * N; ++i) {
        if (hC[i] != hRef[i]) {
            ok = false;
            break;
        }
    }

    float naive_ms =
        time_kernel(launch_naive, dA, dB, dC, N, grid, block, 5, 20);
    float tiled_ms =
        time_kernel(launch_tiled, dA, dB, dC, N, grid, block, 5, 20);

    printf("N=%d TILE=%d\n", N, TILE);
    printf("correctness: %s\n", ok ? "PASS" : "FAIL");
    printf("naive mean ms: %.3f\n", naive_ms);
    printf("tiled mean ms: %.3f\n", tiled_ms);
    printf("speedup (naive/tiled): %.2fx\n", naive_ms / tiled_ms);

    free(hA);
    free(hB);
    free(hC);
    free(hRef);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return ok ? 0 : 1;
}
