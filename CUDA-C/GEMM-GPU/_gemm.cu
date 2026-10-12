#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define N 256

/*
 * Naive square GEMM with a CPU golden check.
 *
 * Important correctness detail: C must start at zero before `+=` accumulation
 * on both CPU and GPU. Forgetting cudaMemset on device C is a classic bug.
 *
 * This is intentionally naive (global-memory K loop). For tiled shared-memory
 * GEMM, see the sibling LeetGPU-Sols repo: Mat-Mult/tiled_impl.cu
 */

__global__ void gemmGPU(const int* a, const int* b, int* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

void gemmCPU(const int* a, const int* b, int* c, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

int main() {
    size_t bytes = static_cast<size_t>(N) * N * sizeof(int);

    int* a = static_cast<int*>(malloc(bytes));
    int* b = static_cast<int*>(malloc(bytes));
    int* c_cpu = static_cast<int*>(malloc(bytes));
    int* c_gpu = static_cast<int*>(malloc(bytes));

    for (int i = 0; i < N * N; ++i) {
        a[i] = (i % 7) + 1;
        b[i] = (i % 5) + 1;
    }

    gemmCPU(a, b, c_cpu, N);

    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, bytes);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    gemmGPU<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < N * N; ++i) {
        if (c_cpu[i] != c_gpu[i]) {
            ok = false;
            break;
        }
    }

    printf("%s\n", ok ? "equal" : "not equal");

    free(a);
    free(b);
    free(c_cpu);
    free(c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return ok ? 0 : 1;
}
