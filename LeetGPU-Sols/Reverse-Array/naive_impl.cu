#include "solve.h"
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < N/2)
    {
        float temp = input[tid];
        input[tid] = input[N - tid - 1];
        input[N - tid - 1] = temp;
    }
}

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
