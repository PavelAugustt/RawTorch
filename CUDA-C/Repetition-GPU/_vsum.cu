#include <iostream>
#define N 10

__global__ void v_sum(int *A, int *B, int *C)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while(tid < N)
    {
        C[tid] = A[tid] + B[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(void)
{

    size_t size = N * sizeof(int);
    int A[N], B[N], C[N];

    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    for(int i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = -i;
    }

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    

    v_sum<<<32, 256>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", A[i], B[i], C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    return 0;
}