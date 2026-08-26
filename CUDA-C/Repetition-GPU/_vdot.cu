#include <iostream>
#define N 1000

__global__ void _vDot(float *a, float *b, float *c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while(tid < N)
    {
        c[tid] = a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(void)
{
    float a[N], b[N], c[N];

    float *d_a, *d_b, *d_c;

    size_t size = N * sizeof(float);
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    for(int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, a, size, cudaMemcpyHostToDevice);

    _vDot<<<32, 256>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    float res = 0;
    for(int i = 0; i < N; i++)
    {
        printf("%f * %f", a[i], b[i]);
        printf(" +");
        res += c[i];
    }

    printf(" =%f\n", res);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}