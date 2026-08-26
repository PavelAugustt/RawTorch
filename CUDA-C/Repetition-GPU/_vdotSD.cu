#include <iostream>
#define N 1000

__global__ void _vDot(float *a, float *b, float *c)
{

    __shared__ float cache[256];

    int cacheId = threadIdx.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double temp = 0;
    while(tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    __syncthreads();
    cache[cacheId] = temp;

    int i = blockDim.x / 2;

    while(i != 0)
    {
        if(cacheId < i)
        {
            cache[cacheId] += cache[cacheId + i];
        }
        __syncthreads();
        i = i/2;
    }

    if(cacheId == 0)
    {
        c[blockIdx.x] = cache[0];
    }
}

int main(void)
{
    float a[N], b[N], c[N];

    float *d_a, *d_b, *d_c;

    size_t size = N * sizeof(float);

    size_t csize = (N+255)/256 * sizeof(float);
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, csize);

    for(int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    _vDot<<<(N+255)/256, 256>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, csize, cudaMemcpyDeviceToHost);

    double res = 0;
    for(int i = 0; i < (N+255)/256; i++)
    {
        res += c[i];
    }

    printf(" =%f\n", res);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}