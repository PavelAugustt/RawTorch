#include <iostream>

# define N 10000

__global__ void add(int *a, int *b, int *c)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    while (id < N)
    {
        c[id] = a[id] + b[id];
        id += blockDim.x * gridDim.x;
    }
}

int main(void)
{
    int a[N], b[N], c[N];

    int *device_a, *device_b, *device_c;

    cudaMalloc((void**)&device_a, N * sizeof(int));
    cudaMalloc((void**)&device_b, N * sizeof(int));
    cudaMalloc((void**)&device_c, N * sizeof(int));

    for(int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    cudaMemcpy(device_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    add<<<128, 128>>>(device_a, device_b, device_c);


    cudaMemcpy(c, device_c, N * sizeof(int), cudaMemcpyDeviceToHost);


    for(int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}