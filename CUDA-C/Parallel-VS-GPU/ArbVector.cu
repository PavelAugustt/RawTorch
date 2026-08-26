#include <iostream>

# define N 10000

__global__ void add(int *a, int *b, int *c)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // i think this implementation got a whole lot fucking simpler becuase i understood how the sequential magic formula works
    // basically how we linearize it

    // lets think of it this way
    // in our past implementation when we called the function we explicity defined the total number of blocks as N/128
    // but what if N/128 was beyond the total number of blocks we can conujure up, in other words we wnat to tod it for 
    // any arbitrary number parallely

    // its good to think of it like this, in each iteration due to our linearization 
    // all operations happen seqeuntially thanks to the id in parallel, so lets say our goal is 1000 and in the first iterastion
    // all of them ran a 100 thread or operations each thread of execution should know the number it must process next
    // that is pretty easy we just increment our id by the total number of threads in the entire execution of our parallel program
    // in other words the total number of threads in each block multiplied by the total number of blocks in our entire grid
    
    // this will result with the total number of threads executing our kernel in the entire grid, if we increment our current id by that
    // it should logically end up in its next sequential position posed to just do its one operation
    while(id < N)
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
        b[i] = -i;
    }

    cudaMemcpy(device_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // here our total number of blocks dont matter it can be arbitrary we can do more blocks in parallel to be faster
    // or not even if we didn't it would work because its a while loop, our threads wont stop executing until each kernel id reaches past
    // the max vector length count
    add<<<4,128>>>(device_a, device_b, device_c);

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