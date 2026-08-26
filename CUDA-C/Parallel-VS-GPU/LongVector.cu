#include <iostream>

# define N 10000

__global__ void add(int *a, int *b, int *c)
{

    // so basically this formula makes it so that each id across each thread in each fucking block sequential and incremental
    // how? i have no fucking clue how it fucking works but it just does i even did it on pen & paper to figure out intuiton
    // god in each block if we have 4 threads itll repeat with 0, 1, 2, 3 for threadIdx.x
    // if we add the block number lets say we have 4 blocks, then it would be 0 + 0, 1+0... 0 + 1, 1 + 1, do u see how it can create overlap
    // here when we multiply it by the number of threads which is blockDim with our block id
    // then 0 + 0 * 4, 1+ 0 * 4... 3 + 0 * 4, 0 + 1 * 4, 1 + 1 * 4, do u see how it makes it incremental?
    // so yeah its cool but i dont get it just that it works ill just remember it for later

    // i think i got it after so long, perseverance! this is my perseverance *insert fang yuan quote*
    // i believe instead of looking at thread id look at it this way the blockDim represents the number of threads in each block
    // in other words it represents one full block, block id instead of looking at it as an identifier look at it as the total
    // number of blocks that came before it, when you multiply the total number of blocks already processed + the total number of threads in each block
    // we get the total number of threads processed in the threads that sequentially came before the current thread by id
    // and when we add thread id into the mix it adds the number of threads processed in our current block to the total number of threads 
    // processed in the previous blocks giving us a truly incremental unique id across the grid parallely just amazing
    // how does one evne think of this
    // should i have not spent so much time understanding the magic formula?
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < N)
    {
        c[id] = a[id] + b[id];
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

    // here there is a funny, we declare 128 arbitray number of threads per block to make sure it doesnt go out of bounds beyond the nunber of threads we have in that block
    // but our blocks seem weirder, let me make it simpler
    // lets say we're launching 128 number of threads for our thing, our total number of operations to do is N which is our vector length
    // so if each block is running 128 threads in other words 128 operations of N, then we just do N/128 to figure out the number
    // of blocks we need but there is a problem, we're doing integer division
    // in other words its not a float or a double, when you divide for example 127/128 where the value is 0.99 ish, itll round it to 0
    // thats basically what integer division is, if this happens when our N is 127 for example then instead of launching just one block with an extra unused thread
    // our shit would just not run, so this is why we add 127 to our N, as a safe guard just in case, itll make sure that itll always launch excess kernels on top of what we need
    // it might be wasteful but it gets the job done!
    add<<<(N + 127)/128,128>>>(device_a, device_b, device_c);

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