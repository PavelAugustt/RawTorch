#include <iostream>

// ill try to explain the logic to make sure i understand so ignore the shiczo ramblings (this is meant to be private anyways)

// i fixed the issue apparently i switched threads per grid (block actually)
// with the n value, no fucking wonder i got result = 0 skull. if u dont get it
// my gpu clearly doesnt have 33 * 100 threads much less in a single fucking block
// also its threads per block btw, grid is where the blocks are arranged in 2d, threads are 3d in grid
# define threadsPerGrid 256
# define N 1000

# define min(a, b) {a<b?a:b}

// just in case, right now we have N == 1000 what if N was something smaller than the predefined number of bloks pre grid
// that is 32, we do not want that as its a waste of resources i think
const int blocksPerGrid = min(32, N+threadsPerGrid-1/threadsPerGrid);

__global__ void dot(int *a, int *b, float *c)
{
    // shared memory is just memory that is shared across every thread in a block
    // here we define an array called cache with length of a threads per block, to store corresponding values for the dot product
    __shared__ float cache[threadsPerGrid];
    // simple linearization to add all the threads that came before this block and in this block, this will make each thread sequential
    // in an incremental manner
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // im defining cache index here
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while (id < N)
    {
        temp += a[id] * b[id];
        // we're offsetting the current id by every single id that came before it including it, which means it makes the id 2x
        // meaning next iteration fucker
        id += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x/2;

    while(i != 0)
    {
        if(threadIdx.x < i)
        {
            // essentially this part basically just kinda counterintuitive
            // what uoriginally thought of was i = i + 1, this wont work exactly as we want 
            // for example look at index 0 and index 1 in that approach of the reduction algorithm
            // 0 will add  0 + 1, and 1 will add 1 + 2, but in the next iteration the window ill be halved
            // but ht eproblem is our both 0 and 1 will still be the same, the value at index 1 will be added twice
            // into our final result, which can skew our result

            // instead what we do is looka t current cache index and add it with current index + its double
            // why wont this overlap exactly? because in each iteration our search window is halved
            // in other words the 2x value of the current index we're adding will never be involved in a second operation again
            // this will keep happening untl we hit 0 and it wotn skew the results i think
            cache[cacheIndex] += cache[cacheIndex + i];
           
            
        }
        __syncthreads();
        // there is an intersting fact btw about sync threads there is a spefici type of infintie loop called thread divergence
        // lets look at it this way the above if coniditional basically makes sure that our kernels only do the operations
        // when its in the search window and matters for the summation to get our result
        // in other words it doesnt make unecessary kernels do its bidding giving a minor performance improvement i believe
        // why not add syncthreads into thsi optimization method cus we only hav eot sync with threads that are actually
        // doing any work. that wont work itll lead to a for loop, why? mainly because 
        // when sync threads are called basically our gpu waits for every single thread to call the syncthread function
        // only when every single executor has called the syncthreads will it continue, so in our conditional
        // some threads wont actually pass the condition, so if our sync hreads were in there, they wont acc end up syncing
        // in othe words our gpu will end up waiting for syncthreads to be called forever
        // how can u fix this i think u shouldnt put this in a conditional unless u expect 100% of the conditionals to pass
        // which kinda defeats the purpose of conditionals
        // in other words make sure its in ur else or else if in all possibilities jsut in case ur using syncthreads in a conditional
        
        i = i / 2;
    }

    // hee its pretty simple we only check 0 cus who cares not exactly lol we only check 0 cus well 
    // if our results get halved then our summation will exist at the first index which is 0 
    // so we need only one kernel to add this, so we do that with kernel 0 in that block
    if(cacheIndex == 0)
    {
        c[blockIdx.x] = cache[0];
    }
}

int main(void)
{
    printf("hello world\n");
    int a[N], b[N];

    float c[blocksPerGrid];

    int *device_a, *device_b;

    float *device_c;

    cudaMalloc((void**)&device_a, N * sizeof(int));
    cudaMalloc((void**)&device_b, N * sizeof(int));
    cudaMalloc((void**)&device_c, blocksPerGrid* sizeof(float));

    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    cudaMemcpy(device_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    dot<<<blocksPerGrid, threadsPerGrid>>>(device_a, device_b, device_c);

    cudaMemcpy(c, device_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    float result = 0;
    for(int i = 0; i < blocksPerGrid; i++)
    {
        result += c[i];

    }

    printf("result: %f\n", result);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}
