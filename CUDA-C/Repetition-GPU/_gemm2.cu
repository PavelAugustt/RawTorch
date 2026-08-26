#include <iostream>

# define N 2

__global__ void transposeGPU(int *a, int *c, int columns, int rows)
{
    int rowId = threadIdx.x + blockIdx.x * blockDim.x;
    int columnId = threadIdx.y + blockIdx.y * blockDim.y;

    while(rowId < columns && columnId < columns)
    {
        c[rowId * columns + columnId] = a[columnId * rows + rowId];

        rowId += blockDim.x * gridDim.x;
        columnId += blockDim.y * gridDim.y;
    }
}

void printMatrix(int *matrix, int columns)
{   
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            printf(" %d ", matrix[i * columns + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void gemmCPU(int *a, int *b, int *c, int columns, int rows)
{
    for(int rowId = 0; rowId < N; rowId++)
    {
        for(int columnId = 0; columnId < N; columnId++)
        {
            for(int elementId = 0; elementId < N; elementId++)
            {
                // i donmt really get this clusterfuck but im sure if i can visualize this i can write the kernel myself without outside refernce
                // lol this was just throwing darts at a wall until i hit th eright combination although i understand the memory layout its still fucked
                // very confusing
                c[rowId * columns + columnId] += a[rowId * columns + elementId] * b[elementId * rows + columnId]; 
            }
        }
    }
}

int main(void)
{
    size_t size = N * N * sizeof(int);

    int *a, *b, *d;

    a = (int*) malloc(size);
    b = (int*) malloc(size);

    d = (int*) malloc(size);

    for(int i = 0; i < N * N; i++)
    {
        a[i] = i;
        b[i] = i + 1;
    }

    gemmCPU(a, b, d, N, N);

    printMatrix(a, N);
    printMatrix(b, N);

    printMatrix(d, N);

    /*
    int *c;
    int *d_a, *d_c;
    c = (int*) malloc(size);

    cudaMalloc((void**)&d_c, size);
    cudaMalloc((void**)&d_a, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(12, 12, 1);

    int blockCount = 0;
    if (N > blockDim.x)
    {
        blockCount = (N+11)/12;
    }
    else
    {
        blockCount = 32;
    }
    dim3 gridDim(blockCount, blockCount, 1);
    transposeGPU<<<gridDim, blockDim>>>(d_a, d_c, N, N);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printMatrix(c, N);
    cudaFree(d_c);
    cudaFree(d_a);

    free(a);
    free(b);
    free(c);

    return 0;
    */
}