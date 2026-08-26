#include <iostream>
# define Count 2    

__global__ void gemmGPU(int *a, int *b, int *c, int M, int N, int K)
{
    int colId = threadIdx.y + blockIdx.y * blockDim.y;
    int rowId = threadIdx.x + blockIdx.x * blockDim.x;

    while(colId < N && rowId < M)
    {
        for(int k = 0; k < K; k++)
        {
            c[rowId * N + colId] += a[rowId * K + k] * b[k * N + colId];
        }

        colId += blockDim.y * gridDim.y;
        rowId += blockDim.x * gridDim.x;
    }
}

void printMatrix(int *a, int M, int N)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            printf(" %d ", a[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void gemmCPU(int *a, int *b, int *c, int M, int N, int K)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            for(int k = 0; k < K; k++)
            {
                // N * k cus well total number of columns = total number of elements in a row
                // when u multiply it with k itll be shifting to each corresponding element in that column
                // when u add j to it itll offset column by 1

                // i * K basically gets the same as well it gets the row but i is slow to incrmeent
                // i wont incrmeent until i hits max, in other words adding k in eachiteration
                // will add each element in a row before i increments and takes it to thenext row
                c[i * N + j] += a[i * K + k] * b[k * N + j];
            }
        }
    }
}

int main(void)
{

    int *a, *b, *c, *d;
    size_t size = Count * Count * sizeof(int);
    a = (int*) malloc(size);
    b = (int*) malloc(size);
    c = (int*) malloc(size);
    d = (int*) malloc(size);

    for(int i = 0; i < Count * Count; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    // just in case our values wont get stacked on garbage values
    memset(c, 0, size);
    memset(d, 0, size);

    gemmCPU(a, b, c, Count, Count, Count);

    printMatrix(a, Count, Count);
    printMatrix(b, Count, Count);
    printMatrix(c, Count, Count);

    dim3 gridDim((Count+11)/12, (Count+11)/12);
    dim3 blockDim(12, 12);

    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_c, size);
    cudaMalloc((void**)&d_b, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    gemmGPU<<<gridDim, blockDim>>>(d_a, d_b, d_c, Count, Count, Count);

    cudaMemcpy(d, d_c, size, cudaMemcpyDeviceToHost);

    printMatrix(d, Count, Count);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
    free(d);
    return 0;
}