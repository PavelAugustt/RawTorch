# include <iostream>
# define N 2


__global__ void gemmGPU(int *a, int *b, int *c, int m, int n, int K)
{
    /* flipping row/col id doesnt have an effect on correctness but it canimpact 
    impact performance due to something called memory coaelscing and data locality
    which i have no idea what it means but i will learn soon. */
    int colId = threadIdx.x + blockIdx.x * blockDim.x;
    int rowId = threadIdx.y + blockIdx.y * blockDim.y;

    while(rowId < m && colId < n)
    {
        c[rowId * n + colId] = 0;
        for(int k = 0; k < K; k++)
        {
            c[rowId * n + colId] += a[rowId * K + k] * b[k * n + colId];
        }

        colId += blockDim.x * gridDim.x;
        rowId += blockDim.y * gridDim.y;
    }

}

void printMatrix(int *a, int m, int n)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            printf(" %d ", a[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void gemmCPU(int *a, int *b, int *c, int m, int n, int K)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            for(int k = 0; k < K; k++)
            {
                c[i * n + j] += a[i * K + k] * b[k * n + j];
            }
        }
    }
}

int main(void)
{
    int *a, *b, *c;

    size_t size = N * N * sizeof(int);

    a = (int*) malloc(size);
    b = (int*) malloc(size);
    c = (int*) malloc(size);

    for(int i = 0; i < N * N; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    printMatrix(a, N, N);

    printMatrix(b, N, N);

    gemmCPU(a, b, c, N, N, N);

    printMatrix(c, N, N);

    int *d;
    int *d_a, *d_b, *d_c;
    d = (int*) malloc(size);

    cudaMalloc((void**)&d_c, size);
    cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
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
    gemmGPU<<<gridDim, blockDim>>>(d_a, d_b, d_c, N, N, N);

    cudaMemcpy(d, d_c, size, cudaMemcpyDeviceToHost);

    printMatrix(d, N, N);
    cudaFree(d_c);
    cudaFree(d_a);
	cudaFree(d_b);
	free(d);

    free(a);
    free(b);
    free(c);
    return 0;
}