# include <iostream>
# define count 1024


__global__ void gemmGPU(int *a, int *b, int *c, int M, int N, int K)
{
    int colId = threadIdx.x + blockIdx.x * blockDim.x;
    int rowId = threadIdx.y + blockIdx.y * blockDim.y;

    
    // this is wrong
    while(rowId < M && colId < N)
    {
        for(int k = 0; k < K; k++)
        {
            c[rowId * N + colId] += a[rowId * K + k] * b[k * N + colId];
        }
        rowId += blockDim.y * gridDim.y;

        colId += blockDim.x * gridDim.x;
    }
}
void printMatrix(int *a, int M, int N)
{
    for(int i = 0; i < M; i++){
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
                c[i * N + j] += a[i * K + k] * b[k * N + j];
            }
        }
    }
}

int main(void)
{
    int *a, *b, *c;

    size_t size = count * count * sizeof(int);

    a = (int*) malloc(size);
    b = (int*) malloc(size);
    c = (int*) malloc(size);

    for (int i = 0;  i < count * count; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    gemmCPU(a, b, c, count, count, count);

    int *d = (int*) malloc(size);

    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((count+15)/16, (count+15)/16);
    gemmGPU<<<gridDim, blockDim>>>(d_a, d_b, d_c, count, count, count);

    cudaMemcpy(d, d_c, size, cudaMemcpyDeviceToHost);

    if(memcmp(c, d, size) == 0)
    {
        printf("Winner winner chicken dinner!\n");
    }
    else
    {
        printf("Fuck you loser\n");
    }

    free(a);
    free(b);
    free(c);
    free(d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;

}