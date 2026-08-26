# include <iostream>
# define count 1024


__global__ void gemmGPU(int *a, int *b, int *c, int M, int N, int K, float alpha, float beta)
{
    int colId = threadIdx.x + blockIdx.x * blockDim.x;
    int rowId = threadIdx.y + blockIdx.y * blockDim.y;

    
    // this is wrong
    if(rowId < M && colId < N)
    {
        float temp = 0;
        for(int k = 0; k < K; k++)
        {
            temp += a[rowId * K + k] * b[k * N + colId];
        }
        // its pretty simple scaling factors are essentially just adjusting how much of the data we need 
        // formula for GEMM is somethign like this C[I, J] = alpha x (a[i, j] * b[j, i]) + beta x c[i, j]
        // essentially the core intuition iswe assume our cmatrix is not empty
        // for example it could be containing weights ina neural network or it could be cntaining image data pixel values
        // we specify alpha and beta as something between 0 and 1, for example if we give beta = 0 we dont watn any data from our previous c matrix
        // or our original image, just make the change required, or if we give something like alpha = 0.7 and beta = 0.3 preserve
        // 30% of the information of previous matrix weights and nudge it in the new direction of our matix mult at 70% essentially its also like normalization
        c[rowId * N + colId] = alpha * temp + beta * c[rowId * N + colId]; 
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gemmGPU<<<gridDim, blockDim>>>(d_a, d_b, d_c, count, count, count, 0.7, 0.3);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel elapsed time: %f \n", milliseconds);

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