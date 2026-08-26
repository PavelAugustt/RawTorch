# include <iostream>
# define N 2


__global__ void gemmGPU(int *a, int *b, int *c, int m, int n, int K)
{
    // flipping row/col id doesnt have an effect on correctness but it canimpact 
    // impact performance due to something called memory coaelscing and data locality
    // which i have no idea what it means but i will learn soon.
    int colId = threadIdx.x + blockIdx.x * blockDim.x;
    int rowId = threadIdx.y + blockIdx.y * blockDim.y;

    /* although this logic works. i think it could be expanded even further for by breaking down
    entire matrices into chunks or tiles, which means each thread would process much lesser 
    numbner of elements and elements that are closer to each other in memory,this would mean that
    they accumulate the dot product for the same index i, j, then finally when all tiles compute for i, j
    they just add to the whole. achieving more concurrency, i think thats how tiling should work
    but fyi i haveno idea how actual tiling works this is just my idea for further optimiztion
    */
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
    // m x k -> dimension of first matrix a
    // k x n -> dimension of second matrix b
    // m x n -> dimension of resulting matrix c

    // for each row in matrix a
    for(int i = 0; i < m; i++)
    {
        // for each column in matrix b
        for(int j = 0; j < n; j++)
        {
            // for each element in both row and column of matrix a, b
            for(int k = 0; k < K; k++)
            {
                // for each element i, j in matrix c, for each element k 
                // in row i * K of matrix a, we multiply with each element j
                // in each row k * N of matrix b. 
                // in other words one we iteratue thru each element in row of matrix 1
                // by offsetting each elemnt by k for each row i * k 
                // simultaneously we use row major order with k * n, to jump to
                // each row and get the j corresponding element of matrix b which would give us columns
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