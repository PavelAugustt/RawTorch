# Profiling

## Elapsed Term for this for nested NAIVE stride loop impementation
```
Kernel elapsed time: 32.848255
Kernel elapsed time: 32.683743

```

```
__global__ void gemmGPU(int *a, int *b, int *c, int M, int N, int K)
{
    for(int row = threadIdx.x + blockIdx.x * blockDim.x; row < M; row += blockDim.x * gridDim.x)
    {
        for(int col = threadIdx.y + blockIdx.y * blockDim.y; col < N; col += blockDim.y * gridDim.y)
        {
            for(int k = 0; k < K; k++)
            {
                c[ row * N + col ] += a[row * K + k] * b[k * N + col];
            }
            
        }
    }
}
```

I think its wrong personally tbh atleast the earlier one will look into it

## Elapsed Term for this simple if based NAIVE implementation would be:
```
Kernel elapsed time: 9.918016
Kernel elapsed time: 9.742080

```
__global__ void gemmGPU(int *a, int *b, int *c, int M, int N, int K)
{
    int colId = threadIdx.x + blockIdx.x * blockDim.x;
    int rowId = threadIdx.y + blockIdx.y * blockDim.y;

    
    // this is wrong
    if(rowId < M && colId < N)
    {
        for(int k = 0; k < K; k++)
        {
            c[rowId * N + colId] += a[rowId * K + k] * b[k * N + colId];
        }
    }
}
```
in terms of efficiency this is better but why?

