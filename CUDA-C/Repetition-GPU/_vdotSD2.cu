# include <iostream>
# define N 1000

__global__ void _vdot(int *a, int *b, int *c)
{
	__shared__ int cache[128];

	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int temp = 0;

	int cacheID = threadIdx.x;
	while(tid < N)
	{
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	__syncthreads();

	cache[cacheID] = temp;

	int i = blockDim.x / 2;

	while(i != 0)
	{
		if(cacheID < i)
		{
			cache[cacheID] += cache[cacheID + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (cacheID == 0)
	{
		c[blockIdx.x] = cache[cacheID];
	}
}

int main(void)
{
	int a[N], b[N], c[(N+127)/128];

	int *d_a, *d_b, *d_c;

	size_t size = N * sizeof(int);
	for(int i = 0; i < N; i++)
	{
		a[i] = i;

		b[i] = i;
	}
	
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, ((N+127)/128) * sizeof(int));

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	_vdot<<<(N+127)/128, 128>>>(d_a, d_b, d_c);

	cudaMemcpy(c, d_c, ((N+127)/128) * sizeof(int), cudaMemcpyDeviceToHost);

	int res = 0;
	for(int i = 0; i < ((N+127)/128) ; i++)
	{
		res += c[i];
	}
	
	printf("result: %d\n", res);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}
