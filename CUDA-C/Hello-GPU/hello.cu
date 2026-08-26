#include <iostream>

__global__ void helloFromGPU()
{
	printf("Hello dude!\n");
}

int main()
{
	helloFromGPU<<<1, 10>>>();
	cudaDeviceSynchronize();
	return 0;
}
