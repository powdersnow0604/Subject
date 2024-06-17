#include "kernelCall.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void kernel(void)
{
	printf("Device code running \
	on the GPU\n");
}

void kernelCall(void)
{
	kernel <<<1, 10>>> ();
	cudaDeviceSynchronize();
}