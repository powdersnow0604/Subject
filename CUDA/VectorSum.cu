#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Profiler.h"

// The size of the vector
#define NUM_DATA 1048576

// Simple vector sum kernel (Max vector size : 1024)
__global__ void vecAdd(int *_a, int *_b, int *_c)
{
	int tID = threadIdx.x + blockDim.x * blockIdx.x;
	if(tID < NUM_DATA) _c[tID] = _a[tID] + _b[tID];
}

using namespace std::chrono;

int main(void)
{
	int *a, *b, *c, *hc; // Vectors on the host
	int *da, *db, *dc;	 // Vectors on the device

	Profiler sec_total("CUDA total");
	Profiler sec_kernel("Computation(Kernel)");
	Profiler sec_transHD("Data Trans.(Hoat -> Device)");
	Profiler sec_transDH("CUDA total.(Device -> Host)");
	Profiler sec_vecaddH("VecAdd on Host");

	int memSize = sizeof(int) * NUM_DATA;
	printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

	// Memory allocation on the host-side
	a = new int[NUM_DATA];
	memset(a, 0, memSize);
	b = new int[NUM_DATA];
	memset(b, 0, memSize);
	c = new int[NUM_DATA];
	memset(c, 0, memSize);
	hc = new int[NUM_DATA];
	memset(hc, 0, memSize);

	// Data generation
	for (int i = 0; i < NUM_DATA; i++)
	{
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	// Vector sum on host (for performance comparision)
	{
		auto _ = sec_vecaddH.profile();

		for (int i = 0; i < NUM_DATA; i++)
			hc[i] = a[i] + b[i];
	}

	// Memory allocation on the device-side
	cudaMalloc(&da, memSize);
	cudaMemset(da, 0, memSize);
	cudaMalloc(&db, memSize);
	cudaMemset(db, 0, memSize);
	cudaMalloc(&dc, memSize);
	cudaMemset(dc, 0, memSize);

	{
		auto P1 = sec_total.profile();

		// Data copy : Host -> Device
		{
			auto _ = sec_transHD.profile();
			cudaMemcpy(da, a, memSize, cudaMemcpyHostToDevice);
			cudaMemcpy(db, b, memSize, cudaMemcpyHostToDevice);
		}

		// Kernel call
		{
			auto _ = sec_kernel.profile();
			vecAdd<<<(uint64_t)ceil(NUM_DATA/1024.), 1024>>>(da, db, dc);
			cudaDeviceSynchronize();
		}

		// Copy results : Device -> Host
		{
			auto _ = sec_transDH.profile();
			cudaMemcpy(c, dc, memSize, cudaMemcpyDeviceToHost);
		}
	}
	
	// Release device memory
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	// Check results
	bool result = true;
	for (int i = 0; i < NUM_DATA; i++)
	{
		if (hc[i] != c[i])
		{
			printf("[%d] The result is not matched! (%d, %d)\n", i, hc[i], c[i]);
			result = false;
		}
	}

	if (result)
		printf("GPU works well!\n");

	sec_total.report<microseconds>();
	sec_kernel.report<microseconds>();
	sec_transHD.report<microseconds>();
	sec_transDH.report<microseconds>();
	sec_vecaddH.report<microseconds>();

	// Release host memory
	delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}