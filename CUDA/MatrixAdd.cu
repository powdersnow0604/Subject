#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "Profiler.h"

#define NUM_ROW 8192
#define NUM_COL 8192
#define NUM_MAT NUM_ROW*NUM_COL


using namespace std::chrono;


__global__ void MatAdd(float* A, float* B, float* C, size_t size)
{
    int idx_thread = threadIdx.y * blockDim.x + threadIdx.x;
    int idx_block = blockIdx.y * gridDim.x + blockIdx.x;
    int block_size = blockDim.x * blockDim.y;
    int idx = idx_block * block_size + idx_thread;

    if(idx < size) C[idx] = A[idx] + B[idx];
}


int main()
{
    float *A, *B, *C, *hC;
    float *dA, *dB, *dC;
    size_t memSize = sizeof(float) * NUM_MAT;

    Profiler sec_total("CUDA total");
	Profiler sec_kernel("Computation(Kernel)");
	Profiler sec_transHD("Data Trans.(Hoat -> Device)");
	Profiler sec_transDH("CUDA total.(Device -> Host)");
	Profiler sec_vecaddH("VecAdd on Host");

    A = new float[NUM_MAT];
    B = new float[NUM_MAT];
    C = new float[NUM_MAT];
    hC = new float[NUM_MAT];
    

    for(int i = 0; i < NUM_MAT; i++){
        A[i] = rand() % 1024;
        B[i] = rand() % 1024;
    }

    {
        auto _ = sec_vecaddH.profile();
        for(int i = 0; i <  NUM_MAT; i++){
            hC[i] = A[i] + B[i];
        }
    }

    cudaMalloc(&dA, memSize);
	cudaMemset(dA, 0, memSize);
	cudaMalloc(&dB, memSize);
	cudaMemset(dB, 0, memSize);
	cudaMalloc(&dC, memSize);
	cudaMemset(dC, 0, memSize);

    size_t block_y = (size_t)ceil((double)NUM_ROW / 32);
    size_t block_x = (size_t)ceil((double)NUM_COL / 32);

    dim3 dimBlock(32,32);
    dim3 dimGrid(block_x, block_y);

    {
		auto P1 = sec_total.profile();

		// Data copy : Host -> Device
		{
			auto _ = sec_transHD.profile();
			cudaMemcpy(dA, A, memSize, cudaMemcpyHostToDevice);
			cudaMemcpy(dB, B, memSize, cudaMemcpyHostToDevice);
		}

		// Kernel call
		{
			auto _ = sec_kernel.profile();
			MatAdd<<<dimGrid, dimBlock>>>(dA, dB, dC, NUM_MAT);
			cudaDeviceSynchronize();
		}

		// Copy results : Device -> Host
		{
			auto _ = sec_transDH.profile();
			cudaMemcpy(C, dC, memSize, cudaMemcpyDeviceToHost);
		}
	}

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    bool result = true;
	for (int i = 0; i < NUM_MAT; i++)
	{
		if (hC[i] != C[i])
		{
			printf("[%d] The result is not matched! (%f, %f)\n", i, hC[i], C[i]);
			result = false;
		}
	}

	if (result)
		printf("GPU works well!\n");

	sec_total.report<milliseconds>();
	sec_kernel.report<milliseconds>();
	sec_transHD.report<milliseconds>();
	sec_transDH.report<milliseconds>();
	sec_vecaddH.report<milliseconds>();

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] hC;

    return 0;
}