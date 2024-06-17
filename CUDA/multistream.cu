#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Profiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_BLOCK (128 * 1024)
#define ARRAY_SIZE (1024 * NUM_BLOCK)

#define NUM_STREAMS 4

#define WORK_LOAD 256

__global__ void myKernel(int *_in, int *_out)
{
    int tID = blockDim.x * blockIdx.x + threadIdx.x;

    int temp = 0;
    int in = _in[tID];
    for (int i = 0; i < WORK_LOAD; i++)
    {
        temp = (temp + in * 5) % 10;
    }
    _out[tID] = temp;
}

int main(void)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    printf(
        "  Concurrent copy and kernel execution:          %s with %d copy "
        "engine(s)\n",
        (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);

    Profiler profiler;

    int *in = NULL, *out = NULL, *out2 = NULL;

    cudaMallocHost(&in, sizeof(int) * ARRAY_SIZE);
    memset(in, 0, sizeof(int) * ARRAY_SIZE);

    cudaMallocHost(&out, sizeof(int) * ARRAY_SIZE);
    memset(out, 0, sizeof(int) * ARRAY_SIZE);

    cudaMallocHost(&out2, sizeof(int) * ARRAY_SIZE);
    memset(out2, 0, sizeof(int) * ARRAY_SIZE);

    int *dIn, *dOut;
    cudaMalloc(&dIn, sizeof(int) * ARRAY_SIZE);
    cudaMalloc(&dOut, sizeof(int) * ARRAY_SIZE);

    for (int i = 0; i < ARRAY_SIZE; i++)
        in[i] = rand() % 10;

    // Single stram version
    {
        auto out_prof = profiler.profile("Single stream");

        {
            auto in_prof = profiler.profile("  * Host -> Device");
            cudaMemcpy(dIn, in, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);
        }

        {
            auto in_prof = profiler.profile("  * Kernel execution");
            myKernel<<<NUM_BLOCK, 1024>>>(dIn, dOut);
            cudaDeviceSynchronize();
        }

        {
            auto in_prof = profiler.profile("  * Device -> Host");
            cudaMemcpy(out, dOut, sizeof(int) * ARRAY_SIZE, cudaMemcpyDeviceToHost);
        }
    }

    // Multiple stream version
    cudaStream_t stream[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamCreate(&stream[i]);

    int chunkSize = ARRAY_SIZE / NUM_STREAMS;

    {
        auto prof = profiler.profile("Multiple Streams");

        for (int i = 0; i < NUM_STREAMS; i++)
        {
            int offset = chunkSize * i;
            cudaMemcpyAsync(dIn + offset, in + offset, sizeof(int) * chunkSize, cudaMemcpyHostToDevice, stream[i]);
        }

        for (int i = 0; i < NUM_STREAMS; i++)
        {
            int offset = chunkSize * i;
            myKernel<<<NUM_BLOCK / NUM_STREAMS, 1024, 0, stream[i]>>>(dIn + offset, dOut + offset);
        }

        for (int i = 0; i < NUM_STREAMS; i++)
        {
            int offset = chunkSize * i;
            cudaMemcpyAsync(out2 + offset, dOut + offset, sizeof(int) * chunkSize, cudaMemcpyDeviceToHost, stream[i]);
        }

        cudaDeviceSynchronize();
    }

    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        if (out[i] != out2[i])
            printf("!");
    }

    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamDestroy(stream[i]);

    profiler.report();

    cudaFree(dIn);
    cudaFree(dOut);

    cudaFreeHost(in);
    cudaFreeHost(out);
    cudaFreeHost(out2);

    return 0;
}