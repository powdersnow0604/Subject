#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

void checkDeviceMemory(void)
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Device memory (free/total) = %zu/%zu bytes\n", free, total);
}


int main(void)
{
    int *dPtr;
    cudaError_t errorCode;

    checkDeviceMemory();
    errorCode = cudaMalloc(&dPtr, sizeof(int) << 20);
    printf("cudaMalloc - %s\n", cudaGetErrorName(errorCode));
    checkDeviceMemory();

    errorCode = cudaMemset(dPtr, 0, sizeof(int) <<  20);
    printf("cudaMemset - %s\n", cudaGetErrorName(errorCode));

    errorCode = cudaFree(dPtr);
    printf("cudaFree - %s\n", cudaGetErrorName(errorCode));
    checkDeviceMemory();

    return 0;
}