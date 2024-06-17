#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


__global__ void checkIndex()
{
    printf("threadIdx:(%d, %d, %d), blockIdx:(%d, %d, %d), blockDim:(%d, %d, %d), gridDim:(%d,%d,%d)\n", 
    threadIdx.x, threadIdx.y, threadIdx.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z);
}


int main()
{
    dim3 dimBlock(3);
    dim3 dimGrid(2);

    checkIndex<<<dimGrid, dimBlock>>>();
    cudaDeviceSynchronize();
    
    return 0;
}