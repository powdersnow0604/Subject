#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"

#define BLOCK_SIZE 64

__global__ void syncwarp_test()
{
    int tID = threadIdx.x;
    int warpID = (int)(tID / 32);
    __shared__ int masterID[BLOCK_SIZE];
    
    if (threadIdx.x % 32 == 0){
        masterID[warpID] = tID;
    }
    __syncwarp();

    printf("[T%d] The master of our warp is %d\n", tID, masterID[warpID]);
}

__global__ void threadCounting_atomicShared(int* a)
{
    __shared__ int sa;
    if(threadIdx.x == 0)
        sa = 0;

    __syncthreads();
    atomicAdd(&sa, 1);

    __syncthreads();

    if(threadIdx.x == 0)
        atomicAdd(a, sa);

}


int main()
{
    int *ptr, res;
    cudaMalloc(&ptr, sizeof(int));
    cudaMemset(ptr, 0, sizeof(int));

    threadCounting_atomicShared<<<4, 1024>>>(ptr);

    cudaMemcpy(&res, ptr, sizeof(int), cudaMemcpyDeviceToHost);

    printf("the result is %d\n",  res);

    return 0;
}