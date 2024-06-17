#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "Profiler.h"

using namespace std::chrono;

typedef float MATTYPE;

const int block_size = 32;

__global__ void MatMul(MATTYPE *A, MATTYPE *B, MATTYPE *C, int m, int n, int k)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int index = row * n + col;

    if(row >= m || col >= n) return;

    C[index] = 0;
    for(int offset = 0; offset < k; offset++){
        C[index] += __fmul_rn(A[row*k + offset], B[col + offset*n]);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        puts("arguments must be three (path , m, k, n)");
        return 1;
    }

    int i;
    int m = atoi(argv[1]), k = atoi(argv[2]), n = atoi(argv[3]);
    int sizeA = m*k, sizeB = k*n, sizeC = m*n;
    MATTYPE *A, *B, *C;
    MATTYPE *dA, *dB, *dC;

    Profiler p_device("CUDA total");

    A = new MATTYPE[sizeA];
    B = new MATTYPE[sizeB];
    C = new MATTYPE[sizeC];

    for (i = 0; i < sizeA; i++)
    {
        A[i] = rand() % 1024;
    }

    for (i = 0; i < sizeB; i++)
    {
        B[i] = rand() % 1024;
    }

    cudaMalloc(&dA, sizeA);
    cudaMemset(dA, 0, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMemset(dB, 0, sizeB);
    cudaMalloc(&dC, sizeC);
    cudaMemset(dC, 0, sizeC);

    int block_x = (int)ceil((double)m / block_size);
    int block_y = (int)ceil((double)n / block_size);

    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid(block_x, block_y);

    // Data copy : Host -> Device
    {
        auto _ = p_device.profile("Data Trans.(Hoat -> Device)");
        cudaMemcpy(dA, A, sizeA * sizeof(MATTYPE), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, sizeB * sizeof(MATTYPE), cudaMemcpyHostToDevice);
    }

    // Kernel call
    {
        auto _ = p_device.profile("Computation(Kernel)");
        MatMul<<<dimGrid, dimBlock>>>(dA, dB, dC, m, n, k);
        cudaDeviceSynchronize();
    }

    // Copy results : Device -> Host
    {
        auto _ = p_device.profile("Data Trans.(Device -> Host)");
        cudaMemcpy(C, dC, sizeC * sizeof(MATTYPE), cudaMemcpyDeviceToHost);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    p_device.report();

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}