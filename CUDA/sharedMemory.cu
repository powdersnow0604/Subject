#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "eigen3/Eigen/Core"
#include "Profiler.h"


//example of matrix multiplication of matrix whose size is less than 1024 by using shared memory


using namespace Eigen;


#define ROW_SIZE (32)
#define K_SIZE   (128)
#define COL_SIZE (32)

#define MAT_SIZE_A (ROW_SIZE*K_SIZE)
#define MAT_SIZE_B (K_SIZE*COL_SIZE)
#define MAT_SIZE_C (ROW_SIZE*COL_SIZE)


__global__ void matMul_kernel(float* _A, float* _B, float* _C)
{
	int row = threadIdx.x;
	int col = threadIdx.y;
	int index = row * blockDim.y + col;

	float result = 0;
	for (int k = 0; k < K_SIZE; k++)
		result += _A[row * K_SIZE + k] * _B[col + k * COL_SIZE];
	_C[index] = result;
}

__global__ void matMul_kernel_shared(float* _A, float* _B, float* _C)
{
	int row = threadIdx.x;
	int col = threadIdx.y;
	int index = row * blockDim.y + col;

	__shared__ float sA[ROW_SIZE][K_SIZE];	// 32 * 256 * 4 bytes = 16 KB
	__shared__ float sB[K_SIZE][COL_SIZE];	// 16 KB

	if (row == 0) { // read matrix B
		for (int k = 0; k < K_SIZE; k++)
			sB[k][col] = _B[col + k * COL_SIZE];
	}
	if (col == 0 ) { // read matrix A
		for (int k = 0; k < K_SIZE; k++)
			sA[row][k] = _A[row * K_SIZE + k];

	}

	__syncthreads(); // wait until all threads load the matrix

	float result = 0;
	for (int k = 0; k < K_SIZE; k++)
		result += sA[row][k] * sB[k][col];
	_C[index] = result;
}


int main()
{
    Profiler prof("CUDA profiler");
    MatrixXf A(ROW_SIZE, K_SIZE), B(K_SIZE, COL_SIZE), C(ROW_SIZE, COL_SIZE), dC(ROW_SIZE, COL_SIZE);
    float *da, *db, *dc;

    A.setRandom();
    B.setRandom();

    cudaMalloc(&da, sizeof(float)*MAT_SIZE_A);
    cudaMalloc(&db, sizeof(float)*MAT_SIZE_B);
    cudaMalloc(&dc, sizeof(float)*MAT_SIZE_C);

    {
        auto _ = prof.profile("MM on Host");

        C = A * B;
    }

    {
        auto _ = prof.profile("memcpy host -> device");

        cudaMemcpy(da, A.data(), sizeof(float)*MAT_SIZE_A, cudaMemcpyHostToDevice);
        cudaMemcpy(db, B.data(), sizeof(float)*MAT_SIZE_B, cudaMemcpyHostToDevice);
    }

    dim3 blockdim(ROW_SIZE, COL_SIZE);

    {
        auto _ = prof.profile("MM with kernel without shared memory");

        matMul_kernel<<<1,blockdim>>>(da,db,dc);
        cudaDeviceSynchronize();
    }

    {
        auto _ = prof.profile("MM with kernel with shared memory");

        matMul_kernel_shared<<<1,blockdim>>>(da,db,dc);
        cudaDeviceSynchronize();
    }

    {
        auto _ = prof.profile("memcpy device -> host");

        cudaMemcpy(dC.data(), dc, sizeof(float)*MAT_SIZE_C, cudaMemcpyDeviceToHost);
    }

    prof.report();

    return 0;
}