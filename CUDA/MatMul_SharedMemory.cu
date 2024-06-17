#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "eigen3/Eigen/Core"
#include "Profiler.h"


//example of matrix multiplication of matrix whose size is less than 1024 by using shared memory


using namespace Eigen;

typedef int DATA_TYPE;


#define ROW_SIZE (512*2)
#define K_SIZE   (512*2)
#define COL_SIZE (512*4)

#define MAT_SIZE_A (ROW_SIZE*K_SIZE)
#define MAT_SIZE_B (K_SIZE*COL_SIZE)
#define MAT_SIZE_C (ROW_SIZE*COL_SIZE)
#define BLOCK_SIZE 32


__global__ void matMul_kernel(DATA_TYPE* _A, DATA_TYPE* _B, DATA_TYPE* _C)
{
	int row = threadIdx.x;
	int col = threadIdx.y;
	int index = row * blockDim.y + col;

	DATA_TYPE result = 0;
	for (int k = 0; k < K_SIZE; k++)
		result += _A[row * K_SIZE + k] * _B[col + k * COL_SIZE];
	_C[index] = result;
}

__global__ void MatMul_SharedMem(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	DATA_TYPE val = 0;
	__shared__ DATA_TYPE subA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ DATA_TYPE subB[BLOCK_SIZE][BLOCK_SIZE];

	int localRow = threadIdx.x;
	int localCol = threadIdx.y;

	for (int bID = 0; bID < ceil((float)k / BLOCK_SIZE); bID++) {
		int offset = bID * BLOCK_SIZE;

		// load A and B
		if (row >= m || offset + localCol >= k)
			subA[localRow][localCol] = 0;
		else
			subA[localRow][localCol] = matA[row * k + (offset + localCol)];

		if (col >= n || offset + localRow >= k)
			subB[localRow][localCol] = 0;
		else
			subB[localRow][localCol] = matB[(offset + localRow) * n + col];

		__syncthreads();

		// compute
		for (int i = 0; i < BLOCK_SIZE; i++) {
			val += subA[localRow][i] * subB[i][localCol];
		}
		__syncthreads();
	}

	if (row >= m || col >= n)
		return;

	matC[row * n + col] = val;
}

__global__ void MatMul_SharedMem_NoBankconflict(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	DATA_TYPE val = 0;
	__shared__ DATA_TYPE subA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ DATA_TYPE subB[BLOCK_SIZE][BLOCK_SIZE];

	int localRow = threadIdx.x;
	int localCol = threadIdx.y;

	for (int bID = 0; bID < ceil((float)k / BLOCK_SIZE); bID++) {
		int offset = bID * BLOCK_SIZE;

		// load A and B
		if (row >= m || offset + localCol >= k)
			subA[localCol][localRow] = 0;
		else
			subA[localCol][localRow] = matA[row * k + (offset + localCol)];

		if (col >= n || offset + localRow >= k)
			subB[localRow][localCol] = 0;
		else
			subB[localRow][localCol] = matB[(offset + localRow) * n + col];

		__syncthreads();

		// compute
		for (int i = 0; i < BLOCK_SIZE; i++) {
			val += subA[i][localRow] * subB[localCol][i];
		}
		__syncthreads();
	}

	if (row >= m || col >= n)
		return;

	matC[row * n + col] = val;
}


int main()
{
    Profiler prof("CUDA profiler");
    Matrix<DATA_TYPE,-1,-1,AutoAlign | ColMajor> A(ROW_SIZE, K_SIZE), B(K_SIZE, COL_SIZE), C(ROW_SIZE, COL_SIZE), dC(ROW_SIZE, COL_SIZE);
    DATA_TYPE *da, *db, *dc;

    A.setRandom();
    B.setRandom();

    cudaMalloc(&da, sizeof(DATA_TYPE)*MAT_SIZE_A);
    cudaMalloc(&db, sizeof(DATA_TYPE)*MAT_SIZE_B);
    cudaMalloc(&dc, sizeof(DATA_TYPE)*MAT_SIZE_C);

    {
        auto _ = prof.profile("MM on Host");

        C = A * B;
    }

    {
        auto _ = prof.profile("memcpy host -> device");

        cudaMemcpy(da, A.data(), sizeof(DATA_TYPE)*MAT_SIZE_A, cudaMemcpyHostToDevice);
        cudaMemcpy(db, B.data(), sizeof(DATA_TYPE)*MAT_SIZE_B, cudaMemcpyHostToDevice);
    }

    dim3 blockdim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridddim(ceil((float)ROW_SIZE / BLOCK_SIZE), ceil((float)COL_SIZE / BLOCK_SIZE));

    {
        auto _ = prof.profile("MM with kernel without shared memory");

        matMul_kernel<<<1,blockdim>>>(da,db,dc);
        cudaDeviceSynchronize();
    }

    {
        auto _ = prof.profile("MM with kernel with shared memory");

        MatMul_SharedMem<<<1,blockdim>>>(da,db,dc,ROW_SIZE,COL_SIZE,K_SIZE);
        cudaDeviceSynchronize();
    }

    {
        auto _ = prof.profile("MM with kernel with shared memory avoiding bank conflict");

        MatMul_SharedMem_NoBankconflict<<<1,blockdim>>>(da,db,dc,ROW_SIZE,COL_SIZE,K_SIZE);
        cudaDeviceSynchronize();
    }

    {
        auto _ = prof.profile("memcpy device -> host");

        cudaMemcpy(dC.data(), dc, sizeof(DATA_TYPE)*MAT_SIZE_C, cudaMemcpyDeviceToHost);
    }

    prof.report();

    return 0;
}