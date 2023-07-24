#include "DFT.h"
#include <malloc.h>
#include <errno.h>
#include <stdio.h>

/*------------------선언부-------------------*/

static void OnShuffle(complex_num* X, int N, int log2N);
static void OnShuffle_4(complex_num* X, int N, int log2N);
static void OnShuffle_4_old(complex_num* X, int N, int log2N);
static int OnReverseBitOrder(int index, int LOG2N);
static void relocate(complex_num* X, int N, int radix, complex_num* temp);
static int OnReverseBitOrder4(int x, int log2N);
static void OnButterfly_radix_2(complex_num* P, int n,int mode);
static void OnButterfly_radix_4(complex_num* P, int n, int mode);
#pragma region OnButterfly_radix_k
static void OnButterfly_radix_k(complex_num* P, int n, int radix, int mode);
static void OnButterfly_radix_k_s(complex_num* P, int n, int radix, complex_num* temp, int mode);
#pragma endregion
#pragma region OnButterfly_common_factor
static void OnButterfly_common_factor(complex_num* P, int N, int mode);
static void OnButterfly_common_factor_s(complex_num* P, int N, complex_num* temp, int mode);
#pragma endregion

/*------------------정의부-------------------*/


/*			[suffle]		*/
static void OnShuffle(complex_num* X, int N, int log2N)
{
	int i;
	complex_num* temp = (complex_num*)malloc(sizeof(complex_num) * N);

	if (temp == NULL) return;

	for (i = 0; i < N; i++) {
		temp[i].Re = X[OnReverseBitOrder(i, log2N)].Re;
		temp[i].Im = X[OnReverseBitOrder(i, log2N)].Im;
	}

	for (i = 0; i < N; i++) {
		X[i].Re = temp[i].Re;
		X[i].Im = temp[i].Im;
	}

	free(temp);
}

static void OnShuffle_4(complex_num* X, int N, int log2N)
{
	int i;
	complex_num* temp = (complex_num*)malloc(sizeof(complex_num) * N);

	if (temp == NULL) return;

	for (i = 0; i < N; i++) {
		temp[i].Re = X[OnReverseBitOrder4(i, log2N)].Re;
		temp[i].Im = X[OnReverseBitOrder4(i, log2N)].Im;
	}

	for (i = 0; i < N; i++) {
		X[i].Re = temp[i].Re;
		X[i].Im = temp[i].Im;
	}

	free(temp);
}

static void OnShuffle_4_old(complex_num* X, int N, int log2N)
{
	int i, j;
	complex_num* temp = (complex_num*)malloc(sizeof(complex_num) * N);

	if (temp == NULL) return;
	
	for (i = 0; i < N; i++) {
		temp[i].Re = X[OnReverseBitOrder(i, log2N)].Re;
		temp[i].Im = X[OnReverseBitOrder(i, log2N)].Im;
	}

	for (i = 0; i < N; i++) {
		X[i].Re = temp[i].Re;
		X[i].Im = temp[i].Im;
	}
	
	for (i = N; i > 1; i /= 4)
	{
		memcpy(temp, X, N * sizeof(complex_num));
		for (j = 0; j < N; j += i)
		{
			memcpy(X + j, temp + j, i / 4 * sizeof(complex_num));
			memcpy(X + j + i / 4, temp + j + i / 4 * 2, i / 4 * sizeof(complex_num));
			memcpy(X + j + i / 4 * 2, temp + j + i / 4, i / 4 * sizeof(complex_num));
			memcpy(X + j + i / 4 * 3, temp + j + i / 4 * 3, i / 4 * sizeof(complex_num));

		}
	}
	free(temp);
}

static int OnReverseBitOrder(int index, int LOG2N)
{
	int i, X, Y = 0;

	for (i = 0; i < LOG2N; i++) {
		X = (index & (1 << i)) >> i;
		Y = (Y << 1) | X;
	}
	return Y;
}

static int OnReverseBitOrder4(int index, int log2N)
{
	int i, X, Y = 0;

	for (i = 0; i < log2N; i+=2) {
		X = (index & (3 << i)) >> i;
		Y = (Y << 2) | X;
	}
	return Y;
}

static void relocate(complex_num* X, int N, int radix, complex_num* temp)
{
	int i, j, pos, N_radix = N / radix;
	
	for (i = 0; i < N; i += radix) {
		pos = i / radix;
		for (j = 0; j < radix; j++) {
			temp[pos + N_radix * j].Re = X[i + j].Re;
			temp[pos + N_radix * j].Im = X[i + j].Im;
		}
	}

	for (i = 0; i < N; i++) {
		X[i].Re = temp[i].Re;
		X[i].Im = temp[i].Im;
	}

}
//////////////////////////////


/*			[usable]		*/
void DFT_1d(complex_num* X)
{
	int i, k, n,N;
	N = (int)(_msize(X) / sizeof(complex_num));
	
	complex_num* temp = (complex_num*)calloc(N,sizeof(complex_num));

	for (k = 0; k < N; k++)
	{
		for (n = 0; n < N; n++)
		{
			temp[k] = complexAdd(temp[k], complexMul(X[n], twiddle_factor(N, k * n % N)));
		}
	}

	for (i = 0; i < N; i++) {
		X[i].Re = temp[i].Re;
		X[i].Im = temp[i].Im;
	}

	free(temp);
}

double* calcAmplitude(complex_num* X)
{
	//동적 할당된 메모리 반환
	int i;
	int N = (int)(_msize(X)/sizeof(complex_num));

	double* temp = (double*)malloc(sizeof(double) * N);

	for (i = 0; i < N; i++) {
		temp[i] = complexAbs(X[i]);
	}

	return temp;
}

#pragma region FFT_radix_2
void FFT_radix_2(complex_num* P)
{
	int N;

	N = (int)(_msize(P) / sizeof(complex_num));

	OnShuffle(P, N, (int)log2(N));

	if (errno == ENOMEM) return;

	OnButterfly_radix_2(P, N, 0);
}

void IFFT_radix_2(complex_num* P)
{
	int i, N;

	N = (int)(_msize(P) / sizeof(complex_num));

	OnShuffle(P, N, (int)log2(N));

	if (errno == ENOMEM) return;

	OnButterfly_radix_2(P, N, 1);

	for (i = 0; i < N; i++) {
		P[i].Re /= N;
		P[i].Im /= N;
	}
}
#pragma endregion

#pragma region FFT_radix_4
void FFT_radix_4(complex_num* P)
{
	int N;

	N = (int)(_msize(P) / sizeof(complex_num));

	OnShuffle_4(P, N, (int)log2(N));
	
	if (errno == ENOMEM) return;

	OnButterfly_radix_4(P, N, 0);
}

void IFFT_radix_4(complex_num* P)
{
	int i,N;

	N = (int)(_msize(P) / sizeof(complex_num));

	OnShuffle_4(P, N, (int)log2(N));

	if (errno == ENOMEM) return;

	OnButterfly_radix_4(P, N, 1);

	for (i = 0; i < N; i++) {
		P[i].Re /= N;
		P[i].Im /= N;
	}
}
#pragma endregion

#pragma region FFT_radix_k
void FFT_radix_k(complex_num* P)
{
	int N, radix;

	N = (int)(_msize(P) / sizeof(complex_num));

	radix = minFactor(N);

	OnButterfly_radix_k(P, N, radix, 0);
}

void FFT_radix_k_s(complex_num* P)
{
	int N, radix;

	N = (int)(_msize(P) / sizeof(complex_num));

	radix = minFactor(N);

	complex_num* temp = (complex_num*)malloc(sizeof(complex_num) * N);

	if (temp == NULL) return;

	OnButterfly_radix_k_s(P, N, radix, temp, 0);

	free(temp);
}

void IFFT_radix_k(complex_num* P)
{
	int i, N, radix;

	N = (int)(_msize(P) / sizeof(complex_num));

	radix = minFactor(N);

	OnButterfly_radix_k(P, N, radix, 1);

	for (i = 0; i < N; i++) {
		P[i].Re /= N;
		P[i].Im /= N;
	}
}

void IFFT_radix_k_s(complex_num* P)
{
	int i, N, radix;

	N = (int)(_msize(P) / sizeof(complex_num));

	radix = minFactor(N);

	complex_num* temp = (complex_num*)malloc(sizeof(complex_num) * N);

	if (temp == NULL) return;

	OnButterfly_radix_k_s(P, N, radix, temp, 1);

	free(temp);

	for (i = 0; i < N; i++) {
		P[i].Re /= N;
		P[i].Im /= N;
	}
}
#pragma endregion

#pragma region FFT_common_factor
void FFT_common_factor(complex_num* P)
{
	int N;

	N = (int)(_msize(P) / sizeof(complex_num));

	OnButterfly_common_factor(P, N, 0);
}

void FFT_common_factor_s(complex_num* P)
{
	int N;

	N = (int)(_msize(P) / sizeof(complex_num));

	complex_num* temp = (complex_num*)malloc(sizeof(complex_num) * N);

	if (temp == NULL) return;

	OnButterfly_common_factor_s(P, N, temp, 0);

	free(temp);
}

void IFFT_common_factor(complex_num* P)
{
	int i,N;

	N = (int)(_msize(P) / sizeof(complex_num));

	OnButterfly_common_factor(P, N, 1);

	for (i = 0; i < N; i++) {
		P[i].Re /= N;
		P[i].Im /= N;
	}
}

void IFFT_common_factor_s(complex_num* P)
{
	int i, N;

	N = (int)(_msize(P) / sizeof(complex_num));

	complex_num* temp = (complex_num*)malloc(sizeof(complex_num) * N);

	if (temp == NULL) return;

	OnButterfly_common_factor_s(P, N, temp, 1);

	free(temp);

	for (i = 0; i < N; i++) {
		P[i].Re /= N;
		P[i].Im /= N;
	}
}
#pragma endregion
///////////////////////////////


/*			[butterfly]			*/
static void OnButterfly_radix_2(complex_num* P, int n, int mode)
{
	if (n == 1)
		return;
	int i, s = 1-2*mode;
	complex_num temp;

	OnButterfly_radix_2(P, n / 2,mode);
	OnButterfly_radix_2(P + n / 2, n / 2,mode);

	for (i = 0; i < n / 2; i++)
	{
		temp = complexMul(twiddle_factor(n, s*i), (P + n / 2)[i]);

		P[i+n/2] = complexSub(P[i], temp);
		P[i] = complexAdd(P[i], temp);
	}
}

static void OnButterfly_radix_4(complex_num* P, int n, int mode)
{
	if (n == 1)
		return;
	int i, s = 1 - 2 * mode;
	complex_num temp1, temp2, temp3;

	for (i = 0; i < 4; i++)
	{
		OnButterfly_radix_4(P + n / 4 * i, n / 4 , mode);
	}

	for (i = 0; i < n / 4; i++)
	{
		temp2 = complexMul(twiddle_factor(n, s * ((2 * i) % n)), (P + n / 4 * 2)[i]);
		temp1 = complexMul(twiddle_factor(n, s * (i % n)), (P + n / 4)[i]);
		temp3 = complexMul(twiddle_factor(n, s * ((3 * i) % n)), (P + n / 4 * 3)[i]);

		P[i + n / 4 * 3] = complexAdd(complexSub(P[i], temp2), complexMul(I, complexSub(temp1, temp3)));
		P[i + n / 4 * 1] = complexAdd(complexSub(P[i], temp2), complexMul(I, complexSub(temp3, temp1)));
		P[i + n / 4 * 2] = complexSub(complexAdd(P[i], temp2), complexAdd(temp1, temp3));
		P[i] = complexAdd(complexAdd(P[i], temp2), complexAdd(temp1, temp3));
	}
}

#pragma region OnButterfly_radix_k
static void OnButterfly_radix_k(complex_num* P, int n, int radix, int mode)
{
	//setting
	if (n == 1)
		return;

	int i,j,k,s=1-2*mode;
	complex_num* temp = (complex_num*)malloc(sizeof(complex_num) * n);

	//relocate
	if(n != radix)
		relocate(P, n, radix,temp);
	free(temp);

	//decompose
	for (i = 0; i < radix; i++) {
		OnButterfly_radix_k(P + i * n / radix, n / radix, radix,mode);
	}

	temp = (complex_num*)malloc(sizeof(complex_num) * (n+radix-1));

	//conquer
	for (i = 0; i < n / radix; i++)
	{
		for (j = 0; j < radix-1; j++) {
			temp[j] = P[i + n / radix * (j+1)];
		}

		for (j = radix-1; j >= 0; j--)
		{
			P[i + n / radix * j] = P[i];
			for (k = 1; k < radix; k++) {
				P[i + n / radix * j] = complexAdd(P[i + n / radix * j], complexMul(temp[k-1], twiddle_factor(n, s * ((i + n / radix * j) * k) % n)));
			}
		}
	}
	free(temp);
}

static void OnButterfly_radix_k_s(complex_num* P, int n, int radix, complex_num* temp, int mode)
{
	//setting
	if (n == 1)
		return;

	int i, j, k, s=1-2*mode;

	//relocate
	if (n != radix)
		relocate(P, n, radix, temp);

	//decompose
	for (i = 0; i < radix; i++) {
		OnButterfly_radix_k_s(P + i * n / radix, n / radix, radix, temp, mode);
	}

	//conquer
	for (i = 0; i < n / radix; i++)
	{
		for (j = 0; j < radix - 1; j++) {
			temp[j] = P[i + n / radix * (j + 1)];
		}

		for (j = radix - 1; j >= 0; j--)
		{
			P[i + n / radix * j] = P[i];
			for (k = 1; k < radix; k++) {
				P[i + n / radix * j] = complexAdd(P[i + n / radix * j], complexMul(temp[k - 1], twiddle_factor(n, s * ((i + n / radix * j) * k) % n)));
			}
		}
	}
}
#pragma endregion

#pragma region OnButterfly_common_factor
static void OnButterfly_common_factor(complex_num* P, int N, int mode)
{
	//setting
	if (N == 1)
		return;

	int i,j,k,radix = minFactor(N), s = 1 - 2*mode;

	complex_num* temp = (complex_num*)malloc(sizeof(complex_num) * N);

	if(N != radix)
		relocate(P, N, radix, temp);

	free(temp);

	//decompose
	for (i = 0; i < radix; i++)
	{
		OnButterfly_common_factor(P + N / radix * i, N / radix, mode);
	}

	temp = (complex_num*)malloc(sizeof(complex_num) * (N + radix - 1));

	//conquer
	for (i = 0; i < N / radix; i++)
	{
		for (j = 0; j < radix-1; j++) {
			temp[j] = P[i + N / radix * (j + 1)];
		}

		for (j = radix - 1; j >= 0; j--)
		{
			
			P[i + N / radix * j] = P[i];
			for (k = 1; k < radix; k++) {
				P[i + N / radix * j] = complexAdd(P[i + N / radix * j], complexMul(temp[k - 1], twiddle_factor(N, s*((i + N / radix * j) * k) % N)));
			}
		}
	}
	free(temp); 
}

static void OnButterfly_common_factor_s(complex_num* P, int N, complex_num* temp, int mode)
{
	//setting
	if (N == 1)
		return;

	int i, j, k, radix = minFactor(N), s = 1 - 2*mode;

	if (N != radix)
		relocate(P, N, radix, temp);

	//decompose
	for (i = 0; i < radix; i++)
	{
		OnButterfly_common_factor_s(P + N / radix * i, N / radix, temp, mode);
	}

	//conquer
	for (i = 0; i < N / radix; i++)
	{
		for (j = 0; j < radix - 1; j++) {
			temp[j] = P[i + N / radix * (j + 1)];
		}

		for (j = radix - 1; j >= 0; j--)
		{

			P[i + N / radix * j] = P[i];
			for (k = 1; k < radix; k++) {
				P[i + N / radix * j] = complexAdd(P[i + N / radix * j], complexMul(temp[k - 1],twiddle_factor(N,s*((i + N / radix * j) * k) % N)));
			}
		}
	}
}
#pragma endregion
///////////////////////////////////