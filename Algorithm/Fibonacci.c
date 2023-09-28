#include "Fibonacci.h"
#include <stdlib.h>

static inline void matrixMul(long long**, const long long*, long long**);

long long k_n(long long k, long long n)
{
	long long kn = 1;
	if (n & 1) kn *= k;

	for (int i = 2; i <= n; i *= 2) {
		k *= k;
		if (i & n) kn *= k;
	}

	return kn;
}

long long fibonacci(int n)
{
	int i;
	long long f[4] = { 0,1,1,1 };
	long long* temp = (long long*)malloc(4 * sizeof(long long));
	long long* x = (long long*)malloc(4 * sizeof(long long));
	x[0] = 1; x[1] = 0; x[2] = 0; x[3] = 1;

	if ((n - 2) & 1) matrixMul(&x, f, &temp);

	for (i = 2; i <= n - 2; i *= 2) {
		//start
		f[2] = f[1] * (f[0] + f[3]);
		f[1] *= f[1];
		f[0] = f[0] * f[0] + f[1];
		f[3] = f[3] * f[3] + f[1];
		f[1] = f[2];

		if (i & (n - 2)) matrixMul(&x, f, &temp);
	}

	long long res = x[2] + x[3];

	free(temp); free(x);

	return res;
}

static void matrixMul(long long** opr1, const long long* opr2, long long** temp)
{
	(*temp)[0] = (*opr1)[0] * opr2[0] + (*opr1)[1] * opr2[2];
	(*temp)[1] = (*opr1)[0] * opr2[1] + (*opr1)[1] * opr2[3];
	(*temp)[2] = (*temp)[1];
	(*temp)[3] = (*opr1)[2] * opr2[1] + (*opr1)[3] * opr2[3];

	long long* t = *temp;
	*temp = *opr1;
	*opr1 = t;
}