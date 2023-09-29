#include "Fibonacci.h"


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
	long long f[3] = { 0,1,1 };
	long long x[3] = { 1,0,1 };
	long long temp;

	if ((n - 2) & 1)
	{
		temp = x[0] * f[1] + x[1] * f[2];
		x[1] = x[1] * f[1];
		x[0] = x[0] * f[0] + x[1];
		x[2] = x[2] * f[2] + x[1];
		x[1] = temp;
	}

	for (i = 2; i <= n - 2; i *= 2) {
		//start
		temp = f[1] * (f[0] + f[2]);
		f[1] *= f[1];
		f[0] = f[0] * f[0] + f[1];
		f[2] = f[2] * f[2] + f[1];
		f[1] = temp;

		if (i & (n - 2)) {
			temp = x[0] * f[1] + x[1] * f[2];
			x[1] = x[1] * f[1];
			x[0] = x[0] * f[0] + x[1];
			x[2] = x[2] * f[2] + x[1];
			x[1] = temp;
		}
	}

	return x[1] + x[2];
}
