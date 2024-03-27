#include "Fibonacci.h"

typedef unsigned long long size_t;

size_t k_n(size_t k, size_t n)
{
	int i;
	size_t kn = 1;
	if (n & 1) kn *= k;

	for (i = 2; i <= n; i <<= 1) {
		k *= k;
		if (i & n) kn *= k;
	}

	return kn;
}

long long fibonacci(int n)
{
	switch (n)
	{
	case 0: return 0;
	case 1:
	case 2: return 1;
	}

	int i;
	size_t f[3] = { 0,1,1 };
	size_t x[3];
	size_t temp;

	if ((n - 3) & 1) { x[0] = 0; x[1] = 1; x[2] = 1; }
	else { x[0] = 1; x[1] = 0; x[2] = 1; }
	

	for (i = 2; i <= n - 3; i <<= 1) {
		//start
		temp = f[1] * (f[0] + f[2]);
		f[1] *= f[1];
		f[0] = f[0] * f[0] + f[1];
		f[2] = f[2] * f[2] + f[1];
		f[1] = temp;

		if (i & (n - 3)) {
			temp = x[0] * f[1] + x[1] * f[2];
			x[1] = x[1] * f[1];
			x[0] = x[0] * f[0] + x[1];
			x[2] = x[2] * f[2] + x[1];
			x[1] = temp;
		}
	}

	return x[1] + x[2] + x[2];
}
