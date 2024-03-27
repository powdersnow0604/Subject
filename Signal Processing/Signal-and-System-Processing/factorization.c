#include "factorization.h"
#include <math.h>

int minFactor(int num)
{
	int k = 2;
	double sqrtnum = sqrt(num);
	while (k <= sqrtnum && num != 1)
	{
		if (num % k == 0)
		{
			return k;
		}
		k++;
	}
	return num;
}

void primeFactorization(int num)
{
	int k = 2;
	double sqrtNum = sqrt(num);
	while (k <= sqrtNum && num != 1)
	{
		if (num % k == 0)
		{
			/*do something*/
			num /= k;
		}
		else
			k++;
	}

	if (num > 1)
	{
		/*last factor*/
	}
}
