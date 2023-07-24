#include "mymath.h"

static int circularShift(unsigned int x, int num, char direction, int log2x) {
	log2x++;
	switch (direction)
	{
	case 0:
	{
		x = ((x << num) | (x >> (log2x - num))) & (unsigned int)((1ll << log2x) - 1);
		return (int)x;
	}
	case 1:
	{
		x = ((x >> num) | (x << (log2x - num))) & (unsigned int)((1ll << log2x) - 1);
		return (int)x;
	}
	default:
		return 0;
	}
}

int LOG2N(int Num)
{
	int log2n = 0;
	while (Num >= 2) {
		Num >>= 1;
		log2n++;
	}
	return log2n;
}
