#include "dynamic_programming.h"
#include <stdlib.h>

typedef struct ks_struct_ {
	int weight;
	int profit;
	int index;
}ks_struct;

static int ks_compare(const void* arg1, const void* arg2) {
	int w1 = ((const ks_struct*)arg1)->weight;
	int w2 = ((const ks_struct*)arg2)->weight;

	if (w1 < w2) return -1;
	if (w1 > w2) return 1;
	return 0;
}

static ks_helper(int i, int n, int capacity, ks_struct* wp, int sum_w, int sum_p, int* res) 
{
	if (n == i) {
		if (capacity >= wp[i].weight) return wp[i].profit;
		else return 0;
	}

	if (capacity < wp[i].weight) return 0;

	if (capacity > sum_w) {
		for (int j = i; j <= n; ++j) {
			res[wp[j].index] = 1;
		}
		return sum_p;
	}

	int* res2 = (int*)malloc(sizeof(int) * 6);
	if (res2 == NULL) return -100000;

	for (int j = 0; j < 6; ++j) {
		res2[j] = res[j];
	}

	int select = ks_helper(i + 1, n, capacity - wp[i].weight, wp, sum_w - wp[i].weight, sum_p - wp[i].profit, res) + wp[i].profit;
	int unselect = ks_helper(i + 1, n, capacity, wp, sum_w - wp[i].weight, sum_p - wp[i].profit, res2);

	if (select > unselect) {
		res[wp[i].index] = 1;
		free(res2);
		return select;
	}
	else {
		for (int j = 0; j < 6; ++j) {
			res[j] = res2[j];
		}
		free(res2);
		return unselect;
	}
}

int binary_knapsack(int i, int n, int capacity, int* w, int* p, int* res)
{
	ks_struct* wp = (ks_struct*)malloc(sizeof(ks_struct) * (n - i + 1));
	if (wp == NULL) return -1;
	int* res2 = (int*)malloc(sizeof(int) * 6);
	if (res2 == NULL) return -1;

	for (int j = 0; j < n - i + 1; ++j) {
		wp[j].weight = w[i + j];
		wp[j].profit = p[i + j];
		wp[j].index = j;
		res[j] = 0;
	}
	
	qsort(wp, (size_t)(n - i + 1), sizeof(ks_struct), ks_compare);

	int sum_p = 0, sum_w = 0;
	for(int j = 0; j < n - i + 1; ++j) {
		sum_p += wp[j].profit;
		sum_w += wp[j].weight;
	}

	int result = ks_helper(i, n, capacity, wp, sum_w, sum_p, res);

	free(wp);

	return result;
}