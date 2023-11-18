#include "greedy.h"
#include <stdlib.h>
#include "algorithm.h"
#include <stdio.h>


static int knapsack_helper(const void* arg1, const void* arg2)
{
	return *((double*)arg1) < *((double*)arg2);
}


void knapsack_greedy(int* p, int* w, int M, double* x, int n)
{
	typedef struct pw_ {
		double efficiency;
		int job_id;
	}pw;

	int i, cu = M;

	pw* jobs = (pw*)malloc(sizeof(pw) * n);

	if (jobs == NULL) return;

	for (i = 0; i < n; ++i) {
		x[i] = 0.;
		jobs[i].job_id = i;
		jobs[i].efficiency = (double)p[i] / w[i];
	}

	Qsort(jobs, 0, (size_t)n - 1, sizeof(pw), knapsack_helper);

	for (i = 0; i < n; ++i) {
		if (w[jobs[i].job_id] > cu) break;
		x[jobs[i].job_id] = 1.;
		cu -= w[jobs[i].job_id];
	}

	if (i < n) x[jobs[i].job_id] = (double)cu / w[jobs[i].job_id];

	free(jobs);
}


void JobSchedule_greedy(int* D, int* J, int n)
{
	D[0] = J[0] = 0;
	int k = 1, r, l; J[1] = 1;
	for (int i = 2; i <= n; ++i) {
		r = k;
		while (D[J[r]] > D[i] && D[J[r]] != r) --r;

		if (D[J[r]] <= D[i] && D[i] > r) {
			for (l = k; l >= r + 1; --l)
				J[l + 1] = J[l];
			J[r + 1] = i; 
			++k;
		}
	}
}


int optimalMergePattern(int n, int* L)
{
	typedef struct node_ {
		struct node_* left;
		struct node_* right;
		int weight;
	}node;

	//for()

	int sum = 0;
	return 0;
}

