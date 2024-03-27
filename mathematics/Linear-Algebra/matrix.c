#include "matrix.h"
#include <stdio.h>
#include <string.h>

/*			Declaration			*/	
static void swap(double* a, double* b, size_t num);
static void div1row(double* matrix, size_t column, int div);
static void add2row(double* matrix, double* addMat, int start, int end);
static double** rref_sol(double** mat, pivot* piv);
static void delPivot(pivot* pivot);


/*			Definition			*/	
double** mkMatrix(size_t row, size_t column)
{
	double** matrix;
	matrix = (double**)malloc(sizeof(double*) * row);
	matrix[0] = (double*)calloc(column * row, sizeof(double));
	for (int i = 1; i < row; i++)
	{
		matrix[i] = matrix[i - 1] + column;
	}
	return matrix;
}

void delMatrix(double** matrix)
{
	free(matrix[0]);
	free(matrix);
}

void delPivot(pivot* pivot)
{
	free(pivot->row);
	free(pivot->column);
}

void printMatrix(double** matrix)
{
	size_t row = _msize(matrix) / sizeof(double*);
	size_t column = _msize(matrix[0]) / sizeof(double) / row;

	puts("");
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column; j++)
		{
			printf("%lf ", matrix[i][j]);
		}
		puts("");
	}
}

double** rref_sol(double** mat, pivot* piv)
{
	size_t row = _msize(mat) / sizeof(double*);
	size_t column = _msize(mat[0]) / sizeof(double) / row - 1;
	pivot pivot;
	pivot.rank = 0;
	pivot.row = (int*)malloc(sizeof(int) * column);
	pivot.column = (int*)malloc(sizeof(int) * column);

	//duplicate
	double** matrix = mkMatrix(row, column + 1);
	memcpy(matrix[0], mat[0], _msize(mat[0]));

	// ~gaussion elimination
	for (int i = 0; i < row; i++)
	{

		// finding pivot		
		int j;
		for (j = pivot.rank == 0 ? 0 : pivot.column[pivot.rank - 1] + 1; j < column; j++)
		{
			if (matrix[i][j] != 0)
			{
				div1row(matrix[i], column, j);
				pivot.row[pivot.rank] = i;
				pivot.column[pivot.rank] = j;
				pivot.rank += 1;
				break;
			}
			else
			{
				for (int k = i + 1; k < row; k++)
				{
					if (matrix[k][j] != 0)
					{
						swap(matrix[i], matrix[k], column + 1);
						j--;
						break;
					}
				}
			}
		}
		if (j == column)
		{
			for (int k = i; k < row; k++)
			{
				if (matrix[k][column] != 0)
				{
					pivot.rank = -1;
					if (piv)
					{
						piv->rank = pivot.rank;
					}
					delPivot(&pivot);
					delMatrix(matrix);
					return mat;
				}
			}
			break;
		}
		

		// gaussion elemination
		for (int j = i + 1; j < row; j++)
		{
			double temp = matrix[j][pivot.column[pivot.rank - 1]];
			for (int k = pivot.column[pivot.rank - 1]; k <= column; k++)
			{
				matrix[j][k] -= matrix[i][k] * temp;
			}
		}

	}

	// jordan elimination
	for (int i = 0; i < pivot.rank; i++)
	{
		for (int j = pivot.row[i] - 1; j >= 0; j--)
		{
			double temp = matrix[j][pivot.column[i]];
			for (int k = pivot.column[i]; k <= column; k++)
			{
				matrix[j][k] -= matrix[pivot.row[i]][k] * temp;
			}
		}
	}

	if (piv)
	{
		piv->rank = pivot.rank;
		piv->row = pivot.row;
		piv->column = pivot.column;
	}
	else
	{
		delPivot(&pivot);
	}
	return matrix;
}

void div1row(double* matrix, size_t column, int div)
{
	double temp = matrix[div];
	for (int i = div; i <= column; i++)
	{
		matrix[i] = matrix[i] / temp;
	}

}

void swap(double* a, double* b, size_t num)
{
	double* temp = (double*)malloc(sizeof(double) * num);
	memcpy(temp, a, sizeof(double) * num);
	memcpy(a, b, sizeof(double) * num);
	memcpy(b, temp, sizeof(double) * num);
	free(temp);
}

double** rref(double** matrix, int *check)
{
	pivot pivot;
	double** rref_matrix = rref_sol(matrix, &pivot);
	size_t row = _msize(rref_matrix) / sizeof(double*);
	size_t column = _msize(rref_matrix[0]) / sizeof(double) / row;

	if (pivot.rank == column - 1)
	{
		/*
		puts("unique solution\n");
		for (int i = 0; i < row; i++)
		{
			printf("   %lf\n", rref_matrix[i][column - 1]);
		}
		*/

		if (check != NULL)
			*check = 1;

		delPivot(&pivot);
		return rref_matrix;
		//delMatrix(rref_matrix);
	}
	else if (pivot.rank == -1)
	{
		if (check != NULL)
			*check = -1;

		return leastSquareMethod(matrix);
	}
	else
	{
		/*
		puts("infinite solution\n");
		int ppos = 0;
		for (int i = 0; i < column - 1; i++)
		{
			if (i == pivot.column[ppos])
			{
				ppos++;
				continue;
			}
			printf("x%d * [ ", i + 1);
			for (int j = 0; j < column - 1; j++)
			{


				if (j >= pivot.rank)
				{
					for (int k = j; k < column - 1; k++)
					{
						if (k == i)
							printf("%lf ", (double)1);
						else
							printf("%lf ", (double)0);
					}
					break;
				}
				else
				{
					if (j < row)
						printf("%lf ", -1 * rref_matrix[j][i]);
					else
						printf("%lf ", (double)0);
				}

			}
			puts("]T");
		}
		printf("const [ ");
		for (int j = 0; j < column - 1; j++)
		{
			if (j < row)
				printf("%lf ", rref_matrix[j][column - 1]);
			else
				printf("%lf ", (double)0);
		}
		puts("]T");
		*/

		if (check != NULL)
			*check = 0;

		delPivot(&pivot);
		return rref_matrix;
		//delMatrix(rref_matrix);
	}
}

double** matrixMul(double** a, double** b)
{
	size_t lm = _msize(a) / sizeof(double*);
	size_t ln = _msize(a[0]) / sizeof(double) / lm;

	size_t rm = _msize(b) / sizeof(double*);
	size_t rn = _msize(b[0]) / sizeof(double) / rm;

	if (ln != rm) return a;
	double** matrix = mkMatrix(lm, rn);
	for (int i = 0; i < lm; i++)
	{
		for (int j = 0; j < rn; j++)
		{
			matrix[i][j] = 0;
			for (int k = 0; k < ln; k++)
			{
				matrix[i][j] += a[i][k] * b[k][j];
			}
		}
	}

	return matrix;
}

double** matrixAdd(double** a, double** b)
{
	size_t lm = _msize(a) / sizeof(double*);
	size_t ln = _msize(a[0]) / sizeof(double) / lm;

	size_t rm = _msize(b) / sizeof(double*);
	size_t rn = _msize(b[0]) / sizeof(double) / rm;

	if (ln != rm) return a;
	double** matrix = mkMatrix(lm, rn);
	for (int i = 0; i < lm; i++)
	{
		for (int j = 0; j < rn; j++)
		{
			matrix[i][j] = a[i][j] + b[i][j];
		}
	}

	return matrix;
}

double** matrixScalarMul(double** matrix, double Scalar)
{
	int i, j;
	int col = (int)(_msize(matrix) / sizeof(double*));
	int row = (int)(_msize(matrix[0]) / sizeof(double)) / col;

	double** res = mkMatrix(row, col);

	for (i = 0; i < col; i++) {
		for (j = 0; j < row; j++) {
			res[i][j] = matrix[i][j] * Scalar;
		}
	}

	return res;
}

double** transpose(double** matrix)
{
	size_t row = _msize(matrix) / sizeof(double*);
	size_t column = _msize(matrix[0]) / sizeof(double) / row;

	double** trans = mkMatrix(column, row);

	for (int i = 0; i < column; i++)
	{
		for (int j = 0; j < row; j++)
		{
			trans[i][j] = matrix[j][i];
		}
	}

	return trans;
}

double** augment(double** A, double** c)
{
	size_t row = _msize(A) / sizeof(double*);
	size_t column = _msize(A[0]) / sizeof(double) / row;

	double** aug = mkMatrix(row, column + 1);

	for (int i = 0; i < row; i++)
	{
		memcpy(aug[i], A[i], column * sizeof(double));
		aug[i][column] = c[i][0];
	}

	return aug;
}

double** decomposeAugment(double** matrix, double*** c)
{
	size_t row = _msize(matrix) / sizeof(double*);
	size_t column = _msize(matrix[0]) / sizeof(double) / row - 1;

	double** A = mkMatrix(row, column);
	*c = mkMatrix(row, 1);

	for (int i = 0; i < row; i++)
	{
		memcpy(A[i], matrix[i], column * sizeof(double));
		(*c)[i][0] = matrix[i][column];
	}

	return A;

}

double** leastSquareMethod(double** matrix)
{
	double** c;
	double** A = decomposeAugment(matrix, &c);
	double** AT = transpose(A);

	double** ATA = matrixMul(AT, A);
	double** ATc = matrixMul(AT, c);

	delMatrix(c);
	delMatrix(A);
	delMatrix(AT);

	double** aug = augment(ATA, ATc);

	delMatrix(ATA);
	delMatrix(ATc);

	puts("LeastSquareMethod");
	double** res =  rref(aug, NULL);

	delMatrix(aug);

	return res;
}

double determinant(double** mat)
{
	int i, j, k;
	int row = (int)(_msize(mat) / sizeof(double*));
	int col = (int)(_msize(mat[0]) / sizeof(double) / row);
	double temp = 0, **matrix;

	if (row != col)
		return 0;

	matrix = mkMatrix(row, col);
	memcpy(matrix[0], mat[0], sizeof(double) * row * col);
	
	for (i = 0; i < row; i++) {
		if (matrix[i][i] == 0) {
			for (j = i + 1; j < row; j++) {
				if (matrix[j][i] != 0) {
					add2row(matrix[i], matrix[j], col, i);
					goto gaussian_elimainattion;
				}
			}
			return 0;
		}

	gaussian_elimainattion:
		temp = 1;
		
		for (j = i + 1; j < row; j++) {
			temp = matrix[j][i] / matrix[i][i];
			for (k = i; k < col; k++)
				matrix[j][k] -= temp * matrix[i][k];
		}
		
	}
	
	temp = 1;
	for (i = 0; i < row; i++) {
		temp *= matrix[i][i];
	}
	
	delMatrix(matrix);

	return temp;
}

void add2row(double* matrix, double* addMat, int end, int start)
{
	int i;

	for (i = start; i < end; i++) {
		matrix[i] += addMat[i];
	}
}

double** inverse(double** mat)
{
	int i, j, k;
	int row = (int)(_msize(mat) / sizeof(double*));
	int col = (int)(_msize(mat[0]) / sizeof(double) / row);
	double det, **matrix, **invMat, temp;

	if ((det = determinant(mat)) == 0) {
		return mat;
	}

	matrix = mkMatrix(row, col);
	invMat = mkMatrix(row, col);

	memcpy(matrix[0], mat[0], sizeof(double) * row * col);

	for (i = 0; i < row; i++)
		invMat[i][i] = 1;

	for (i = 0; i < row; i++) {
		if (matrix[i][i] == 0) {
			for (j = i + 1; j < row; j++) {
				if (matrix[j][i] != 0) {
					add2row(matrix[i], matrix[j], col, i);
					add2row(invMat[i], invMat[j], col, 0);
					goto gaussian_elimainattion;
				}
			}
		}

	gaussian_elimainattion:
		temp = matrix[i][i];
		div1row(matrix[i], (size_t)col-1, i);
		for (j = 0; j < col; j++)
			invMat[i][j] /= temp;

		for (j = i + 1; j < row; j++) {
			temp = matrix[j][i];
			for (k = i; k < col; k++)
				matrix[j][k] -= temp * matrix[i][k];
			for (k = 0; k < col; k++)
				invMat[j][k] -= temp * invMat[i][k];
		}
	}
	
	for (i = row - 1; i > 0; i--) {
		for (j = i - 1; j >= 0; j--) {
			temp = matrix[j][i];
			matrix[j][i] = 0;
			for (k = 0; k < col; k++) {
				invMat[j][k] -= invMat[i][k] * temp;
			}
		}
	}
	
	delMatrix(matrix);

	return invMat;
}
