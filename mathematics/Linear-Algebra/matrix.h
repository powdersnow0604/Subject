#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <stdlib.h>

typedef struct {
	int* row;
	int* column;
	int rank;
}pivot;

double** mkMatrix(size_t row, size_t column);
void delMatrix(double** matrix);
void printMatrix(double** matrix);

double** rref(double** matrix, int* check);
double** leastSquareMethod(double** matrix);

double** matrixMul(double** a, double** b);
double** matrixAdd(double** a, double** b);
double** matrixScalarMul(double** matrix, double Scalar);
double** transpose(double** matrix);
double** augment(double** A, double** c);
double** decomposeAugment(double** matrix, double*** c);
double determinant(double** matrix);
double** inverse(double** matrix);

#endif
