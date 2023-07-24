#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int matrix1[4][4] = { 1,3,0,5,
							 0,0,0,0,
							 0,0,0,0,
							 0,3,0,0 };

static int matrix2[4][4] = { 0,6,0,7,
							 0,0,0,0,
							 0,0,0,2,
							 0,0,9,0 };


typedef struct {
	int column;
	int data;
}coordinate;

typedef struct {
	coordinate* value;
	int* row;
	int rnum;
	int cnum;
}csrform;

static void printCsr(csrform csr);
static void freeCsr(csrform* csr);
static csrform csr(int* a,int r, int c);
static csrform sparseAddition(csrform a, csrform b);
static csrform sparseMultiplication(csrform a, csrform b);
static void insert(coordinate* temp, int *cnt, int data, int column);
static csrform sparseTranspose(csrform a);

int main_sparse(void)
{
	csrform csr1 = csr((int*)matrix1, sizeof(matrix1)/sizeof(matrix1[0]), sizeof(matrix1[0]) / sizeof(int));
	csrform csr2 = csr((int*)matrix2, sizeof(matrix2) / sizeof(matrix2[0]), sizeof(matrix2[0]) / sizeof(int));
	csrform add = sparseAddition(csr1, csr2);
	csrform mul = sparseMultiplication(csr1, csr2);
	csrform tp = sparseTranspose(csr2);

	printCsr(add);
	printCsr(mul);
	printCsr(tp);

	freeCsr(&csr1);
	freeCsr(&csr2);
	freeCsr(&add);
	freeCsr(&mul);
	freeCsr(&tp);

	return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////
static void printCsr(csrform csr)
{
	int row = 0;
	int index = 0;
	for (int i = 0; i < csr.rnum; i++)
	{
		for (int j = 0; j < csr.cnum; j++)
		{
			while (index == csr.row[row + 1]) row++;

			if (row == i && csr.value[index].column == j)
			{
				printf("%d ", csr.value[index++].data);
			}
			else
			{
				printf("0 ");
			}
		}
		puts("");
	}
	puts("");
}

static void freeCsr(csrform* csr)
{
	free(csr->row);
	free(csr->value);
}

static csrform csr(int* a, int r, int c)
{
	csrform temp;
	temp.cnum = c; 
	temp.rnum = r; 
	temp.value = (coordinate*)malloc(sizeof(coordinate) * (temp.cnum*temp.rnum));
	temp.row = (int*)malloc(sizeof(int) * (temp.rnum+1));

	int index = 0;
	for (int i = 0; i < temp.rnum; i++)
	{
		temp.row[i] = index;
		for (int j = 0; j < temp.cnum; j++)
		{
			if (a[i*r+j] != 0)
			{
				temp.value[index].column = j;
				temp.value[index++].data = a[i*r+j];
			}
		}
	}
	temp.row[temp.rnum] = index;

	return temp;
}

static csrform sparseAddition(csrform a, csrform b)
{
	if (a.cnum != b.cnum || a.rnum != b.rnum)
	{
		puts("dimention of matrixes does not match");
		puts("return fisrt argument");
		return a;
	}

	csrform add;
	add.rnum = a.rnum;	add.cnum = b.cnum;
	add.value = (coordinate*)malloc((a.rnum * b.cnum) * sizeof(coordinate));
	add.row = (int*)calloc((add.rnum+1), sizeof(int));

	int indexa = 0;
	int indexb = 0;
	int arow = 0;
	int brow = 0;
	int indexadd = 0;

	while (indexa < a.row[a.rnum] && indexb < b.row[b.rnum])
	{
		while (indexa == a.row[arow + 1])
		{
			arow++;
		}
		while (indexb == b.row[brow + 1])
		{
			brow++;
		}
		if (arow * a.cnum + a.value[indexa].column < brow * b.cnum + b.value[indexb].column)
		{
			add.row[arow + 1]++;
			add.value[indexadd++] = a.value[indexa++];
		}
		else if (arow * a.cnum + a.value[indexa].column > brow * b.cnum + b.value[indexb].column)
		{
			add.row[brow + 1]++;
			add.value[indexadd++] = b.value[indexb++];
		}
		else
		{
			add.row[brow + 1]++;
			add.value[indexadd].column = b.value[indexb].column;
			add.value[indexadd++].data = b.value[indexb++].data + a.value[indexa++].data;
		}
	}

	if (indexa >= a.row[a.rnum])
	{
		for (int i = indexb; i < b.row[b.rnum]; i++)
		{
			while (indexb == b.row[brow + 1])
			{
				brow++;
			}
			add.row[brow + 1]++;
			add.value[indexadd++] = b.value[i];
		}
	}
	if (indexb >= b.row[b.rnum])
	{
		for (int i = indexa; i < a.row[a.rnum]; i++)
		{
			while (indexa == a.row[arow + 1])
			{
				arow++;
			}
			add.row[arow + 1]++;
			add.value[indexadd++] = a.value[i];
		}
	}

	for (int i = 1; i <= add.rnum; i++)
	{
		add.row[i] += add.row[i - 1];
	}

	return add;
}

static csrform sparseMultiplication(csrform a, csrform b)
{
	if (a.cnum != b.rnum) return a;

	csrform mul;
	mul.rnum = a.rnum;	mul.cnum = b.cnum;
	mul.value = (coordinate*)malloc((a.rnum * b.cnum) * sizeof(coordinate));
	mul.row = (int*)calloc((mul.rnum + 1), sizeof(int));

	coordinate* temp = (coordinate*)malloc(sizeof(coordinate) * mul.cnum);
	int indexrow = 0;
	int cnt = 0;
	while (indexrow < mul.rnum)
	{
		cnt = 0;
		memset(temp, 0, sizeof(coordinate) * mul.cnum);
		for (int i = a.row[indexrow]; i <  a.row[indexrow + 1]; i++)
		{
			for (int j = b.row[a.value[i].column]; j < b.row[a.value[i].column + 1]; j++)
			{
				insert(temp, &cnt, a.value[i].data * b.value[j].data, b.value[j].column);
			}

		}
		mul.row[indexrow + 1] = cnt;
		if (indexrow != 0)
		{
			mul.row[indexrow + 1] += mul.row[indexrow];
		}
		memcpy(mul.value+mul.row[indexrow++], temp, sizeof(coordinate) * cnt);
	}
	free(temp);

	return mul;
}

static void insert(coordinate* rowarr, int *cnt, int data, int column)
{
	coordinate temp = { column, data };
	/*if (cnt == 0)
	{
		rowarr[0] = temp;
		return;
	}*/

	int i = 0;
	for (; i < *cnt; i++)
	{
		if (rowarr[i].column == temp.column)
		{
			rowarr[i].data += data;
			return;
		}
		else if (rowarr[i].column > temp.column)
		{
			break;
		}
	}

	for (int j = *cnt - 1; j >= i; j--)
	{
		rowarr[j + 1] = rowarr[j];
	}

	rowarr[i] = temp;
	(*cnt)++;
}

static csrform sparseTranspose(csrform a)
{
	csrform tx;
	tx.cnum = a.rnum; tx.rnum = a.cnum;
	tx.value = (coordinate*)malloc(_msize(a.value));
	tx.row = (int*)calloc(tx.rnum+1, sizeof(int));
	int* temp = (int*)malloc(sizeof(int)*a.row[a.rnum]);

	int index = 0;
	int indexrow = 1;
	for (index = a.row[0]; index < a.row[1]; index++)
	{
		tx.value[index] = a.value[index];
		temp[index] = 0;
		tx.row[tx.value[index].column+1]++;
	}
	while (indexrow < a.rnum)
	{
		for (int i = a.row[indexrow]; i < a.row[indexrow + 1]; i++)
		{
			int j = 0;
			for (; j < index; j++)
			{
				if (a.value[i].column < tx.value[j].column)
				{
					break;
				}
			}

			for (int k = index-1;  k>= j; k--)
			{
				tx.value[k + 1] = tx.value[k];
				temp[k + 1] = temp[k];
			}

			tx.value[j] = a.value[i];
			temp[j] = indexrow;
			tx.row[tx.value[j].column + 1]++;
			index++;
		}
		indexrow++;
	}

	for (int i = 1; i <= tx.rnum; i++)
	{
		tx.row[i] += tx.row[i - 1];
	}

	for (int i = 0; i <= tx.row[tx.rnum]; i++)
	{
		tx.value[i].column = temp[i];
	}

	free(temp);
	return tx;
}
