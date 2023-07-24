#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "complex_num.h"
#define window_size 4096      //data 개수 



static complex_num* FFT(complex_num* P);

static char path[260];
static int max = 0;
static int second = 0;

int main_FFT_radix_2(void)
{
	FILE* file = fopen("sample1.snd", "r+b");
	if (file == NULL)
	{
		perror("file open error");
	}
	char* data = (char*)malloc(sizeof(char) * window_size);
	complex_num* cxtypeData = (complex_num*)malloc(sizeof(complex_num) * window_size);
	double* amplitude = (double*)malloc(sizeof(double) * window_size);

	//read data
	fread(data, sizeof(char), window_size, file);
	fclose(file);


	//change type to complex_num
	for (int i = 0; i < window_size; i++)
	{
		cxtypeData[i].real = data[i];
		cxtypeData[i].imaginary = 0;
	}
	free(data);


	//calc FFT
	complex_num *FFTres = FFT(cxtypeData);


	//calc amplitude
	for (int i = 0; i < window_size; i++)
	{
		amplitude[i] = sqrt(pow(FFTres[i].real, 2) + pow(FFTres[i].imaginary, 2));
	}
	free(FFTres);

	//finding freq with max amplitude
	for (int i = 1; i <= window_size / 2; i++)
	{
		if (amplitude[i] > amplitude[max])
		{
			second = max;
			max = i;
		}
	}
	printf("\n[최대 크기 주파수: %lfHz,	크기: %lf]\n", (double)max * 8000 / window_size, amplitude[max]);


	//plot
	FILE* nfile = fopen("DFTres.txt", "w+");
	if (nfile == NULL)
	{
		perror("nfile error");
	}
	for (int i = 0; i < window_size; i++)
	{
		fprintf(nfile, "%lf\n", amplitude[i]);
	}
	fclose(nfile);

	strcpy(path, "DFTplot.py ");
	char buf[8];
	strcat(path, _itoa(window_size, buf, 10));
	strcat(path, " DFTres.txt ");
	strcat(path, _itoa(max, buf, 10));
	strcat(path, " ");
	strcat(path, _itoa(second, buf, 10));

	system(path);

	//free
	free(amplitude);
	free(cxtypeData);

	return 0;
}

static complex_num* FFT(complex_num* P)
{
	size_t n = _msize(P) / sizeof(complex_num);
	if (n == 1)
		return P;

	complex_num W;
	
	complex_num* evenTerm = (complex_num*)malloc(sizeof(complex_num) * (n / 2));
	complex_num* oddTerm = (complex_num*)malloc(sizeof(complex_num) * (n / 2));

	for (int i = 0; i < n / 2; i++)
	{
		evenTerm[i] = P[2 * i];
		oddTerm[i] = P[2 * i + 1];
	}

	complex_num* evenFFTRes = FFT(evenTerm);
	complex_num* oddFFTRes = FFT(oddTerm);
	complex_num* FFTRes = (complex_num*)malloc(sizeof(complex_num) * n);

	for (int i = 0; i < n / 2; i++)
	{
		W.real = cos(2 * M_PI / n * i);
		W.imaginary = -1 * sin(2 * M_PI / n * i);
		FFTRes[i] = complexAdd(evenFFTRes[i], complexMul(W, oddFFTRes[i]));
		FFTRes[i+n/2] = complexSub(evenFFTRes[i], complexMul(W, oddFFTRes[i]));
	}

	
	free(evenTerm);
	free(oddTerm);
	
	if (n != 2) {
		free(evenFFTRes);
		free(oddFFTRes);
	}

	return FFTRes;
}

