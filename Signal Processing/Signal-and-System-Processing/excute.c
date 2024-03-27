#pragma warning(disable : 4996)
#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include "DFT.h"
#define FILEINPUTl

const int window_size = 8;     //data size 
const int frequencyRate = 8000;   //frequecy rate
typedef double type;              //choose data type

#pragma region declaration
#define checkType(x) _Generic((x),	char : "char", short : "short", \
									int : "int", long : "long", \
									float : "float", long long : "long long", \
									double : "double")

static char path[260];
static int max = 0;
static int second = 0;
#pragma endregion

int main_excute(void)
{
	complex_num* FFTRes;
	double* amplitude;

	//making data
	#ifdef FILEINPUT
	//file open
	FILE* file = fopen("sample1.snd", "r+b");
	if (file == NULL)
	{
		perror("file open error");
	}
	type* data = (type*)malloc(sizeof(type) * window_size);
	
	//read data
	fread(data, sizeof(type), window_size, file);
	fclose(file);
	#pragma endregion
	#else
	//user-customized function
	type* data = (type*)malloc(sizeof(type) * window_size);
	for (int i = 0; i < window_size; i++)
	{
		data[i] = 5* cos(M_PI/2*i) + 5 * cos(M_PI * 3 / 4 * i);
	}
	#endif

	//change type to complex_num
	type check = 0;
	FFTRes = data2complex(data, window_size, checkType(check));
	free(data);


	//calc FFT
	clock_t start = clock();

	FFT_radix_2(FFTRes);        //change here for another version of FFT
	
	if (errno == ENOMEM) {
		perror("오류 발생");
		return 0;
	}
	printf("time passed: %ld ms\n", clock() - start);


	//calc amplitude
	amplitude = calcAmplitude(FFTRes);
	free(FFTRes);


	//finding freq with max amplitude
	#pragma region Max Amplitude
	for (int i = 1; i <= window_size -1; i++)
	{
		if (amplitude[i] > amplitude[max])
		{
			second = max;
			max = i;
		}
	}
	printf("\n[최대 크기 주파수: %lfHz,	크기: %lf]\n", (double)max * frequencyRate / window_size, amplitude[max]);
	#pragma endregion


	//store
	#pragma region Store
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
	#pragma endregion
	

	//plot
	plot(path, "DFTplot.py", "DFTres.txt", window_size, frequencyRate, 2, max, second);


	//free
	free(amplitude);

	return 0;
}

