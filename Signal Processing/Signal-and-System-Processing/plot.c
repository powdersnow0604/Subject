#include "plot.h"

void plot(char path[], char filePath[], char dataPath[], int dataSize, int samplingRate, int num, ...)
{
	int i, temp;
	char buf[8];
	va_list va;

	strcpy(path, filePath);
	strcat(path, " ");
	strcat(path, _itoa(dataSize, buf, 10));
	strcat(path, " ");
	strcat(path, dataPath);
	strcat(path, " ");
	strcat(path, _itoa(samplingRate, buf, 10));
	strcat(path, " ");
	strcat(path, _itoa(num, buf, 10));

	va_start(va, num);
	for (i = 0; i < num; i++) {
		temp = va_arg(va, int);
		strcat(path, " ");
		strcat(path, _itoa(temp, buf, 10));
	}
	va_end(va);

	system(path);

}
