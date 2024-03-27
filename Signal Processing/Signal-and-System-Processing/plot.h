#ifndef  __PLOT_H__
#define  __PLOT_H__

#define _CRT_SECURE_NO_WARNINGS
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

void plot(char path[], char filePath[], char dataPath[], int dataSize, int samplingRate, int num, ...);

#endif