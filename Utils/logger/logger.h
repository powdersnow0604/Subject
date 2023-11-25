#ifndef __LOGGER_H__
#define __LOGGER_H__

#include "stdio.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef enum LOGLEVEL_ {
	DEBUG,
	INFO,
	WARNING,
	ERROR,
	CRITICAL
}LOGLEVEL;

void logger(FILE* file, const char* log, LOGLEVEL level);

#ifdef __cplusplus
}
#endif

#endif