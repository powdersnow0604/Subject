#include "logger.h"
#include "time.h"

#ifdef __cplusplus
extern "C" {
#endif


void logger(FILE* file, const char* log, LOGLEVEL level)
{
	static char log_levels[][9] = { "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "UNDEFINED" };

	time_t current_time;
	time(&current_time);

	char* timestamp = asctime(localtime(&current_time));
	timestamp[24] = '\0';

	fprintf(file, "[%s][%s]\t%s\n", timestamp, log_levels[level], log);
}

#ifdef __cplusplus
}
#endif