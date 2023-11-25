#include "logger.h"
#include "time.h"

#ifdef __cplusplus
extern "C" {
#endif

static char log_levels[][9] = { "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "UNDEFINED"};
static char _timestamp[26];

void logger(FILE* file, const char* log, LOGLEVEL level)
{
	time_t current_time = time(NULL); 
	ctime_s(_timestamp, sizeof(_timestamp), &current_time);
	_timestamp[24] = '\0';
	fprintf_s(file, "[%s][%s]\t%s\n", _timestamp, log_levels[level], log);
}

#ifdef __cplusplus
}
#endif