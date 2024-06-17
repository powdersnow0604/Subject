#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <unistd.h>
#include "Profiler.h"

using namespace std::chrono;

int main(void)
{
    milliseconds sec = 0s;

    {
        Profiler<milliseconds> P(sec);

        sleep(5);
    }

    {
        Profiler<milliseconds> P(sec);

        sleep(2);
    }

    printf("Time has been %ld s\n", sec.count());

    return 0;
}