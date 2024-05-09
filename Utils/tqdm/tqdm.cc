#include "tqdm.h"
#include <sstream>

namespace tqdmHELPER {
    std::string toNumUnitString(float X)
    {
        static const char *arr[] = {"K", "M", "B", "T", "Q"};
        uint8_t cnt = 0;
        std::ostringstream out;
        out.precision(2);

        while (X > 1000.f && cnt < 5)
        {
            cnt++;
            X /= 1000.f;
        }

        if (X > 1000.f)
            cnt++;

        out << std::fixed << X;

        switch (cnt)
        {
        case 0:
            return out.str();
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
            return out.str() + arr[cnt - 1];
        default:
            return out.str() + "Q+";
        }
    }
}

