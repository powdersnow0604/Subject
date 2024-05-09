#ifndef __PROFILER_H__
#define __PROFILER_H__

#include <chrono>
#include <string>
#include <cstdio>
#include <map>

#define PREFIX std::chrono

namespace ProfilerHELPER
{
    template <typename Unit>
    class TimeUnit
    {
    public:
        static const char *postfix() { return "ns"; }
        static double weight() { return 1.; }
    };

    template <>
    class TimeUnit<PREFIX::nanoseconds>
    {
    public:
        static const char *postfix() { return "ns"; }
        static double weight() { return 1.; }
    };

    template <>
    class TimeUnit<PREFIX::microseconds>
    {
    public:
        static const char *postfix() { return "us"; }
        static double weight() { return 1.e-3; }
    };

    template <>
    class TimeUnit<PREFIX::milliseconds>
    {
    public:
        static const char *postfix() { return "ms"; }
        static double weight() { return 1.e-6; }
    };

    template <>
    class TimeUnit<PREFIX::seconds>
    {
    public:
        static const char *postfix() { return "s"; }
        static double weight() { return 1.e-9; }
    };

    class ProfilerBase
    {
    protected:
        PREFIX::nanoseconds total;
        uint32_t cnt;

    public:
        ProfilerBase() : total(0), cnt(0) {}

        template <typename Unit>
        double getAverage()
        {
            return total.count() * ProfilerHELPER::TimeUnit<Unit>::weight() / cnt;
        }

        template <typename Unit>
        double getTotal()
        {
            return total.count() * ProfilerHELPER::TimeUnit<Unit>::weight();
        }

        PREFIX::nanoseconds &getTotal_()
        {
            return total;
        }

        uint32_t getCnt()
        {
            return cnt;
        }

        uint32_t &getCnt_()
        {
            return cnt;
        }
    };

}

class Profiler : public ProfilerHELPER::ProfilerBase
{
    //////////////////// private class definition ////////////////////

    class TemporalProfiler
    {
        PREFIX::nanoseconds &total;
        PREFIX::nanoseconds *total_named;
        PREFIX::_V2::system_clock::time_point start;
        uint32_t &cnt;
        uint32_t *cnt_named;

    public:
        TemporalProfiler(PREFIX::nanoseconds &__total, uint32_t &__cnt) : total(__total), start(PREFIX::high_resolution_clock::now()),
                                                                          cnt(__cnt), total_named(NULL), cnt_named(NULL) {}
        TemporalProfiler(PREFIX::nanoseconds &__total, uint32_t &__cnt, PREFIX::nanoseconds *__total_named, uint32_t *__cnt_named) : total(__total), start(PREFIX::high_resolution_clock::now()),
                                                                                                                                     cnt(__cnt), total_named(__total_named), cnt_named(__cnt_named) {}
        ~TemporalProfiler() noexcept
        {
            auto temp = PREFIX::high_resolution_clock::now() - start;
            total += temp;
            cnt++;

            if (total_named)
                *total_named += temp;
            if (cnt_named)
                (*cnt_named)++;
        }
    };

    //////////////////// end of definition ////////////////////

    PREFIX::_V2::system_clock::time_point start;
    std::string name;
    std::map<std::string, ProfilerBase> namedP;
    bool not_started;

public:
    Profiler() : ProfilerBase() {}
    Profiler(std::string __name) : ProfilerBase(), name(__name), not_started(true) {}

    void onTimer()
    {
        start = PREFIX::high_resolution_clock::now();
        not_started = false;
    }

    void offTimer()
    {
        if (not_started)
            return;

        total += PREFIX::high_resolution_clock::now() - start;
        cnt++;
        not_started = true;
    }

    void offTimer(std::string profiling_name)
    {
        if (not_started)
            return;

        auto temp = PREFIX::high_resolution_clock::now() - start;
        total += temp;
        cnt++;
        namedP[profiling_name].getTotal_() += temp;
        namedP[profiling_name].getCnt_()++;
        not_started = true;
    }

    TemporalProfiler profile()
    {
        return TemporalProfiler(total, cnt);
    }

    TemporalProfiler profile(std::string profiling_name)
    {
        return TemporalProfiler(total, cnt, &namedP[profiling_name].getTotal_(), &namedP[profiling_name].getCnt_());
    }

    void setName(std::string nName)
    {
        name = nName;
    }

    template <typename Unit = PREFIX::milliseconds>
    void report(bool print_average = false)
    {
        if (name.length())
            std::printf("\n**** Profiler [%s] Report ****\n", name.c_str());
        else
            std::puts("\n**** Profiler report ****");

        std::printf("%-19s: %d\n", "Number of Profiling", cnt);

        if (print_average)
            std::printf("%-19s: %.4lf %s\n", "average time", this->getAverage<Unit>(), ProfilerHELPER::TimeUnit<Unit>::postfix());

        std::printf("%-19s: %.4lf %s\n", "total time", this->getTotal<Unit>(), ProfilerHELPER::TimeUnit<Unit>::postfix());

        for (auto iter = namedP.begin(); iter != namedP.end(); iter++)
        {
            printf(" - %s: %.4lf %s (%d)\n", iter->first.c_str(), iter->second.getTotal<Unit>(), ProfilerHELPER::TimeUnit<Unit>::postfix(), iter->second.getCnt());
        }

        std::puts("**** End of the report ****\n");
    }
};

#undef PREFIX

#endif