#ifndef __TQDM_H__
#define __TQDM_H__

#include <type_traits>
#include <stdint.h>
#include <iostream>
#include <string>
#include <cstring>
#include "Profiler.h"
#include <cmath>
#include <limits>

namespace tqdmHELPER
{
    typedef uint32_t NUMTYPE;
    typedef uint8_t STEPTYPE;

    template <class T, class = void>
    struct num_iter
    {
        NUMTYPE num;
        num_iter(T &iterable) : num(0)
        {
            for (auto iter = iterable.begin(); iter != iterable.end(); iter++)
            {
                num++;
            }
        }
    };

    template <class T>
    struct num_iter<T, std::enable_if_t<std::is_integral<decltype(std::declval<T>().size())>::value, void>>
    {
        NUMTYPE num;
        num_iter(T &iterable) : num(iterable.size()) {}
    };

    template <class, class = void>
    struct has_begin : std::false_type
    {
    };

    template <class T>
    struct has_begin<T, std::enable_if_t<std::is_class<decltype(std::declval<T>().begin())>::value, void>> : std::true_type
    {
    };

    template <class, class = void>
    struct has_end : std::false_type
    {
    };

    template <class T>
    struct has_end<T, std::enable_if_t<std::is_class<decltype(std::declval<T>().end())>::value, void>> : std::true_type
    {
    };

    std::string toNumUnitString(float X);
}

template <typename Iterable, typename Unit = std::chrono::seconds,
          std::enable_if_t<tqdmHELPER::has_begin<Iterable>::value, bool> = true,
          std::enable_if_t<tqdmHELPER::has_end<Iterable>::value, bool> = true>
class tqdm
{
    typedef tqdmHELPER::NUMTYPE NUMTYPE;
    typedef tqdmHELPER::STEPTYPE STEPTYPE;

    Iterable &iterable;
    NUMTYPE num;
    float step;
    STEPTYPE print_inc;
    const STEPTYPE total_step;
    const char *desc;

    class Iterator;

public:
    tqdm(Iterable &__iterable, STEPTYPE __step = 20, const char *__desc = NULL) : iterable(__iterable), num(tqdmHELPER::num_iter<Iterable>(iterable).num),
                                                                                  total_step(__step), desc(__desc)
    {
        if (total_step > num)
        {
            print_inc = total_step / num;
            step = 1.f;
        }
        else
        {
            print_inc = 1;
            step = (float)num / total_step;
        }
    }

    Iterator begin()
    {
        return Iterator(iterable.begin(), num, step, print_inc, total_step, desc, true);
    }

    Iterator end()
    {
        return Iterator(iterable.end(), num, step, print_inc, total_step, desc, false);
    }
};

#endif

template <typename Iterable, typename Unit,
          std::enable_if_t<tqdmHELPER::has_begin<Iterable>::value, bool> J,
          std::enable_if_t<tqdmHELPER::has_end<Iterable>::value, bool> K>
class tqdm<Iterable, Unit, J, K>::Iterator
{
    const char *ascii = "\u2588";
    decltype(std::declval<Iterable>().begin()) iter;
    const NUMTYPE num;
    const float step;
    const STEPTYPE print_inc;
    const STEPTYPE total_step;
    STEPTYPE curr_pos;
    NUMTYPE curr_iter;
    Profiler profiler;
    const char *desc;
    float next_step;
    std::string str;
    std::string margin;

    void print()
    {
        if (desc)
        {
            std::printf("\n%s: %3u%% |%s%s| %u/%u [%sit/%s]\n", desc, 100 * curr_iter / num, str.c_str(), margin.c_str(), curr_iter, num,
                        tqdmHELPER::toNumUnitString(profiler.getCnt() / ((float)profiler.getTotal<Unit>() + std::numeric_limits<float>::epsilon())).c_str(),
                        ProfilerHELPER::TimeUnit<Unit>::postfix());
        }
        else
        {
            std::printf("\n%3u%% |%s%s| %u/%u [%sit/%s]\n", 100 * curr_iter / num, str.c_str(), margin.c_str(), curr_iter, num,
                        tqdmHELPER::toNumUnitString(profiler.getCnt() / ((float)profiler.getTotal<Unit>() + std::numeric_limits<float>::epsilon())).c_str(),
                        ProfilerHELPER::TimeUnit<Unit>::postfix());
        }
    }

public:
    Iterator(decltype(std::declval<Iterable>().begin()) __iter, NUMTYPE __num, float __step, STEPTYPE __print_inc,
             STEPTYPE __total_step, const char *__desc, bool is_begin) : iter(__iter), num(__num), step(__step), print_inc(__print_inc),
                                                                         curr_pos(0), curr_iter(0), total_step(__total_step),
                                                                         desc(__desc), next_step(__step), margin(total_step, ' ')
    {
        if (is_begin)
        {
            str.reserve((std::strlen(ascii) - 1) * total_step);
            print();
            str += ascii;
            margin.pop_back();
            profiler.onTimer();
        }
    }

    // ~Iterator() noexcept
    // {
    //     profiler.offTimer();
    // }

    bool operator==(const Iterator &other)
    {
        return iter == other.iter;
    }

    bool operator==(const decltype(std::declval<Iterable>().begin()) &other)
    {
        return iter == other;
    }

    bool operator!=(const Iterator &other)
    {
        return !(iter == other.iter);
    }

    bool operator!=(const decltype(std::declval<Iterable>().begin()) &other)
    {
        return !(iter == other);
    }

    decltype(*iter) &operator*()
    {
        return *iter;
    }

    decltype(iter.operator->()) operator->()
    {
        return iter.operator->();
    }

    Iterator &operator++()
    {
        profiler.offTimer();
        ++iter;

        // if (curr_iter == 0)
        // {
        //     print();
        //     ++curr_iter;
        //     str += ascii;
        //     margin.pop_back();
        //     profiler.onTimer();
        //     return *this;
        // }

        ++curr_iter;

        if (curr_iter == num)
        {
            curr_pos = total_step;
            print();
        }
        else if (curr_iter >= lround(next_step))
        {
            curr_pos += print_inc;
            print();
            next_step += step;
            str += ascii;
            margin.pop_back();
        }

        profiler.onTimer();
        return *this;
    }

    Iterator operator++(int _)
    {
        auto temp = *this;
        this->operator++();
        return temp;
    }
};
