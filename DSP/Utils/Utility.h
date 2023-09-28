#ifndef __UTILITY__
#define __UTILITY__

#include "SupportVector.hpp"
#include "Signals.h"
#include <map>
#include <vector>

using namespace SupportVector;
using namespace DSP::Signals;
using std::vector;

namespace DSP {
	namespace Utility {

		constexpr double M_PI = 3.14159265358979323846;

		signal diff(const signal& sig);

		signal cummSum(const signal& sig);
	}
}

#endif