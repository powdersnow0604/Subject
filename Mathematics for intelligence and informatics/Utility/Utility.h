#ifndef __UTILITY__
#define __UTILITY__

#include "DataModel.h"
#include "SupportVector.hpp"
#include <functional>

using namespace BasicAi::DataModels;
using namespace SupportVector;

namespace BasicAi {
	namespace Utility {
		//class
		class StandardScaler {
		public:
			static DataModel transform(const DataModel& dm);
		};

		//fuction
		vector<double> mean(const DataModel& dm);
		vector<double> stdev(const DataModel& dm, const std::vector<double>& mean);
		double EuclidianDistance(const vector<double>& point1, const vector<double>& point2);
		vector<double> absDiff(const vector<double>& point1, const vector<double>& point2);
		

		//templates
		template <typename iter>
		iter max(iter start, iter end, std::function<bool(const iter&, const iter&)> comp = 
			[](const iter& opr1, const iter& opr2) const -> constexpr bool{ return *opr1 > *opr2 })
		{
			iter max = start;
			
			for (auto i = ++start; i != end; ++i) {
				if (!comp(max, i)) max = i;
			}

			return max;
		}

		template <typename iter>
		iter min(iter start, iter end, std::function<bool(const iter&, const iter&)> comp = 
			[](const iter& opr1, const iter& opr2) const -> constexpr bool{ return *opr1 < *opr2 })
		{
			iter min = start;
			for (auto i = ++start; i != end; ++i) {
				if (!comp(min, i)) min = i;
			}

			return min;
		}

		
	}
}
#endif
