#ifndef __UTILITY__
#define __UTILITY__

#include "DataModel.h"

using namespace BasicAi::DataModels;

namespace BasicAi {
	namespace Utility {

		class StandardScaler {
		public:
			static DataModel transform(DataModel& dm);
		};

		std::vector<double> mean(const DataModel& dm);
		std::vector<double> stdev(const DataModel& dm, const std::vector<double>& mean);
		double EuclidianDistance(const std::vector<double>& point1, const std::vector<double>& point2);
		std::vector<double> absDiff(const std::vector<double>& point1, const std::vector<double>& point2);
	}
}
#endif
