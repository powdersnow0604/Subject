#ifndef __UTILITY__
#define __UTILITY__

#include "DataModel.h"
#include <functional>

using namespace BasicAi::DataModels;

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
		#pragma region support vectorize operation	
		vector<double> operator+(const vector<double>& vec1, const vector<double>& vec2);
		vector<double> operator-(const vector<double>& vec1, const vector<double>& vec2);
		vector<double> operator*(const vector<double>& vec1, const vector<double>& vec2);
		vector<double> operator/(const vector<double>& vec1, const vector<double>& vec2);
		vector<double> operator+(const vector<double>& vec1, double scalar);
		vector<double> operator-(const vector<double>& vec1, double scalar);
		vector<double> operator*(const vector<double>& vec1, double scalar);
		vector<double> operator/(const vector<double>& vec1, double scalar);
		vector<double>& operator+=(vector<double>& vec1, const vector<double>& vec2);
		vector<double>& operator-=(vector<double>& vec1, const vector<double>& vec2);
		vector<double>& operator*=(vector<double>& vec1, const vector<double>& vec2);
		vector<double>& operator/=(vector<double>& vec1, const vector<double>& vec2);
		vector<double>& operator+=(vector<double>& vec1, double scalar);
		vector<double>& operator-=(vector<double>& vec1, double scalar);
		vector<double>& operator*=(vector<double>& vec1, double scalar);
		vector<double>& operator/=(vector<double>& vec1, double scalar);
		#pragma endregion

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

		template <typename T>
		double vector_sum(const vector<T>& vec) {
			double sum = 0;
			for (auto& i : vec) {
				sum += i;
			}
			return sum;
		}
	}
}
#endif
