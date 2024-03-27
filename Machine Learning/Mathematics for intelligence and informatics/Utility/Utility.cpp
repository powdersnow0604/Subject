#include "Utility.h"
#include <algorithm>
#include <cmath>

namespace BasicAi {
	namespace Utility {

		DataModel StandardScaler::transform2(const DataModel& dm)
		{
			std::vector<std::vector<double>> scaled_input(dm.input->size(), std::vector<double>(dm[0].size()));
			auto mean_dm = mean(dm);
			auto stdev_dm = stdev(dm, mean_dm);

			for (size_t i = 0; i < dm.size; ++i) {
				for (size_t j = 0; j < dm[0].size(); ++j) {
					scaled_input[i][j] = (dm[i][j] - mean_dm[j]) / stdev_dm[j];
				}
			}

			return { scaled_input, *dm.target };
		}

		void StandardScaler::fit(const DataModel& Dm, bool include_target)
		{
			calc_min_max(Dm, include_target);
		}

		DataModel StandardScaler::transform(const DataModel& Dm, bool include_target)
		{
			DataModel res = Dm.copy();

			for (size_t i = 0; i < Dm.size; ++i) {
				for (size_t j = 0; j < Dm[0].size(); ++j) {
					res[i][j] = (res[i][j] - min[j]) / (max[j] - min[j]);
				}
				if (include_target) {
					res(i) = (res(i) - min.back()) / (max.back() - min.back());
				}
			}

			return res;
		}

		Vector StandardScaler::calc_min_max(const DataModel& Dm, bool include_target)
		{
			min.clear(); max.clear();
			if (include_target) {
				min = Dm[0]; min.push_back(Dm(0));
				max = Dm[0]; max.push_back(Dm(0));

				for (size_t i = 1; i < Dm.size; ++i) {
					for (size_t j = 0; j < Dm[0].size(); ++j) {
						if (min[j] > Dm[i][j]) min[j] = Dm[i][j];
						else if (max[j] < Dm[i][j]) max[j] = Dm[i][j];
					}
					if (min.back() > Dm(i)) min.back() = Dm(i);
					else if (max.back() < Dm(i)) max.back() = Dm(i);
				}
			}
			else {
				min = Dm[0];
				max = Dm[0];

				for (size_t i = 1; i < Dm.size; ++i) {
					for (size_t j = 0; j < Dm[0].size(); ++j) {
						if (min[j] > Dm[i][j]) min[j] = Dm[i][j];
						else if (max[j] < Dm[i][j]) max[j] = Dm[i][j];
					}
				}
			}

		}


		std::vector<double> BasicAi::Utility::mean(const DataModel& dm)
		{
			std::vector<double> sum(dm[0].size(), 0.);

			for (auto& i : *dm.input) {
				auto iter = i.cbegin();
				std::transform(sum.begin(), sum.end(), sum.begin(), [&iter](double& n) {return n + *iter++; });
			}

			std::transform(sum.begin(), sum.end(), sum.begin(), [&dm](double& n) {return n / dm.input->size(); });

			return sum;
		}

		std::vector<double> BasicAi::Utility::stdev(const DataModel& dm, const std::vector<double>& mean)
		{
			std::vector<double> sum(dm[0].size(), 0);
			std::vector<double> stdev(dm[0].size(), 0);

			for (auto& i : *dm.input) {
				auto iter = i.cbegin();
				auto iter_mean = mean.cbegin();
				std::transform(sum.begin(), sum.end(), sum.begin(), [&iter, &iter_mean](const double& n)
					//{return n + std::pow(*iter++ - *iter_mean++, 2); });
					{return n + (*iter++ - *iter_mean++) * (*iter - *iter_mean); });
			}

			std::transform(sum.begin(), sum.end(), stdev.begin(), [&dm](const double& n) {return std::sqrt(n / dm.size); });

			return stdev;
		}

		double BasicAi::Utility::EuclidianDistance(const std::vector<double>& point1, const std::vector<double>& point2)
		{
			if (point1.size() != point2.size()) return 999999999999999.;

			double sum = 0.;
			for (auto i = point1.begin(), j = point2.begin(); i != point1.end(); ++i, ++j)
			{
				sum += (*i - *j) * (*i - *j);
			}

			return std::sqrt(sum);
		}

		std::vector<double> BasicAi::Utility::absDiff(const std::vector<double>& point1, const std::vector<double>& point2)
		{
			if (point1.size() != point2.size()) return {};
			std::vector<double> res;
			res.reserve(point1.size());

			for (size_t i = 0; i < point1.size(); ++i) {
				res.push_back(std::abs(point1[i] - point2[i]));
			}

			return res;
		}
	}
}